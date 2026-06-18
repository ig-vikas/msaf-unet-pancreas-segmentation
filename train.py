"""Train and resume the same experiment for each model variant."""

import json
import os
import random
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from pickle import UnpicklingError

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

import config
from dataset import TrainingSlices, audited_data_digest, read_splits, verify_dataset
from evaluate import CLASSES, predict_and_score, write_table
from model import PancreasUNet, VARIANTS
from predict import load_model
from prepare_data import file_digest, split_digest, write_csv, write_json


def segmentation_loss(logits, labels):
    cross_entropy = F.cross_entropy(logits, labels)
    probabilities = logits.float().softmax(dim=1)
    targets = F.one_hot(labels, num_classes=3).movedim(-1, 1)
    probabilities, targets = probabilities[:, 1:], targets[:, 1:]
    dims = (0, 2, 3)
    overlap = (probabilities * targets).sum(dims)
    total = probabilities.sum(dims) + targets.sum(dims)
    present = targets.sum(dims) > 0
    if present.any():
        dice_loss = 1 - ((2 * overlap + 1e-5) / (total + 1e-5))[present].mean()
        return (cross_entropy + dice_loss) / 2
    return cross_entropy


def save_checkpoint(payload, path):
    """Write a complete file before replacing a checkpoint."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as handle:
        temporary = Path(handle.name)
        try:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        except BaseException:
            handle.close()
            temporary.unlink(missing_ok=True)
            raise
    try:
        checked = torch.load(temporary, map_location="cpu", weights_only=True)
        if checked["epoch"] != payload["epoch"]:
            raise ValueError("The new checkpoint could not be verified.")
        del checked
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def validate(model, root, cases, device):
    dice = {name: [] for name in CLASSES.values()}
    for case in cases:
        rows, _, _, _, _ = predict_and_score(model, root, case, device)
        for row in rows:
            if row["dice"] is not None:
                dice[row["label"]].append(row["dice"])
    if not all(dice.values()):
        raise ValueError("Validation must include each foreground class before ranking checkpoints.")
    return {name: float(np.mean(values)) for name, values in dice.items()}


def check_baseline_learning(root, device):
    """Repeat a small real batch to verify that the baseline can learn."""
    require_training_device(device)
    require_preflight()
    torch.manual_seed(config.SPLIT_SEED)
    train_cases = verify_dataset(root, ("train",))["train"]
    case = next((case for case in train_cases if case["tumor_group"] != "absent"), None)
    if case is None:
        raise ValueError("The learning check needs a training case with a reference tumour.")
    samples = TrainingSlices(root, [case], config.SPLIT_SEED, 0)
    first = samples[0]
    positive = None
    for index in range(len(samples)):
        sample = samples[index]
        if (sample[1] == 2).any():
            positive = sample
            break
    if positive is None:
        raise ValueError(f"{case['case_id']} has no tumour in the sampled slices.")
    images, masks = zip(first, positive)
    images = torch.stack(images).to(device)
    masks = torch.stack(masks).to(device)
    model = PancreasUNet("unet").to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)
    losses = []
    for _ in range(12):
        optimizer.zero_grad(set_to_none=True)
        loss = segmentation_loss(model(images), masks)
        if not torch.isfinite(loss):
            raise ValueError("The repeated-batch check produced a non-finite loss.")
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
    with torch.inference_mode():
        final_loss = segmentation_loss(model(images), masks).item()
    if not np.isfinite(final_loss) or final_loss >= losses[0]:
        raise ValueError("The baseline loss did not fall on the repeated batch.")
    return {"case_id": case["case_id"], "steps": len(losses),
            "initial_loss": losses[0], "final_loss": final_loss}


def experiment_settings(variant, seed, pilot):
    return {
        "architecture_version": config.ARCHITECTURE_VERSION,
        "source_sha256": config.source_digest(),
        "ct_window": list(config.CT_WINDOW),
        "variant": variant,
        "seed": seed,
        "pilot": pilot,
        "split_sha256": split_digest(),
        "data_summary_sha256": audited_data_digest(),
        "slices_per_case": config.SLICES_PER_CASE,
        "batch_size": config.BATCH_SIZE,
        "learning_rate": config.LEARNING_RATE,
        "weight_decay": config.WEIGHT_DECAY,
        "amp": config.AMP,
    }


def export_training_files(state, folder):
    """Rebuild the best weights and log from the latest committed epoch."""
    best = {
        "model": state["best_model"],
        "epoch": state["best_epoch"],
        "validation_score": state["best_score"],
        **state["settings"],
    }
    best_path = folder / "best.pt"
    matches = False
    if best_path.is_file():
        try:
            previous = torch.load(best_path, map_location="cpu", weights_only=True)
            matches = (
                isinstance(previous, dict)
                and all(previous.get(key) == best[key] for key in best if key != "model")
                and previous["model"].keys() == best["model"].keys()
                and all(torch.equal(previous["model"][key], tensor)
                        for key, tensor in best["model"].items())
            )
        except (OSError, AttributeError, KeyError, TypeError, RuntimeError, ValueError, EOFError, UnpicklingError):
            pass
    if not matches:
        save_checkpoint(best, best_path)
    write_csv(folder / "epochs.csv", state["history"])
    write_json(folder / "settings.json", {**state["settings"], **state["environment"]})


def run_name(variant, seed, pilot=False):
    return f"{'pilot_' if pilot else ''}{variant}_s{seed}"


def require_preflight():
    path = config.RESULTS_DIR / "preflight" / "checks.json"
    if not path.is_file():
        raise FileNotFoundError("Run preflight and review the preparation before training.")
    checks = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "source_sha256": config.source_digest(),
        "split_sha256": split_digest(),
        "data_summary_sha256": audited_data_digest(),
    }
    if any(checks.get(key) != value for key, value in expected.items()):
        raise ValueError("The code, audit or case lists changed after preflight. Repeat the checks.")


def require_training_device(device):
    if not (config.IN_COLAB or config.REMOTE_RUN):
        raise RuntimeError("Use the training commands on the remote machine.")
    if torch.device(device).type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Training needs a CUDA-enabled PyTorch install and an NVIDIA GPU.")


def experiment_runs(include_repeats=False):
    runs = [(variant, config.SPLIT_SEED) for variant in VARIANTS]
    if include_repeats:
        runs += [(variant, seed) for variant in ("unet", "attention_fusion") for seed in (43, 44)]
    return runs


def require_pilot():
    path = config.RESULTS_DIR / "pilot.json"
    if not path.is_file():
        raise FileNotFoundError("Finish the training pilot before starting the comparison.")
    report = json.loads(path.read_text(encoding="utf-8"))
    settings = experiment_settings("attention_fusion", config.SPLIT_SEED, True)
    if not report.get("baseline_resumed") or any(
        report["settings"].get(key) != value for key, value in settings.items()
    ):
        raise ValueError("Repeat the pilot with the current code, audited data and settings.")
    return report


def run_pilot(root, device):
    """Check learning, resume a baseline epoch and time the full model."""
    learning = check_baseline_learning(root, device)
    baseline = fit("unet", root, device, epochs=1, pilot=True)
    fit("unet", root, device, epochs=2, pilot=True)
    state = torch.load(baseline / "last.pt", map_location="cpu", weights_only=True)
    if state["epoch"] != 2 or state["history"][1]["resumed_from"] != 1:
        raise RuntimeError("The second pilot epoch did not resume from epoch one.")
    del state

    folder = fit("attention_fusion", root, device, epochs=1, pilot=True)
    saved = torch.load(folder / "last.pt", map_location="cpu", weights_only=True)
    model, _ = load_model(folder / "best.pt", device)
    del model
    cases = read_splits()
    timing = saved["history"][-1]
    training_time = timing["training_seconds"]
    validation_time = timing["seconds"] - training_time
    training_estimate = training_time * len(cases["train"]) / len(saved["settings"]["train_cases"])
    validation_estimate = validation_time * len(cases["validation"]) / len(saved["settings"]["validation_cases"])
    report = {
        "split_sha256": split_digest(),
        "settings": saved["settings"],
        "learning": learning,
        "baseline_resumed": True,
        "full_epoch": timing,
        "estimated_full_split_epoch_minutes": round((training_estimate + validation_estimate) / 60, 1),
        "estimate_method": "Scale training and validation separately by case count. Scan sizes and runtime vary.",
        "gpu": saved["environment"]["gpu"],
    }
    write_json(config.RESULTS_DIR / "pilot.json", report)
    return report


def comparison_plan(experiments, epochs, create=False):
    """Save the shared run list and epoch budget before the first fit."""
    if not isinstance(epochs, int) or isinstance(epochs, bool) or epochs < 1:
        raise ValueError("Choose a positive number of epochs.")
    if len(set(experiments)) != len(experiments) or not {(name, 42) for name in VARIANTS}.issubset(experiments):
        raise ValueError("Include each seed-42 variant once in the comparison.")
    settings = experiment_settings("attention_fusion", 42, False)
    shared = {key: value for key, value in settings.items() if key not in ("variant", "seed", "pilot")}
    plan = {
        "epochs": epochs,
        "settings": shared,
        "runs": [{"variant": variant, "seed": seed} for variant, seed in experiments],
    }
    path = config.RESULTS_DIR / "comparison_plan.json"
    if create:
        try:
            with path.open("x", encoding="utf-8") as handle:
                json.dump(plan, handle, indent=2)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
        except FileExistsError:
            pass
    if not path.is_file():
        raise FileNotFoundError("Start the comparison training runs first.")
    if json.loads(path.read_text(encoding="utf-8")) != plan:
        raise ValueError("The epoch budget, run list or settings changed. Use results/comparison_plan.json.")
    return plan


def run_comparison(experiments, run_index, root, device, epochs):
    require_training_device(device)
    require_preflight()
    require_pilot()
    comparison_plan(experiments, epochs, create=True)
    variant, seed = experiments[run_index]
    try:
        return fit(variant, root, device, epochs=epochs, seed=seed)
    except (Exception, KeyboardInterrupt) as exc:
        failure = {
            "time_utc": datetime.now(timezone.utc).isoformat(),
            "run": run_name(variant, seed),
            "target_epochs": epochs,
            "error": f"{type(exc).__name__}: {exc}",
        }
        with (config.RESULTS_DIR / "run_errors.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(failure) + "\n")
        raise


def fit(variant, root, device, epochs=config.TRAINING_EPOCHS, seed=42, pilot=False):
    require_training_device(device)
    device = torch.device(device)
    if not isinstance(epochs, int) or epochs < 1:
        raise ValueError("Epochs must be a positive integer.")
    if config.SLICES_PER_CASE % 3 or config.SLICES_PER_CASE % config.BATCH_SIZE:
        raise ValueError("Choose slices per case divisible by three and by the batch size.")
    if variant not in VARIANTS:
        raise ValueError(f"Unknown model: {variant}")
    require_preflight()
    lists = verify_dataset(root, ("train", "validation"))
    train_cases, val_cases = lists["train"], lists["validation"]
    if pilot:
        train_cases = train_cases[:8]
        # Include a labelled mass in the small pilot validation list.
        val_cases = [case for case in val_cases if case["tumor_group"] != "absent"][:2]
        if not val_cases:
            raise ValueError("The pilot needs a validation case with a reference tumour.")

    settings = experiment_settings(variant, seed, pilot)
    settings["train_cases"] = [case["case_id"] for case in train_cases]
    settings["validation_cases"] = [case["case_id"] for case in val_cases]
    folder = config.CHECKPOINTS_DIR / run_name(variant, seed, pilot)
    frozen = config.RESULTS_DIR / "frozen_comparison.json"
    if frozen.is_file() and not pilot:
        runs = json.loads(frozen.read_text(encoding="utf-8"))["runs"]
        if any(entry["run"] == folder.name for entry in runs):
            raise ValueError(f"{folder.name} is frozen for test evaluation.")
    latest = folder / "last.pt"
    random.seed(seed)
    torch.manual_seed(seed)
    model = PancreasUNet(variant).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )
    scaler = torch.amp.GradScaler("cuda", enabled=config.AMP)
    state = {"epoch": 0, "history": [], "best_score": -1.0, "settings": settings}
    if latest.is_file():
        state = torch.load(latest, map_location="cpu", weights_only=True)
        if state["settings"] != settings:
            raise ValueError(f"The split or settings changed for {folder.name}.")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        torch.set_rng_state(state["cpu_rng"])
        torch.cuda.set_rng_state(state["cuda_rng"], device)
        export_training_files(state, folder)
        print(f"Resumed {folder.name} at completed epoch {state['epoch']}.", flush=True)

    environment = {
        "torch": str(torch.__version__), "numpy": str(np.__version__),
        "gpu": torch.cuda.get_device_name(device),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
    }
    first_epoch = state["epoch"] + 1
    for epoch in range(first_epoch, epochs + 1):
        started = time.perf_counter()
        torch.cuda.reset_peak_memory_stats(device)
        batches = DataLoader(
            TrainingSlices(root, train_cases, seed, epoch),
            batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True
        )
        model.train()
        losses = []
        for images, masks in batches:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", enabled=config.AMP):
                logits = model(images)
                loss = segmentation_loss(logits, masks)
            if not torch.isfinite(loss):
                raise ValueError(f"Non-finite loss at epoch {epoch}.")
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            losses.append(loss.item())

        training_seconds = time.perf_counter() - started
        scores = validate(model, root, val_cases, device)
        selection = float(np.mean(list(scores.values())))
        if selection > state["best_score"]:
            state["best_epoch"] = epoch
            state["best_score"] = selection
            state["best_model"] = {
                key: tensor.detach().cpu().clone() for key, tensor in model.state_dict().items()
            }
        row = {
            "epoch": epoch, "training_loss": float(np.mean(losses)),
            "validation_pancreas_dice": scores["pancreas"],
            "validation_tumour_dice": scores["tumour"], "selection_score": selection,
            "seconds": time.perf_counter() - started,
            "training_seconds": training_seconds,
            "resumed_from": first_epoch - 1 if epoch == first_epoch and first_epoch > 1 else None,
            "peak_gpu_gib": torch.cuda.max_memory_allocated(device) / 1024**3,
            "gpu": environment["gpu"],
        }
        state["history"].append(row)
        state.update(
            epoch=epoch, model=model.state_dict(), optimizer=optimizer.state_dict(),
            scaler=scaler.state_dict(), cpu_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state(device), environment=environment,
        )
        save_checkpoint(state, latest)
        export_training_files(state, folder)
        print(folder.name, row, flush=True)
    return folder


def freeze_comparison(experiments, epochs=config.TRAINING_EPOCHS):
    """Commit completed run checkpoints before releasing held-out evaluation."""
    path = config.RESULTS_DIR / "frozen_comparison.json"
    digest = split_digest()
    entries = []
    plan = comparison_plan(experiments, epochs)
    for variant, seed in experiments:
        name = run_name(variant, seed)
        folder = config.CHECKPOINTS_DIR / name
        last = torch.load(folder / "last.pt", map_location="cpu", weights_only=True)
        expected = experiment_settings(variant, seed, False)
        if last["epoch"] != epochs or any(last["settings"].get(key) != value for key, value in expected.items()):
            raise ValueError(f"{name} has incomplete training or mismatched settings.")
        export_training_files(last, folder)
        _, best = load_model(folder / "best.pt", "cpu")
        if best["epoch"] != last["best_epoch"]:
            raise ValueError(f"{name} has a stale best checkpoint.")
        entries.append({
            "run": name, "variant": variant, "seed": seed, "best_epoch": best["epoch"],
            "validation_score": best["validation_score"], "checkpoint_sha256": file_digest(folder / "best.pt"),
        })
    full_model = [entry for entry in entries if entry["variant"] == "attention_fusion"]
    if not full_model:
        raise ValueError("Include the full model in the comparison.")
    demo = max(full_model, key=lambda entry: entry["validation_score"])
    manifest = {
        "split_sha256": digest, "source_sha256": config.source_digest(),
        "data_summary_sha256": plan["settings"]["data_summary_sha256"],
        "comparison_plan_sha256": file_digest(config.RESULTS_DIR / "comparison_plan.json"),
        "epochs": epochs, "runs": entries, "demo_run": demo["run"]
    }
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != manifest:
            raise ValueError("A different comparison is already frozen.")
    save_validation_summary(entries)
    if not path.exists():
        write_json(path, manifest)
    return path


def save_validation_summary(entries):
    import matplotlib.pyplot as plt

    rows = []
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    metrics = {
        "training_loss": "Training loss",
        "validation_pancreas_dice": "Validation pancreas Dice",
        "validation_tumour_dice": "Validation tumour Dice",
    }
    for entry in entries:
        folder = config.CHECKPOINTS_DIR / entry["run"]
        state = torch.load(folder / "last.pt", map_location="cpu", weights_only=True)
        history = state["history"]
        best = history[entry["best_epoch"] - 1]
        rows.append({
            **entry,
            "pancreas_dice": best["validation_pancreas_dice"],
            "tumour_dice": best["validation_tumour_dice"],
            "parameters": state["environment"]["parameters"],
            "total_seconds": sum(row["seconds"] for row in history),
            "peak_gpu_gib": max(row["peak_gpu_gib"] for row in history),
        })
        for axis, (key, title) in zip(axes, metrics.items()):
            axis.plot([row["epoch"] for row in history], [row[key] for row in history], label=entry["run"])
            axis.set(xlabel="Epoch", title=title)
    axes[0].legend(fontsize=7)
    fig.savefig(config.RESULTS_DIR / "training_curves.png", dpi=150)
    plt.close(fig)
    write_table(config.RESULTS_DIR / "validation_comparison.csv", rows)
