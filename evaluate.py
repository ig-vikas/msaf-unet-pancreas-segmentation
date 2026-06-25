"""Per-case metrics and held-out comparisons on the original CT grid."""

import json
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import nibabel as nib
import monai
import numpy as np
import torch

import config
from dataset import audited_data_digest, read_pair, verify_dataset
from predict import load_model, predict_volume
from prepare_data import file_digest, split_digest, write_csv, write_json

CLASSES = {1: "pancreas", 2: "tumour"}
METRICS = ("dice", "iou", "precision", "recall", "hd95_mm")


def hd95(prediction, reference, spacing):
    from monai.metrics import compute_hausdorff_distance

    region = prediction | reference
    bounds = []
    for axis in range(3):
        others = tuple(i for i in range(3) if i != axis)
        active = np.flatnonzero(region.any(axis=others))
        bounds.append(slice(max(0, active[0] - 1), active[-1] + 2))
    box = tuple(bounds)
    pred = torch.from_numpy(np.ascontiguousarray(prediction[box]))[None, None]
    ref = torch.from_numpy(np.ascontiguousarray(reference[box]))[None, None]
    distance = compute_hausdorff_distance(
        pred, ref, include_background=True, percentile=95, spacing=spacing.tolist()
    ).item()
    if not np.isfinite(distance):
        raise ValueError("The surface distance is undefined for two nonempty masks.")
    return distance


def score_volume(prediction, reference, affine, surface_distance=False):
    if prediction.shape != reference.shape:
        raise ValueError("The prediction and reference must share the same voxel grid.")
    if not np.isin(prediction, [0, 1, 2]).all() or not np.isin(reference, [0, 1, 2]).all():
        raise ValueError("Expected integer label values 0, 1 or 2.")
    voxel_ml = abs(np.linalg.det(affine[:3, :3])) / 1000
    spacing = nib.affines.voxel_sizes(affine)
    rows = []
    for label, name in CLASSES.items():
        pred, ref = prediction == label, reference == label
        predicted = int(np.count_nonzero(pred))
        actual = int(np.count_nonzero(ref))
        matched = int(np.count_nonzero(pred & ref))
        row = {
            "label": name,
            "reference_ml": actual * voxel_ml,
            "predicted_ml": predicted * voxel_ml,
            "false_positive_ml": (predicted - matched) * voxel_ml,
            "dice": 2 * matched / (actual + predicted) if actual else None,
            "iou": matched / (actual + predicted - matched) if actual else None,
            "precision": matched / predicted if actual and predicted else None,
            "recall": matched / actual if actual else None,
            "hd95_mm": None,
            "hd95_status": "pending" if surface_distance else "skipped",
            "status": "overlap" if matched else "disjoint",
        }
        if not actual:
            row["status"] = "false_positive" if predicted else "both_empty"
            row["hd95_status"] = "empty_reference"
        elif not predicted:
            row["status"] = "missed"
            row["hd95_status"] = "empty_prediction"
        elif surface_distance:
            axes = affine[:3, :3] / spacing
            if not np.allclose(axes.T @ axes, np.eye(3), atol=1e-4):
                row["hd95_status"] = "nonorthogonal_grid"
            else:
                row["hd95_mm"] = hd95(pred, ref, spacing)
                row["hd95_status"] = "measured"
        rows.append(row)
    return rows


def predict_and_score(model, root, case, device, surface_distance=False):
    image, mask = read_pair(root, case)
    if surface_distance and image.header.get_xyzt_units()[0] != "mm":
        raise ValueError(f"{case['case_id']} has no millimetre spatial units.")
    start = time.perf_counter()
    prediction = predict_volume(model, image, device)
    elapsed = time.perf_counter() - start
    reference = np.asanyarray(mask.dataobj)
    rows = score_volume(prediction, reference, image.affine, surface_distance)
    return rows, prediction, reference, image, elapsed


def summarize(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["run"], row["label"])].append(row)
    result = []
    for (run, label), cases in sorted(grouped.items()):
        summary = {
            "run": run, "label": label, "cases": len(cases),
            "reference_positive": sum(row["reference_ml"] > 0 for row in cases),
            "misses": sum(row["status"] == "missed" for row in cases),
            "absent_reference_false_positives": sum(row["status"] == "false_positive" for row in cases),
            "hd95_nonorthogonal_cases": sum(row["hd95_status"] == "nonorthogonal_grid" for row in cases),
        }
        absent = [row["false_positive_ml"] for row in cases if row["reference_ml"] == 0]
        summary["mean_absent_reference_fp_ml"] = float(np.mean(absent)) if absent else None
        summary["mean_prediction_seconds"] = float(np.mean([row["prediction_seconds"] for row in cases]))
        summary["parameters"] = cases[0]["parameters"]
        for metric in METRICS:
            values = [row[metric] for row in cases if row[metric] is not None]
            summary[f"mean_{metric}"] = float(np.mean(values)) if values else None
            summary[f"n_{metric}"] = len(values)
        result.append(summary)
    return result


def write_table(path, rows):
    if not rows:
        raise ValueError(f"No rows to write to {path.name}.")
    write_csv(path, rows)


def save_comparison_overlay(model, root, case, label, device, path, show_false_positive=False):
    import matplotlib.pyplot as plt

    image, mask = read_pair(root, case)
    prediction = predict_volume(model, image, device)
    ct = nib.as_closest_canonical(image)
    reference = np.asanyarray(nib.as_closest_canonical(mask).dataobj)
    orientation = nib.orientations.io_orientation(image.affine)
    ras = nib.orientations.axcodes2ornt(("R", "A", "S"))
    prediction = nib.orientations.apply_orientation(
        prediction, nib.orientations.ornt_transform(orientation, ras)
    )
    active = (prediction == label) & (reference != label) if show_false_positive else reference == label
    z = int(active.sum(axis=(0, 1)).argmax())
    low, high = config.CT_WINDOW
    spacing = ct.header.get_zooms()
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.6), constrained_layout=True)
    for axis, labels, title in zip(axes, (None, reference, prediction), ("CT", "Reference", "Prediction")):
        axis.imshow(
            np.asarray(ct.dataobj[:, :, z]).T, cmap="gray", vmin=low, vmax=high,
            origin="lower", aspect=spacing[1] / spacing[0]
        )
        if labels is not None:
            for value, color in ((1, "#26a880"), (2, "#ee6873")):
                plane = labels[:, :, z].T == value
                if plane.any() and not plane.all():
                    axis.contour(plane, levels=[0.5], colors=[color], linewidths=0.9)
        axis.set_title(title)
        axis.set_axis_off()
    fig.suptitle(f"{case['case_id']} · slice {z} · green pancreas, pink tumour")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def read_completed_run(path, identity, run, cases):
    if not path.is_file():
        return None
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
        rows = record["rows"]
        expected = {(case["case_id"], label) for case in cases for label in CLASSES.values()}
        actual = {(row["case_id"], row["label"]) for row in rows if row["run"] == run}
        if record["identity"] == identity and len(rows) == len(expected) and actual == expected:
            return rows
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return None


def evaluate_test(root, manifest_path, device):
    """Use the checkpoint hashes frozen before any test scores are viewed."""
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if manifest["split_sha256"] != split_digest():
        raise ValueError("The current patient lists differ from the frozen comparison.")
    if manifest.get("source_sha256") != config.source_digest():
        raise ValueError("Use the code snapshot recorded in the frozen comparison.")
    if manifest.get("data_summary_sha256") != audited_data_digest():
        raise ValueError("The dataset audit differs from the one used to train the frozen comparison.")
    if manifest.get("comparison_plan_sha256") != file_digest(config.RESULTS_DIR / "comparison_plan.json"):
        raise ValueError("The experiment plan differs from the frozen comparison.")
    cases = verify_dataset(root, ("test",))["test"]
    device = torch.device(device)
    comparison_hash = file_digest(manifest_path)
    hardware = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
    runtime = {"device": str(device), "hardware": hardware, "torch": str(torch.__version__),
               "monai": str(monai.__version__)}
    folder = config.RESULTS_DIR / "held_out"
    folder.mkdir(parents=True, exist_ok=True)
    results = []
    for entry in manifest["runs"]:
        path = config.CHECKPOINTS_DIR / entry["run"] / "best.pt"
        if file_digest(path) != entry["checkpoint_sha256"]:
            raise ValueError(f"The frozen checkpoint changed for {entry['run']}.")
        identity = {"comparison_sha256": comparison_hash, **runtime}
        completed = folder / f"{entry['run']}.json"
        previous = read_completed_run(completed, identity, entry["run"], cases)
        if previous is not None:
            print(f"Reusing the completed test scores for {entry['run']}.", flush=True)
            results.extend(previous)
            continue
        model, saved = load_model(path, device)
        if saved["split_sha256"] != manifest["split_sha256"]:
            raise ValueError(f"{entry['run']} used a different patient split.")
        parameters = sum(parameter.numel() for parameter in model.parameters())
        run_rows = []
        for index, case in enumerate(cases, 1):
            rows, _, _, _, elapsed = predict_and_score(model, root, case, device, surface_distance=True)
            for row in rows:
                run_rows.append({
                    "run": entry["run"], "case_id": case["case_id"], **row,
                    "parameters": parameters, "prediction_seconds": elapsed,
                })
            print(f"{entry['run']}  {index}/{len(cases)}  {case['case_id']}", flush=True)
        write_table(folder / f"{entry['run']}.csv", run_rows)
        write_json(completed, {"identity": identity, "rows": run_rows})
        results.extend(run_rows)
        del model
    write_table(config.RESULTS_DIR / "per_case_metrics.csv", results)
    comparison = summarize(results)
    write_table(config.RESULTS_DIR / "model_comparison.csv", comparison)

    # Pick examples by measured errors only after the test comparison is fixed.
    selected_run = manifest["demo_run"]
    tumour = [row for row in results if row["run"] == selected_run and row["label"] == "tumour"]
    positives = [row for row in tumour if row["reference_ml"] > 0]
    missed = [row for row in tumour if row["status"] == "missed"]
    false_positives = [row for row in tumour if row["false_positive_ml"] > 0]
    examples = {
        "Smallest reference tumour": min(positives, key=lambda row: row["reference_ml"], default=None),
        "Complete miss": min(missed, key=lambda row: row["reference_ml"], default=None),
        "Largest false-positive tumour volume": max(
            false_positives, key=lambda row: row["false_positive_ml"], default=None
        ),
    }
    model, _ = load_model(config.CHECKPOINTS_DIR / selected_run / "best.pt", device)
    cases_by_id = {case["case_id"]: case for case in cases}
    notes = [f"# Review cases for {selected_run}", ""]
    for number, (title, row) in enumerate(examples.items(), start=1):
        if row is None:
            notes.append(f"- {title} — no matching case in this run.")
            continue
        case_id = row["case_id"]
        path = folder / f"{selected_run}_{number}_{case_id}.png"
        save_comparison_overlay(model, root, cases_by_id[case_id], 2, device, path,
                                show_false_positive=number == 3)
        notes.append(
            f"- {title} — {case_id}. Reference {row['reference_ml']:.2f} mL, "
            f"false-positive volume {row['false_positive_ml']:.2f} mL. [{path.name}](held_out/{path.name})"
        )
    notes.extend(("", "Review the displayed slices and neighbouring slices before describing the cause of each error."))
    examples_path = config.RESULTS_DIR / "failure_examples.md"
    examples_path.write_text("\n".join(notes) + "\n", encoding="utf-8")
    review_path = config.RESULTS_DIR / "failure_notes.md"
    try:
        with review_path.open("x", encoding="utf-8") as handle:
            handle.write(
                "# Held-out scan review\n\n"
                "Use the [selected cases](failure_examples.md) to record what appears in the CT, "
                "reference and predicted masks. Add the case IDs and observed errors after viewing "
                "the overlays.\n"
            )
    except FileExistsError:
        pass
    record = {
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "comparison_sha256": comparison_hash,
        "runtime": runtime,
        "cases": len(cases),
        "runs": len(manifest["runs"]),
        "metrics_sha256": file_digest(config.RESULTS_DIR / "per_case_metrics.csv"),
        "summary_sha256": file_digest(config.RESULTS_DIR / "model_comparison.csv"),
        "examples_sha256": file_digest(examples_path),
    }
    write_json(config.RESULTS_DIR / "evaluation_run.json", record)
    return comparison
