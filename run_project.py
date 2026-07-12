"""Run Team 9's experiments from a terminal on a remote Linux machine."""

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path


def positive_integer(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("Use a positive whole number.")
    return number


def read_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("environment", help="Check imports, memory, disk and CUDA.")
    commands.add_parser("audit", help="Audit the CTs and create the patient split.")
    preflight = commands.add_parser("preflight", help="Check preparation and all four models on real data.")
    preflight.add_argument("--device", choices=("cpu", "cuda"), default="cuda",
                           help="Use cpu to run the checks on a remote machine without a GPU.")
    commands.add_parser("pilot", help="Check learning, checkpoint resume, and GPU use.")
    training = commands.add_parser("train", help="Train or resume the planned runs.")
    training.add_argument("--epochs", required=True, type=positive_integer, help="Shared epoch budget.")
    training.add_argument("--repeats", action="store_true", help="Add the optional seed 43 and 44 runs.")
    training.add_argument("--run", type=positive_integer, help="Run only this number. With no number, run all.")
    commands.add_parser("freeze", help="Record the completed checkpoints and validation results.")
    commands.add_parser("evaluate", help="Score the frozen checkpoints on the held-out cases.")
    demo = commands.add_parser("demo", help="Save an aligned mask and an axial preview.")
    demo.add_argument("--scan", type=Path, help="Input NIfTI CT. Defaults to a training example.")
    check = commands.add_parser("check-model", help="Verify a saved prediction checkpoint on CPU.")
    check.add_argument("--checkpoint", type=Path, help="Path to the trained best.pt.")
    predict = commands.add_parser("predict", help="Run standalone prediction with a project checkpoint.")
    predict.add_argument("--scan", required=True, type=Path, help="Input NIfTI CT.")
    predict.add_argument("--checkpoint", type=Path, help="Path to the trained best.pt.")
    export = commands.add_parser("export-model", help="Bundle a trained checkpoint with its prediction code.")
    export.add_argument("--checkpoint", type=Path, help="Path to the trained best.pt.")
    package = commands.add_parser("package", help="Collect evidence into a ZIP.")
    package.add_argument("stage", choices=("checks", "review", "submission"))
    return parser.parse_args()


def require_cuda():
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Run the environment check on the GPU machine.")
    return torch.device("cuda")


def export_prediction(scan, destination, checkpoint=None):
    import matplotlib.pyplot as plt
    import numpy as np

    from app import demo_cases, draw_slice, run_demo
    from prepare_data import file_digest, write_json

    if scan is None:
        examples = demo_cases()
        if not examples:
            raise FileNotFoundError("Supply a CT with --scan. The audited training scans are unavailable.")
        scan = next(iter(examples.values()))
    if Path(scan).resolve() == (destination / "segmentation.nii.gz").resolve():
        raise ValueError("Choose an input CT outside the output mask path.")
    destination.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination) as folder:
        temporary = Path(folder)
        mask = temporary / "segmentation.nii.gz"
        ct, labels, spacing, report = run_demo(scan, mask, checkpoint=checkpoint)
        z = labels.shape[2] // 2
        for label in (2, 1):
            counts = np.count_nonzero(labels == label, axis=(0, 1))
            if counts.any():
                z = int(counts.argmax())
                break
        figure = draw_slice(ct, labels, spacing, z)
        overlay = temporary / "overlay.png"
        try:
            figure.savefig(overlay, dpi=150)
        finally:
            plt.close(figure)
        report["overlay_sha256"] = file_digest(overlay)
        write_json(temporary / "run.json", report)
        # Publish the run record last because it identifies the mask and preview.
        for name in ("segmentation.nii.gz", "overlay.png", "run.json"):
            (temporary / name).replace(destination / name)
    print(f"Saved demo files to {destination}")


def run(args, config):
    dataset_dir = config.DATASET_DIR
    results_dir = config.RESULTS_DIR
    if args.command == "environment":
        from check_environment import run_checks

        return 0 if run_checks()["checks_passed"] else 1

    if args.command == "audit":
        from prepare_data import run_audit

        run_audit(dataset_dir)
    elif args.command == "preflight":
        from check_pipeline import run_preflight

        device = "cpu" if args.device == "cpu" else require_cuda()
        run_preflight(dataset_dir, device)
        print(f"Saved {results_dir / 'preflight' / 'checks.json'}")
    elif args.command == "pilot":
        from train import run_pilot

        report = run_pilot(dataset_dir, require_cuda())
        print(json.dumps(report, indent=2))
    elif args.command == "train":
        from train import experiment_runs, run_comparison, run_name

        experiments = experiment_runs(args.repeats)
        for number, (variant, seed) in enumerate(experiments, start=1):
            print(number, run_name(variant, seed))
        if args.run is not None and args.run > len(experiments):
            raise ValueError("Choose a run number from the list above.")
        device = require_cuda()
        indices = range(len(experiments)) if args.run is None else [args.run - 1]
        for index in indices:
            run_comparison(experiments, index, dataset_dir, device, args.epochs)
    elif args.command == "freeze":
        from train import freeze_comparison

        path = results_dir / "comparison_plan.json"
        if not path.is_file():
            raise FileNotFoundError("Start the planned training runs first.")
        plan = json.loads(path.read_text(encoding="utf-8"))
        experiments = [(row["variant"], row["seed"]) for row in plan["runs"]]
        print(f"Saved {freeze_comparison(experiments, epochs=plan['epochs'])}")
    elif args.command == "evaluate":
        from evaluate import evaluate_test

        manifest = results_dir / "frozen_comparison.json"
        if not manifest.is_file():
            raise FileNotFoundError("Freeze the completed comparison before evaluating it.")
        for row in evaluate_test(dataset_dir, manifest, require_cuda()):
            print(row)
    elif args.command == "demo":
        export_prediction(args.scan, results_dir / "demo")
    elif args.command in ("check-model", "predict", "export-model"):
        from app import demo_checkpoint
        from model_files import check_model, export_model

        checkpoint = args.checkpoint or config.MODEL_CHECKPOINT
        if (args.checkpoint is None and not config.MODEL_CHECKPOINT_SETTING
                and not checkpoint.is_file() and (results_dir / "frozen_comparison.json").is_file()):
            checkpoint, _ = demo_checkpoint()
        if args.command == "check-model":
            print(json.dumps(check_model(checkpoint), indent=2))
        elif args.command == "predict":
            export_prediction(args.scan, results_dir / "predictions", checkpoint)
        else:
            print(f"Saved {export_model(checkpoint)}")
    elif args.command == "package":
        from collect_evidence import collect_evidence

        print(f"Saved {collect_evidence(args.stage)}")
    return 0


def main():
    args = read_arguments()
    if not sys.platform.startswith("linux"):
        raise SystemExit("Run this command in a terminal on the remote Linux machine.")
    if sys.version_info < (3, 11):
        raise SystemExit("Use Python 3.11 or later.")

    import fcntl

    os.environ["PANCREAS_REMOTE_RUN"] = "1"
    os.environ.setdefault("MPLBACKEND", "Agg")
    import config

    try:
        config.prepare_folders()
        # Keep two terminal sessions from writing the same audit or checkpoints.
        with (config.RESULTS_DIR / ".remote-run.lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise RuntimeError("A project command is already running in this folder.") from None
            return run(args, config)
    except (ImportError, OSError, RuntimeError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
