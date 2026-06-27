"""Check a project checkpoint and package it with the code for prediction."""

import hashlib
import json
import tempfile
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import config
from predict import load_model
from prepare_data import file_digest


def check_model(checkpoint):
    checkpoint = Path(checkpoint)
    digest = file_digest(checkpoint) if checkpoint.is_file() else None
    model, saved = load_model(checkpoint, "cpu")
    if digest != file_digest(checkpoint):
        raise ValueError("The weights changed while they were being checked. Try again after the copy finishes.")
    details = {key: saved[key] for key in (
        "variant", "architecture_version", "ct_window", "epoch", "seed", "pilot",
        "validation_score", "source_sha256", "split_sha256", "data_summary_sha256",
    )}
    details["parameters"] = sum(parameter.numel() for parameter in model.parameters())
    details["checkpoint_sha256"] = digest
    details["labels"] = {"0": "background", "1": "pancreas", "2": "tumour/mass"}
    return details


def export_model(checkpoint):
    """Save the checked weights and matching source files for standalone prediction."""
    details = check_model(checkpoint)
    if details["pilot"]:
        raise ValueError("Use a trained comparison checkpoint for the model bundle.")
    project = config.PROJECT_ROOT
    bundled_checkpoint = "weights/model.pt"
    details["checkpoint"] = bundled_checkpoint
    source_files = (*config.EXPERIMENT_FILES, "app.py", "run_project.py", "collect_evidence.py",
                    "report_results.py", "check_environment.py", "requirements-demo.txt",
                    "MODEL_HANDOVER.md")
    destination = config.RESULTS_DIR / "team9_model.zip"
    destination.parent.mkdir(parents=True, exist_ok=True)
    instructions = (
        "# Team 9 prediction bundle\n\n"
        "Use Python 3.11 or later on the remote Linux machine. Install the requirements, "
        "then follow the standalone prediction steps in MODEL_HANDOVER.md. "
        "The model details and checksum are in model_details.json.\n"
    )
    with tempfile.TemporaryDirectory(dir=destination.parent) as folder:
        temporary = Path(folder) / destination.name
        with ZipFile(temporary, "w", compression=ZIP_DEFLATED) as archive:
            for name in source_files:
                archive.write(project / name, name)
            archive.write(checkpoint, bundled_checkpoint)
            archive.writestr("README.md", instructions)
            archive.writestr("model_details.json", json.dumps(details, indent=2, allow_nan=False) + "\n")
        with ZipFile(temporary) as archive:
            with archive.open(bundled_checkpoint) as weights:
                if hashlib.file_digest(weights, "sha256").hexdigest() != details["checkpoint_sha256"]:
                    raise ValueError("The checkpoint changed while the model ZIP was written.")
            source = hashlib.sha256()
            for name in config.EXPERIMENT_FILES:
                source.update(name.encode("utf-8"))
                source.update(archive.read(name))
            if source.hexdigest() != details["source_sha256"] or archive.testzip():
                raise ValueError("The model ZIP does not contain the matching source files.")
        temporary.replace(destination)
    return destination
