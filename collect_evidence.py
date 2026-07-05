"""Check and bundle the audit, measured results, and final submission."""

import hashlib
import json
import tempfile
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import config
from report_results import build_result_notes

REPORT_FILE = "REPORTS/Team 9 Final Project Report.docx"
SLIDES_FILE = "REPORTS/Team 9 Final Review.pptx"
DEMO_RECORDING = "results/demo/demo.mp4"


def project_file(project, name):
    path = (project / name).resolve()
    if not path.is_relative_to(project) or not path.is_file():
        raise FileNotFoundError(f"Missing {name} in the project folder.")
    return path


def read_json(project, name):
    return json.loads(project_file(project, name).read_text(encoding="utf-8"))


def file_hash(project, name):
    with project_file(project, name).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def collect_evidence(
    stage, report_file=REPORT_FILE, slides_file=SLIDES_FILE, demo_recording=DEMO_RECORDING
):
    """Write a checked ZIP under results and return its path."""
    if stage not in ("checks", "review", "submission"):
        raise ValueError("Choose checks, review or submission.")

    project = config.PROJECT_ROOT.resolve()
    files_to_pack = [
        "results/data_audit.json", "results/data_summary.csv",
        "results/data_exclusions.csv", "results/duplicate_scans.csv",
        f"results/environment_{config.PROJECT_RUNTIME}.json", "results/preflight/checks.json",
        "splits/train.csv", "splits/validation.csv", "splits/test.csv",
    ]
    audit = read_json(project, "results/data_audit.json")
    preflight = read_json(project, "results/preflight/checks.json")
    summary_hash = file_hash(project, "results/data_summary.csv")
    if audit["summary_sha256"] != summary_hash or preflight["data_summary_sha256"] != summary_hash:
        raise ValueError("The audit and preparation checks use different case summaries.")
    source_hash = config.source_digest()
    if preflight["source_sha256"] != source_hash:
        raise ValueError("The current code differs from the preparation checks.")
    split_hash = hashlib.sha256()
    for name in ("train", "validation", "test"):
        split_hash.update(project_file(project, f"splits/{name}.csv").read_bytes())
    if audit["split_sha256"] != split_hash.hexdigest() or preflight["split_sha256"] != split_hash.hexdigest():
        raise ValueError("The audit and preparation checks must use the current case lists.")
    files_to_pack.extend(f"results/data_overlays/{name}" for name in audit["overlay_files"])
    files_to_pack.extend(f"results/preflight/{row['overlay']}" for row in preflight["preparation"])

    if stage in ("review", "submission"):
        manifest_name = "results/frozen_comparison.json"
        manifest = read_json(project, manifest_name)
        evaluation = read_json(project, "results/evaluation_run.json")
        if manifest["split_sha256"] != split_hash.hexdigest():
            raise ValueError("The frozen runs use a different case split.")
        if manifest["source_sha256"] != source_hash or manifest["data_summary_sha256"] != summary_hash:
            raise ValueError("The frozen runs use different code or audited files.")
        plan_name = "results/comparison_plan.json"
        if manifest["comparison_plan_sha256"] != file_hash(project, plan_name):
            raise ValueError("The comparison plan differs from the frozen runs.")
        if evaluation["comparison_sha256"] != file_hash(project, manifest_name):
            raise ValueError("The test evidence uses a different frozen comparison.")
        for field, name in (
            ("metrics_sha256", "results/per_case_metrics.csv"),
            ("summary_sha256", "results/model_comparison.csv"),
            ("examples_sha256", "results/failure_examples.md"),
        ):
            if evaluation[field] != file_hash(project, name):
                raise ValueError(f"{name} does not match the saved evaluation.")
        files_to_pack += [
            "results/model_settings.csv", "results/pilot.json", plan_name,
            manifest_name, "results/evaluation_run.json",
            "results/validation_comparison.csv", "results/training_curves.png",
            "results/per_case_metrics.csv", "results/model_comparison.csv",
            "results/failure_examples.md", "results/failure_notes.md",
        ]
        result_notes = build_result_notes(project)
        files_to_pack.append("report_results.py")
        files_to_pack.append(result_notes.relative_to(project).as_posix())
        if (project / "results/run_errors.jsonl").is_file():
            files_to_pack.append("results/run_errors.jsonl")
        if stage == "submission":
            files_to_pack += [
                "README.md", "BASE_PAPER.md", "requirements.txt", "plan.md",
                "prompts.md", "progress.md", "PROJECT_PLAN.md", "MODEL_HANDOVER.md",
                "requirements-demo.txt",
            ]
            files_to_pack.extend(
                project_file(project, name).relative_to(project).as_posix()
                for name in (report_file, slides_file, demo_recording)
            )
            for draft in ("Final review notes.md", "Team 9 Pretraining Project Report.docx",
                          "Team 9 Pretraining Review.pptx"):
                name = f"REPORTS/{draft}"
                if (project / name).is_file():
                    files_to_pack.append(name)
            files_to_pack.extend(path.name for path in project.glob("*.py"))
            if config.PROJECT_RUNTIME == "colab":
                files_to_pack.extend(
                    f"notebooks/{path.name}" for path in (project / "notebooks").glob("*.ipynb")
                )
        files_to_pack.extend(f"results/held_out/{path.name}" for path in (project / "results/held_out").glob("*.png"))
        locations = ["# Checkpoint locations", "", "Paths are relative to the project folder.", ""]
        for entry in manifest["runs"]:
            folder = f"checkpoints/{entry['run']}"
            weights = f"{folder}/best.pt"
            digest = file_hash(project, weights)
            if digest != entry["checkpoint_sha256"]:
                raise ValueError(f"The best checkpoint changed for {entry['run']}.")
            locations.append(f"- {weights} (SHA-256 {digest})")
            files_to_pack.extend((f"{folder}/epochs.csv", f"{folder}/settings.json"))
            if entry["run"] == manifest["demo_run"]:
                if stage == "submission":
                    files_to_pack.append(weights)
                demo_name = "results/demo/run.json"
                mask_name = "results/demo/segmentation.nii.gz"
                try:
                    demo = read_json(project, demo_name)
                    if not isinstance(demo, dict):
                        raise ValueError("The demo report is incomplete.")
                    if demo.get("frozen_comparison") is not True or demo.get("export_grid_matches") is not True:
                        raise ValueError("Run the official demo with the frozen checkpoint before packaging it.")
                    if demo.get("checkpoint_sha256") != digest or demo.get("mask_sha256") != file_hash(project, mask_name):
                        raise ValueError("The demo checkpoint or mask differs from the saved demo record.")
                except (OSError, ValueError) as error:
                    if stage == "submission":
                        raise
                    print(f"Demo evidence left out of the review ZIP: {error}")
                else:
                    files_to_pack.extend((demo_name, mask_name))
                    overlay_name = "results/demo/overlay.png"
                    if ((project / overlay_name).is_file() and demo.get("overlay_sha256")
                            and demo["overlay_sha256"] == file_hash(project, overlay_name)):
                        files_to_pack.append(overlay_name)

    destination = project / "results" / f"team9_{stage}.zip"
    with tempfile.NamedTemporaryFile(dir=destination.parent, suffix=".tmp", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        with ZipFile(temporary, "w", compression=ZIP_DEFLATED) as archive:
            for name in sorted(set(files_to_pack)):
                source = project_file(project, name)
                archive.write(source, arcname=source.relative_to(project).as_posix())
            if stage in ("review", "submission"):
                archive.writestr("CHECKPOINTS.md", "\n".join(locations) + "\n")
        with ZipFile(temporary) as archive:
            damaged = archive.testzip()
            if damaged:
                raise ValueError(f"The ZIP contains a damaged file: {damaged}")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination
