"""Check the result tables and write a factual summary for the review."""

import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean

LABELS = ("pancreas", "tumour")
METRICS = ("dice", "iou", "precision", "recall", "hd95_mm")
VARIANT_NAMES = {
    "unet": "Plain U-Net",
    "attention": "Attention U-Net",
    "fusion": "Fusion U-Net",
    "attention_fusion": "Attention and fusion",
}


def read_csv(path):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def metric_value(text, name):
    if text == "":
        return None
    value = float(text)
    upper = 1 if name != "hd95_mm" else math.inf
    if not math.isfinite(value) or not 0 <= value <= upper:
        raise ValueError(f"Invalid {name} value: {text}")
    return value


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def check_splits(project, audit, manifest):
    summary = project / "results" / "data_summary.csv"
    summary_hash = digest(summary)
    if summary_hash != audit["summary_sha256"] or summary_hash != manifest["data_summary_sha256"]:
        raise ValueError("The audit summary differs from the frozen comparison.")
    usable = {}
    for row in read_csv(summary):
        if row["source"] == "labelled" and row["status"] == "usable":
            case = row["case_id"]
            if case in usable:
                raise ValueError(f"Repeated usable case in the audit: {case}")
            usable[case] = (row["patient_id"], row["image_voxel_sha256"])

    splits, patients, hashes = {}, set(), set()
    split_hash = hashlib.sha256()
    for split in ("train", "validation", "test"):
        path = project / "splits" / f"{split}.csv"
        split_hash.update(path.read_bytes())
        rows = read_csv(path)
        if not rows:
            raise ValueError(f"No cases in {split}.csv.")
        cases = set()
        for row in rows:
            case, patient, image_hash = row["case_id"], row["patient_id"], row["image_voxel_sha256"]
            if usable.get(case) != (patient, image_hash) or not patient or not image_hash:
                raise ValueError(f"{case} does not match a usable audit row.")
            if case in cases or patient in patients or image_hash in hashes:
                raise ValueError(f"A case, patient or CT occurs more than once: {case}")
            cases.add(case)
            patients.add(patient)
            hashes.add(image_hash)
        splits[split] = cases
    if set.union(*splits.values()) != set(usable):
        raise ValueError("Split files do not cover exactly the usable labelled cases.")
    split_digest = split_hash.hexdigest()
    if split_digest != audit["split_sha256"] or split_digest != manifest["split_sha256"]:
        raise ValueError("The case lists differ from the audit or frozen comparison.")
    return splits


def check_test_scores(project, entries, test_cases):
    """Require both labels for every test case and recompute each reported mean."""
    results = project / "results"
    names = {entry["run"] for entry in entries}
    groups = defaultdict(dict)
    for row in read_csv(results / "per_case_metrics.csv"):
        key = (row["run"], row["label"])
        case = row["case_id"]
        if key[0] not in names or key[1] not in LABELS or case not in test_cases or case in groups[key]:
            raise ValueError(f"Unexpected or repeated test result: {key}, {case}")
        groups[key][case] = row

    summaries = {}
    for row in read_csv(results / "model_comparison.csv"):
        key = (row["run"], row["label"])
        if key in summaries:
            raise ValueError(f"Repeated comparison row: {key}")
        summaries[key] = row
    expected = {(name, label) for name in names for label in LABELS}
    if set(summaries) != expected or set(groups) != expected:
        raise ValueError("The summary and per-case tables must cover every run and both labels.")

    for key, cases in groups.items():
        summary = summaries[key]
        if set(cases) != test_cases or int(summary["cases"]) != len(test_cases):
            raise ValueError(f"Incomplete held-out case coverage for {key}.")
        for row in cases.values():
            reference = float(row["reference_ml"])
            prediction = float(row["predicted_ml"])
            if not all(math.isfinite(value) and value >= 0 for value in (reference, prediction)):
                raise ValueError(f"Invalid volume in {key}, {row['case_id']}.")
            if not reference:
                expected_status = "false_positive" if prediction else "both_empty"
                if any(row[name] != "" for name in METRICS):
                    raise ValueError(f"An absent reference has a reported metric in {key}, {row['case_id']}.")
            elif not prediction:
                expected_status = "missed"
                if any(metric_value(row[name], name) != 0 for name in ("dice", "iou", "recall")):
                    raise ValueError(f"A missed reference needs zero overlap and recall in {key}, {row['case_id']}.")
                if row["precision"] or row["hd95_mm"]:
                    raise ValueError(f"A missed reference has a precision or distance in {key}, {row['case_id']}.")
            else:
                score = metric_value(row["dice"], "dice")
                if score is None:
                    raise ValueError(f"Missing Dice in {key}, {row['case_id']}.")
                expected_status = "overlap" if score > 0 else "disjoint"
            if row["status"] != expected_status:
                raise ValueError(f"Invalid overlap status in {key}, {row['case_id']}.")
        positives = sum(float(row["reference_ml"]) > 0 for row in cases.values())
        misses = sum(row["status"] == "missed" for row in cases.values())
        absent_false_positives = sum(row["status"] == "false_positive" for row in cases.values())
        for name, count in (
            ("reference_positive", positives), ("misses", misses),
            ("absent_reference_false_positives", absent_false_positives),
        ):
            if int(summary[name]) != count:
                raise ValueError(f"Incorrect {name} count for {key}.")
        for metric in METRICS:
            values = [metric_value(row[metric], metric) for row in cases.values()]
            scores = [score for score in values if score is not None]
            actual = metric_value(summary[f"mean_{metric}"], metric)
            if int(summary[f"n_{metric}"]) != len(scores):
                raise ValueError(f"Incorrect {metric} count for {key}.")
            if scores:
                if actual is None or not math.isclose(mean(scores), actual, abs_tol=1e-9):
                    raise ValueError(f"Incorrect mean {metric} for {key}.")
            elif actual is not None:
                raise ValueError(f"Expected a blank {metric} mean for {key}.")
    return summaries


def check_epoch_logs(project, entries, target):
    """Verify that each selected epoch agrees with the first best validation score."""
    validation = {}
    for row in read_csv(project / "results" / "validation_comparison.csv"):
        name = row["run"]
        if name in validation:
            raise ValueError(f"Repeated validation summary for {name}.")
        validation[name] = row
    if set(validation) != {entry["run"] for entry in entries}:
        raise ValueError("The validation comparison and frozen runs differ.")
    for entry in entries:
        name = entry["run"]
        log = read_csv(project / "checkpoints" / name / "epochs.csv")
        if [int(row["epoch"]) for row in log] != list(range(1, target + 1)):
            raise ValueError(f"{name} does not have a full epoch log.")
        for row in log:
            details = [metric_value(row[f"validation_{label}_dice"], "dice") for label in LABELS]
            score = metric_value(row["selection_score"], "dice")
            if None in details or score is None or not math.isclose(mean(details), score, abs_tol=1e-9):
                raise ValueError(f"Epoch {row['epoch']} has an incorrect validation score for {name}.")
        selected = max(log, key=lambda row: float(row["selection_score"]))
        summary = validation[name]
        if int(selected["epoch"]) != entry["best_epoch"] or int(summary["best_epoch"]) != entry["best_epoch"]:
            raise ValueError(f"The selected epoch for {name} differs from its validation log.")
        for label in LABELS:
            score = metric_value(selected[f"validation_{label}_dice"], "dice")
            if score is None or not math.isclose(score, float(summary[f"{label}_dice"]), abs_tol=1e-9):
                raise ValueError(f"Selected validation {label} Dice differs for {name}.")
        expected = mean(float(selected[f"validation_{label}_dice"]) for label in LABELS)
        for value in (selected["selection_score"], summary["validation_score"], entry["validation_score"]):
            if not math.isclose(expected, float(value), abs_tol=1e-9):
                raise ValueError(f"Selected validation score differs for {name}.")
    return validation


def score_cell(summary, metric):
    value = summary[f"mean_{metric}"]
    count = summary[f"n_{metric}"]
    return f"{float(value):.3f} ({count})" if value else "— (0)"


def build_result_notes(project):
    """Write a draft based on the recorded audit, validation logs and test scores."""
    project = Path(project)
    results = project / "results"
    manifest_path = results / "frozen_comparison.json"
    manifest = read_json(manifest_path)
    audit = read_json(results / "data_audit.json")
    receipt = read_json(results / "evaluation_run.json")
    if receipt["comparison_sha256"] != digest(manifest_path):
        raise ValueError("The held-out evaluation is tied to a different comparison.")
    plan_path = results / "comparison_plan.json"
    if manifest["comparison_plan_sha256"] != digest(plan_path):
        raise ValueError("The comparison plan differs from the frozen runs.")
    plan = read_json(plan_path)
    for field, name in (
        ("metrics_sha256", "per_case_metrics.csv"), ("summary_sha256", "model_comparison.csv"),
        ("examples_sha256", "failure_examples.md"),
    ):
        if receipt[field] != digest(results / name):
            raise ValueError(f"{name} differs from the evaluation receipt.")
    splits = check_splits(project, audit, manifest)
    counts = {name: len(cases) for name, cases in splits.items()}
    if counts != audit["split_counts"] or receipt["cases"] != counts["test"]:
        raise ValueError("The case counts differ from the audit or evaluation record.")
    entries = manifest["runs"]
    if receipt["runs"] != len(entries) or len({entry["run"] for entry in entries}) != len(entries):
        raise ValueError("Run names are repeated or do not match the completed evaluation.")
    run_pairs = [(entry["variant"], entry["seed"]) for entry in entries]
    planned_pairs = [(entry["variant"], entry["seed"]) for entry in plan["runs"]]
    if run_pairs != planned_pairs or len(set(run_pairs)) != len(run_pairs) or plan["epochs"] != manifest["epochs"]:
        raise ValueError("The frozen run list or epoch budget differs from the comparison plan.")
    seed42 = {entry["variant"]: entry["run"] for entry in entries if entry["seed"] == 42}
    if set(seed42) != set(VARIANT_NAMES):
        raise ValueError("The completed evaluation needs all four seed-42 models.")
    validation = check_epoch_logs(project, entries, manifest["epochs"])
    summaries = check_test_scores(project, entries, splits["test"])

    lines = [
        "# Measured results for the final review", "",
        f"The audit retained {sum(counts.values())} labelled cases. The saved lists contain "
        f"{counts['train']} training, {counts['validation']} validation and {counts['test']} test cases. "
        f"The audit recorded {audit['status_counts'].get('excluded', 0)} exclusions across all source volumes.",
        "", "## Validation", "",
        f"Each listed run completed {manifest['epochs']} epochs. The checkpoint was chosen with the "
        "mean of the two validation Dice scores.", "",
        "| Run | Seed | Selected epoch | Pancreas Dice | Tumour Dice | Parameters |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for entry in entries:
        row = validation[entry["run"]]
        name = VARIANT_NAMES[entry["variant"]]
        lines.append(
            f"| {name} | {entry['seed']} | {entry['best_epoch']} | "
            f"{float(row['pancreas_dice']):.3f} | {float(row['tumour_dice']):.3f} | {int(row['parameters']):,} |"
        )
    lines.extend(["", "## Held-out evaluation", "",
                  "Each cell shows the mean and the number of test cases included in that mean.", "",
                  "| Run | Seed | Label | Dice | IoU | Precision | Recall | HD95 (mm) |",
                  "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |"])
    for entry in entries:
        for label in LABELS:
            row = summaries[(entry["run"], label)]
            values = " | ".join(score_cell(row, metric) for metric in METRICS)
            lines.append(f"| {VARIANT_NAMES[entry['variant']]} | {entry['seed']} | {label} | {values} |")
    lines.extend(["", "### Full model and baseline at seed 42", ""])
    for label in LABELS:
        baseline = summaries[(seed42["unet"], label)]["mean_dice"]
        full = summaries[(seed42["attention_fusion"], label)]["mean_dice"]
        if baseline and full:
            difference = float(full) - float(baseline)
            lines.append(f"- {label.capitalize()} Dice changed by {difference * 100:+.2f} percentage points on the saved test split.")
    lines.extend(["", "## Figures and observed errors", "",
                  "Use the [training curves](training_curves.png) and review the "
                  "[selected test examples](failure_examples.md) before writing the discussion. "
                  "Record observations separately in [failure notes](failure_notes.md).", ""])
    path = results / "review_results.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
