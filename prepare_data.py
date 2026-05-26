"""Check the original CT volumes and save a reproducible case split."""

import csv
import hashlib
import json
import os
import random
import re
import tempfile
import time
import zlib
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import nibabel as nib
import numpy as np

import config


def write_csv(path, rows, columns=None):
    """Write a small table without adding a dataframe dependency."""
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = columns or list(dict.fromkeys(key for row in rows for key in row))
    with tempfile.NamedTemporaryFile(
        mode="w", dir=path.parent, suffix=".tmp", newline="", encoding="utf-8", delete=False
    ) as handle:
        temporary = Path(handle.name)
        try:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        except BaseException:
            handle.close()
            temporary.unlink(missing_ok=True)
            raise
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def write_json(path, value):
    """Replace a report only after its new contents have been written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def nifti_files(folder):
    """Ignore the Apple metadata files included in some extracted downloads."""
    return {p.name: p for p in folder.glob("*.nii.gz") if not p.name.startswith("._")}


def file_digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def split_digest(folder=config.SPLITS_DIR):
    digest = hashlib.sha256()
    for name in ("train", "validation", "test"):
        digest.update((Path(folder) / f"{name}.csv").read_bytes())
    return digest.hexdigest()


def ras_array(image):
    """Reorder and flip voxel axes to the nearest RAS orientation."""
    return np.asanyarray(nib.as_closest_canonical(image).dataobj)


def restore_labels(labels, image):
    """Undo the orientation change and verify the original voxel shape."""
    if labels.shape != nib.as_closest_canonical(image).shape:
        raise ValueError("The prediction does not match the prepared scan shape.")
    ras = nib.orientations.axcodes2ornt(("R", "A", "S"))
    original = nib.orientations.io_orientation(image.affine)
    inverse = nib.orientations.ornt_transform(ras, original)
    restored = nib.orientations.apply_orientation(labels, inverse)
    if restored.shape != image.shape:
        raise ValueError("The mask could not be restored to the original scan shape.")
    return restored


def normalize_slice(plane):
    low, high = config.CT_WINDOW
    plane = np.array(plane, dtype=np.float32, copy=True)
    np.clip(plane, low, high, out=plane)
    plane -= low
    plane /= high - low
    return plane


def safe_path(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"Metadata path leaves the dataset folder: {relative}")
    return path


def inspect_inventory(root):
    """Compare actual files with the lists supplied in dataset.json."""
    metadata_path = root / "dataset.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Dataset metadata is missing: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    labels = {int(key): value for key, value in metadata["labels"].items()}
    if set(labels) != {0, 1, 2} or metadata.get("modality") != {"0": "CT"}:
        raise ValueError("Expected one CT modality and labels 0, 1, 2. Review dataset.json.")

    issues = []
    pairs = {}
    for entry in metadata["training"]:
        image = safe_path(root, entry["image"])
        mask = safe_path(root, entry["label"])
        if image.parent != root / "imagesTr" or mask.parent != root / "labelsTr":
            issues.append(f"Training entry has unexpected folders: {entry}")
        if image.name != mask.name or not re.fullmatch(r"pancreas_\d+\.nii\.gz", image.name):
            issues.append(f"Training filenames do not form a known case pair: {entry}")
        if image.name in pairs:
            issues.append(f"Repeated training metadata entry: {image.name}")
        pairs[image.name] = mask.name

    images = nifti_files(root / "imagesTr")
    masks = nifti_files(root / "labelsTr")
    unlabelled = nifti_files(root / "imagesTs")
    test_names = []
    for entry in metadata["test"]:
        image = safe_path(root, entry)
        if image.parent != root / "imagesTs" or not re.fullmatch(r"pancreas_\d+\.nii\.gz", image.name):
            issues.append(f"Unlabelled metadata entry has an unexpected path: {entry}")
        test_names.append(image.name)
    for description, actual, expected in (
        ("imagesTr", set(images), set(pairs)),
        ("labelsTr", set(masks), set(pairs.values())),
        ("imagesTs", set(unlabelled), set(test_names)),
    ):
        if actual != expected:
            issues.append(f"{description}: missing={sorted(expected-actual)}, extra={sorted(actual-expected)}")
    if len(pairs) != metadata["numTraining"] or len(test_names) != metadata["numTest"]:
        issues.append("Metadata entry counts do not match numTraining/numTest.")
    if len(test_names) != len(set(test_names)):
        issues.append("Repeated unlabelled metadata entry.")
    if set(images) & set(unlabelled):
        issues.append("A case filename occurs in both imagesTr and imagesTs.")

    records = []
    for name in sorted(set(images) | set(masks) | set(pairs)):
        records.append({"case_id": name.removesuffix(".nii.gz"), "source": "labelled",
                        "image": f"imagesTr/{name}", "mask": f"labelsTr/{pairs.get(name, name)}"})
    for name in sorted(set(unlabelled) | set(test_names)):
        records.append({"case_id": name.removesuffix(".nii.gz"), "source": "unlabelled",
                        "image": f"imagesTs/{name}", "mask": ""})
    return metadata, labels, records, issues


def check_geometry(image, prefix, row, errors, warnings):
    row[f"{prefix}_shape"] = "x".join(map(str, image.shape))
    spacing = np.asarray(image.header.get_zooms()[:3], dtype=float)
    row[f"{prefix}_spacing"] = ";".join(f"{value:.8g}" for value in spacing)
    row[f"{prefix}_spatial_unit"] = image.header.get_xyzt_units()[0]
    row[f"{prefix}_qform_code"] = int(image.header["qform_code"])
    row[f"{prefix}_sform_code"] = int(image.header["sform_code"])
    if len(image.shape) != 3 or min(image.shape) < 1:
        raise ValueError(f"{prefix} is not a nonempty 3D volume")
    affine = image.affine
    if affine is None or not np.all(np.isfinite(affine)) or abs(np.linalg.det(affine[:3, :3])) < 1e-10:
        raise ValueError(f"{prefix} has an invalid affine")
    row[f"{prefix}_orientation"] = "".join(code or "?" for code in nib.aff2axcodes(affine))
    if not np.all(np.isfinite(spacing)) or np.any(spacing <= 0):
        errors.append(f"{prefix} has invalid voxel spacing")
    if row[f"{prefix}_spatial_unit"] != "mm":
        errors.append(f"{prefix} spatial units are not explicitly mm")
    affine_spacing = nib.affines.voxel_sizes(image.affine)
    if not np.allclose(spacing, affine_spacing, atol=1e-4, rtol=1e-4):
        errors.append(f"{prefix} header and affine spacing differ")
    qform, qcode = image.get_qform(coded=True)
    sform, scode = image.get_sform(coded=True)
    if qcode and scode and not np.allclose(qform, sform, atol=1e-3, rtol=0):
        warnings.append(f"{prefix} qform/sform differ; NiBabel-selected affine is recorded")
    row[f"{prefix}_affine"] = json.dumps(image.affine.tolist(), separators=(",", ":"))


def voxel_hash(data, affine):
    """Compare decoded float32 voxels in RAS order, ignoring gzip headers."""
    orientation = nib.orientations.io_orientation(affine)
    ras = nib.orientations.axcodes2ornt(("R", "A", "S"))
    canonical = nib.orientations.apply_orientation(
        data, nib.orientations.ornt_transform(orientation, ras)
    )
    digest = hashlib.sha256(str(canonical.shape).encode("ascii"))
    for plane in canonical:
        digest.update(np.asarray(plane, dtype="<f4", order="C").tobytes())
    return digest.hexdigest()


def inspect_case(root, record, label_values):
    """Load one volume at a time so the audit fits on an ordinary computer."""
    row = dict(record, patient_id=record["case_id"])
    errors, warnings = [], []
    try:
        image_path = root / record["image"]
        image = nib.load(image_path)
        row["image_bytes"] = image_path.stat().st_size
        row["image_file_sha256"] = file_digest(image_path)
        check_geometry(image, "image", row, errors, warnings)
        if errors:
            raise ValueError("; ".join(errors))
        data = np.asanyarray(image.dataobj)
        if not np.isfinite(data).all():
            raise ValueError("CT contains non-finite values")
        row["ct_min"] = float(data.min())
        row["ct_max"] = float(data.max())
        if row["ct_min"] == row["ct_max"]:
            errors.append("CT is constant")
        row["image_voxel_sha256"] = voxel_hash(data, image.affine)
        del data

        if record["mask"]:
            mask_path = root / record["mask"]
            row["mask_file_sha256"] = file_digest(mask_path)
            mask = nib.load(mask_path)
            check_geometry(mask, "mask", row, errors, warnings)
            row["shape_match"] = image.shape == mask.shape
            row["spacing_match"] = bool(np.allclose(image.header.get_zooms()[:3], mask.header.get_zooms()[:3], atol=1e-4, rtol=1e-4))
            row["orientation_match"] = row["image_orientation"] == row["mask_orientation"]
            row["affine_match"] = bool(np.allclose(image.affine, mask.affine, atol=1e-4, rtol=0))
            if not all(row[key] for key in ("shape_match", "spacing_match", "orientation_match", "affine_match")):
                errors.append("CT and mask do not share the same voxel grid")
            mask_data = np.asanyarray(mask.dataobj)
            values, counts = np.unique(mask_data, return_counts=True)
            row["label_values"] = ";".join(map(str, values.tolist()))
            if not np.isfinite(values).all() or not set(values.tolist()).issubset(label_values):
                errors.append(f"Invalid mask labels. Observed {len(values)} values, first 16: {values[:16].tolist()}")
            else:
                voxel_mm3 = abs(float(np.linalg.det(image.affine[:3, :3])))
                row["voxel_volume_mm3"] = voxel_mm3
                counts_by_label = dict(zip(values.astype(int).tolist(), counts.tolist()))
                for value, name in ((0, "background"), (1, "pancreas"), (2, "tumor")):
                    count = counts_by_label.get(value, 0)
                    row[f"{name}_voxels"] = count
                    row[f"{name}_volume_ml"] = count * voxel_mm3 / 1000
                row["whole_pancreas_volume_ml"] = row["pancreas_volume_ml"] + row["tumor_volume_ml"]
                if row["pancreas_voxels"] == 0:
                    warnings.append("No pancreas-label voxels")
                row["mask_voxel_sha256"] = voxel_hash(mask_data, mask.affine)
    except (OSError, ValueError, EOFError, zlib.error, nib.filebasedimages.ImageFileError,
            nib.spatialimages.HeaderDataError, nib.orientations.OrientationError) as error:
        message = f"{type(error).__name__}: {error}"
        if not errors:
            errors.append(message)
        elif message not in errors:
            errors.append(message)
    row["status"] = "excluded" if errors else "usable" if record["mask"] else "unlabelled"
    row["exclusion_reason"] = "; ".join(errors)
    row["warnings"] = "; ".join(warnings)
    return row


def mark_duplicates(rows):
    """Retain one labelled copy of an identical scan; quarantine public overlap."""
    by_hash = defaultdict(list)
    for row in rows:
        if row.get("image_voxel_sha256"):
            by_hash[row["image_voxel_sha256"]].append(row)
    duplicates = []
    for matching in by_hash.values():
        if len(matching) < 2:
            continue
        labelled = sorted((r for r in matching if r["source"] == "labelled"), key=lambda r: r["case_id"])
        public_overlap = any(r["source"] == "unlabelled" for r in matching)
        mask_hashes = {r.get("mask_voxel_sha256") for r in labelled}
        conflict = len(mask_hashes) > 1
        for row in matching:
            duplicates.append({"case_id": row["case_id"], "source": row["source"],
                               "matching_cases": ";".join(r["case_id"] for r in matching if r is not row),
                               "cross_source": public_overlap, "conflicting_masks": conflict})
            if row["source"] == "labelled" and (public_overlap or conflict or row is not labelled[0]):
                row["status"] = "excluded"
                reason = "Exact decoded CT duplicate: " + ";".join(r["case_id"] for r in matching if r is not row)
                row["exclusion_reason"] = "; ".join(filter(None, (row["exclusion_reason"], reason)))
    return duplicates


def split_counts(total):
    """Largest-remainder rounding gives 197/42/42 when total is 281."""
    exact = [total * fraction for fraction in config.SPLIT_FRACTIONS]
    counts = [int(value) for value in exact]
    for index in sorted(range(3), key=lambda i: (-(exact[i] - counts[i]), i))[:total - sum(counts)]:
        counts[index] += 1
    return counts


def make_splits(rows):
    """Balance tumour presence/size while keeping entire case IDs together."""
    usable = sorted((row for row in rows if row["status"] == "usable"), key=lambda r: r["case_id"])
    if len(usable) < 3 or len({row["patient_id"] for row in usable}) != len(usable):
        raise ValueError("Need at least three distinct usable patient/case IDs.")
    positive = sorted(row["tumor_volume_ml"] for row in usable if row["tumor_voxels"] > 0)
    cutoffs = np.quantile(positive, [1/3, 2/3]).tolist() if positive else []
    groups = defaultdict(list)
    for row in usable:
        if row["tumor_voxels"] == 0:
            group = "absent"
        else:
            group = ("small", "medium", "large")[int(np.searchsorted(cutoffs, row["tumor_volume_ml"], side="right"))]
        row["tumor_group"] = group
        groups[group].append(row)

    rng = random.Random(config.SPLIT_SEED)
    targets = split_counts(len(usable))
    splits = [[], [], []]
    # Allocate each stratum's expected share, then fill remaining capacity.
    for group in sorted(groups):
        members = groups[group][:]
        rng.shuffle(members)
        expected = [len(members) * count / len(usable) for count in targets]
        allocated = [0, 0, 0]
        for row in members:
            available = [i for i in range(3) if len(splits[i]) < targets[i]]
            index = max(available, key=lambda i: (expected[i] - allocated[i], targets[i] - len(splits[i]), -i))
            splits[index].append(row)
            allocated[index] += 1

    for subset in splits:
        subset.sort(key=lambda row: row["case_id"])
    validate_splits(splits, usable)
    return dict(zip(("train", "validation", "test"), splits)), cutoffs


def validate_splits(splits, usable):
    patients, hashes = set(), set()
    for subset in splits:
        for row in subset:
            if row["patient_id"] in patients or row["image_voxel_sha256"] in hashes:
                raise ValueError("Patient ID or duplicate CT overlaps between split rows.")
            patients.add(row["patient_id"])
            hashes.add(row["image_voxel_sha256"])
    if patients != {row["patient_id"] for row in usable}:
        raise ValueError("Split coverage differs from the usable cases.")
    if list(map(len, splits)) != split_counts(len(usable)):
        raise ValueError("Split counts differ from the planned usable-case proportions.")


def save_splits(splits, folder):
    """Never silently replace a different frozen patient assignment."""
    columns = ["case_id", "patient_id", "image", "mask", "tumor_group", "image_voxel_sha256"]
    planned = {name: [{key: str(row[key]) for key in columns} for row in rows] for name, rows in splits.items()}
    for name, rows in planned.items():
        path = folder / f"{name}.csv"
        if path.exists():
            with path.open(newline="", encoding="utf-8") as handle:
                if list(csv.DictReader(handle)) != rows:
                    raise ValueError(f"Existing {path.name} differs. Review before changing a frozen split.")
    for name, rows in planned.items():
        write_csv(folder / f"{name}.csv", rows, columns)


def save_overlays(root, training_rows, folder):
    """Choose QA examples only from training, including the smallest mass."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    positive = sorted((r for r in training_rows if r["tumor_voxels"] > 0), key=lambda r: r["tumor_volume_ml"])
    selected = [positive[i] for i in sorted({0, len(positive)//2, len(positive)-1})] if positive else []
    absent = next((r for r in training_rows if r["tumor_voxels"] == 0), None)
    if absent:
        selected.append(absent)
    folder.mkdir(parents=True, exist_ok=True)
    files = []
    for row in selected:
        image = nib.as_closest_canonical(nib.load(root / row["image"]))
        mask = nib.as_closest_canonical(nib.load(root / row["mask"]))
        labels = np.asanyarray(mask.dataobj)
        tumor_area = np.count_nonzero(labels == 2, axis=(0, 1))
        pancreas_area = np.count_nonzero(labels == 1, axis=(0, 1))
        z = int(np.argmax(tumor_area if tumor_area.max() else pancreas_area))
        present = np.flatnonzero(tumor_area)
        edge = int(present[np.argmin(tumor_area[present])]) if len(present) else z
        figure, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
        spacing = image.header.get_zooms()
        for axis, slice_index, title, overlay in zip(axes, (z, z, edge), ("CT", "Reference mask", "Smallest visible tumour slice" if len(present) else "Reference mask"), (False, True, True)):
            plane = np.asarray(image.dataobj[:, :, slice_index])
            axis.imshow(plane.T, cmap="gray", vmin=-100, vmax=250, origin="lower", aspect=float(spacing[1]/spacing[0]))
            if overlay:
                mask_plane = labels[:, :, slice_index].T
                axis.imshow(np.ma.masked_equal(mask_plane, 0), cmap=ListedColormap(["black", "#39c987", "#ed6572"]), vmin=0, vmax=2, alpha=0.55, origin="lower", aspect=float(spacing[1]/spacing[0]), interpolation="nearest")
            axis.set_title(f"{title}\nRAS axial slice {slice_index}", fontsize=10)
            axis.set_axis_off()
        figure.suptitle(f"{row['case_id']} | training QA | mass {row['tumor_volume_ml']:.2f} mL", fontsize=12)
        figure.legend(handles=[Patch(color="#39c987", label="Pancreas (1)"), Patch(color="#ed6572", label="Tumour/mass (2)")], loc="outside lower center", ncol=2)
        path = folder / f"{row['case_id']}.png"
        figure.savefig(path, dpi=140)
        plt.close(figure)
        files.append(path.name)
    return files


def read_audit_progress(path, identity):
    if not path.is_file():
        return {}
    try:
        saved = json.loads(path.read_text(encoding="utf-8"))
        if saved["identity"] == identity:
            if not isinstance(saved["rows"], list):
                raise ValueError("Invalid saved audit rows")
            return {row["image"]: row for row in saved["rows"]}
        print("The dataset or audit settings changed. Starting the audit again.", flush=True)
    except (OSError, ValueError, KeyError, TypeError):
        print("The saved audit progress could not be read. Starting the audit again.", flush=True)
    return {}


def reuse_audit_row(root, record, previous):
    if previous is None or previous.get("status") not in ("usable", "unlabelled"):
        return None
    if any(previous.get(key) != value for key, value in record.items()):
        return None
    try:
        for kind in ("image", "mask"):
            if record[kind] and file_digest(root / record[kind]) != previous.get(f"{kind}_file_sha256"):
                return None
    except OSError:
        return None
    return previous


def run_audit(dataset_dir=None, results_dir=None, splits_dir=None):
    root = Path(dataset_dir or config.DATASET_DIR).resolve()
    results = Path(results_dir or config.RESULTS_DIR)
    splits_folder = Path(splits_dir or config.SPLITS_DIR)
    started = time.monotonic()
    metadata, labels, records, inventory_issues = inspect_inventory(root)
    if inventory_issues:
        issue_path = results / "inventory_issues.json"
        write_json(issue_path, inventory_issues)
        raise ValueError(f"Dataset inventory does not match metadata. See {issue_path}.")
    progress_path = results / ".audit_progress.json"
    identity = {
        "dataset_root": str(root),
        "metadata_sha256": file_digest(root / "dataset.json"),
        "audit_code_sha256": file_digest(__file__),
        "numpy": np.__version__,
        "nibabel": nib.__version__,
    }
    saved_rows = read_audit_progress(progress_path, identity)
    print(f"Dataset: {root}\nAuditing {len(records)} original CT volumes, one at a time.", flush=True)
    if saved_rows:
        print(f"Checking {len(saved_rows)} saved entries against the source files.", flush=True)
    rows = []
    reused = 0
    for index, record in enumerate(records, 1):
        row = reuse_audit_row(root, record, saved_rows.get(record["image"]))
        if row is None:
            row = inspect_case(root, record, set(labels))
        else:
            reused += 1
        rows.append(row)
        saved_rows[record["image"]] = row
        # Save raw rows before duplicate status and split labels are added.
        if index % 10 == 0 or index == len(records):
            write_json(progress_path, {"identity": identity, "rows": list(saved_rows.values())})
        if index % 10 == 0 or index == len(records) or row["status"] == "excluded":
            print(f"{index}/{len(records)} | {record['case_id']} | {row['status']} | reused {reused} | {time.monotonic()-started:.1f}s", flush=True)
    duplicates = mark_duplicates(rows)
    write_csv(results / "duplicate_scans.csv", duplicates, ["case_id", "source", "matching_cases", "cross_source", "conflicting_masks"])
    exclusions = [row for row in rows if row["status"] == "excluded"]
    write_csv(results / "data_exclusions.csv", exclusions, ["case_id", "source", "image", "mask", "exclusion_reason"] if not exclusions else None)
    splits, cutoffs = make_splits(rows)
    save_splits(splits, splits_folder)
    overlay_files = save_overlays(root, splits["train"], results / "data_overlays")
    summary_path = results / "data_summary.csv"
    write_csv(summary_path, rows)
    report = {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_folder": root.name,
        "metadata_sha256": identity["metadata_sha256"],
        "summary_sha256": file_digest(summary_path),
        "metadata_labels": labels,
        "metadata_training_count": metadata["numTraining"],
        "metadata_unlabelled_count": metadata["numTest"],
        "audited_volumes": len(rows),
        "reused_audit_rows": reused,
        "status_counts": dict(Counter(row["status"] for row in rows)),
        "inventory_issues": inventory_issues,
        "duplicate_rows": len(duplicates),
        "warning_case_count": sum(bool(row["warnings"]) for row in rows),
        "patient_id_rule": "Each unique MSD pancreas_NNN case ID is treated as one patient; no separate patient linkage table is supplied.",
        "duplicate_rule": "SHA-256 of decoded RAS-ordered float32 voxels and shape; not proof against non-identical repeat acquisitions.",
        "split_seed": config.SPLIT_SEED,
        "split_fractions": config.SPLIT_FRACTIONS,
        "split_counts": {name: len(subset) for name, subset in splits.items()},
        "split_sha256": split_digest(splits_folder),
        "tumor_volume_tertile_cutoffs_ml": cutoffs,
        "stratum_counts": {name: dict(Counter(row["tumor_group"] for row in subset)) for name, subset in splits.items()},
        "split_policy": "Project 70/15/15 holdout; seed 42; tumour presence and volume tertiles; not a reproduced external protocol. Reference labels used only for split stratification and training QA.",
        "overlap_and_coverage_checks_passed": True,
        "overlay_files": overlay_files,
        "elapsed_seconds": round(time.monotonic()-started, 2),
        "versions": {"numpy": np.__version__, "nibabel": nib.__version__},
    }
    write_json(results / "data_audit.json", report)
    progress_path.unlink(missing_ok=True)
    (results / "inventory_issues.json").unlink(missing_ok=True)
    print(json.dumps(report, indent=2), flush=True)
    return report


if __name__ == "__main__":
    run_audit()
