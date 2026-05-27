"""Native axial CT slices for the models and whole-volume prediction."""

import csv
import json
import re
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset

import config
from prepare_data import file_digest, normalize_slice, ras_array, split_digest

SPLIT_NAMES = ("train", "validation", "test")


def read_splits(folder=config.SPLITS_DIR, audit_path=config.RESULTS_DIR / "data_audit.json"):
    """Load the saved case lists and check that they are still disjoint."""
    lists = {}
    seen = {name: set() for name in ("case_id", "patient_id", "image_voxel_sha256")}
    for split in SPLIT_NAMES:
        path = Path(folder) / f"{split}.csv"
        if not path.is_file():
            raise FileNotFoundError(f"Missing {path.name}. Finish the dataset audit first.")
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            raise ValueError(f"{path.name} has no cases.")
        for row in rows:
            for column, values in seen.items():
                value = row.get(column)
                if not value or value in values:
                    raise ValueError(f"Missing or repeated {column} in {path.name}: {value}")
                values.add(value)
            if (not re.fullmatch(r"pancreas_\d+", row["case_id"])
                    or row.get("image") != f"imagesTr/{row['case_id']}.nii.gz"
                    or row.get("mask") != f"labelsTr/{row['case_id']}.nii.gz"):
                raise ValueError(f"Unexpected labelled file paths for {row['case_id']}.")
        lists[split] = rows
    if not Path(audit_path).is_file():
        raise FileNotFoundError("Run the complete audit before using the patient lists.")
    audit = json.loads(Path(audit_path).read_text(encoding="utf-8"))
    counts = {split: len(rows) for split, rows in lists.items()}
    if (audit.get("split_counts") != counts or audit.get("split_sha256") != split_digest(folder)
            or audit.get("overlap_and_coverage_checks_passed") is not True):
        raise ValueError("The patient lists do not match the completed audit.")
    return lists


def open_scan(source):
    """Keep the original header and affine for the exported mask."""
    image = source
    if isinstance(source, (str, Path)):
        try:
            image = nib.load(source)
        except nib.filebasedimages.ImageFileError as error:
            raise ValueError(f"Could not read {Path(source).name} as a NIfTI CT.") from error
    if not isinstance(image, (nib.Nifti1Image, nib.Nifti2Image)):
        raise ValueError("Choose a 3D NIfTI CT scan.")
    if len(image.shape) != 3 or min(image.shape[:]) < 1:
        raise ValueError("Choose a 3D NIfTI CT scan.")
    affine = image.affine
    if affine is None or not np.isfinite(affine).all() or abs(np.linalg.det(affine[:3, :3])) < 1e-10:
        raise ValueError("The scan has an invalid spatial affine.")
    return image


def audited_data_digest():
    """Tie training and scoring to the completed scan and mask inventory."""
    audit = json.loads((config.RESULTS_DIR / "data_audit.json").read_text(encoding="utf-8"))
    summary_hash = file_digest(config.RESULTS_DIR / "data_summary.csv")
    if audit.get("summary_sha256") != summary_hash:
        raise ValueError("The case summary differs from the completed audit. Run the audit again.")
    return summary_hash


def verify_dataset(root, subsets=SPLIT_NAMES):
    """Check the actual compressed files against the completed audit."""
    root = Path(root)
    lists = read_splits()
    audit = json.loads((config.RESULTS_DIR / "data_audit.json").read_text(encoding="utf-8"))
    if audit.get("metadata_sha256") != file_digest(root / "dataset.json"):
        raise ValueError("The dataset metadata differs from the completed audit. Run the audit again.")
    audited_data_digest()
    with (config.RESULTS_DIR / "data_summary.csv").open(newline="", encoding="utf-8") as handle:
        summary = {row["case_id"]: row for row in csv.DictReader(handle) if row["source"] == "labelled"}
    for split in subsets:
        cases = lists[split]
        for case in cases:
            row = summary.get(case["case_id"], {})
            if row.get("status") != "usable" or row.get("image_voxel_sha256") != case["image_voxel_sha256"]:
                raise ValueError(f"{case['case_id']} does not match a usable case in the audit.")
            for kind in ("image", "mask"):
                path = root / case[kind]
                if row.get(f"{kind}_file_sha256") != file_digest(path):
                    raise ValueError(f"{case['case_id']} {kind} changed after the audit.")
        print(f"Verified {len(cases)} {split} scan/mask pairs.", flush=True)
    return lists


def read_pair(root, case):
    image = open_scan(Path(root) / case["image"])
    mask = nib.load(Path(root) / case["mask"])
    if mask.shape != image.shape or not np.allclose(image.affine, mask.affine, atol=1e-4, rtol=0):
        raise ValueError(f"{case['case_id']} has a scan and mask on different grids.")
    return image, mask


class TrainingSlices(Dataset):
    """Read one case at a time and sample 12 slices from its reference mask."""

    def __init__(self, root, cases, seed, epoch):
        self.root = Path(root)
        self.cases = list(cases)
        self.rng = np.random.default_rng(np.random.SeedSequence([seed, epoch]))
        self.rng.shuffle(self.cases)
        self.current_case = None

    def __len__(self):
        return len(self.cases) * config.SLICES_PER_CASE

    def load_case(self, index):
        self.current_case = None
        self.ct = self.labels = None
        image, mask = read_pair(self.root, self.cases[index])
        self.ct, self.labels = ras_array(image), ras_array(mask)
        tumour = (self.labels == 2).any(axis=(0, 1))
        pancreas = (self.labels == 1).any(axis=(0, 1))
        groups = (tumour, pancreas & ~tumour, ~(pancreas | tumour))
        available = np.arange(self.labels.shape[2])
        slices = []
        for group in groups:
            candidates = np.flatnonzero(group)
            if not candidates.size:
                candidates = available
            count = config.SLICES_PER_CASE // len(groups)
            slices.extend(self.rng.choice(candidates, size=count, replace=len(candidates) < count))
        self.rng.shuffle(slices)
        self.slices = slices
        self.current_case = index

    def __getitem__(self, index):
        case_index, sample = divmod(index, config.SLICES_PER_CASE)
        if self.current_case != case_index:
            self.load_case(case_index)
        z = self.slices[sample]
        plane = normalize_slice(self.ct[:, :, z])
        labels = self.labels[:, :, z]
        if self.rng.random() < 0.5:
            plane, labels = plane[::-1], labels[::-1]
        return (
            torch.from_numpy(np.ascontiguousarray(plane[None])),
            torch.from_numpy(np.ascontiguousarray(labels, dtype=np.int64)),
        )
