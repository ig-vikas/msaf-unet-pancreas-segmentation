"""Run real-data checks before starting the training pilot."""

import csv
import io
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torch

import config
from dataset import TrainingSlices, audited_data_digest, read_pair, read_splits, verify_dataset
from model import PancreasUNet, VARIANTS
from predict import predict_volume, save_mask
from prepare_data import normalize_slice, ras_array, restore_labels, split_digest, write_json
from train import segmentation_loss
from evaluate import score_volume


def check_geometry_and_metrics():
    """Exercise axis reversals and the empty-class and distance rules."""
    labels = np.zeros((9, 11, 13), dtype=np.uint8)
    labels[2:5, 3:7, 4:8] = 1
    labels[3, 4, 5] = 2
    affines = (
        np.diag([0.7, 0.8, 2.5, 1.0]),
        np.array([[0, -0.8, 0, 20], [0.7, 0, 0, 10], [0, 0, -2.5, 30], [0, 0, 0, 1]]),
    )
    for affine in affines:
        image = nib.Nifti1Image(labels, affine)
        restored = restore_labels(ras_array(image), image)
        if not np.array_equal(restored, labels):
            raise ValueError("A flipped or permuted grid changed labels in the round trip.")
    # Encode the expected classes as intensities to check prediction without weights.
    class IntensityLabels(torch.nn.Module):
        def forward(self, images):
            classes = torch.tensor([0.0, 0.5, 1.0], device=images.device).view(1, 3, 1, 1)
            return -(images - classes).abs()

    low, high = config.CT_WINDOW
    ct = (low + labels.astype(np.float32) * (high - low) / 2).astype(np.float32)
    image = nib.Nifti1Image(ct, affines[1])
    prediction = predict_volume(IntensityLabels(), image, "cpu", batch_size=4)
    if not np.array_equal(prediction, labels):
        raise ValueError("Prediction changed labels during slice batching or orientation restoration.")
    empty = np.zeros_like(labels)
    missed = score_volume(empty, labels, affines[0])
    assert all(row["dice"] == 0 and row["recall"] == 0 for row in missed)
    absent = score_volume(labels, empty, affines[0])
    assert all(row["dice"] is None and row["false_positive_ml"] > 0 for row in absent)
    perfect = score_volume(labels, labels, affines[0], surface_distance=True)
    assert all(row["dice"] == 1 and row["hd95_mm"] == 0 for row in perfect)
    one = np.zeros_like(labels)
    shifted = np.zeros_like(labels)
    one[3, 4, 5] = shifted[3, 4, 6] = 2
    distance = score_volume(one, shifted, affines[0], surface_distance=True)[1]
    assert np.isclose(distance["hd95_mm"], 2.5)
    return {"orientation_round_trip": True, "prediction_batch_and_round_trip": True, "empty_class_rules": True,
            "perfect_overlap": True, "anisotropic_distance": True}


def check_preparation(root):
    train_cases = read_splits()["train"]
    by_id = {case["case_id"]: case for case in train_cases}
    with (config.RESULTS_DIR / "data_summary.csv").open(newline="", encoding="utf-8") as handle:
        positive = [
            row for row in csv.DictReader(handle)
            if row["case_id"] in by_id and row["source"] == "labelled" and float(row["tumor_volume_ml"]) > 0
        ]
    if not positive:
        raise ValueError("The training lists need a reference tumour for the small-mass check.")
    positive.sort(key=lambda row: float(row["tumor_volume_ml"]))
    selected = [positive[i] for i in sorted({0, len(positive) // 2})]
    results = config.RESULTS_DIR / "preflight"
    results.mkdir(parents=True, exist_ok=True)
    records = []
    for row in selected:
        case_id = row["case_id"]
        image, mask = read_pair(root, by_id[case_id])
        ct, labels = ras_array(image), ras_array(mask)
        original = np.asanyarray(mask.dataobj)
        restored = restore_labels(labels, image)
        if not np.array_equal(restored, original):
            raise ValueError(f"{case_id}: the orientation round trip changed the mask.")
        with tempfile.TemporaryDirectory() as temporary:
            exported = nib.load(save_mask(restored, image, Path(temporary) / "mask.nii.gz"))
            if not np.array_equal(np.asanyarray(exported.dataobj), original):
                raise ValueError(f"{case_id}: the exported NIfTI changed label values.")
            if not np.allclose(exported.affine, image.affine, atol=1e-4, rtol=0):
                raise ValueError(f"{case_id}: the exported NIfTI changed the affine.")

        z = int((labels == 2).sum(axis=(0, 1)).argmax())
        plane = normalize_slice(ct[:, :, z])
        if not np.isfinite(plane).all() or plane.min() < 0 or plane.max() > 1:
            raise ValueError(f"{case_id}: the prepared slice has invalid intensities.")
        fig, axes = plt.subplots(1, 2, figsize=(8, 4), constrained_layout=True)
        spacing = nib.as_closest_canonical(image).header.get_zooms()
        for axis, data, low, high, title in zip(
            axes, (ct[:, :, z], plane), (config.CT_WINDOW[0], 0),
            (config.CT_WINDOW[1], 1), ("Original CT", "Prepared slice")
        ):
            axis.imshow(data.T, cmap="gray", vmin=low, vmax=high,
                        origin="lower", aspect=spacing[1] / spacing[0])
            for label, color in ((1, "#26a880"), (2, "#ee6873")):
                active = labels[:, :, z].T == label
                if active.any() and not active.all():
                    axis.contour(active, levels=[0.5], colors=[color])
            axis.set_title(title)
            axis.set_axis_off()
        fig.suptitle(f"{case_id} · axial slice {z} · {float(row['tumor_volume_ml']):.2f} mL tumour")
        name = f"{case_id}_preparation.png"
        fig.savefig(results / name, dpi=150)
        plt.close(fig)
        records.append({
            "case_id": case_id, "original_shape": image.shape, "ras_shape": ct.shape,
            "tumour_voxels": int(np.count_nonzero(labels == 2)),
            "round_trip_and_export_passed": True, "overlay": name,
        })
    return records


def check_models(root, device):
    cases = read_splits()["train"]
    case = next((row for row in cases if row["tumor_group"] != "absent"), cases[0])
    samples = TrainingSlices(root, [case], config.SPLIT_SEED, 0)
    size = config.BATCH_SIZE if torch.device(device).type == "cuda" else min(config.BATCH_SIZE, 2)
    if size < 1 or size > len(samples):
        raise ValueError("Choose a batch size between one and the number of slices per case.")
    batch = [samples[index] for index in range(size)]
    if not any((mask == 2).any() for _, mask in batch):
        for index in range(size, len(samples)):
            candidate = samples[index]
            if (candidate[1] == 2).any():
                batch[-1] = candidate
                break
        else:
            raise ValueError(f"{case['case_id']} has no tumour in the sampled training slices.")
    images, masks = zip(*batch)
    images, masks = torch.stack(images).to(device), torch.stack(masks).to(device)
    if not torch.isin(masks, torch.tensor([0, 1, 2], device=device)).all():
        raise ValueError("The loader returned unexpected labels.")
    records = []
    for variant in VARIANTS:
        model = PancreasUNet(variant).to(device)
        logits = model(images)
        expected = (images.shape[0], 3, *images.shape[-2:])
        if logits.shape != expected or not torch.isfinite(logits).all():
            raise ValueError(f"{variant} returned an invalid prediction tensor.")
        loss = segmentation_loss(logits, masks)
        if not torch.isfinite(loss):
            raise ValueError(f"{variant} returned a non-finite loss.")
        loss.backward()
        gradients = [parameter.grad for parameter in model.parameters()]
        if any(grad is None or not torch.isfinite(grad).all() for grad in gradients):
            raise ValueError(f"{variant} has missing or non-finite gradients.")
        if not any(torch.count_nonzero(grad).item() for grad in gradients):
            raise ValueError(f"{variant} has only zero gradients.")
        with io.BytesIO() as buffer:
            torch.save(model.state_dict(), buffer)
            buffer.seek(0)
            copy = PancreasUNet(variant).to(device).eval()
            copy.load_state_dict(torch.load(buffer, map_location="cpu", weights_only=True))
        with torch.inference_mode():
            torch.testing.assert_close(copy(images), model.eval()(images))
        records.append({
            "variant": variant, "case_id": case["case_id"], "input_shape": list(images.shape),
            "tumour_input_voxels": int(torch.count_nonzero(masks == 2)),
            "parameters": sum(p.numel() for p in model.parameters()), "loss": loss.item(),
            "gradient_and_reload_passed": True,
        })
        del model, copy, logits, loss, gradients
    return records


def run_preflight(root, device):
    rules = check_geometry_and_metrics()
    verify_dataset(root, ("train", "validation"))
    preparation = check_preparation(root)
    models = check_models(root, device)
    report = {"preparation": preparation, "models": models, "metric_and_geometry_checks": rules,
              "source_sha256": config.source_digest(), "device": str(device),
              "data_summary_sha256": audited_data_digest(),
              "split_sha256": split_digest(), "architecture_version": config.ARCHITECTURE_VERSION}
    write_json(config.RESULTS_DIR / "preflight" / "checks.json", report)
    return report


if __name__ == "__main__":
    run_preflight(config.DATASET_DIR, "cpu")
    print(f"Saved {config.RESULTS_DIR / 'preflight' / 'checks.json'}")
