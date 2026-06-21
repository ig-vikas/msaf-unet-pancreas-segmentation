"""Predict every slice and put the result back on the input CT grid."""

import math
import re
from collections.abc import Mapping
from pathlib import Path
from pickle import UnpicklingError

import nibabel as nib
import numpy as np
import torch

import config
from dataset import open_scan
from model import PancreasUNet, VARIANTS
from prepare_data import normalize_slice, ras_array, restore_labels


def load_model(checkpoint, device):
    checkpoint = Path(checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Weights are missing at {checkpoint}. Use the project's trained best.pt.")
    try:
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    except (EOFError, OSError, RuntimeError, ValueError, UnpicklingError) as error:
        raise ValueError(f"Could not read {checkpoint.name}. Copy the original project best.pt again.") from error
    if not isinstance(saved, Mapping) or "settings" in saved:
        raise ValueError("Prediction needs the project's best.pt. The last.pt file is for resuming training.")
    if saved.get("architecture_version") != config.ARCHITECTURE_VERSION:
        raise ValueError("The checkpoint uses a different model version.")
    window = saved.get("ct_window")
    if not isinstance(window, (tuple, list)) or tuple(window) != config.CT_WINDOW:
        raise ValueError("The checkpoint and current CT window differ.")
    if saved.get("source_sha256") != config.source_digest():
        raise ValueError("Use the preparation and model files saved with these weights.")
    epoch = saved.get("epoch")
    score = saved.get("validation_score")
    if (saved.get("variant") not in VARIANTS or type(epoch) is not int or epoch < 1
            or type(saved.get("seed")) is not int or type(saved.get("pilot")) is not bool
            or type(score) not in (int, float) or not math.isfinite(score) or not 0 <= score <= 1):
        raise ValueError("The checkpoint is missing valid model or training details.")
    for field in ("split_sha256", "data_summary_sha256"):
        digest = saved.get(field)
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError(f"The checkpoint is missing a valid {field}.")
    state = saved.get("model")
    if not isinstance(state, Mapping) or not state or any(
        not isinstance(value, torch.Tensor) or not value.is_floating_point()
        or not torch.isfinite(value).all().item() for value in state.values()
    ):
        raise ValueError("The checkpoint contains missing or invalid weights.")
    model = PancreasUNet(saved["variant"])
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as error:
        raise ValueError("The checkpoint weights do not fit the saved model variant.") from error
    return model.to(device).eval(), saved


@torch.inference_mode()
def predict_volume(model, image, device, batch_size=config.PREDICTION_BATCH_SIZE):
    if not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("Prediction batch size must be a positive integer.")
    image = open_scan(image)
    ct = ras_array(image)
    if min(ct.shape[:2]) < 8 or not np.isfinite(ct).all():
        raise ValueError("The CT has invalid intensities or is too small for the model.")
    labels = np.empty(ct.shape, dtype=np.uint8)
    model.eval()
    for start in range(0, ct.shape[2], batch_size):
        stop = min(start + batch_size, ct.shape[2])
        planes = [normalize_slice(ct[:, :, z]) for z in range(start, stop)]
        batch = torch.from_numpy(np.stack(planes)[:, None]).to(device)
        predictions = model(batch).argmax(dim=1).cpu().numpy()
        labels[:, :, start:stop] = predictions.transpose(1, 2, 0)
    return restore_labels(labels, image)


def save_mask(labels, image, path):
    if labels.shape != image.shape:
        raise ValueError("The exported mask must match the original CT shape.")
    header = image.header.copy()
    header.set_data_dtype(np.uint8)
    header.set_slope_inter(1, 0)
    header.set_intent("label")
    result = image.__class__(labels.astype(np.uint8, copy=False), image.affine, header)
    result.header.set_slope_inter(1, 0)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(result, path)
    return path
