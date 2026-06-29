"""Generate segmentation masks and display them in the scan viewer."""

import json
import tempfile
import time
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

import config
from dataset import audited_data_digest, open_scan, read_splits
from predict import load_model, predict_volume, save_mask
from prepare_data import file_digest, ras_array, split_digest


def demo_checkpoint():
    path = config.RESULTS_DIR / "frozen_comparison.json"
    if not path.is_file():
        raise FileNotFoundError("Freeze the comparison or add your trained project checkpoint to weights/model.pt.")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest["split_sha256"] != split_digest():
        raise ValueError("The demo checkpoint uses a different patient split.")
    if manifest["source_sha256"] != config.source_digest() or manifest["data_summary_sha256"] != audited_data_digest():
        raise ValueError("Use the source and data audit saved with the frozen comparison.")
    run = manifest["demo_run"]
    entry = next(row for row in manifest["runs"] if row["run"] == run)
    checkpoint = config.CHECKPOINTS_DIR / run / "best.pt"
    digest = entry["checkpoint_sha256"]
    if file_digest(checkpoint) != digest:
        raise ValueError("The saved demo weights differ from the frozen checkpoint.")
    return checkpoint, digest


def demo_cases():
    examples = {}
    for case in read_splits()["train"]:
        path = config.DATASET_DIR / case["image"]
        if path.is_file():
            examples[case["case_id"]] = path
    return examples


def run_demo(scan_path, destination, device="cpu", checkpoint=None):
    frozen_comparison = checkpoint is None
    if frozen_comparison:
        weights, digest = demo_checkpoint()
    else:
        weights = Path(checkpoint)
        digest = file_digest(weights) if weights.is_file() else None
    model, saved = load_model(weights, device)
    if digest != file_digest(weights):
        raise ValueError("The weights changed while they were being loaded. Try again after the copy finishes.")
    if saved["pilot"]:
        raise ValueError("Use trained comparison weights to predict a scan.")
    image = open_scan(scan_path)
    started = time.perf_counter()
    prediction = predict_volume(model, image, device)
    elapsed = time.perf_counter() - started
    path = save_mask(prediction, image, destination)
    exported = nib.load(path)
    if exported.shape != image.shape or not np.allclose(exported.affine, image.affine, atol=1e-4, rtol=0):
        raise ValueError("The saved mask does not match the original scan grid.")
    ct = ras_array(image)
    labels = ras_array(exported)
    return ct, labels, nib.as_closest_canonical(image).header.get_zooms()[:2], {
        "variant": saved["variant"],
        "scan_file": Path(scan_path).name,
        "checkpoint_epoch": saved["epoch"],
        "checkpoint_sha256": digest,
        "frozen_comparison": frozen_comparison,
        "mask_sha256": file_digest(path),
        "device": str(device),
        "prediction_seconds": elapsed,
        "shape": list(image.shape),
        "export_grid_matches": True,
    }


def draw_slice(ct, labels, spacing, z, pancreas=True, tumour=True):
    low, high = config.CT_WINDOW
    fig, axis = plt.subplots(figsize=(6, 5), constrained_layout=True)
    axis.imshow(ct[:, :, z].T, cmap="gray", vmin=low, vmax=high,
                origin="lower", aspect=spacing[1] / spacing[0])
    for value, color, visible in ((1, "#26a880", pancreas), (2, "#ee6873", tumour)):
        active = labels[:, :, z].T == value
        if visible and active.any() and not active.all():
            axis.contour(active, levels=[0.5], colors=[color], linewidths=1)
    axis.set_title(f"Axial slice {z + 1} / {ct.shape[2]}")
    axis.set_axis_off()
    return fig


def main():
    import streamlit as st

    st.set_page_config(page_title="Pancreas CT viewer", layout="centered")
    st.title("Pancreas CT viewer")
    st.caption("Research demonstration · green pancreas · pink tumour/mass")
    def clear_prediction():
        st.session_state.pop("prediction", None)
        st.session_state.pop("preview_slice", None)

    checkpoint = None
    if config.MODEL_CHECKPOINT_SETTING or config.MODEL_CHECKPOINT.is_file():
        checkpoint = config.MODEL_CHECKPOINT

    try:
        examples = demo_cases()
    except (OSError, ValueError):
        examples = {}
    options = ("Training example", "Upload CT") if examples else ("Upload CT",)
    source = st.radio("Scan source", options, horizontal=True, on_change=clear_prediction)
    upload = None
    scan = None
    if source == "Training example":
        case = st.selectbox("Training case", list(examples), on_change=clear_prediction)
        scan = examples[case]
    else:
        upload = st.file_uploader("CT volume (.nii or .nii.gz)", max_upload_size=512,
                                  on_change=clear_prediction)
    ready = checkpoint.is_file() if checkpoint is not None else (config.RESULTS_DIR / "frozen_comparison.json").is_file()
    if not ready:
        st.info("Add the trained checkpoint to weights/model.pt, or finish and freeze the comparison.")
    if st.button("Predict mask", disabled=not ready or (scan is None and upload is None)):
        clear_prediction()
        try:
            with st.spinner("Predicting each slice…"), tempfile.TemporaryDirectory() as directory:
                directory = Path(directory)
                if upload is not None:
                    name = upload.name.lower()
                    if not name.endswith((".nii", ".nii.gz")):
                        raise ValueError("Choose a .nii or .nii.gz file.")
                    scan = directory / ("scan.nii.gz" if name.endswith(".gz") else "scan.nii")
                    scan.write_bytes(upload.getvalue())
                destination = directory / "segmentation.nii.gz"
                ct, labels, spacing, report = run_demo(scan, destination, checkpoint=checkpoint)
                if upload is not None:
                    report["scan_file"] = upload.name
                st.session_state.prediction = ct, labels, spacing, report, destination.read_bytes()
        except (OSError, ValueError, KeyError, RuntimeError, nib.filebasedimages.ImageFileError) as error:
            st.error(f"Could not predict this scan. {error}")
    if "prediction" not in st.session_state:
        return
    ct, labels, spacing, report, file_bytes = st.session_state.prediction
    z = 0
    if ct.shape[2] > 1:
        z = st.slider("Axial slice", 0, ct.shape[2] - 1, ct.shape[2] // 2, key="preview_slice")
    pancreas = st.checkbox("Pancreas", value=True)
    tumour = st.checkbox("Tumour/mass", value=True)
    fig = draw_slice(ct, labels, spacing, z, pancreas, tumour)
    st.pyplot(fig)
    plt.close(fig)
    st.caption(f"{report['scan_file']} · {report['prediction_seconds']:.1f} seconds on {report['device']}")
    st.download_button("Download mask", file_bytes, "segmentation.nii.gz", "application/gzip")


if __name__ == "__main__":
    main()
