# Using the trained model

Use these commands in the project folder on the remote Linux machine. The project saves two files in each run folder. Use `best.pt` for prediction and keep `last.pt` with the run folder if training may need to resume. Keep the exact code used for training, especially `config.py`, because the checkpoint records its checksum.

## Check a returned model

Copy the trained `best.pt` to `weights/model.pt`, or pass its location with `--checkpoint`. For seed 42, the full model's run normally writes `checkpoints/attention_fusion_s42/best.pt`. When the whole comparison is frozen, `results/frozen_comparison.json` identifies the full-model run selected by validation.

```bash
python run_project.py check-model --checkpoint checkpoints/attention_fusion_s42/best.pt
```

Record the printed variant, seed, epoch and checkpoint checksum. The command loads the matching model on CPU and checks its tensor shapes and values. It also reports whether the file came from the pilot. Use a completed comparison checkpoint for prediction and export.

## Predict on one CT

Install the core requirements on the remote machine. CPU works for this step. With the weights at `weights/model.pt`, run these commands with the path to your scan.

```bash
python run_project.py check-model
python run_project.py predict --scan /path/to/scan.nii.gz
```

The output is under `results/predictions/`. Open `overlay.png` to inspect the example slice and `run.json` for the time and export check. The mask is `segmentation.nii.gz`. It uses the scan's original shape and spatial alignment, with 0 for background, 1 for pancreas and 2 for tumour or mass.

For the viewer, install the optional requirements and run Streamlit on the remote machine.

```bash
python -m pip install -r requirements-demo.txt
python -m streamlit run app.py --server.address 127.0.0.1
```

If using SSH, forward port 8501 to view the app. Upload a CT, select **Predict mask**, move through the slices and turn each label on and off. Download the mask and inspect it with the source scan. To use a differently named file, set `PANCREAS_MODEL_CHECKPOINT` before launching the viewer.

## Transfer the model

```bash
python run_project.py export-model --checkpoint checkpoints/attention_fusion_s42/best.pt
```

Transfer `results/team9_model.zip`. It contains the model as `weights/model.pt`, the matching code, requirements and `model_details.json` with the model and data checksums. Extract it into a fresh folder, provide a CT, and follow the prediction commands above. Standalone prediction does not require the training dataset or patient split files.

## Finish the comparison and submission

Keep the original `splits/`, `results/` and all planned run folders under `checkpoints/` on persistent storage. A model ZIP supports prediction. The held-out study also requires those run artifacts and the audited Task07 dataset. On the training machine, finish the comparison using the same run plan, then run these commands.

```bash
python run_project.py freeze
python run_project.py evaluate
python run_project.py demo
python run_project.py package review
```

Review the test overlays and write the observed errors in `results/failure_notes.md`. Add the measured results to the report and slides, record the viewer and run `python run_project.py package submission`. The final names are listed in README.md.
