# Pancreas and pancreatic tumour segmentation

Team 9 is comparing a plain U-Net with versions that add attention, multi-scale fusion, or both. All four models use the same axial CT slices and patient lists. Each produces a three-label mask on the original CT grid.

## Set up the remote machine

Use a remote Linux machine with Python 3.11 or later and persistent storage. A CPU machine can start the audit and preflight. The pilot, training and held-out evaluation use an NVIDIA GPU. Keep the project folder on persistent storage so the audit, split lists and weights survive a disconnect.

Use this repository as the project folder. Put the extracted `Task07_Pancreas` folder next to `config.py`. It should contain `dataset.json`, `imagesTr/`, `labelsTr/` and `imagesTs/`.

If the dataset is only on your computer, you can download the public archive directly on the remote machine instead of uploading your copy. Project MONAI lists the download and checksum. [Dataset source](https://raw.githubusercontent.com/Project-MONAI/MONAI/dev/monai/apps/datasets.py)

```bash
curl -fL -C - -o Task07_Pancreas.tar https://msd-for-monai.s3-us-west-2.amazonaws.com/Task07_Pancreas.tar
echo '4f7080cfca169fa8066d17ce6eb061e4  Task07_Pancreas.tar' | md5sum -c - &&
  tar -xf Task07_Pancreas.tar
```

From the project folder, create and activate a Python environment. Use the Linux CPU or CUDA command from the PyTorch install selector, then install the remaining requirements. On a GPU machine, run `nvidia-smi` to check that the device is visible. [PyTorch install selector](https://pytorch.org/get-started/locally/)

```bash
python3 -m venv .venv
source .venv/bin/activate
```

```bash
python -m pip install -r requirements.txt
python run_project.py environment
```

Read `results/environment_remote.json` for the measured versions, free disk space and RAM. The CUDA tensor check should say `passed` on a GPU machine. A CPU machine reports `not_available` for CUDA.

The audit and preflight can run before any weights are available. The pilot is the first step that updates weights. Start it when the team is ready to train.

## Audit the data and check the models

```bash
python run_project.py audit
python run_project.py preflight --device cpu
python run_project.py package checks
```

On a GPU machine, `python run_project.py preflight` checks all four models at the configured training batch size. CPU preflight uses up to two slices per batch and records its device in `results/preflight/checks.json`. The GPU pilot measures memory and time for training.

With the dataset and Python environment set up on a laptop, run `python check_environment.py`, `python prepare_data.py` and `python check_pipeline.py` for the same checks on CPU.

The local audit checked 281 labelled CTs and 139 unlabelled CTs. It excluded four labelled cases after finding duplicate volumes and saved 194 training, 42 validation and 41 test cases using seed 42 and a 70/15/15 split. MSD case IDs are the grouping keys. These identifiers cannot link different scans belonging to the same person. Scored evaluation uses the labelled test split.

Review the exclusions and duplicates, the three CSV files in `splits/`, the training overlays in `results/data_overlays/` and the preparation examples in `results/preflight/`. The preflight report also records each model's gradient and checkpoint reload checks. `results/team9_checks.zip` collects these files for review.

The loader clips CT values to −100 to 250 HU and scales them to 0–1. It keeps the native in-plane size and uses axis reordering and flips for orientation. Each training case supplies 12 slices per epoch from tumour, pancreas-only and background groups when available. Validation and prediction cover every slice.

If the audit stops, rerun the same command in the project folder. It verifies source hashes and reuses unchanged entries from its last saved progress file. For long runs through SSH, keep the terminal session open in a tool such as `tmux` if available.

To reuse the completed laptop audit on the GPU machine, extract the laptop's `results/team9_checks.zip` into the project folder. Keep the same Task07 files and saved `splits/`. GPU preflight verifies the training and validation file hashes against the audit.

## Train and evaluate

First run the pilot and read `results/pilot.json` for the learning check, checkpoint resume, peak GPU memory and estimated time per complete epoch.

```bash
python run_project.py pilot
```

Choose one epoch budget for the whole comparison after the pilot. The planned default is 30 per model. The first `train` command records the budget and run list in `results/comparison_plan.json`. Add `--repeats` to that first command if time allows the optional seed 43 and 44 runs. For one run at a time, add `--run NUMBER` using the printed list.

```bash
python run_project.py train --epochs 30
python run_project.py freeze
python run_project.py evaluate
```

After a disconnect, repeat the same training command and settings. Each run saves `last.pt`, `best.pt`, `epochs.csv` and `settings.json` under `checkpoints/`. The freeze step records the selected weights and validation curves. Evaluation writes per-case scores, model comparisons, and overlays for held-out cases. Review the selected examples and write observations in `results/failure_notes.md`.

## Show a prediction and collect the results

```bash
python run_project.py demo
python run_project.py package review
```

The demo command uses the frozen comparison and a training case by default. Pass `--scan /path/to/scan.nii.gz` to use a specific CT. The output folder `results/demo/` contains the mask, an axial overlay and a timed run record. For an interactive recording, install `requirements-demo.txt` on the remote machine and start `python -m streamlit run app.py --server.address 127.0.0.1`. Open it over an SSH tunnel if using SSH. [Streamlit configuration](https://docs.streamlit.io/develop/api-reference/configuration/config.toml)

Use `results/team9_review.zip` and its `review_results.md` to fill in the measured findings. Save the final files with these names before running `python run_project.py package submission`:

- `REPORTS/Team 9 Final Project Report.docx`
- `REPORTS/Team 9 Final Review.pptx`
- `results/demo/demo.mp4`

The final ZIP contains the evidence, executable code and the weights selected for the demo. See [the method worksheet](BASE_PAPER.md) for the model design. Local review drafts and run outputs are kept separately.

## Use a returned checkpoint

If you receive just the trained weights, copy the project's `best.pt` to `weights/model.pt`. The following commands work with that file and a CT, including on the remote CPU. An explicit `--checkpoint` can also be used for each command.

```bash
python run_project.py check-model
python run_project.py predict --scan /path/to/scan.nii.gz
python run_project.py export-model
```

See [MODEL_HANDOVER.md](MODEL_HANDOVER.md) for the standalone output files, the portable model ZIP and what to keep for the four-model evaluation.
