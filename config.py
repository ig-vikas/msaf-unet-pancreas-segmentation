"""Shared project paths and experiment settings."""

import os
import hashlib
from importlib.util import find_spec
from pathlib import Path


# Set this environment variable only when the project is in a different folder.
PROJECT_ROOT = Path(
    os.environ.get("PANCREAS_PROJECT_ROOT", Path(__file__).resolve().parent)
).expanduser().resolve()

try:
    IN_COLAB = find_spec("google.colab") is not None
except ModuleNotFoundError:
    IN_COLAB = False

REMOTE_RUN = os.environ.get("PANCREAS_REMOTE_RUN") == "1"
if REMOTE_RUN:
    PROJECT_RUNTIME = "remote"
else:
    PROJECT_RUNTIME = "colab" if IN_COLAB else "local"

DATASET_NAME = "Task07_Pancreas"
DATA_DIR = PROJECT_ROOT / "data"
# By default the extracted download sits beside this file.
DATASET_DIR = PROJECT_ROOT / DATASET_NAME
PROCESSED_DIR = DATA_DIR / "processed"
SPLITS_DIR = PROJECT_ROOT / "splits"
RESULTS_DIR = PROJECT_ROOT / "results"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
# Point this to a returned project checkpoint without editing the experiment files.
MODEL_CHECKPOINT_SETTING = os.environ.get("PANCREAS_MODEL_CHECKPOINT")
MODEL_CHECKPOINT = Path(MODEL_CHECKPOINT_SETTING or "weights/model.pt").expanduser()
if not MODEL_CHECKPOINT.is_absolute():
    MODEL_CHECKPOINT = PROJECT_ROOT / MODEL_CHECKPOINT

# In Colab, working data can live on the VM while results stay in Drive.
RUNTIME_ROOT = Path("/content/pancreas_runtime") if PROJECT_RUNTIME == "colab" else PROJECT_ROOT
RUNTIME_DATASET_DIR = RUNTIME_ROOT / "data" / DATASET_NAME

SPLIT_SEED = 42
SPLIT_FRACTIONS = (0.70, 0.15, 0.15)

# Shared by all four models.
CT_WINDOW = (-100.0, 250.0)
ARCHITECTURE_VERSION = 1
SLICES_PER_CASE = 12
BATCH_SIZE = 4
PREDICTION_BATCH_SIZE = 4
LEARNING_RATE = 0.0003
WEIGHT_DECAY = 0.0001
TRAINING_EPOCHS = 30
AMP = False

EXPERIMENT_FILES = (
    "config.py", "prepare_data.py", "dataset.py", "model.py", "predict.py",
    "train.py", "evaluate.py", "check_pipeline.py", "model_files.py", "requirements.txt",
)


def source_digest():
    """Identify the preparation, model and scoring code used for an experiment."""
    digest = hashlib.sha256()
    for name in EXPERIMENT_FILES:
        digest.update(name.encode("utf-8"))
        digest.update((PROJECT_ROOT / name).read_bytes())
    return digest.hexdigest()


def prepare_folders():
    """Create the folders needed for setup and the next data-preparation phase."""
    if not (PROJECT_ROOT / "config.py").is_file():
        raise FileNotFoundError(
            f"Project files were not found in {PROJECT_ROOT}. "
            "Check PANCREAS_PROJECT_ROOT."
        )

    for folder in (DATA_DIR, PROCESSED_DIR, SPLITS_DIR, RESULTS_DIR):
        folder.mkdir(parents=True, exist_ok=True)
