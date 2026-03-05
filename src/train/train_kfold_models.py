from ultralytics import YOLO, settings
import os
import shutil
import torch


# ==========================
# EXPERIMENT CONFIG
# ==========================

# Resolve project paths from this file location: src/train/train_kfold_models.py.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
FOLDS_DIR = os.path.join(PROJECT_ROOT, "folds")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
WEIGHTS_DIR = os.path.join(SRC_DIR, "weights")

# Use GPU 0 when CUDA is available; otherwise use CPU.
DEVICE = 0 if torch.cuda.is_available() else "cpu"
CLEAN_RESULTS_DIR = False  # Set True to delete results/ before a new run.

# YOLO checkpoints to train on each fold.
MODEL_NAMES = [
    "yolo26n.pt",
    "yolo26s.pt",
    "yolo26m.pt",
    "yolo26l.pt",
    "yolo26x.pt",
]

# Training hyperparameters shared across all runs.
TRAIN_CONFIG = {
    "epochs": 1,          # Number of epochs over the training dataset.
    "imgsz": 640,         # Input image resolution used during training.
    "batch": 8,           # Number of images processed per training step.
    "patience": 1,        # Early stopping patience in epochs without improvement.
    "optimizer": "AdamW", # Optimization algorithm used to update weights.
    "momentum": 0.937,    # Momentum factor for supported optimizers.
    "lr0": 0.001,         # Initial learning rate.
}


def configure_runtime() -> None:
    """Prepare directories and Ultralytics settings before training."""
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    if CLEAN_RESULTS_DIR and os.path.isdir(RESULTS_DIR):
        shutil.rmtree(RESULTS_DIR)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    settings.update({"weights_dir": WEIGHTS_DIR})


def list_folds() -> list[str]:
    """Return folds with valid data.yaml in deterministic order."""
    if not os.path.isdir(FOLDS_DIR):
        raise FileNotFoundError(f"Folds directory not found: {FOLDS_DIR}")

    folds = []
    for fold_name in sorted(os.listdir(FOLDS_DIR)):
        fold_path = os.path.join(FOLDS_DIR, fold_name)
        data_yaml = os.path.join(fold_path, "data.yaml")
        if os.path.isdir(fold_path) and os.path.isfile(data_yaml):
            folds.append(fold_name)

    if not folds:
        raise RuntimeError(f"No valid folds with data.yaml found in: {FOLDS_DIR}")

    return folds


def train_model_on_fold(model_name: str, fold_name: str, data_yaml: str) -> None:
    """Train one model checkpoint on one fold."""
    model_id = model_name.replace(".pt", "")
    print(f"\nTraining {model_id} - {fold_name}\n")

    model = YOLO(model_name)
    model.train(
        data=data_yaml, # Dataset YAML file for the current fold.
        epochs=TRAIN_CONFIG["epochs"],
        imgsz=TRAIN_CONFIG["imgsz"],
        batch=TRAIN_CONFIG["batch"],
        patience=TRAIN_CONFIG["patience"],
        optimizer=TRAIN_CONFIG["optimizer"],
        momentum=TRAIN_CONFIG["momentum"],
        lr0=TRAIN_CONFIG["lr0"], # Initial learning rate.
        device=DEVICE, # Uses GPU 0 if available, otherwise CPU.
        project=RESULTS_DIR, # Directory where training outputs are saved.
        name=f"{fold_name}_{model_id}",
    )

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> None:
    """Train all checkpoints across all generated folds."""
    configure_runtime()
    folds = list_folds()

    for fold_name in folds:
        fold_path = os.path.join(FOLDS_DIR, fold_name)
        data_yaml = os.path.join(fold_path, "data.yaml")

        for model_name in MODEL_NAMES:
            train_model_on_fold(model_name, fold_name, data_yaml)


if __name__ == "__main__":
    main()
