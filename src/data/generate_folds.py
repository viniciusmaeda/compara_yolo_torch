import shutil
from sklearn.model_selection import KFold
import os
import yaml

# ==========================
# EXPERIMENT CONFIG
# ==========================

# Resolve project root from this file: src/data/generate_folds.py -> project root.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Dataset source directories.
DATASET_DIR = os.path.join(PROJECT_ROOT, "dataset")
IMAGES_DIR = os.path.join(DATASET_DIR, "train", "images")
LABELS_DIR = os.path.join(DATASET_DIR, "train", "labels")
DATASET_YAML = os.path.join(DATASET_DIR, "data.yaml")

# Output directory for generated K-Fold splits.
FOLDS_DIR = os.path.join(PROJECT_ROOT, "folds")

# K-Fold configuration.
K = 5
RANDOM_STATE = 42
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


def load_class_config() -> tuple[int, list[str]]:
    """Load class count and names from dataset/data.yaml."""
    if not os.path.isfile(DATASET_YAML):
        raise FileNotFoundError(f"Dataset YAML not found: {DATASET_YAML}.")

    with open(DATASET_YAML, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    names = data.get("names")
    nc = data.get("nc")

    if isinstance(names, dict):
        names = [names[idx] for idx in sorted(names)]

    if not isinstance(names, list) or not names:
        raise ValueError(f"Invalid or missing 'names' in {DATASET_YAML}.")

    if nc is None:
        nc = len(names)
    nc = int(nc)

    if nc != len(names):
        raise ValueError(
            f"Inconsistent classes in {DATASET_YAML}: nc={nc}, names={len(names)}."
        )

    return nc, [str(name) for name in names]


def write_data_yaml(
    fold_path: str,
    train_images_dir: str,
    val_images_dir: str,
    nc: int,
    class_names: list[str],
) -> None:
    """Write Ultralytics-compatible data.yaml for one fold."""
    data_yaml = os.path.join(fold_path, "data.yaml")
    with open(data_yaml, "w", encoding="utf-8") as f:
        f.write(
            "\n".join(
                [
                    f"train: {train_images_dir}",
                    f"val: {val_images_dir}",
                    "",
                    f"nc: {nc}",
                    f"names: {class_names}",
                    "",
                ]
            )
        )


def copy_split_files(files: list[str], images_dst: str, labels_dst: str) -> None:
    """Copy image files and their matching YOLO label files."""
    for image_name in files:
        image_src = os.path.join(IMAGES_DIR, image_name)
        image_stem, _ = os.path.splitext(image_name)
        label_src = os.path.join(LABELS_DIR, f"{image_stem}.txt")

        if not os.path.exists(label_src):
            raise FileNotFoundError(f"Missing label file for image: {image_name}")

        shutil.copy2(image_src, images_dst)
        shutil.copy2(label_src, labels_dst)


def main() -> None:
    nc, class_names = load_class_config()

    # Collect image files in deterministic order.
    images = sorted(
        file_name
        for file_name in os.listdir(IMAGES_DIR)
        if os.path.isfile(os.path.join(IMAGES_DIR, file_name))
        and os.path.splitext(file_name)[1].lower() in IMAGE_EXTENSIONS
    )
    if not images:
        raise RuntimeError(f"No images found in: {IMAGES_DIR}")

    # Create K-Fold splitter with shuffle for better distribution.
    kf = KFold(n_splits=K, shuffle=True, random_state=RANDOM_STATE)

    # Recreate folds directory to guarantee a clean output.
    if os.path.exists(FOLDS_DIR):
        shutil.rmtree(FOLDS_DIR)
    os.makedirs(FOLDS_DIR, exist_ok=True)

    # Build each fold with train/val images, labels, and data.yaml.
    for fold, (train_idx, val_idx) in enumerate(kf.split(images), start=1):
        fold_path = os.path.join(FOLDS_DIR, f"fold_{fold}")

        train_img_dir = os.path.join(fold_path, "train", "images")
        train_lbl_dir = os.path.join(fold_path, "train", "labels")
        val_img_dir = os.path.join(fold_path, "val", "images")
        val_lbl_dir = os.path.join(fold_path, "val", "labels")

        os.makedirs(train_img_dir, exist_ok=True)
        os.makedirs(train_lbl_dir, exist_ok=True)
        os.makedirs(val_img_dir, exist_ok=True)
        os.makedirs(val_lbl_dir, exist_ok=True)

        train_files = [images[i] for i in train_idx]
        val_files = [images[i] for i in val_idx]

        copy_split_files(train_files, train_img_dir, train_lbl_dir)
        copy_split_files(val_files, val_img_dir, val_lbl_dir)
        write_data_yaml(fold_path, train_img_dir, val_img_dir, nc, class_names)

        print(f"Fold {fold} created.")

    print("All folds generated successfully.")


if __name__ == "__main__":
    main()
