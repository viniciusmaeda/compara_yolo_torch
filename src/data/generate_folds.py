import shutil
from sklearn.model_selection import KFold
import os
import yaml
import json
from collections import defaultdict

# ==========================
# EXPERIMENT CONFIG
# ==========================

# Resolve project root from this file: src/data/generate_folds.py -> project root.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Dataset source directories.
DATASET_DIR = os.path.join(PROJECT_ROOT, "dataset")
DATASET_TRAIN_DIR = os.path.join(DATASET_DIR, "train")
IMAGES_DIR = os.path.join(DATASET_TRAIN_DIR, "images")
LABELS_DIR = os.path.join(DATASET_TRAIN_DIR, "labels")
DATASET_YAML = os.path.join(DATASET_DIR, "data.yaml")
COCO_JSON = os.path.join(DATASET_TRAIN_DIR, "_annotations.coco.json")

# Output directory for generated K-Fold splits.
FOLDS_DIR = os.path.join(PROJECT_ROOT, "folds")

# K-Fold configuration.
K = 2
RANDOM_STATE = 42
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


def detect_dataset_mode() -> str:
    """Detect whether dataset source is YOLO dirs or COCO JSON."""
    if os.path.isfile(COCO_JSON):
        return "coco"
    if os.path.isdir(IMAGES_DIR) and os.path.isdir(LABELS_DIR) and os.path.isfile(DATASET_YAML):
        return "yolo"
    raise FileNotFoundError(
        "Dataset source not found. Expected either:\n"
        f"- COCO: {COCO_JSON}\n"
        f"- YOLO: {IMAGES_DIR}, {LABELS_DIR}, {DATASET_YAML}"
    )


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


def load_coco_data() -> tuple[list[dict], list[dict], dict[int, list[dict]], dict[int, int], list[str]]:
    """Load and validate COCO JSON data needed to generate YOLO labels."""
    if not os.path.isfile(COCO_JSON):
        raise FileNotFoundError(f"COCO annotations not found: {COCO_JSON}")

    with open(COCO_JSON, "r", encoding="utf-8") as f:
        coco = json.load(f)

    images = coco.get("images", [])
    annotations = coco.get("annotations", [])
    categories = coco.get("categories", [])

    if not images or not categories:
        raise ValueError("COCO JSON must contain non-empty 'images' and 'categories'.")

    categories_sorted = sorted(categories, key=lambda c: c.get("id", 0))
    category_to_idx = {}
    class_names = []
    for idx, category in enumerate(categories_sorted):
        category_id = int(category["id"])
        category_to_idx[category_id] = idx
        class_names.append(str(category.get("name", f"class_{category_id}")))

    anns_by_image: dict[int, list[dict]] = defaultdict(list)
    for ann in annotations:
        image_id = int(ann.get("image_id"))
        anns_by_image[image_id].append(ann)

    return images, annotations, anns_by_image, category_to_idx, class_names


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


def coco_bbox_to_yolo_line(
    bbox: list[float],
    img_w: int,
    img_h: int,
    class_id: int,
) -> str:
    """Convert COCO [x,y,w,h] box to YOLO normalized line."""
    x, y, w, h = [float(v) for v in bbox]

    x1 = max(0.0, min(x, float(img_w)))
    y1 = max(0.0, min(y, float(img_h)))
    x2 = max(0.0, min(x + w, float(img_w)))
    y2 = max(0.0, min(y + h, float(img_h)))

    box_w = x2 - x1
    box_h = y2 - y1
    if box_w <= 0 or box_h <= 0:
        return ""

    cx = (x1 + x2) / 2.0 / float(img_w)
    cy = (y1 + y2) / 2.0 / float(img_h)
    nw = box_w / float(img_w)
    nh = box_h / float(img_h)
    return f"{class_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}"


def copy_split_from_coco(
    split_images: list[dict],
    anns_by_image: dict[int, list[dict]],
    category_to_idx: dict[int, int],
    images_dst: str,
    labels_dst: str,
) -> None:
    """Copy images and build YOLO label files from COCO annotations."""
    for image_info in split_images:
        file_name = str(image_info["file_name"])
        image_id = int(image_info["id"])
        img_w = int(image_info["width"])
        img_h = int(image_info["height"])

        image_src = os.path.join(DATASET_TRAIN_DIR, file_name)
        if not os.path.isfile(image_src):
            raise FileNotFoundError(f"Image referenced in COCO not found: {image_src}")

        shutil.copy2(image_src, images_dst)

        label_stem, _ = os.path.splitext(file_name)
        label_dst = os.path.join(labels_dst, f"{label_stem}.txt")

        yolo_lines = []
        for ann in anns_by_image.get(image_id, []):
            if int(ann.get("iscrowd", 0)) == 1:
                continue
            category_id = int(ann["category_id"])
            if category_id not in category_to_idx:
                continue
            bbox = ann.get("bbox", [])
            if len(bbox) != 4:
                continue
            yolo_line = coco_bbox_to_yolo_line(bbox, img_w, img_h, category_to_idx[category_id])
            if yolo_line:
                yolo_lines.append(yolo_line)

        with open(label_dst, "w", encoding="utf-8") as f:
            if yolo_lines:
                f.write("\n".join(yolo_lines) + "\n")


def main() -> None:
    dataset_mode = detect_dataset_mode()

    if dataset_mode == "coco":
        images_data, _, anns_by_image, category_to_idx, class_names = load_coco_data()
        nc = len(class_names)
        images_data = sorted(images_data, key=lambda item: str(item.get("file_name", "")))
        if not images_data:
            raise RuntimeError(f"No images found in COCO: {COCO_JSON}")
    else:
        nc, class_names = load_class_config()
        image_names = sorted(
            file_name
            for file_name in os.listdir(IMAGES_DIR)
            if os.path.isfile(os.path.join(IMAGES_DIR, file_name))
            and os.path.splitext(file_name)[1].lower() in IMAGE_EXTENSIONS
        )
        if not image_names:
            raise RuntimeError(f"No images found in: {IMAGES_DIR}")
        images_data = image_names

    # Create K-Fold splitter with shuffle for better distribution.
    kf = KFold(n_splits=K, shuffle=True, random_state=RANDOM_STATE)

    # Recreate folds directory to guarantee a clean output.
    if os.path.exists(FOLDS_DIR):
        shutil.rmtree(FOLDS_DIR)
    os.makedirs(FOLDS_DIR, exist_ok=True)

    # Build each fold with train/val images, labels, and data.yaml.
    for fold, (train_idx, val_idx) in enumerate(kf.split(images_data), start=1):
        fold_path = os.path.join(FOLDS_DIR, f"fold_{fold}")

        train_img_dir = os.path.join(fold_path, "train", "images")
        train_lbl_dir = os.path.join(fold_path, "train", "labels")
        val_img_dir = os.path.join(fold_path, "val", "images")
        val_lbl_dir = os.path.join(fold_path, "val", "labels")

        os.makedirs(train_img_dir, exist_ok=True)
        os.makedirs(train_lbl_dir, exist_ok=True)
        os.makedirs(val_img_dir, exist_ok=True)
        os.makedirs(val_lbl_dir, exist_ok=True)

        train_files = [images_data[i] for i in train_idx]
        val_files = [images_data[i] for i in val_idx]

        if dataset_mode == "coco":
            copy_split_from_coco(
                train_files, anns_by_image, category_to_idx, train_img_dir, train_lbl_dir
            )
            copy_split_from_coco(
                val_files, anns_by_image, category_to_idx, val_img_dir, val_lbl_dir
            )
        else:
            copy_split_files(train_files, train_img_dir, train_lbl_dir)
            copy_split_files(val_files, val_img_dir, val_lbl_dir)
        write_data_yaml(fold_path, train_img_dir, val_img_dir, nc, class_names)

        print(f"Fold {fold} created.")

    print("All folds generated successfully.")


if __name__ == "__main__":
    main()
