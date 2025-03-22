from pathlib import Path
import pandas as pd
from collections import Counter
from sklearn.model_selection import KFold
import datetime
import shutil
from ultralytics import YOLO
import yaml
import random
from tqdm import tqdm
from natsort import natsorted

def kfold_split(dataset_path, yaml_file, ksplit):
    dataset_path = Path(dataset_path)  # replace with 'path/to/dataset' for your custom data
    labels = natsorted(dataset_path.rglob("*labels/*.txt"))  # all data in 'labels'

    with open(yaml_file, encoding="utf8") as y:
        classes = yaml.safe_load(y)["names"]
    cls_idx = sorted(classes.keys())

    index = [label.stem for label in labels]  # uses base filename as ID (no extension)
    labels_df = pd.DataFrame([], columns=cls_idx, index=index)

    for label in labels:
        lbl_counter = Counter()

        with open(label) as lf:
            lines = lf.readlines()

        for line in lines:
            # classes for YOLO label uses integer at first position of each line
            lbl_counter[int(line.split(" ")[0])] += 1

        labels_df.loc[label.stem] = lbl_counter

    labels_df = labels_df.fillna(0.0)  # replace `nan` values with `0.0`

    random.seed(0)  # for reproducibility
    kf = KFold(n_splits=ksplit, shuffle=True, random_state=20)  # setting random_state for repeatable results
    kfolds = list(kf.split(labels_df))

    folds = [f"split_{n}" for n in range(1, ksplit + 1)]
    folds_df = pd.DataFrame(index=index, columns=folds)

    for i, (train, val) in enumerate(kfolds, start=1):
        folds_df[f"split_{i}"].loc[labels_df.iloc[train].index] = "train"
        folds_df[f"split_{i}"].loc[labels_df.iloc[val].index] = "val"

    fold_lbl_distrb = pd.DataFrame(index=folds, columns=cls_idx)

    for n, (train_indices, val_indices) in enumerate(kfolds, start=1):
        train_totals = labels_df.iloc[train_indices].sum()
        val_totals = labels_df.iloc[val_indices].sum()

        # To avoid division by zero, we add a small value (1E-7) to the denominator
        ratio = val_totals / (train_totals + 1e-7)
        fold_lbl_distrb.loc[f"split_{n}"] = ratio

    supported_extensions = [".jpg", ".jpeg", ".png"]

    # Initialize an empty list to store image file paths
    images = []

    # Loop through supported extensions and gather image files
    for ext in supported_extensions:
        images.extend(natsorted((dataset_path / "images").rglob(f"*{ext}")))

    # Create the necessary directories and dataset YAML files
    save_path = Path(dataset_path / f"{datetime.date.today().isoformat()}_{ksplit}-Fold_Cross-val")
    save_path.mkdir(parents=True, exist_ok=True)
    ds_yamls = []

    for split in folds_df.columns:
        # Create directories
        split_dir = save_path / split
        split_dir.mkdir(parents=True, exist_ok=True)
        (split_dir / "train" / "images").mkdir(parents=True, exist_ok=True)
        (split_dir / "train" / "labels").mkdir(parents=True, exist_ok=True)
        (split_dir / "val" / "images").mkdir(parents=True, exist_ok=True)
        (split_dir / "val" / "labels").mkdir(parents=True, exist_ok=True)

        # Create dataset YAML files
        dataset_yaml = split_dir / f"{split}_dataset.yaml"
        ds_yamls.append(dataset_yaml)

        with open(dataset_yaml, "w") as ds_y:
            yaml.safe_dump(
                {
                    "path": split_dir.as_posix(),
                    "train": "train",
                    "val": "val",
                    "names": classes,
                },
                ds_y,
            )

    for image in tqdm(images, total=len(images), desc="Copying files"):
        for split, k_split in folds_df.loc[image.stem].items():
            # Destination directory
            img_to_path = save_path / split / k_split / "images"
            lbl_to_path = save_path / split / k_split / "labels"
            label_name = (f"{image.stem}.txt")
            label = str(image).replace("images", "labels")
            label = label.split(".")[0] + ".txt"

            # Copy image and label files to new directory (SamefileError if file already exists)
            shutil.copy(image, img_to_path / image.name)
            shutil.copy(label, lbl_to_path / label_name)

    folds_df.to_csv(save_path / "kfold_datasplit.csv")
    fold_lbl_distrb.to_csv(save_path / "kfold_label_distribution.csv")

    return ds_yamls

if __name__ == "__main__":
    weights_path = "E:\\YOLO11-Vertebrae-Detection\\yolo11s.pt"
    model = YOLO(weights_path)

    results = {}
    ksplit = 5

    dataset_path = Path("E:\\YOLO11-Vertebrae-Detection\\train_data\\total")
    dataset_yaml = Path("E:\\YOLO11-Vertebrae-Detection\\kfold_train_config.yaml")
    kfolds_yaml = kfold_split(dataset_path, dataset_yaml, ksplit)

    batch = 16
    epochs = 500

    for k in range(ksplit):
        dataset_yaml = kfolds_yaml[k]
        model.train(data=dataset_yaml, epochs=epochs, batch=batch, imgsz = 640, project="/")
        results[k] = model.metrics