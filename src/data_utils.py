from __future__ import annotations

import json
import random
import shutil
from collections import Counter
from pathlib import Path
from typing import Iterable

import pandas as pd
from sklearn.model_selection import train_test_split

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
DEFAULT_CLASSES = ["anthracnose", "gumosis", "leaf miner", "red rust", "healthy"]


def scan_dataset(dataset_root: Path, classes: Iterable[str] = DEFAULT_CLASSES) -> pd.DataFrame:
    rows = []
    for class_name in classes:
        class_dir = dataset_root / class_name
        if not class_dir.exists():
            raise FileNotFoundError(f"Missing class directory: {class_dir}")
        for file_path in class_dir.iterdir():
            if file_path.suffix.lower() in IMAGE_EXTENSIONS:
                rows.append({"path": str(file_path), "label": class_name})
    if not rows:
        raise RuntimeError(f"No image files found under {dataset_root}")
    return pd.DataFrame(rows)


def stratified_split(
    df: pd.DataFrame,
    train_size: float = 0.70,
    val_size: float = 0.15,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    test_size = 1.0 - train_size - val_size
    if test_size <= 0:
        raise ValueError("train_size + val_size must be less than 1.0")

    train_df, temp_df = train_test_split(
        df, test_size=(1.0 - train_size), random_state=seed, stratify=df["label"]
    )
    relative_test_size = test_size / (val_size + test_size)
    val_df, test_df = train_test_split(
        temp_df, test_size=relative_test_size, random_state=seed, stratify=temp_df["label"]
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


def copy_split(split_df: pd.DataFrame, output_dir: Path, split_name: str) -> None:
    for _, row in split_df.iterrows():
        source = Path(row["path"])
        target_dir = output_dir / split_name / row["label"]
        target_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target_dir / source.name)


def compute_class_weights(labels: Iterable[str], classes: Iterable[str] = DEFAULT_CLASSES) -> dict[str, float]:
    counts = Counter(labels)
    total = sum(counts.values())
    num_classes = len(list(classes))
    return {class_name: float(total / (num_classes * max(1, counts[class_name]))) for class_name in classes}


def save_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
