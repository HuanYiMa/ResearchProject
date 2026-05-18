"""Build stratified 5-fold splits for a YOLO-format broccoli dataset.

Each image filename starts with a date prefix like "07-11-..."; we stratify on
that prefix so every fold sees a similar date distribution.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import yaml
from sklearn.model_selection import StratifiedKFold

DATE_RE = re.compile(r"^(\d{2}-\d{2})")


@dataclass
class Fold:
    index: int
    train_images: list[Path]
    val_images: list[Path]


def list_image_label_pairs(images_dir: Path) -> list[tuple[Path, Path]]:
    pairs: list[tuple[Path, Path]] = []
    for img in sorted(images_dir.iterdir()):
        if img.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
            continue
        lbl = img.with_suffix(".txt")
        if not lbl.exists():
            continue
        pairs.append((img, lbl))
    if not pairs:
        raise RuntimeError(f"No image/label pairs found in {images_dir}")
    return pairs


def date_of(path: Path) -> str:
    m = DATE_RE.match(path.name)
    if not m:
        raise ValueError(f"Cannot extract date from {path.name}")
    return m.group(1)


def make_folds(images_dir: Path, n_splits: int = 5, seed: int = 42) -> list[Fold]:
    pairs = list_image_label_pairs(images_dir)
    images = [p[0] for p in pairs]
    dates = [date_of(p) for p in images]

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds: list[Fold] = []
    for i, (train_idx, val_idx) in enumerate(skf.split(images, dates)):
        folds.append(
            Fold(
                index=i,
                train_images=[images[j] for j in train_idx],
                val_images=[images[j] for j in val_idx],
            )
        )
    return folds


def write_fold_files(fold: Fold, out_dir: Path) -> tuple[Path, Path, Path]:
    """Write train.txt, val.txt, data.yaml for a fold. Returns their paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    train_txt = out_dir / "train.txt"
    val_txt = out_dir / "val.txt"
    train_txt.write_text("\n".join(str(p.resolve()) for p in fold.train_images))
    val_txt.write_text("\n".join(str(p.resolve()) for p in fold.val_images))

    data_yaml = out_dir / "data.yaml"
    data_yaml.write_text(
        yaml.safe_dump(
            {
                "path": str(out_dir.resolve()),
                "train": str(train_txt.resolve()),
                "val": str(val_txt.resolve()),
                "names": {0: "broccoli"},
                "nc": 1,
            }
        )
    )
    return train_txt, val_txt, data_yaml
