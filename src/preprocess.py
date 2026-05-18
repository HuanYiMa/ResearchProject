"""Generate preprocessed dataset variants.

Each variant is a sibling folder under ``annotated_broccoli_datasets/`` next to
``raw/``. Labels are copied verbatim (preprocessing is image-only); only images
are transformed. Register new variants by adding to ``VARIANTS``.
"""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

Transform = Callable[[np.ndarray], np.ndarray]


def identity(img: np.ndarray) -> np.ndarray:
    return img


def clahe(img: np.ndarray, clip_limit: float = 2.0, tile_grid: int = 8) -> np.ndarray:
    """Apply CLAHE on the L channel of LAB color space."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    op = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid, tile_grid))
    l2 = op.apply(l)
    return cv2.cvtColor(cv2.merge((l2, a, b)), cv2.COLOR_LAB2BGR)


VARIANTS: dict[str, Transform] = {
    "clahe": clahe,
    # add new ones here, e.g. "gamma": lambda im: ...,
}


def build_variant(name: str, src_dir: Path, dst_root: Path) -> Path:
    if name == "raw":
        return src_dir
    if name not in VARIANTS:
        raise KeyError(f"Unknown variant '{name}'. Known: {list(VARIANTS)}")
    fn = VARIANTS[name]
    dst = dst_root / name
    dst.mkdir(parents=True, exist_ok=True)

    for f in sorted(src_dir.iterdir()):
        out = dst / f.name
        if f.suffix.lower() in {".png", ".jpg", ".jpeg"}:
            if out.exists():
                continue
            img = cv2.imread(str(f), cv2.IMREAD_COLOR)
            if img is None:
                raise RuntimeError(f"Failed to read {f}")
            cv2.imwrite(str(out), fn(img))
        elif f.suffix.lower() == ".txt":
            if not out.exists():
                shutil.copy2(f, out)
    return dst
