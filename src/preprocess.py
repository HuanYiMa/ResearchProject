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
import pywt

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


def bilateral(img: np.ndarray, d: int = 9, sigma_color: float = 50.0, sigma_space: float = 50.0) -> np.ndarray:
    """Edge-preserving smoothing. Removes leaf-texture noise while keeping
    broccoli-head edges sharp."""
    return cv2.bilateralFilter(img, d, sigma_color, sigma_space)


def unsharp(img: np.ndarray, radius: float = 1.5, amount: float = 1.0) -> np.ndarray:
    """Unsharp masking: sharpen by adding back the high-frequency residual.
    radius = Gaussian sigma (smaller -> sharper fine detail).
    amount = strength of the boost (0.5-2.0 typical)."""
    blurred = cv2.GaussianBlur(img, (0, 0), sigmaX=radius)
    return cv2.addWeighted(img, 1.0 + amount, blurred, -amount, 0)


def _wavelet_denoise_channel(ch: np.ndarray, wavelet: str, level: int) -> np.ndarray:
    coeffs = pywt.wavedec2(ch, wavelet, level=level)
    # Noise estimate from finest-scale HH (Donoho's MAD estimator)
    hh = coeffs[-1][2]
    sigma = np.median(np.abs(hh)) / 0.6745 if hh.size else 0.0
    # Universal soft threshold
    thr = sigma * np.sqrt(2.0 * np.log(ch.size)) if sigma > 0 else 0.0
    new_coeffs = [coeffs[0]]
    for detail in coeffs[1:]:
        new_coeffs.append(tuple(pywt.threshold(c, thr, mode="soft") for c in detail))
    out = pywt.waverec2(new_coeffs, wavelet)
    return out[: ch.shape[0], : ch.shape[1]]  # crop pywt's edge padding


def median(img: np.ndarray, ksize: int = 5) -> np.ndarray:
    """Median filter: replaces each pixel with the median of its k x k
    neighborhood. Kills impulse/speckle noise while keeping edges sharp.
    ksize must be odd; 3 is gentle, 5 is moderate, 7+ starts erasing detail."""
    if ksize % 2 == 0:
        raise ValueError("ksize must be odd")
    return cv2.medianBlur(img, ksize)


def wavelet(img: np.ndarray, wavelet_name: str = "db4", level: int = 2) -> np.ndarray:
    """Wavelet denoising via soft thresholding on detail coefficients.
    Per-channel, with noise estimated from the finest HH band."""
    f = img.astype(np.float32) / 255.0
    out = np.stack(
        [_wavelet_denoise_channel(f[..., c], wavelet_name, level) for c in range(f.shape[2])],
        axis=-1,
    )
    return np.clip(out * 255.0, 0, 255).astype(np.uint8)


def compose(*fns: Transform) -> Transform:
    """Apply preprocessing functions left-to-right: compose(a, b)(x) == b(a(x))."""
    def composed(img: np.ndarray) -> np.ndarray:
        for fn in fns:
            img = fn(img)
        return img
    return composed


VARIANTS: dict[str, Transform] = {
    "clahe": clahe,
    "bilateral": bilateral,
    "unsharp": unsharp,
    "wavelet": wavelet,
    "median": median,
    # Combinations — order matters (denoise -> enhance -> sharpen).
    "bilateral_unsharp": compose(bilateral, unsharp),
    "clahe_unsharp": compose(clahe, unsharp),
    "median_clahe": compose(median, clahe),
    "wavelet_unsharp": compose(wavelet, unsharp),
    "bilateral_clahe_unsharp": compose(bilateral, clahe, unsharp),
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
