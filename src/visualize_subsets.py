"""For each subset image, render the prediction of every variant side-by-side.

Output layout (one folder per subset image, one file per variant):

    subsets/<subset-stem>/<image-stem>/
        raw.png
        clahe.png
        wavelet_unsharp.png
        ...

Each rendered image is the variant's preprocessed image (what the model
actually saw) with GT boxes in green and predictions in red. Predictions come
from the fold model that did NOT train on that image (out-of-fold), matching
the evaluation script.

Usage:
    uv run python src/visualize_subsets.py --subset subsets/blurry.txt \\
        --variants raw,clahe,unsharp,wavelet_unsharp --run-name aug-default
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO

from dataset import make_folds

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "annotated_broccoli_datasets"
RUNS_ROOT = ROOT / "runs"

GT_COLOR = (0, 200, 0)
PRED_COLOR = (0, 0, 255)


def _stem_to_fold(images_dir: Path, n_splits: int, seed: int) -> dict[str, int]:
    folds = make_folds(images_dir, n_splits=n_splits, seed=seed)
    return {img.stem: f.index for f in folds for img in f.val_images}


def _draw_gt(img, label_path: Path) -> None:
    if not label_path.exists():
        return
    H, W = img.shape[:2]
    for line in label_path.read_text().splitlines():
        p = line.strip().split()
        if len(p) < 5:
            continue
        cx, cy, w, h = (float(x) for x in p[1:5])
        x1 = int((cx - w / 2) * W); y1 = int((cy - h / 2) * H)
        x2 = int((cx + w / 2) * W); y2 = int((cy + h / 2) * H)
        cv2.rectangle(img, (x1, y1), (x2, y2), GT_COLOR, 2)


def visualize_subset_variant(
    variant: str,
    subset_stems: list[str],
    out_root: Path,
    run_name: str | None,
    n_splits: int,
    seed: int,
    conf: float,
    iou_nms: float,
    imgsz: int,
    device: int | str,
) -> None:
    images_dir = DATA_ROOT / "raw" if variant == "raw" else DATA_ROOT / variant
    variant_out = RUNS_ROOT / variant / run_name if run_name else RUNS_ROOT / variant
    stem_to_fold = _stem_to_fold(images_dir, n_splits, seed)
    by_stem = {p.stem: p for p in images_dir.iterdir()
               if p.suffix.lower() in {".png", ".jpg", ".jpeg"}}

    models: dict[int, YOLO] = {}
    missing = 0
    for stem in subset_stems:
        if stem not in stem_to_fold or stem not in by_stem:
            missing += 1
            continue
        fold = stem_to_fold[stem]
        if fold not in models:
            best = variant_out / f"fold{fold}" / "train" / "weights" / "best.pt"
            if not best.exists():
                raise FileNotFoundError(f"Missing weights for {variant} fold {fold}: {best}")
            models[fold] = YOLO(str(best))

        img_path = by_stem[stem]
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        _draw_gt(img, img_path.with_suffix(".txt"))

        res = models[fold].predict(source=str(img_path), conf=conf, iou=iou_nms,
                                   imgsz=imgsz, device=device, verbose=False)[0]
        for box, score in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.conf.cpu().numpy()):
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(img, (x1, y1), (x2, y2), PRED_COLOR, 2)
            cv2.putText(img, f"{score:.2f}", (x1, max(0, y1 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, PRED_COLOR, 1, cv2.LINE_AA)
        cv2.putText(img, f"{variant}  GT=green Pred=red", (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

        img_dir = out_root / stem
        img_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(img_dir / f"{variant}.png"), img)

    if missing:
        print(f"  [{variant}] {missing} entries not found in fold map / image dir")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", required=True, help="Newline-delimited list of image filenames/stems")
    ap.add_argument("--variants", required=True, help="Comma-separated variant names")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--device", default=0)
    ap.add_argument("--out", default=None,
                    help="Output root (default: subsets/<subset-stem>/)")
    args = ap.parse_args()

    try:
        device: int | str = int(args.device)
    except (TypeError, ValueError):
        device = args.device

    subset_path = Path(args.subset)
    subset_stems = [Path(l.strip()).stem for l in subset_path.read_text().splitlines()
                    if l.strip() and not l.lstrip().startswith("#")]
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    out_root = Path(args.out) if args.out else subset_path.with_suffix("")

    print(f"Subset '{subset_path.name}': {len(subset_stems)} images | variants: {variants}")
    print(f"Output: {out_root}/<image-stem>/<variant>.png")

    for variant in variants:
        print(f"  rendering {variant} ...")
        visualize_subset_variant(variant, subset_stems, out_root, args.run_name,
                                 args.folds, args.seed, args.conf, args.iou,
                                 args.imgsz, device)

    print("Done.")


if __name__ == "__main__":
    main()
