"""Evaluate preprocessing variants on manually-defined image subsets.

Each image in the 5-fold split is in exactly one fold's val set. We score each
subset image with the fold model that did NOT train on it (out-of-fold), pool
all subset predictions, and compute precision / recall / mAP50 / mAP50-95 once
over the pool with Ultralytics' own ap_per_class (so numbers match CV metrics).

Subsets are manual newline-delimited lists of image filenames (or stems);
matching is by filename stem so the same list works across every variant.

Usage:
    uv run python src/evaluate_subsets.py --subset subsets/blurry.txt \\
        --variants raw,clahe,bilateral --run-name aug-off
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.utils.metrics import ap_per_class, box_iou

from dataset import make_folds

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "annotated_broccoli_datasets"
RUNS_ROOT = ROOT / "runs"

IOUV = torch.linspace(0.5, 0.95, 10)  # mAP50 .. mAP50-95, matches Ultralytics


def _match_predictions(pred_cls: torch.Tensor, true_cls: torch.Tensor, iou: torch.Tensor) -> torch.Tensor:
    """Replicates BaseValidator.match_predictions. iou is (M_gt, N_pred).
    Returns (N_pred, 10) bool correctness matrix."""
    correct = np.zeros((pred_cls.shape[0], IOUV.shape[0])).astype(bool)
    correct_class = true_cls[:, None] == pred_cls  # (M, N)
    iou = (iou * correct_class).cpu().numpy()
    for i, threshold in enumerate(IOUV.tolist()):
        matches = np.array(np.nonzero(iou >= threshold)).T  # (K, 2): [gt, pred]
        if matches.shape[0]:
            if matches.shape[0] > 1:
                matches = matches[iou[matches[:, 0], matches[:, 1]].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            correct[matches[:, 1].astype(int), i] = True
    return torch.tensor(correct, dtype=torch.bool)


def _read_gt_xyxy(label_path: Path, w: int, h: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (boxes_xyxy_pixels (M,4), classes (M,))."""
    boxes, classes = [], []
    if label_path.exists():
        for line in label_path.read_text().splitlines():
            p = line.strip().split()
            if len(p) < 5:
                continue
            cls, cx, cy, bw, bh = int(p[0]), *(float(x) for x in p[1:5])
            boxes.append([(cx - bw / 2) * w, (cy - bh / 2) * h,
                          (cx + bw / 2) * w, (cy + bh / 2) * h])
            classes.append(cls)
    if not boxes:
        return torch.zeros((0, 4)), torch.zeros((0,))
    return torch.tensor(boxes, dtype=torch.float32), torch.tensor(classes, dtype=torch.float32)


def _stem_to_fold(images_dir: Path, n_splits: int, seed: int) -> dict[str, int]:
    folds = make_folds(images_dir, n_splits=n_splits, seed=seed)
    mapping: dict[str, int] = {}
    for fold in folds:
        for img in fold.val_images:
            mapping[img.stem] = fold.index
    return mapping


def evaluate_subset(
    variant: str,
    subset_stems: list[str],
    run_name: str | None,
    n_splits: int = 5,
    seed: int = 42,
    conf: float = 0.25,
    iou_nms: float = 0.5,
    imgsz: int = 1280,
    device: int | str = 0,
) -> dict[str, float]:
    images_dir = DATA_ROOT / "raw" if variant == "raw" else DATA_ROOT / variant
    variant_out = RUNS_ROOT / variant / run_name if run_name else RUNS_ROOT / variant
    stem_to_fold = _stem_to_fold(images_dir, n_splits, seed)

    # one image per stem in the variant dir
    by_stem = {p.stem: p for p in images_dir.iterdir()
               if p.suffix.lower() in {".png", ".jpg", ".jpeg"}}

    models: dict[int, YOLO] = {}
    tp_all, conf_all, pcls_all, tcls_all = [], [], [], []
    missing = 0

    for stem in subset_stems:
        if stem not in stem_to_fold or stem not in by_stem:
            missing += 1
            continue
        fold = stem_to_fold[stem]
        if fold not in models:
            best = variant_out / f"fold{fold}" / "train" / "weights" / "best.pt"
            if not best.exists():
                raise FileNotFoundError(f"Missing weights: {best}")
            models[fold] = YOLO(str(best))

        img_path = by_stem[stem]
        res = models[fold].predict(source=str(img_path), conf=conf, iou=iou_nms,
                                   imgsz=imgsz, device=device, verbose=False)[0]
        h, w = res.orig_shape
        pred_box = res.boxes.xyxy.cpu()
        pred_conf = res.boxes.conf.cpu()
        pred_cls = res.boxes.cls.cpu()
        gt_box, gt_cls = _read_gt_xyxy(img_path.with_suffix(".txt"), w, h)

        if pred_box.shape[0]:
            if gt_box.shape[0]:
                iou = box_iou(gt_box, pred_box)  # (M, N)
                correct = _match_predictions(pred_cls, gt_cls, iou)
            else:
                correct = torch.zeros((pred_box.shape[0], IOUV.shape[0]), dtype=torch.bool)
            tp_all.append(correct.numpy())
            conf_all.append(pred_conf.numpy())
            pcls_all.append(pred_cls.numpy())
        tcls_all.append(gt_cls.numpy())

    if missing:
        print(f"  [warn] {missing} subset entries not found in dataset/fold map (skipped)")

    target_cls = np.concatenate(tcls_all) if tcls_all else np.zeros((0,))
    if not tp_all:  # no predictions at all
        return {"images": 0, "precision": 0.0, "recall": 0.0, "mAP50": 0.0, "mAP50-95": 0.0}

    tp = np.concatenate(tp_all)
    conf_arr = np.concatenate(conf_all)
    pred_cls_arr = np.concatenate(pcls_all)
    out = ap_per_class(tp, conf_arr, pred_cls_arr, target_cls)
    p, r, ap = out[2], out[3], out[5]  # per-class precision, recall, AP (nc,10)
    return {
        "images": len([s for s in subset_stems if s in stem_to_fold and s in by_stem]),
        "precision": float(p.mean()) if p.size else 0.0,
        "recall": float(r.mean()) if r.size else 0.0,
        "mAP50": float(ap[:, 0].mean()) if ap.size else 0.0,
        "mAP50-95": float(ap.mean()) if ap.size else 0.0,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", required=True, help="Path to newline-delimited list of image filenames/stems")
    ap.add_argument("--variants", default="raw", help="Comma-separated variants to compare, e.g. raw,clahe,bilateral")
    ap.add_argument("--run-name", default=None, help="Subfolder under runs/<variant>/ (must match training)")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.5, help="NMS IoU threshold")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--device", default=0)
    args = ap.parse_args()

    try:
        device: int | str = int(args.device)
    except (TypeError, ValueError):
        device = args.device

    subset_path = Path(args.subset)
    subset_stems = [Path(l.strip()).stem for l in subset_path.read_text().splitlines()
                    if l.strip() and not l.lstrip().startswith("#")]
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    print(f"Subset '{subset_path.name}': {len(subset_stems)} images | variants: {variants}")

    rows = []
    for variant in variants:
        m = evaluate_subset(variant, subset_stems, args.run_name, args.folds, args.seed,
                            args.conf, args.iou, args.imgsz, device)
        m["variant"] = variant
        rows.append(m)
        print(f"  {variant:12s} n={m['images']:3d}  P={m['precision']:.4f}  "
              f"R={m['recall']:.4f}  mAP50={m['mAP50']:.4f}  mAP50-95={m['mAP50-95']:.4f}")

    out_csv = subset_path.with_name(f"{subset_path.stem}_results.csv")
    with out_csv.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["variant", "images", "precision", "recall", "mAP50", "mAP50-95"])
        w.writeheader()
        for row in rows:
            w.writerow(row)
    print(f"\nWrote {out_csv}")


if __name__ == "__main__":
    main()
