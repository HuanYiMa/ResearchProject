"""Train YOLOv8n with 5-fold stratified-by-date CV on a chosen dataset variant."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from ultralytics import YOLO

from dataset import make_folds, write_fold_files
from preprocess import build_variant
from visualize import visualize_fold

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "annotated_broccoli_datasets"
RUNS_ROOT = ROOT / "runs"


def run_variant(
    variant: str,
    epochs: int = 100,
    imgsz: int = 1280,
    batch: int = 16,
    n_splits: int = 5,
    seed: int = 42,
    weights: str = "yolov8n.pt",
    patience: int = 30,
    device: int | str = 0,
    workers: int = 4,
    iou: float = 0.5,
    run_name: str | None = None,
    hsv_h: float = 0.015,
    hsv_s: float = 0.7,
    hsv_v: float = 0.4,
    scale: float = 0.5,
    mosaic: float = 1.0,
    translate: float = 0.1,
) -> Path:
    images_dir = build_variant(variant, DATA_ROOT / "raw", DATA_ROOT)
    print(f"[variant={variant}] images dir: {images_dir}")

    folds = make_folds(images_dir, n_splits=n_splits, seed=seed)
    variant_out = RUNS_ROOT / variant / run_name if run_name else RUNS_ROOT / variant
    variant_out.mkdir(parents=True, exist_ok=True)
    print(f"[variant={variant}] output dir: {variant_out}")

    metrics_csv = variant_out / "cv_metrics.csv"
    with metrics_csv.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["fold", "precision", "recall", "mAP50", "mAP50-95"])

        for fold in folds:
            fold_dir = variant_out / f"fold{fold.index}"
            _, val_txt, data_yaml = write_fold_files(fold, fold_dir)

            print(f"\n=== {variant} fold {fold.index}: "
                  f"{len(fold.train_images)} train / {len(fold.val_images)} val ===")

            model = YOLO(weights)
            model.train(
                data=str(data_yaml),
                epochs=epochs,
                imgsz=imgsz,
                batch=batch,
                patience=patience,
                device=device,
                workers=workers,
                hsv_h=hsv_h,
                hsv_s=hsv_s,
                hsv_v=hsv_v,
                scale=scale,
                mosaic=mosaic,
                translate=translate,
                project=str(fold_dir),
                name="train",
                exist_ok=True,
                seed=seed,
                verbose=True,
            )

            val = model.val(
                data=str(data_yaml),
                imgsz=imgsz,
                device=device,
                workers=workers,
                iou=iou,
                project=str(fold_dir),
                name="val",
                exist_ok=True,
                verbose=False,
            )
            p = float(val.box.mp)
            r = float(val.box.mr)
            m50 = float(val.box.map50)
            m5095 = float(val.box.map)
            writer.writerow([fold.index, p, r, m50, m5095])
            fh.flush()
            print(f"fold {fold.index}: P={p:.4f} R={r:.4f} mAP50={m50:.4f} mAP50-95={m5095:.4f}")

            best = fold_dir / "train" / "weights" / "best.pt"
            if best.exists():
                visualize_fold(best, val_txt, fold_dir / "vis", iou=iou, max_images=None)

    _summarize(metrics_csv)
    return variant_out


def _summarize(csv_path: Path) -> None:
    import statistics

    rows = list(csv.DictReader(csv_path.open()))
    if not rows:
        return
    cols = ["precision", "recall", "mAP50", "mAP50-95"]
    summary = {c: [float(r[c]) for r in rows] for c in cols}
    print("\n=== CV summary ===")
    for c, vs in summary.items():
        print(f"{c}: mean={statistics.mean(vs):.4f}  std={statistics.pstdev(vs):.4f}")

    out = csv_path.with_name("cv_summary.csv")
    with out.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["metric", "mean", "std"])
        for c, vs in summary.items():
            w.writerow([c, statistics.mean(vs), statistics.pstdev(vs)])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="raw", help="raw, clahe, ... (see preprocess.VARIANTS)")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--weights", default="yolov8n.pt")
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--device", default=0, help="CUDA device index, or 'cpu'")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--iou", type=float, default=0.5, help="NMS IoU threshold for val and visualizations")
    ap.add_argument("--run-name", default=None,
                    help="Optional subfolder under runs/<variant>/ to keep multiple runs separate")
    ap.add_argument("--hsv-h", type=float, default=0.015, help="HSV hue augmentation (0 to disable)")
    ap.add_argument("--hsv-s", type=float, default=0.7, help="HSV saturation augmentation (0 to disable)")
    ap.add_argument("--hsv-v", type=float, default=0.4, help="HSV value/brightness augmentation (0 to disable)")
    ap.add_argument("--scale", type=float, default=0.5, help="Random scale gain (0 to disable)")
    ap.add_argument("--mosaic", type=float, default=1.0, help="Mosaic augmentation probability (0 to disable)")
    ap.add_argument("--translate", type=float, default=0.1, help="Random translation fraction (0 to disable)")
    args = ap.parse_args()

    device: int | str
    try:
        device = int(args.device)
    except (TypeError, ValueError):
        device = args.device

    run_variant(
        variant=args.variant,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        n_splits=args.folds,
        seed=args.seed,
        weights=args.weights,
        patience=args.patience,
        device=device,
        workers=args.workers,
        iou=args.iou,
        run_name=args.run_name,
        hsv_h=args.hsv_h,
        hsv_s=args.hsv_s,
        hsv_v=args.hsv_v,
        scale=args.scale,
        mosaic=args.mosaic,
        translate=args.translate,
    )


if __name__ == "__main__":
    main()
