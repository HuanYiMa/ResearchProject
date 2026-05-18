"""Draw predicted (red) vs ground-truth (green) bounding boxes for a val split.

Images are saved with a rank prefix so they sort worst-performing first in
the filesystem. Per-image score is the mean IoU across all boxes (greedy
match between preds and GTs; unmatched boxes on either side contribute 0).
"""
from __future__ import annotations

import shutil
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

GT_COLOR = (0, 200, 0)       # green
PRED_COLOR = (0, 0, 255)     # red


def _read_yolo_labels(label_path: Path) -> list[tuple[int, float, float, float, float]]:
    if not label_path.exists():
        return []
    out: list[tuple[int, float, float, float, float]] = []
    for line in label_path.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        out.append((int(parts[0]), *(float(x) for x in parts[1:5])))
    return out


def _yolo_to_xyxy(cx: float, cy: float, w: float, h: float, W: int, H: int) -> tuple[int, int, int, int]:
    x1 = int((cx - w / 2) * W)
    y1 = int((cy - h / 2) * H)
    x2 = int((cx + w / 2) * W)
    y2 = int((cy + h / 2) * H)
    return x1, y1, x2, y2


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def _mean_iou(
    gt_xyxy: list[np.ndarray],
    pred_xyxy: list[np.ndarray],
    pred_conf: list[float],
) -> float:
    """Greedy-match preds (highest-conf first) to GTs by IoU. Mean IoU over
    max(num_gt, num_pred) boxes — unmatched boxes contribute 0."""
    if not gt_xyxy and not pred_xyxy:
        return 1.0
    order = sorted(range(len(pred_xyxy)), key=lambda i: -pred_conf[i])
    used_gt: set[int] = set()
    iou_sum = 0.0
    for pi in order:
        best_iou, best_gi = 0.0, -1
        for gi, gt in enumerate(gt_xyxy):
            if gi in used_gt:
                continue
            v = _iou(pred_xyxy[pi], gt)
            if v > best_iou:
                best_iou, best_gi = v, gi
        if best_gi >= 0 and best_iou > 0:
            used_gt.add(best_gi)
            iou_sum += best_iou
    denom = max(len(gt_xyxy), len(pred_xyxy))
    return iou_sum / denom if denom > 0 else 0.0


def visualize_fold(
    weights: Path,
    val_txt: Path,
    out_dir: Path,
    conf: float = 0.25,
    iou: float = 0.5,
    max_images: int | None = 24,
) -> None:
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(weights))
    image_paths = [Path(p) for p in val_txt.read_text().splitlines() if p.strip()]

    scored: list[tuple[float, np.ndarray, Path]] = []
    for img_path in image_paths:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        H, W = img.shape[:2]

        gt_xyxy: list[np.ndarray] = []
        for _, cx, cy, w, h in _read_yolo_labels(img_path.with_suffix(".txt")):
            x1, y1, x2, y2 = _yolo_to_xyxy(cx, cy, w, h, W, H)
            gt_xyxy.append(np.array([x1, y1, x2, y2], dtype=float))
            cv2.rectangle(img, (x1, y1), (x2, y2), GT_COLOR, 2)

        result = model.predict(source=str(img_path), conf=conf, iou=iou, verbose=False)[0]
        boxes = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        pred_xyxy = [b.astype(float) for b in boxes]
        for box, score in zip(boxes, confs):
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(img, (x1, y1), (x2, y2), PRED_COLOR, 2)
            cv2.putText(
                img, f"{score:.2f}", (x1, max(0, y1 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, PRED_COLOR, 1, cv2.LINE_AA,
            )

        miou = _mean_iou(gt_xyxy, pred_xyxy, confs.tolist())
        scored.append((miou, img, img_path))

    scored.sort(key=lambda t: t[0])
    selected = scored if max_images is None else scored[:max_images]
    width = max(3, len(str(len(selected))))
    for rank, (miou, img, img_path) in enumerate(selected):
        name = f"{rank:0{width}d}_iou-{miou:.2f}_{img_path.name}"
        cv2.imwrite(str(out_dir / name), img)
