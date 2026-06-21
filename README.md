# Improving Broccoli Head Size Estimation — A Systematic Comparison of Image Preprocessing Techniques for YOLOv8n under Dutch Field Conditions

A study performing a systematic comparison of image preprocessing techniques for automated
broccoli head detection in raw Dutch field imagery. A YOLOv8n detector is
trained with 5-fold cross-validation (stratified by capture date) on the raw
images and on a set of preprocessed copies, so that the input transformation is
the only variable that changes between experiments. Each variant is scored both
on the full dataset and on manually curated subsets representing difficult field
conditions (camera blur, harsh lighting, leaf occlusion).

## Data is not included

**The dataset is not committed to this repository.** The field imagery was
provided by a third party (a researcher at Inholland University of Applied
Sciences) and is not redistributed here. To run the experiments, you must place
the data yourself:

```
annotated_broccoli_datasets/
└── raw/
    ├── 07-11-segment1_camera1_slot_0_frame_0000.png
    ├── 07-11-segment1_camera1_slot_0_frame_0000.txt   # YOLO-format label
    └── ...
```

- One folder, `annotated_broccoli_datasets/raw/`, holding images and their
  YOLO-format `.txt` labels side by side (same stem).
- Single object class (`broccoli`, class id `0`).
- Filenames must begin with an `MM-DD-` date prefix — the cross-validation
  folds are stratified on this prefix, so it is required.

Preprocessed variants, training runs, and subset crops are all **generated**
from `raw/` and are likewise not tracked. Everything under
`annotated_broccoli_datasets/`, `runs/`, and the generated subset image folders
should be excluded from version control (see *Reproducing the results* below).

## Requirements

- Python ≥ 3.12
- An NVIDIA GPU with CUDA (the experiments were run on an RTX 5080, CUDA 12.8).
  CPU training is possible but slow — pass `--device cpu`.
- [`uv`](https://docs.astral.sh/uv/) for dependency management.

Install dependencies (creates the virtual environment from `pyproject.toml` /
`uv.lock`):

```bash
uv sync
```

Pretrained YOLOv8n weights (`yolov8n.pt`) are fetched automatically by
Ultralytics on first use.

## Usage

All commands are run through `uv` so they use the locked environment.

### Train a variant (5-fold CV)

```bash
uv run python main.py --variant raw      # baseline (no preprocessing)
uv run python main.py --variant clahe    # any name from preprocess.VARIANTS
```

The preprocessed copy is built on demand under
`annotated_broccoli_datasets/<variant>/` the first time it is trained, then
reused. Useful flags (see `src/train.py` for the full list):

| Flag | Default | Meaning |
|------|---------|---------|
| `--variant` | `raw` | Variant to train (see list below) |
| `--epochs` | `100` | Max epochs (early stopping via `--patience`) |
| `--imgsz` | `1280` | Network input resolution |
| `--batch` | `16` | Batch size |
| `--folds` | `5` | Number of CV folds |
| `--seed` | `42` | Seed for the (fixed) fold split |
| `--device` | `0` | CUDA device index, or `cpu` |
| `--run-name` | `None` | Subfolder under `runs/<variant>/` to keep runs separate |

### Available variants

Defined in [`src/preprocess.py`](src/preprocess.py) (`VARIANTS`):

- **Baseline:** `raw`
- **Single techniques:** `clahe`, `bilateral`, `unsharp`, `wavelet`, `median`
- **Combinations:** `bilateral_unsharp`, `clahe_unsharp`, `median_clahe`,
  `wavelet_unsharp`, `bilateral_clahe_unsharp`

To add a new technique, register a transform in the `VARIANTS` dict.

### Evaluate on the difficult-condition subsets

Subsets are newline-delimited lists of image filenames/stems in `subsets/`.
Each subset image is scored out-of-fold (by the fold model that did not train on
it) and the predictions are pooled:

```bash
uv run python src/evaluate_subsets.py \
    --subset subsets/blurry.txt \
    --variants raw,clahe,unsharp,wavelet_unsharp \
    --run-name aug-default
```

Results are written to `subsets/<subset>_results.csv`.

### Visualize predictions

```bash
# Side-by-side predictions of every variant for each subset image
uv run python src/visualize_subsets.py
```

Per-fold prediction images (predicted vs. ground-truth boxes) are also produced
automatically during training under `runs/<variant>/.../fold<i>/vis/`.

## Outputs

For each variant and run:

```
runs/<variant>/<run-name>/
├── cv_metrics.csv     # per-fold precision / recall / mAP@50 / mAP@50-95
├── cv_summary.csv     # mean and std across folds
└── fold<i>/
    ├── train/weights/best.pt
    ├── val/
    └── vis/           # GT (green) vs prediction (red) overlays
```

Metrics are computed at an IoU threshold of `0.5`; mAP follows the COCO
101-point interpolation used by Ultralytics.

## Project structure

```
.
├── main.py                     # entry point: 5-fold CV training for one variant
├── pyproject.toml / uv.lock    # dependencies (managed by uv)
├── src/
│   ├── dataset.py              # stratified-by-date 5-fold splitting
│   ├── preprocess.py           # preprocessing techniques + VARIANTS registry
│   ├── train.py                # training/eval loop and CLI
│   ├── evaluate_subsets.py     # out-of-fold pooled subset evaluation
│   ├── visualize.py            # per-fold prediction overlays
│   └── visualize_subsets.py    # per-image, per-variant comparison grids
├── subsets/                    # subset definitions (*.txt) and results (*.csv)
├── annotated_broccoli_datasets/  # data — NOT committed (see above)
└── runs/                       # training outputs — NOT committed
```

## Reproducing the results

1. Place the dataset under `annotated_broccoli_datasets/raw/` as described above.
2. `uv sync`
3. Train each variant: `uv run python main.py --variant <name> --run-name aug-default`
4. Evaluate the subsets with `src/evaluate_subsets.py`.

The fold split is fully determined by `--seed` and `--folds`, so every variant
sees the identical partition and the results are reproducible.
