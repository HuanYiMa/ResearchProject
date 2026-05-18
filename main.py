"""Entry point: run 5-fold stratified-by-date CV for YOLOv8n.

Usage:
    uv run python main.py --variant raw
    uv run python main.py --variant clahe
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from train import main  # noqa: E402

if __name__ == "__main__":
    main()
