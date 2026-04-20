from __future__ import annotations

from pathlib import Path

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "out_put"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


__all__ = [
    "OUTPUT_DIR",
]
