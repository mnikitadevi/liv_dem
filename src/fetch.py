# src/fetch.py
"""
One-time dataset fetch for Model A training data.
Run this once tonight while on wifi. Safe to re-run — skips download if already present.
"""

from pathlib import Path
from torchvision.datasets import Imagenette

DATA_ROOT = Path(__file__).resolve().parent.parent / "data" / "raw"

# The 3 classes Model A will learn to distinguish
SELECTED_CLASSES = ["golf ball", "chainsaw", "parachute"]


def fetch():
    DATA_ROOT.mkdir(parents=True, exist_ok=True)

    print(f"Fetching Imagenette into {DATA_ROOT} ...")

    # download=True triggers download only if not already present at root
    train_set = Imagenette(
        root=str(DATA_ROOT),
        split="train",
        size="320px",  # good balance: sharp enough for projector, still small/fast
        download=True,
    )

    val_set = Imagenette(
        root=str(DATA_ROOT),
        split="val",
        size="320px",
        download=True,
    )

    print(f"Train samples: {len(train_set)}")
    print(f"Val samples:   {len(val_set)}")
    print("Classes available:", train_set.classes)
    print("Done. Data cached locally — safe to go offline now.")


if __name__ == "__main__":
    fetch()
