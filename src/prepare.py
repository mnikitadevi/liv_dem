# src/prepare.py
"""
Filters the full Imagenette dataset down to the 3 classes Model A will train on,
and organizes them into a simple folder structure:

    data/processed/train/<class_name>/*.jpg
    data/processed/test/<class_name>/*.jpg

Run this once after fetch.py. Safe to re-run — clears and rebuilds processed/ each time,
so you can change SELECTED_CLASSES and rerun without leftover files from old choices.
"""

import shutil
from pathlib import Path
from torchvision.datasets import Imagenette

RAW_ROOT = Path(__file__).resolve().parent.parent / "data" / "raw"
PROCESSED_ROOT = Path(__file__).resolve().parent.parent / "data" / "processed"

# Must match fetch.py — note torchvision's actual label is "chain saw" (with a space)
SELECTED_CLASSES = ["golf ball", "chain saw", "parachute"]

# How many images per class to actually use for training/testing.
# Small on purpose — Model A is supposed to be a small, undertrained model.
N_TRAIN_PER_CLASS = 25
N_TEST_PER_CLASS = 8


def class_label_to_folder_name(label: str) -> str:
    """Turn 'chain saw' into 'chain_saw' for a clean folder name."""
    return label.replace(" ", "_")


def prepare_split(dataset, split_name: str, n_per_class: int):
    # dataset.classes is a list of tuples, e.g. ('chain saw', 'chainsaw')
    # dataset._samples or similar gives (path, class_idx) — but torchvision's
    # Imagenette exposes `._samples` as a list of (Path, label_idx)
    class_names = [c[0] for c in dataset.classes]  # primary label per class idx

    # map selected class name -> class idx
    selected_idxs = {}
    for wanted in SELECTED_CLASSES:
        if wanted not in class_names:
            raise ValueError(f"Class '{wanted}' not found. Available: {class_names}")
        selected_idxs[class_names.index(wanted)] = wanted

    counts = {idx: 0 for idx in selected_idxs}

    for path, label_idx in dataset._samples:
        if label_idx not in selected_idxs:
            continue
        if counts[label_idx] >= n_per_class:
            continue

        class_name = selected_idxs[label_idx]
        folder_name = class_label_to_folder_name(class_name)
        dest_dir = PROCESSED_ROOT / split_name / folder_name
        dest_dir.mkdir(parents=True, exist_ok=True)

        dest_path = dest_dir / Path(path).name
        shutil.copy(path, dest_path)
        counts[label_idx] += 1

        # stop early once every selected class has enough
        if all(c >= n_per_class for c in counts.values()):
            break

    print(f"  {split_name}: " + ", ".join(f"{selected_idxs[idx]}={counts[idx]}" for idx in counts))


def prepare():
    if PROCESSED_ROOT.exists():
        print(f"Clearing existing {PROCESSED_ROOT} ...")
        shutil.rmtree(PROCESSED_ROOT)

    print("Loading Imagenette metadata (no download, already cached) ...")
    train_set = Imagenette(root=str(RAW_ROOT), split="train", size="320px", download=False)
    val_set = Imagenette(root=str(RAW_ROOT), split="val", size="320px", download=False)

    print("Filtering and copying images ...")
    prepare_split(train_set, "train", N_TRAIN_PER_CLASS)
    prepare_split(val_set, "test", N_TEST_PER_CLASS)

    print(f"Done. Processed data ready at {PROCESSED_ROOT}")


if __name__ == "__main__":
    prepare()
