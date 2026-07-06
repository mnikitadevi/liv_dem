# src/infer.py
"""
Loads Model A (small scratch CNN) and Model B (pretrained ResNet50),
and exposes a simple predict() function for each.

Each model has different preprocessing — this file keeps that straight so
ui.py doesn't have to know or care about the details.
"""

from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision import models, transforms

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


DEVICE = get_device()


# ---------------------------------------------------------------------------
# Model A: small scratch CNN (must match the class definition in train_scratch.py exactly)
# ---------------------------------------------------------------------------

class SmallCNN(torch.nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),

            torch.nn.Conv2d(16, 32, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),

            torch.nn.Conv2d(32, 64, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
        )
        self.classifier = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(64 * 8 * 8, 64),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


MODEL_A_IMAGE_SIZE = 64

model_a_transform = transforms.Compose([
    transforms.Resize((MODEL_A_IMAGE_SIZE, MODEL_A_IMAGE_SIZE)),
    transforms.ToTensor(),
    # No normalization — matches train_scratch.py exactly.
])


def load_model_a():
    classes_path = MODELS_DIR / "model_a_scratch_classes.txt"
    weights_path = MODELS_DIR / "model_a_scratch.pt"

    classes = classes_path.read_text().strip().split("\n")

    model = SmallCNN(num_classes=len(classes))
    state_dict = torch.load(weights_path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()

    return model, classes


def predict_model_a(image, model, classes, topk=3):
    """image: PIL.Image (RGB). Returns list of (label, confidence) sorted desc."""
    img = image.convert("RGB")
    tensor = model_a_transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1)[0]

    topk = min(topk, len(classes))
    top_probs, top_idxs = torch.topk(probs, topk)

    return [
        (classes[idx].replace("_", " "), float(prob))
        for prob, idx in zip(top_probs.tolist(), top_idxs.tolist())
    ]


# ---------------------------------------------------------------------------
# Model B: pretrained ResNet50 (ImageNet, 1000 classes)
# ---------------------------------------------------------------------------

MODEL_B_IMAGE_SIZE = 224

model_b_transform = transforms.Compose([
    transforms.Resize((MODEL_B_IMAGE_SIZE, MODEL_B_IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


def load_model_b():
    weights_path = MODELS_DIR / "model_b.pt"
    classes_path = MODELS_DIR / "imagenet_classes.txt"

    classes = classes_path.read_text().strip().split("\n")

    # weights=None: reconstruct architecture only, no network call.
    # Real weights come from our locally-cached state_dict below.
    model = models.resnet50(weights=None)
    state_dict = torch.load(weights_path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()

    return model, classes


def predict_model_b(image, model, classes, topk=3):
    img = image.convert("RGB")
    tensor = model_b_transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1)[0]

    top_probs, top_idxs = torch.topk(probs, topk)

    return [
        (classes[idx], float(prob))
        for prob, idx in zip(top_probs.tolist(), top_idxs.tolist())
    ]


# ---------------------------------------------------------------------------
# Stats for the comparison panel — read live from disk so numbers stay honest
# even if you retrain or change data counts before the event.
# ---------------------------------------------------------------------------

def count_images(folder: Path) -> int:
    if not folder.exists():
        return 0
    return sum(1 for _ in folder.rglob("*.jpg")) + sum(1 for _ in folder.rglob("*.JPEG"))


def get_model_a_stats(model, classes):
    train_dir = PROJECT_ROOT / "data" / "processed" / "train"
    n_params = sum(p.numel() for p in model.parameters())
    return {
        "Training images": count_images(train_dir),
        "Categories recognized": len(classes),
        "Parameters": f"{n_params:,}",
        "Pretraining": "None — random init",
    }


def get_model_b_stats(model, classes):
    n_params = sum(p.numel() for p in model.parameters())
    return {
        "Training images": "~1.28 million",
        "Categories recognized": len(classes),
        "Parameters": f"{n_params:,}",
        "Pretraining": "Full ImageNet, GPU cluster",
    }


if __name__ == "__main__":
    # Quick smoke test
    from PIL import Image

    model_a, classes_a = load_model_a()
    model_b, classes_b = load_model_b()
    print("Model A classes:", classes_a)
    print("Model A stats:", get_model_a_stats(model_a, classes_a))
    print("Model B stats:", get_model_b_stats(model_b, classes_b))
    print("Both models loaded successfully.")
