# src/train.py
"""
Trains Model A live: a frozen ResNet18 backbone (ImageNet-pretrained) with only
the final layer retrained on your 3 small classes.

This is intentionally small-data / few-epoch — Model A is supposed to be a
visibly undertrained model, not a good one.

Usage:
    uv run src/train.py

Saves:
    models/model_a.pt          -- trained weights (just the final layer + backbone)
    models/model_a_classes.txt -- class names in index order, so infer.py can label predictions
"""

import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torchvision.datasets import ImageFolder

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAIN_DIR = PROJECT_ROOT / "data" / "processed" / "train"
TEST_DIR = PROJECT_ROOT / "data" / "processed" / "test"
MODELS_DIR = PROJECT_ROOT / "models"

EPOCHS = 15
BATCH_SIZE = 8
LR = 0.001
IMAGE_SIZE = 224  # what ResNet expects


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_dataloaders():
    # ImageNet normalization stats — required since we're using an ImageNet-pretrained backbone
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    test_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        normalize,
    ])

    train_dataset = ImageFolder(str(TRAIN_DIR), transform=train_transform)
    test_dataset = ImageFolder(str(TEST_DIR), transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, test_loader, train_dataset.classes


def build_model(num_classes: int, device):
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    # Freeze everything except the final layer
    for param in model.parameters():
        param.requires_grad = False

    # Replace the final layer for our 3 classes — this is the only part that trains
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model.to(device)


def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total


def train():
    device = get_device()
    print(f"Using device: {device}")

    train_loader, test_loader, class_names = build_dataloaders()
    print(f"Classes: {class_names}")
    print(f"Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")

    model = build_model(num_classes=len(class_names), device=device)

    criterion = nn.CrossEntropyLoss()
    # Only the final layer has requires_grad=True, so only it gets optimized
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=LR)

    print("\nStarting training...\n")
    start_time = time.time()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        avg_loss = running_loss / len(train_loader.dataset)
        test_acc = evaluate(model, test_loader, device)

        print(f"Epoch {epoch:2d}/{EPOCHS}  loss={avg_loss:.4f}  test_accuracy={test_acc*100:.1f}%")

    elapsed = time.time() - start_time
    final_acc = evaluate(model, test_loader, device)

    print(f"\nTraining complete in {elapsed:.1f}s")
    print(f"Final test accuracy: {final_acc*100:.1f}%")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), MODELS_DIR / "model_a.pt")

    with open(MODELS_DIR / "model_a_classes.txt", "w") as f:
        f.write("\n".join(class_names))

    print(f"\nSaved model to {MODELS_DIR / 'model_a.pt'}")
    print(f"Saved class labels to {MODELS_DIR / 'model_a_classes.txt'}")


if __name__ == "__main__":
    train()
