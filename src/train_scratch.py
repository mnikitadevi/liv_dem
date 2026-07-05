# src/train_scratch.py
"""
Trains Model A from scratch — a small CNN with randomly-initialized weights,
no pretrained backbone. This is the "honest" version of Model A: whatever it
learns comes only from your ~75 training images, nothing borrowed.

Expect it to be noticeably weaker and less consistent than the transfer-learning
version (train.py) — that's the point. The train/test accuracy gap is itself
part of the lesson (it may memorize training images without generalizing).

Usage:
    uv run src/train_scratch.py

Saves:
    models/model_a_scratch.pt          -- trained weights
    models/model_a_scratch_classes.txt -- class names in index order
"""

import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAIN_DIR = PROJECT_ROOT / "data" / "processed" / "train"
TEST_DIR = PROJECT_ROOT / "data" / "processed" / "test"
MODELS_DIR = PROJECT_ROOT / "models"

EPOCHS = 40
BATCH_SIZE = 8
LR = 0.001
IMAGE_SIZE = 64  # small on purpose — keeps the model tiny and fast


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class SmallCNN(nn.Module):
    """A deliberately small CNN — a few conv layers, no pretraining, no borrowed features."""

    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 64 -> 32

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 32 -> 16

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 16 -> 8
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def build_dataloaders():
    # No ImageNet normalization here on purpose — this model never saw ImageNet,
    # so there's no reason to match its statistics. Simple 0-1 scaling via ToTensor is enough.
    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
    ])

    test_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
    ])

    train_dataset = ImageFolder(str(TRAIN_DIR), transform=train_transform)
    test_dataset = ImageFolder(str(TEST_DIR), transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, test_loader, train_dataset.classes


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


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def train():
    device = get_device()
    print(f"Using device: {device}")

    train_loader, test_loader, class_names = build_dataloaders()
    print(f"Classes: {class_names}")
    print(f"Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")

    model = SmallCNN(num_classes=len(class_names)).to(device)
    print(f"Model parameters: {count_params(model):,}  (no pretraining — random init)")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

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
        train_acc = evaluate(model, train_loader, device)
        test_acc = evaluate(model, test_loader, device)

        print(
            f"Epoch {epoch:2d}/{EPOCHS}  loss={avg_loss:.4f}  "
            f"train_accuracy={train_acc*100:.1f}%  test_accuracy={test_acc*100:.1f}%"
        )

    elapsed = time.time() - start_time
    final_train_acc = evaluate(model, train_loader, device)
    final_test_acc = evaluate(model, test_loader, device)

    print(f"\nTraining complete in {elapsed:.1f}s")
    print(f"Final train accuracy: {final_train_acc*100:.1f}%")
    print(f"Final test accuracy:  {final_test_acc*100:.1f}%")
    if final_train_acc - final_test_acc > 0.15:
        print("(Big train/test gap — model memorized training images rather than generalizing. Good talking point.)")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), MODELS_DIR / "model_a_scratch.pt")

    with open(MODELS_DIR / "model_a_scratch_classes.txt", "w") as f:
        f.write("\n".join(class_names))

    print(f"\nSaved model to {MODELS_DIR / 'model_a_scratch.pt'}")
    print(f"Saved class labels to {MODELS_DIR / 'model_a_scratch_classes.txt'}")


if __name__ == "__main__":
    train()
