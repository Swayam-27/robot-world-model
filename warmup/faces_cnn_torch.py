"""Step 4a warm-up: port the Real vs AI Faces custom CNN from Keras to PyTorch (solved; comments say which Keras line each part replaces).

Original Keras model: github.com/Swayam-27/Real_vs_AI_faces_CNN (cell 19 of the notebook).
Run on Kaggle (GPU on, add the dataset "human-and-ai-photos-generalized" as input):
    !python warmup/faces_cnn_torch.py --data "/kaggle/input/datasets/swayam006/human-and-ai-photos-generalized" --epochs 5
Sanity checks built in: a shape test before training, a loss per epoch, and a reload test at the end.
The goal is learning the PyTorch training loop, not beating the Keras accuracy.
"""
import argparse

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


class FacesCNN(nn.Module):
    """The custom Keras CNN in PyTorch: 3 conv blocks -> global average pooling -> dense head -> 1 logit."""

    def __init__(self) -> None:
        super().__init__()
        # Keras block = Conv2D, Conv2D, BatchNormalization, MaxPooling2D. PyTorch Conv2d needs the INPUT channels too.
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
            nn.BatchNorm2d(32), nn.MaxPool2d(2),                                   # 128 -> 64
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.BatchNorm2d(64), nn.MaxPool2d(2),                                   # 64 -> 32
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(),
            nn.BatchNorm2d(128),                                                   # block 3: no max-pool
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),                                 # GlobalAveragePooling2D -> (batch, 128)
        )
        # Dense(128) -> Dropout -> Dense(64) -> Dropout -> Dense(1). No sigmoid: BCEWithLogitsLoss applies it inside.
        self.head = nn.Sequential(
            nn.Linear(128, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, 3, 128, 128) floats in [0, 1] -> logits of shape (batch,)."""
        return self.head(self.features(x)).squeeze(1)  # (batch, 1) -> (batch,)


def make_loaders(data_dir: str, batch_size: int = 32) -> tuple[DataLoader, DataLoader, list[str]]:
    """Like keras image_dataset_from_directory with validation_split=0.2, seed=42, image_size=(128,128). (Done for you.)"""
    tfm = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor()])  # ToTensor: (3,H,W), scaled to [0,1]
    full = datasets.ImageFolder(data_dir, transform=tfm)                              # labels from folder names
    n_val = int(0.2 * len(full))
    train, val = random_split(full, [len(full) - n_val, n_val], generator=torch.Generator().manual_seed(42))
    return (DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=2),
            DataLoader(val, batch_size=batch_size, num_workers=2), full.classes)


def train_one_epoch(model: nn.Module, loader: DataLoader, loss_fn, optimizer, device) -> float:
    """One pass over the training data; return the mean loss. This is what Keras fit() hides from you."""
    model.train()  # turns ON dropout and batch-norm updates
    total, n = 0.0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.float().to(device)
        optimizer.zero_grad()            # (a) clear old gradients (PyTorch ADDS new gradients to old ones otherwise)
        logits = model(images)           # (b) forward pass
        loss = loss_fn(logits, labels)   #     how wrong are we?
        loss.backward()                  # (c) backpropagation: fills p.grad for every weight
        optimizer.step()                 # (d) Adam updates every weight using p.grad
        total += loss.item() * len(labels); n += len(labels)
    return total / n


@torch.no_grad()  # no gradients needed when only measuring
def evaluate(model: nn.Module, loader: DataLoader, loss_fn, device) -> tuple[float, float]:
    """Mean loss and accuracy on the validation data."""
    model.eval()  # turns OFF dropout, uses batch-norm's stored statistics
    total, correct, n = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.float().to(device)
        logits = model(images)
        total += loss_fn(logits, labels).item() * len(labels)       # loss_fn gives the batch MEAN; weight by batch size
        correct += ((logits > 0).float() == labels).sum().item()    # logit > 0  <=>  sigmoid > 0.5
        n += len(labels)
    return total / n, correct / n


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--patience", type=int, default=3)
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_loader, val_loader, classes = make_loaders(args.data)
    print(f"classes {classes} | train batches {len(train_loader)} | val batches {len(val_loader)} | device {device}")

    model = FacesCNN().to(device)
    out = model(torch.zeros(2, 3, 128, 128, device=device))
    assert out.shape == (2,), f"expected logits of shape (2,), got {tuple(out.shape)}"   # sanity check 1
    print(f"shape test OK | {sum(p.numel() for p in model.parameters()):,} parameters")

    # Keras compile(loss="binary_crossentropy", optimizer=Adam(0.001))
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    best_val, bad_epochs = float("inf"), 0
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, loss_fn, device)
        print(f"epoch {epoch}: train loss {train_loss:.4f} | val loss {val_loss:.4f} | val acc {val_acc:.3f}")
        if val_loss < best_val:                                   # ModelCheckpoint(save_best_only=True)
            best_val, bad_epochs = val_loss, 0
            torch.save(model.state_dict(), "best_faces_cnn.pt")   # state_dict = dict of all weight tensors
            print("  saved best_faces_cnn.pt")
        else:                                                     # EarlyStopping(patience)
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"  no improvement for {args.patience} epochs: stopping")
                break

    # Reload into a NEW model: proves the file holds everything needed.                   # sanity check 2
    reloaded = FacesCNN().to(device)
    reloaded.load_state_dict(torch.load("best_faces_cnn.pt", map_location=device))
    re_loss, re_acc = evaluate(reloaded, val_loader, loss_fn, device)
    print(f"reloaded best model: val loss {re_loss:.4f} | val acc {re_acc:.3f}")
    assert abs(re_loss - best_val) < 1e-4, "reloaded model does not match the best epoch"


if __name__ == "__main__":
    main()
