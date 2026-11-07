"""Step 4a warm-up: port the Real vs AI Faces custom CNN from Keras to PyTorch. YOU fill in the TODOs.

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
        # TODO 1: the conv part, as nn.Sequential(...). Keras block = Conv2D, Conv2D, BatchNormalization, MaxPooling2D.
        #   Conv2D(32, (3,3), padding='same', activation='relu')  ->  nn.Conv2d(IN, 32, 3, padding=1), nn.ReLU()
        #   BatchNormalization()                                  ->  nn.BatchNorm2d(32)
        #   MaxPooling2D(2)                                       ->  nn.MaxPool2d(2)
        #   Blocks: 32, 64, 128 filters. The first conv takes 3 input channels (RGB). Block 3 has no max-pool.
        #   GlobalAveragePooling2D()                              ->  nn.AdaptiveAvgPool2d(1), nn.Flatten()
        self.features = None
        # TODO 2: the dense head, as nn.Sequential(...):
        #   Dense(128, relu) -> Dropout(0.2) -> Dense(64, relu) -> Dropout(0.2) -> Dense(1)
        #   nn.Linear needs the INPUT size too: what comes out of global pooling with 128 channels?
        #   NO sigmoid at the end: BCEWithLogitsLoss applies it inside the loss (more stable).
        self.head = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, 3, 128, 128) floats in [0, 1] -> logits of shape (batch,)."""
        # TODO 3: pass x through features, then head, then .squeeze(1) so the shape is (batch,)
        raise NotImplementedError


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
        # TODO 4: the 4 lines at the heart of every PyTorch training loop:
        #   (a) clear old gradients   (b) forward: logits = model(images); loss = loss_fn(logits, labels)
        #   (c) backpropagate         (d) let the optimizer update the weights
        raise NotImplementedError
        total += loss.item() * len(labels); n += len(labels)
    return total / n


@torch.no_grad()  # no gradients needed when only measuring
def evaluate(model: nn.Module, loader: DataLoader, loss_fn, device) -> tuple[float, float]:
    """Mean loss and accuracy on the validation data."""
    model.eval()  # turns OFF dropout, uses batch-norm's stored statistics
    total, correct, n = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.float().to(device)
        # TODO 5: logits = model(images); add the batch loss to total; a prediction is "1" when logit > 0
        #   (that is sigmoid > 0.5); count correct predictions; n += len(labels)
        raise NotImplementedError
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

    # TODO 6: loss_fn = BCEWithLogitsLoss, optimizer = Adam over model.parameters() with lr=0.001 (same as Keras)
    loss_fn, optimizer = None, None

    best_val, bad_epochs = float("inf"), 0
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, loss_fn, device)
        print(f"epoch {epoch}: train loss {train_loss:.4f} | val loss {val_loss:.4f} | val acc {val_acc:.3f}")
        # TODO 7: EarlyStopping + ModelCheckpoint by hand:
        #   if val_loss improved: remember it, reset bad_epochs, torch.save(model.state_dict(), "best_faces_cnn.pt")
        #   else: bad_epochs += 1, and stop (break) once bad_epochs reaches args.patience

    # TODO 8: reload the best weights into a NEW FacesCNN (load_state_dict + torch.load), move it to device,
    #   evaluate it, and print its val loss/acc. It must equal the best epoch above.          # sanity check 2
    reloaded = None


if __name__ == "__main__":
    main()
