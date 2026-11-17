"""Step 4: the state reader. Two stacked camera frames -> the 23-number Pusher observation.

In the dream there is no true state, only pictures, so this network is the policies' eyes.
Frames arrive as uint8 (batch, 6, 64, 64) = frame t-1 and frame t stacked on the channel axis; scaling to [-1, 1] happens in forward().
The network predicts NORMALISED states; ReaderBundle undoes the normalisation so callers get real units.
"""
import numpy as np
import torch
from torch import nn

from envs.constants import FRAME_SIZE, OBS_DIM


def conv_block(c_in: int, c_out: int) -> nn.Sequential:
    """Two 3x3 convs + ReLU, then batch norm (same block as the faces CNN)."""
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, padding=1), nn.ReLU(),
        nn.Conv2d(c_out, c_out, 3, padding=1), nn.ReLU(),
        nn.BatchNorm2d(c_out),
    )


class Reader(nn.Module):
    """Swayam's reader CNN: 3 conv blocks -> 1x1 conv bottleneck -> flatten (keeps positions) -> dense head."""

    def __init__(self, n_out: int = OBS_DIM) -> None:
        super().__init__()
        self.features = nn.Sequential(
            conv_block(6, 32), nn.MaxPool2d(2),    # 64 -> 32
            conv_block(32, 64), nn.MaxPool2d(2),   # 32 -> 16
            conv_block(64, 128), nn.MaxPool2d(2),  # 16 -> 8
            nn.Conv2d(128, 16, kernel_size=1), nn.ReLU(),  # 128 channels -> 16 per cell, the 8x8 grid stays
        )
        side = FRAME_SIZE // 8
        self.head = nn.Sequential(
            nn.Flatten(),                      # NOT global pooling: the head must know WHERE things are
            nn.Linear(16 * side * side, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, n_out),              # no activation: regression outputs can be any number
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: uint8 (batch, 6, 64, 64) -> normalised states (batch, n_out)."""
        x = x.float() / 127.5 - 1.0            # [0, 255] -> [-1, 1], only inside the model (project rule)
        return self.head(self.features(x))


def stack_pair(prev: np.ndarray, cur: np.ndarray) -> np.ndarray:
    """Two (H, W, 3) uint8 frames -> one (6, H, W) uint8 array, channels first as PyTorch wants."""
    return np.concatenate([prev, cur], axis=-1).transpose(2, 0, 1)


class ReaderBundle:
    """A trained reader plus its normalisation numbers: frames in, real-unit 23-number state out."""

    def __init__(self, model: Reader, mean: np.ndarray, std: np.ndarray, device: str = "cpu") -> None:
        self.model, self.device = model.to(device).eval(), device
        self.mean, self.std = mean.astype(np.float32), std.astype(np.float32)

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "ReaderBundle":
        """Load the checkpoint written by models/train_reader.py."""
        ckpt = torch.load(path, map_location=device, weights_only=False)
        model = Reader(ckpt["n_out"])
        model.load_state_dict(ckpt["model"])
        return cls(model, ckpt["mean"], ckpt["std"], device)

    @torch.no_grad()
    def read(self, prev: np.ndarray, cur: np.ndarray) -> np.ndarray:
        """Two (64, 64, 3) uint8 frames -> estimated observation (23,) in real units."""
        x = torch.from_numpy(stack_pair(prev, cur)[None]).to(self.device)
        return self.model(x)[0].cpu().numpy() * self.std + self.mean
