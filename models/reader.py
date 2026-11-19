"""Step 4: the state reader. The last few camera frames, stacked -> the 23-number Pusher observation.

In the dream there is no true state, only pictures, so this network is the policies' eyes.
Frames arrive as uint8 (batch, 3 * n_frames, 64, 64) = the last n_frames frames (oldest first) stacked on the channel axis;
scaling to [-1, 1] happens in forward(). v1 used n_frames = 2, v2 uses 3 (see DECISIONS.md).
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

    def __init__(self, n_out: int = OBS_DIM, n_frames: int = 2) -> None:
        super().__init__()
        self.n_frames = n_frames
        self.features = nn.Sequential(
            conv_block(3 * n_frames, 32), nn.MaxPool2d(2),    # RGB per frame; 64 -> 32
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
        """x: uint8 (batch, 3 * n_frames, 64, 64) -> normalised states (batch, n_out)."""
        x = x.float() / 127.5 - 1.0            # [0, 255] -> [-1, 1], only inside the model (project rule)
        return self.head(self.features(x))


def stack_frames(frames: list[np.ndarray]) -> np.ndarray:
    """k frames of (H, W, 3) uint8, oldest first -> one (3k, H, W) uint8 array, channels first as PyTorch wants."""
    return np.concatenate(frames, axis=-1).transpose(2, 0, 1)


class FrameWindow:
    """The last k frames of an episode. Right after a reset there is only one frame: it fills every slot,
    exactly as the training data does at t = 0 (and t = 1 for k = 3)."""

    def __init__(self, k: int) -> None:
        self.k, self.frames = k, []

    def reset(self, frame: np.ndarray) -> list[np.ndarray]:
        """Start a new episode with its first frame."""
        self.frames = [frame] * self.k
        return self.frames

    def push(self, frame: np.ndarray) -> list[np.ndarray]:
        """Add the newest frame, drop the oldest."""
        self.frames = self.frames[1:] + [frame]
        return self.frames


class ReaderBundle:
    """A trained reader plus its normalisation numbers: frames in, real-unit 23-number state out."""

    def __init__(self, model: Reader, mean: np.ndarray, std: np.ndarray, device: str = "cpu") -> None:
        self.model, self.device = model.to(device).eval(), device
        self.mean, self.std = mean.astype(np.float32), std.astype(np.float32)

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "ReaderBundle":
        """Load the checkpoint written by models/train_reader.py."""
        ckpt = torch.load(path, map_location=device, weights_only=False)
        model = Reader(ckpt["n_out"], ckpt.get("n_frames", 2))   # v1 checkpoints predate n_frames: they used 2
        model.load_state_dict(ckpt["model"])
        return cls(model, ckpt["mean"], ckpt["std"], device)

    @property
    def n_frames(self) -> int:
        """How many recent frames the reader needs."""
        return self.model.n_frames

    @torch.no_grad()
    def read(self, frames: list[np.ndarray]) -> np.ndarray:
        """The last n_frames (64, 64, 3) uint8 frames, oldest first -> estimated observation (23,) in real units."""
        assert len(frames) == self.n_frames, f"reader needs {self.n_frames} frames, got {len(frames)}"
        x = torch.from_numpy(stack_frames(frames)[None]).to(self.device)
        return self.model(x)[0].cpu().numpy() * self.std + self.mean
