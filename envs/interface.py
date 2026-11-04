"""The contract every data source must follow (MuJoCo now, a real arm + webcam in phase 2).

Recording, training and evaluation code only use these four methods, never MuJoCo directly.
"""
from typing import Protocol

import numpy as np


class ArmEnv(Protocol):
    """A robot arm we can reset, act in, and observe through a camera."""

    def reset(self, seed: int) -> np.ndarray:
        """Start a new episode from a repeatable start position; return the first state."""
        ...

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool]:
        """Apply joint torques for one step; return (state, reward, episode_done)."""
        ...

    def frame(self) -> np.ndarray:
        """Return the current camera image, shape (H, W, 3), dtype uint8."""
        ...

    def state(self) -> np.ndarray:
        """Return the current true state vector (23 numbers for Pusher)."""
        ...
