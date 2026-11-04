"""MuJoCo Pusher-v5 behind the ArmEnv interface (see envs/interface.py)."""
import os
import sys

# Headless rendering must be chosen BEFORE MuJoCo is imported.
# Linux (Kaggle) has no screen -> use EGL (GPU) rendering. Windows uses its default.
if sys.platform.startswith("linux"):
    os.environ.setdefault("MUJOCO_GL", "egl")  # fallback if EGL fails: "osmesa"

import gymnasium as gym
import numpy as np

from envs.constants import CAMERA_CONFIG, ENV_ID, FRAME_SIZE, GOAL_XYZ, OBJECT_XYZ, OBS_DIM, TRAIN_SEEDS


class TrainSeedReset(gym.Wrapper):
    """Every reset starts from a seed drawn from TRAIN_SEEDS, so eval start positions never appear in training."""

    def __init__(self, env: gym.Env, rng_seed: int = 0) -> None:
        super().__init__(env)
        self._rng = np.random.default_rng(rng_seed)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:                     # SB3 passes its seed once at the start
            self._rng = np.random.default_rng(seed)
        start_seed = int(self._rng.choice(TRAIN_SEEDS))
        return self.env.reset(seed=start_seed, options=options)


def make_train_env(rng_seed: int = 0) -> gym.Env:
    """Gymnasium Pusher for RL training (Stable-Baselines3 needs a gym.Env). No rendering: faster."""
    return TrainSeedReset(gym.make(ENV_ID), rng_seed=rng_seed)


def object_goal_distance(obs: np.ndarray) -> float:
    """Distance between the object (can) and the goal in the table plane (x, y), in metres."""
    obj_xy = obs[OBJECT_XYZ][:2]
    goal_xy = obs[GOAL_XYZ][:2]
    return float(np.linalg.norm(obj_xy - goal_xy))


class PusherEnv:
    """Pusher-v5 that renders FRAME_SIZE x FRAME_SIZE RGB frames. Implements ArmEnv."""

    def __init__(self, frame_size: int = FRAME_SIZE) -> None:
        camera = {**CAMERA_CONFIG, "lookat": np.array(CAMERA_CONFIG["lookat"])}
        self._env = gym.make(
            ENV_ID,
            render_mode="rgb_array",
            width=frame_size,
            height=frame_size,
            default_camera_config=camera,
        )
        self._obs: np.ndarray | None = None

    def reset(self, seed: int) -> np.ndarray:
        """Start an episode; the seed fixes where the can and goal start."""
        self._obs, _ = self._env.reset(seed=seed)
        return self._obs.copy()

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool]:
        """Apply 7 joint torques; return (state, reward, done). Done after 100 steps."""
        self._obs, reward, terminated, truncated, _ = self._env.step(action)
        return self._obs.copy(), float(reward), bool(terminated or truncated)

    def frame(self) -> np.ndarray:
        """Current camera image, (64, 64, 3) uint8."""
        return self._env.render()

    def state(self) -> np.ndarray:
        """Current true 23-number state."""
        assert self._obs is not None, "call reset() first"
        assert self._obs.shape == (OBS_DIM,)
        return self._obs.copy()

    def sample_action(self) -> np.ndarray:
        """A random action: the 'floor' policy."""
        return self._env.action_space.sample()

    def close(self) -> None:
        """Free the simulator and renderer."""
        self._env.close()
