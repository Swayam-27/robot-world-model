"""Play one episode with a trained SAC checkpoint and save it as a GIF (sanity check for Step 2).

Run:  python -m policies.watch_policy --ckpt outputs/sac_pusher/sac_final.zip
Uses a TRAINING seed: eval seeds are only for scoring, never for looking/tuning.
"""
import argparse
import os
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from envs.constants import TRAIN_SEEDS
from envs.pusher import PusherEnv, object_goal_distance  # sets MUJOCO_GL before anything else


def play_episode(model, env: PusherEnv, seed: int) -> tuple[list[np.ndarray], float, float]:
    """Run one deterministic episode; return frames and start/end object-goal distance."""
    obs = env.reset(seed)
    start = object_goal_distance(obs)
    frames, done = [env.frame()], False
    while not done:
        action, _ = model.predict(obs, deterministic=True)  # best-guess action, no sampling
        obs, _, done = env.step(action)
        frames.append(env.frame())
    return frames, start, object_goal_distance(obs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--seed", type=int, default=TRAIN_SEEDS[0])
    args = parser.parse_args()
    assert args.seed in TRAIN_SEEDS, "only watch on training seeds"

    # Create the renderer BEFORE importing Stable-Baselines3/torch: loading those graphics/CUDA
    # libraries first crashed EGL rendering on Kaggle (segfault, 2026-11-05).
    env = PusherEnv()
    env.reset(args.seed); env.frame()
    # SB3 -> torch.utils.tensorboard -> tensorboard imports TensorFlow if it is installed (Kaggle has it).
    # TensorFlow ships its own LLVM, which clashes with OSMesa's LLVM ("LLVM ERROR: inconsistency in
    # registered CommandLine options") and crashed EGL too. We never use TensorFlow, so block it.
    sys.modules["tensorflow"] = None
    from stable_baselines3 import SAC

    model = SAC.load(args.ckpt, device="cpu")
    frames, start, end = play_episode(model, env, args.seed)
    env.close()

    out = Path(args.ckpt).with_suffix(f".seed{args.seed}.gif")
    big = [np.kron(f, np.ones((4, 4, 1), dtype=np.uint8)) for f in frames]  # 64 -> 256 for viewing
    imageio.mimsave(out, big, duration=0.05, loop=0)
    print(f"[MUJOCO_GL={os.environ.get('MUJOCO_GL', 'default')}] {Path(args.ckpt).name}: object-goal distance {start:.3f} m -> {end:.3f} m | saved {out}")


if __name__ == "__main__":
    main()
