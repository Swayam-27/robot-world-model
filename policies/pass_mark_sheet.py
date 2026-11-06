"""Step 2d: a picture sheet for choosing the success threshold ("is the can on the goal?") by eye.

Run:  python -m policies.pass_mark_sheet --config configs/pool.yaml
Plays the pool brains on TRAINING seeds, keeps each episode's final frame + final can-to-goal distance,
and saves a grid: one row per target distance, columns = real episodes that ended closest to it.
Frames are rendered at 256x256 for viewing only (the dataset stays 64x64).
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

from envs.constants import TRAIN_SEEDS
from envs.pusher import PusherEnv, object_goal_distance
from policies.pool import build_pool


def final_frames(pool: list, seeds: list[int], env: PusherEnv) -> list[tuple[float, np.ndarray, str, int]]:
    """Play every policy on every seed; return (final distance, final frame, policy name, seed) per episode."""
    episodes = []
    for p in pool:
        for seed in seeds:
            obs, done = env.reset(seed), False
            p.reset(seed)
            while not done:
                obs, _, done = env.step(p.act(obs))
            episodes.append((object_goal_distance(obs), env.frame(), p.name, seed))
    return episodes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pool.yaml")
    parser.add_argument("--n-seeds", type=int, default=30)
    parser.add_argument("--targets", type=float, nargs="*", default=[0.02, 0.04, 0.06, 0.08, 0.10, 0.14])
    parser.add_argument("--per-row", type=int, default=4)
    parser.add_argument("--crop", type=int, nargs=4, default=[100, 205, 120, 225],
                        help="y0 y1 x0 x1 window of the 256x256 frame (the goal never moves, so a fixed zoom works)")
    args = parser.parse_args()
    cfg = yaml.safe_load(open(args.config))

    pool = build_pool(cfg["pool"], Path(cfg["run"]))
    env = PusherEnv(frame_size=256)
    episodes = final_frames(pool, TRAIN_SEEDS[: args.n_seeds], env)  # practice seeds only
    env.close()
    dists = np.array([e[0] for e in episodes])

    fig, axes = plt.subplots(len(args.targets), args.per_row, figsize=(2.6 * args.per_row, 2.8 * len(args.targets)))
    for r, target in enumerate(args.targets):
        closest = np.argsort(np.abs(dists - target))[: args.per_row]  # real episodes nearest this distance
        for c, i in enumerate(closest):
            d, frame, name, seed = episodes[i]
            ax = axes[r, c]
            y0, y1, x0, x1 = args.crop
            ax.imshow(frame[y0:y1, x0:x1]); ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"{d * 100:.1f} cm | {name} s{seed}", fontsize=7)
        axes[r, 0].set_ylabel(f"~{target * 100:.0f} cm", fontsize=11)
    fig.suptitle("Final frames by can-to-goal distance (training seeds): where does 'on the goal' stop?")
    fig.tight_layout()
    out = Path(cfg["out_dir"]) / "pass_mark_sheet.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110); plt.close(fig)
    print(f"{len(episodes)} episodes, final distances {dists.min()*100:.1f} to {dists.max()*100:.1f} cm | saved {out}")


if __name__ == "__main__":
    main()
