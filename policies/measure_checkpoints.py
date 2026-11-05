"""Measure how well every checkpoint of a SAC run actually PUSHES the can (Step 2, before building the pool).

Run:  python -m policies.measure_checkpoints --run outputs/sac_pusher
Plays each checkpoint deterministically on the same N TRAINING seeds (no rendering) and records the
object-goal distance at start and end. Saves checkpoint_measure.csv and checkpoint_measure.png in --run.
Eval seeds are never used here: they are only for the final scoring (ranking A).
"""
import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from envs.constants import TRAIN_SEEDS
from envs.pusher import PusherEnv, object_goal_distance
from policies.sb3_utils import load_sac

MOVED_M = 0.02  # the can counts as "moved" if it ends at least 2 cm closer to the goal


def run_episode(model, env: PusherEnv, seed: int) -> tuple[float, float]:
    """One deterministic episode; return (start, end) object-goal distance in metres."""
    obs = env.reset(seed)
    start, done = object_goal_distance(obs), False
    while not done:
        obs, _, done = env.step(model.predict(obs, deterministic=True)[0])
    return start, object_goal_distance(obs)


def checkpoint_steps(path: Path, total: int) -> int:
    """Training step of a checkpoint file: sac_20000_steps.zip -> 20000; sac_final.zip -> total."""
    m = re.search(r"_(\d+)_steps", path.name)
    return int(m.group(1)) if m else total


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, help="training run folder, e.g. outputs/sac_pusher")
    parser.add_argument("--n-seeds", type=int, default=20)
    args = parser.parse_args()

    run = Path(args.run)
    ckpts = sorted(run.glob("ckpts/*.zip"), key=lambda p: checkpoint_steps(p, 0))
    seeds = TRAIN_SEEDS[: args.n_seeds]
    env = PusherEnv()

    rows = []
    for ckpt in ckpts:
        model = load_sac(ckpt)
        starts, ends = np.array([run_episode(model, env, s) for s in seeds]).T
        moved = starts - ends
        rows.append({"checkpoint": ckpt.name, "steps": checkpoint_steps(ckpt, 0),
                     "start_mean_m": starts.mean(), "end_mean_m": ends.mean(), "end_std_m": ends.std(),
                     "moved_mean_m": moved.mean(), "frac_moved": (moved >= MOVED_M).mean()})
        r = rows[-1]
        print(f"{ckpt.name:24s} end {r['end_mean_m']:.3f} ± {r['end_std_m']:.3f} m | "
              f"moved {r['moved_mean_m']*100:5.1f} cm | can moved in {r['frac_moved']*100:3.0f}% of episodes")
    env.close()

    df = pd.DataFrame(rows)
    df.to_csv(run / "checkpoint_measure.csv", index=False)

    plt.figure(figsize=(7, 4))
    plt.errorbar(df["steps"], df["end_mean_m"], yerr=df["end_std_m"], marker="o", capsize=3, label="end (mean ± std)")
    plt.axhline(df["start_mean_m"].iloc[0], color="gray", ls="--", label="start (can never touched)")
    plt.xlabel("training steps"); plt.ylabel("final object-goal distance (m), lower = better")
    plt.title(f"Pushing skill per checkpoint ({len(seeds)} training seeds)"); plt.legend(); plt.tight_layout()
    plt.savefig(run / "checkpoint_measure.png", dpi=120); plt.close()
    print(f"saved {run / 'checkpoint_measure.csv'} and {run / 'checkpoint_measure.png'}")


if __name__ == "__main__":
    main()
