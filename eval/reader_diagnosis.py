"""Step 4 diagnosis: which of the reader's numbers breaks a brain? Runs on TRAINING seeds only.

Run:  python -m eval.reader_diagnosis --config configs/ranking_b.yaml
One brain plays the same training seeds under several conditions. Each step it sees the TRUE state, except the chosen
group(s) of numbers, which come from the reader instead:
    true          nothing from the reader (should match the brain's normal score)
    reader_all    everything from the reader (what ranking B does)
    only_<group>  one group from the reader, the rest true
The group whose "only" condition hurts most is the one to fix first (change one thing at a time).
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from envs.constants import (FINGERTIP_XYZ, JOINT_ANGLES, JOINT_VELOCITIES, OBJECT_XYZ, SUCCESS_DISTANCE,
                            TRAIN_SEEDS)
from envs.pusher import PusherEnv, object_goal_distance
from models.reader import FrameWindow, ReaderBundle
from policies.pool import build_pool

GROUPS = {"joint_angles": JOINT_ANGLES, "joint_velocities": JOINT_VELOCITIES,
          "fingertip": FINGERTIP_XYZ, "object": OBJECT_XYZ}   # goal is left out: the reader gets it almost exactly


def play_mixed(policy, env: PusherEnv, reader: ReaderBundle, seed: int, from_reader: list[slice]) -> float:
    """One episode where the brain sees the true state, except the slices in from_reader, which come from the reader.

    Returns the TRUE final object-goal distance in metres.
    """
    true_obs, done = env.reset(seed), False
    policy.reset(seed)
    window = FrameWindow(reader.n_frames)
    frames = window.reset(env.frame())
    while not done:
        seen = true_obs.copy()
        if from_reader:
            est = reader.read(frames)
            for sl in from_reader:
                seen[sl] = est[sl]                            # swap in the reader's guess for this group only
        true_obs, _, done = env.step(policy.act(seen))
        frames = window.push(env.frame())
    return object_goal_distance(true_obs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(open(args.config))
    diag = cfg["diagnosis"]
    pool_cfg = yaml.safe_load(open(cfg["pool_config"]))
    seeds = TRAIN_SEEDS[: diag["n_seeds"]]                    # training seeds: we may choose a fix based on these
    torch.set_num_threads(1)

    entry = [c for c in pool_cfg["pool"] if c["name"] == diag["brain"]]
    policy = build_pool(entry, Path(pool_cfg["run"]))[0]
    reader = ReaderBundle.load(cfg["reader"])
    env = PusherEnv()

    conditions = {"true": [], "reader_all": list(GROUPS.values()),
                  **{f"only_{g}": [sl] for g, sl in GROUPS.items()}}
    ends = {}
    for name, slices in conditions.items():
        ends[name] = np.array([play_mixed(policy, env, reader, s, slices) for s in seeds])
        print(f"  {name:22s} end {ends[name].mean() * 100:5.1f} cm | success {(ends[name] < SUCCESS_DISTANCE).mean():.0%}")
    env.close()

    # Sanity check: with nothing from the reader, the score must equal the pool screen (same brain, same seeds).
    screen = pd.read_csv(Path(pool_cfg["out_dir"]) / "pool_screen.csv").set_index("name")
    if len(seeds) == pool_cfg["n_seeds"]:
        assert abs(ends["true"].mean() - screen.loc[diag["brain"], "end_mean_m"]) < 1e-9, "'true' does not match the pool screen"
        print(f"check OK: 'true' matches the pool screen for {diag['brain']} ({ends['true'].mean() * 100:.1f} cm)")

    base = ends["true"]
    rows = []
    for name, e in ends.items():
        diff = e - base                                       # paired: same seeds, so compare seed by seed
        rows.append({"condition": name, "end_mean_cm": e.mean() * 100, "success_rate": (e < SUCCESS_DISTANCE).mean(),
                     "worse_than_true_cm": diff.mean() * 100,
                     "se_cm": diff.std(ddof=1) / np.sqrt(len(diff)) * 100})
    table = pd.DataFrame(rows).sort_values("worse_than_true_cm", ascending=False)
    out = Path(diag["out"]); out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)
    print(f"\n{diag['brain']} on {len(seeds)} training seeds (worse_than_true: + = the reader's numbers hurt):")
    print(table.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
