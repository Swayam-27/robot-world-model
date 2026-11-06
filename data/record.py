"""Step 3: record the world-model dataset: frames, actions and true states from many brains (training seeds only).

Run:  python -m data.record --config configs/dataset.yaml [--limit 10]
One file per episode, ep_XXXXX.npz:
    frames  (101, 64, 64, 3) uint8    frame t = what the camera sees BEFORE action t (frame 0 = after reset)
    actions (100, 7)         float32  the torques actually applied (already clipped)
    states  (101, 23)        float32  true state matching each frame
Plus episodes.csv: file, brain, seed, split (train/val, by episode), final distance.
Re-running skips episodes whose file already exists, so an interrupted run can simply be restarted.
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from envs.constants import ACTION_DIM, EPISODE_STEPS, EVAL_SEEDS, FRAME_SIZE, OBS_DIM, TRAIN_SEEDS
from envs.pusher import PusherEnv, object_goal_distance
from policies.pool import build_pool


def record_episode(policy, env: PusherEnv, seed: int) -> dict[str, np.ndarray]:
    """Play one episode and return its frames, actions and states, aligned so frame t + action t -> frame t+1."""
    obs = env.reset(seed)
    policy.reset(seed)
    frames, states, actions = [env.frame()], [obs], []
    for _ in range(EPISODE_STEPS):
        action = policy.act(obs)
        obs, _, done = env.step(action)
        frames.append(env.frame()); states.append(obs); actions.append(action)
    assert done, "episode should end after EPISODE_STEPS"
    return {"frames": np.array(frames, dtype=np.uint8),
            "actions": np.array(actions, dtype=np.float32),
            "states": np.array(states, dtype=np.float32)}


def plan_episodes(cfg: dict) -> pd.DataFrame:
    """One row per episode: file name, brain, a unique training seed, and train/val split (last 10% of each brain)."""
    rows, i = [], 0
    for brain, n in cfg["episodes"].items():
        assert brain not in cfg["held_out"], f"{brain} is held out and must not be recorded"
        n_val = int(round(n * cfg["val_fraction"]))
        for k in range(n):
            seed = TRAIN_SEEDS[cfg["first_seed_index"] + i]
            rows.append({"file": f"ep_{i:05d}.npz", "brain": brain, "seed": seed,
                         "split": "val" if k >= n - n_val else "train"})
            i += 1
    plan = pd.DataFrame(rows)
    assert plan["seed"].is_unique and not set(plan["seed"]) & set(EVAL_SEEDS), "seeds must be unique training seeds"
    return plan


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--limit", type=int, help="record only the first N planned episodes (speed test)")
    args = parser.parse_args()
    cfg = yaml.safe_load(open(args.config))
    pool_cfg = yaml.safe_load(open(cfg["pool_config"]))

    plan = plan_episodes(cfg)
    if args.limit:
        plan = plan.groupby("brain").head(max(1, args.limit // len(cfg["episodes"])))  # a few of every brain
    entries = [c for c in pool_cfg["pool"] if c["name"] in cfg["episodes"]]
    policies = {p.name: p for p in build_pool(entries, Path(pool_cfg["run"]))}

    out = Path(cfg["out_dir"]); out.mkdir(parents=True, exist_ok=True)
    env, finals, t0, done_now = PusherEnv(), [], time.time(), 0
    for _, row in plan.iterrows():
        path = out / row["file"]
        if path.exists():  # resume: keep what an earlier run recorded
            finals.append(object_goal_distance(np.load(path)["states"][-1])); continue
        ep = record_episode(policies[row["brain"]], env, int(row["seed"]))
        assert ep["frames"].shape == (EPISODE_STEPS + 1, FRAME_SIZE, FRAME_SIZE, 3)
        assert ep["actions"].shape == (EPISODE_STEPS, ACTION_DIM) and ep["states"].shape == (EPISODE_STEPS + 1, OBS_DIM)
        tmp = path.with_suffix(".tmp")
        with open(tmp, "wb") as fh:  # write fully, then rename: a killed run never leaves a half-written .npz
            np.savez_compressed(fh, **ep)
        tmp.replace(path)
        finals.append(object_goal_distance(ep["states"][-1]))
        done_now += 1
        if done_now % 100 == 0:
            print(f"{done_now} episodes recorded, {(time.time() - t0) / done_now:.2f} s per episode")
    env.close()

    plan = plan.assign(final_distance_m=finals)
    plan.to_csv(out / "episodes.csv", index=False)
    size_mb = sum(f.stat().st_size for f in out.glob("ep_*.npz")) / 1e6
    print(plan.groupby(["brain", "split"]).size().unstack(fill_value=0).to_string())
    print(f"{len(plan)} episodes in {out} ({size_mb:.0f} MB) | recorded now: {done_now} in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
