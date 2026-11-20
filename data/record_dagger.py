"""Step 4, reader v4 data (DAgger round 1): brains play through the reader's eyes, and we film it with the TRUE state.

Run:  python -m data.record_dagger --config configs/dagger.yaml [--limit 7]
Why: the reader only ever saw poses reached by brains that read the TRUE state. Driven by its own estimates, the arm
reaches poses it never saw, its errors grow and snowball. Here the brain acts on the reader's estimate (exactly as in
ranking B), but every frame is saved together with the true state, so retraining teaches the reader those poses.
Same file format as data/record.py (frames, actions, states) plus reader_states, the estimate the brain acted on:
    frames        (101, 64, 64, 3) uint8    frame t = what the camera sees BEFORE action t
    actions       (100, 7)         float32  the torques actually applied
    states        (101, 23)        float32  TRUE state of each frame  <- the training target
    reader_states (100, 23)        float32  what the brain saw at step t (reader estimate from frames up to t)
Re-running skips episodes whose file already exists, so an interrupted run can simply be restarted.
"""
import argparse
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from data.record import plan_episodes
from envs.constants import ACTION_DIM, EPISODE_STEPS, FRAME_SIZE, OBJECT_XYZ, OBS_DIM
from envs.pusher import PusherEnv, object_goal_distance
from models.reader import FrameWindow, ReaderBundle
from policies.pool import build_pool


def record_through_reader(policy, env: PusherEnv, reader: ReaderBundle, seed: int) -> dict[str, np.ndarray]:
    """One episode where the brain sees only the reader's estimate; returns frames, actions, TRUE states, estimates."""
    obs = env.reset(seed)
    policy.reset(seed)
    window = FrameWindow(reader.n_frames)
    win = window.reset(env.frame())
    frames, states, actions, seen = [env.frame()], [obs], [], []
    for _ in range(EPISODE_STEPS):
        est = reader.read(win)                    # the brain's eyes: an estimate, not the truth
        action = policy.act(est)
        obs, _, done = env.step(action)
        frame = env.frame()
        win = window.push(frame)
        frames.append(frame); states.append(obs); actions.append(action); seen.append(est)
    assert done, "episode should end after EPISODE_STEPS"
    return {"frames": np.array(frames, dtype=np.uint8), "actions": np.array(actions, dtype=np.float32),
            "states": np.array(states, dtype=np.float32), "reader_states": np.array(seen, dtype=np.float32)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--limit", type=int, help="record only a few episodes of every brain (speed test)")
    parser.add_argument("--out-dir", help="override out_dir (for a test run)")
    args = parser.parse_args()
    cfg = yaml.safe_load(open(args.config))
    pool_cfg = yaml.safe_load(open(cfg["pool_config"]))
    torch.set_num_threads(1)                      # one small image at a time; also keeps the run deterministic

    plan = plan_episodes(cfg)                     # same planner as Step 3: unique training seeds, split by episode
    if args.limit:
        plan = plan.groupby("brain").head(max(1, args.limit // len(cfg["episodes"])))
    entries = [c for c in pool_cfg["pool"] if c["name"] in cfg["episodes"]]
    policies = {p.name: p for p in build_pool(entries, Path(pool_cfg["run"]))}
    reader = ReaderBundle.load(cfg["reader"])

    out = Path(args.out_dir or cfg["out_dir"]); out.mkdir(parents=True, exist_ok=True)
    env, finals, obj_err, t0, done_now = PusherEnv(), [], [], time.time(), 0
    for _, row in plan.iterrows():
        path = out / row["file"]
        if not path.exists():
            ep = record_through_reader(policies[row["brain"]], env, reader, int(row["seed"]))
            assert ep["frames"].shape == (EPISODE_STEPS + 1, FRAME_SIZE, FRAME_SIZE, 3)
            assert ep["actions"].shape == (EPISODE_STEPS, ACTION_DIM) and ep["states"].shape == (EPISODE_STEPS + 1, OBS_DIM)
            tmp = path.with_suffix(".tmp")
            with open(tmp, "wb") as fh:           # write fully, then rename: a killed run never leaves a half file
                np.savez_compressed(fh, **ep)
            tmp.replace(path)
            done_now += 1
            if done_now % 100 == 0:
                print(f"{done_now} episodes recorded, {(time.time() - t0) / done_now:.2f} s per episode", flush=True)
        with np.load(path) as ep:
            finals.append(object_goal_distance(ep["states"][-1]))
            est_obj, true_obj = ep["reader_states"][:, OBJECT_XYZ], ep["states"][:-1, OBJECT_XYZ]   # estimate t vs truth t
            obj_err.append(np.linalg.norm(est_obj - true_obj, axis=1).mean())
    env.close()

    plan = plan.assign(final_distance_m=finals, reader_object_err_m=obj_err)
    plan.to_csv(out / "episodes.csv", index=False)
    size_mb = sum(f.stat().st_size for f in out.glob("ep_*.npz")) / 1e6
    summary = plan.groupby("brain")[["final_distance_m", "reader_object_err_m"]].mean() * 100
    print(summary.rename(columns={"final_distance_m": "final cm", "reader_object_err_m": "reader can err cm"}).round(1).to_string())
    print(f"{len(plan)} episodes in {out} ({size_mb:.0f} MB) | recorded now: {done_now} in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
