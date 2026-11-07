"""Step 3 sanity check: prove recorded frames, actions and states line up, and save a playback GIF.

Run:  python -m data.playback --config configs/dataset.yaml [--n 2]
Checks every file's shapes, then for n random episodes of every brain: reset the simulator to the episode's seed, replay its recorded actions,
and require every state and frame to match the file exactly (frame t + action t -> frame t+1).
"""
import argparse
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import pandas as pd
import yaml

from envs.constants import ACTION_DIM, EPISODE_STEPS, EVAL_SEEDS, FRAME_SIZE, OBS_DIM
from envs.pusher import PusherEnv


def replay_matches(ep: dict, seed: int, env: PusherEnv) -> bool:
    """Re-simulate the recorded actions from the seed; True if every state and frame matches the recording."""
    obs = env.reset(seed)
    ok = np.allclose(obs, ep["states"][0], atol=1e-5) and np.array_equal(env.frame(), ep["frames"][0])
    for t in range(EPISODE_STEPS):
        obs, _, _ = env.step(ep["actions"][t])
        ok &= np.allclose(obs, ep["states"][t + 1], atol=1e-5) and np.array_equal(env.frame(), ep["frames"][t + 1])
    return bool(ok)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--n", type=int, default=2, help="episodes to replay per brain")
    parser.add_argument("--data-dir", help="override out_dir, e.g. the dataset under /kaggle/input")
    args = parser.parse_args()
    cfg = yaml.safe_load(open(args.config))
    out = Path(args.data_dir or cfg["out_dir"])

    index = pd.read_csv(out / "episodes.csv")
    assert not set(index["brain"]) & set(cfg["held_out"]), "a held-out brain is in the dataset"
    print(index.groupby(["brain", "split"]).size().unstack(fill_value=0).to_string())

    assert index["seed"].is_unique and not set(index["seed"]) & set(EVAL_SEEDS), "seeds must be unique training seeds"
    shapes = {"frames": (EPISODE_STEPS + 1, FRAME_SIZE, FRAME_SIZE, 3), "actions": (EPISODE_STEPS, ACTION_DIM),
              "states": (EPISODE_STEPS + 1, OBS_DIM)}
    for f in index["file"]:  # every file opens and has the right shapes
        with np.load(out / f) as ep:
            assert all(ep[k].shape == s for k, s in shapes.items()) and ep["frames"].dtype == np.uint8, f
    print(f"all {len(index)} files load with the right shapes; seeds unique, no eval seeds, no held-out brains")

    env = PusherEnv()
    picks = index.groupby("brain").sample(n=args.n, random_state=0)  # n random episodes of EVERY brain
    results = []
    for _, row in picks.iterrows():
        ep = dict(np.load(out / row["file"]))
        results.append(replay_matches(ep, int(row["seed"]), env))
        print(f"{row['file']} ({row['brain']}, seed {row['seed']}, {row['split']}) | replay matches: {results[-1]}")
    env.close()
    assert all(results), "some replays did not match the recording"
    print(f"replay check: {sum(results)}/{len(results)} episodes match exactly")

    row = picks.iloc[0]
    frames = np.load(out / row["file"])["frames"]
    gif = Path(cfg["out_dir"]) / f"playback_{row['file'].replace('.npz', '')}.gif"  # never write into the data folder
    gif.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(gif, [np.kron(f, np.ones((4, 4, 1), dtype=np.uint8)) for f in frames], duration=0.05, loop=0)
    print(f"saved {gif}")


if __name__ == "__main__":
    main()
