"""Step 3 sanity check: prove recorded frames, actions and states line up, and save a playback GIF.

Run:  python -m data.playback --config configs/dataset.yaml [--n 5]
For a few random episodes: reset the simulator to the episode's seed, replay its recorded actions,
and require every state and frame to match the file exactly (frame t + action t -> frame t+1).
"""
import argparse
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import pandas as pd
import yaml

from envs.constants import EPISODE_STEPS
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
    parser.add_argument("--n", type=int, default=5, help="episodes to replay")
    args = parser.parse_args()
    cfg = yaml.safe_load(open(args.config))
    out = Path(cfg["out_dir"])

    index = pd.read_csv(out / "episodes.csv")
    assert not set(index["brain"]) & set(cfg["held_out"]), "a held-out brain is in the dataset"
    print(index.groupby(["brain", "split"]).size().unstack(fill_value=0).to_string())

    env = PusherEnv()
    picks = index.sample(n=min(args.n, len(index)), random_state=0)
    for _, row in picks.iterrows():
        ep = dict(np.load(out / row["file"]))
        print(f"{row['file']} ({row['brain']}, seed {row['seed']}): frames {ep['frames'].shape} {ep['frames'].dtype}, "
              f"actions {ep['actions'].shape}, states {ep['states'].shape} | replay matches: {replay_matches(ep, int(row['seed']), env)}")
    env.close()

    row = picks.iloc[0]
    frames = np.load(out / row["file"])["frames"]
    gif = out / f"playback_{row['file'].replace('.npz', '')}.gif"
    imageio.mimsave(gif, [np.kron(f, np.ones((4, 4, 1), dtype=np.uint8)) for f in frames], duration=0.05, loop=0)
    print(f"saved {gif}")


if __name__ == "__main__":
    main()
