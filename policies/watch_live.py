"""Watch any brain from the pool push the can in a live 3D window (laptop only, needs a screen).

Run from the project folder (PowerShell):
    .venv\\Scripts\\python.exe -m policies.watch_live --name sac_960k
    .venv\\Scripts\\python.exe -m policies.watch_live --name random --episodes 3
Drag to rotate, scroll to zoom. Plays TRAINING seeds only: the exam seeds stay sealed.
"""
import argparse
from pathlib import Path

import yaml

from envs.constants import TRAIN_SEEDS
from envs.pusher import PusherEnv, object_goal_distance
from policies.pool import build_pool


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pool.yaml")
    parser.add_argument("--name", default="sac_960k", help="a policy name from the config")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--first-seed", type=int, default=0, help="index into TRAIN_SEEDS")
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))
    entry = [c for c in cfg["pool"] if c["name"] == args.name]
    assert entry, f"no policy named {args.name}; choose from {[c['name'] for c in cfg['pool']]}"
    policy = build_pool(entry, Path(cfg["run"]))[0]

    env = PusherEnv(live=True)
    for seed in TRAIN_SEEDS[args.first_seed: args.first_seed + args.episodes]:
        obs, done = env.reset(seed), False
        policy.reset(seed)
        start = object_goal_distance(obs)
        while not done:                          # 100 steps, about 5 seconds in the window
            obs, _, done = env.step(policy.act(obs))
        print(f"{args.name} | seed {seed}: can-to-goal {start:.3f} m -> {object_goal_distance(obs):.3f} m")
    env.close()


if __name__ == "__main__":
    main()
