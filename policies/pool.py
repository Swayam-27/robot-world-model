"""Step 2b: the policy pool, i.e. the "class" of robot brains we rank.

Every brain has the same two buttons, so scoring code and (later) the dream never care which kind it is:
    reset(seed)  - a new episode starts from this start position
    act(state)   - return 7 joint torques
Run:  python -m policies.pool --config configs/pool.yaml
Screens every candidate on TRAINING seeds and flags near-ties, so the final pool is chosen before the exam.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from envs.constants import ACTION_DIM, ACTION_LIMIT, TRAIN_SEEDS
from envs.pusher import PusherEnv, object_goal_distance
from policies.sb3_utils import load_sac


class RandomPolicy:
    """The floor: uniform random torques, repeatable per start position."""

    def __init__(self, name: str) -> None:
        self.name, self._rng = name, np.random.default_rng(0)

    def reset(self, seed: int) -> None:
        self._rng = np.random.default_rng(seed)  # same start position -> same "random" moves

    def act(self, obs: np.ndarray) -> np.ndarray:
        return self._rng.uniform(-ACTION_LIMIT, ACTION_LIMIT, ACTION_DIM).astype(np.float32)


class SACPolicy:
    """A SAC checkpoint's best-guess action, optionally with shaky hands (Gaussian noise)."""

    def __init__(self, name: str, model, noise: float = 0.0) -> None:
        self.name, self.model, self.noise = name, model, noise
        self._rng = np.random.default_rng(0)

    def reset(self, seed: int) -> None:
        self._rng = np.random.default_rng(seed)  # fixed noise seed per start position (hard rule 4)

    def act(self, obs: np.ndarray) -> np.ndarray:
        action = self.model.predict(obs, deterministic=True)[0]  # never sample from SAC itself
        if self.noise > 0:
            action = action + self._rng.normal(0.0, self.noise, ACTION_DIM)
        return np.clip(action, -ACTION_LIMIT, ACTION_LIMIT).astype(np.float32)


def build_pool(candidates: list[dict], run: Path) -> list:
    """Turn config entries into policies; each checkpoint is loaded once and shared by its noisy copies."""
    models, pool = {}, []
    for c in candidates:
        if c["kind"] == "random":
            pool.append(RandomPolicy(c["name"]))
        else:
            path = run / f"{c['ckpt']}.zip"
            if path not in models:
                models[path] = load_sac(path)
            pool.append(SACPolicy(c["name"], models[path], c.get("noise", 0.0)))
    return pool


def final_distance(policy, env: PusherEnv, seed: int) -> float:
    """Play one 100-step episode; return the final object-goal distance in metres (lower = better)."""
    obs, done = env.reset(seed), False
    policy.reset(seed)
    while not done:
        obs, _, done = env.step(policy.act(obs))
    return object_goal_distance(obs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run", help="override the checkpoint folder from the config")
    args = parser.parse_args()
    cfg = yaml.safe_load(open(args.config))

    pool = build_pool(cfg["pool"], Path(args.run or cfg["run"]))
    seeds = TRAIN_SEEDS[: cfg["n_seeds"]]  # practice questions only; the exam is score.py on EVAL_SEEDS
    env = PusherEnv()
    ends = {p.name: np.array([final_distance(p, env, s) for s in seeds]) for p in pool}
    env.close()

    # Rank best -> worst, and compare each brain with the next one on the SAME seeds (paired difference)
    order = sorted(ends, key=lambda n: ends[n].mean())
    rows = []
    for i, name in enumerate(order):
        row = {"name": name, "end_mean_m": ends[name].mean(), "end_std_m": ends[name].std()}
        if i + 1 < len(order):
            diff = ends[order[i + 1]] - ends[name]  # how much worse the next brain is, per seed
            se = diff.std(ddof=1) / np.sqrt(len(diff))
            row.update(gap_to_next_cm=diff.mean() * 100, near_tie=bool(diff.mean() < 2 * se))
        rows.append(row)
    df = pd.DataFrame(rows)

    out = Path(cfg["out_dir"]); out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "pool_screen.csv", index=False)
    print(df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"{len(pool)} candidates on {len(seeds)} training seeds | saved {out / 'pool_screen.csv'}")


if __name__ == "__main__":
    main()
