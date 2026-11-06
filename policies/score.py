"""Step 2c: the sealed exam. Score every pool policy on the 50 EVAL_SEEDS in the real simulator -> ranking A.

Run:  python -m policies.score --config configs/pool.yaml
Saves in scores_dir:
    real_scores_per_seed.csv  every policy x every eval seed (final object-goal distance), for the pass mark later
    real_scores.csv           mean final distance + rank per policy: RANKING A, the ground truth
The exam is played twice and must give identical numbers (hard rule 4: deterministic, fixed noise seeds).
Nothing is tuned on these results: the pool was chosen on training seeds (policies/pool.py).
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from envs.constants import EVAL_SEEDS
from policies.pool import build_pool, play_pool, rank_table


def spearman(a: list[str], b: list[str]) -> float:
    """Spearman rank correlation of two orderings (best first) of the same names, no ties: 1 - 6*sum(d^2)/(n(n^2-1))."""
    rank_b = {name: i for i, name in enumerate(b)}
    d = np.array([i - rank_b[name] for i, name in enumerate(a)])
    n = len(a)
    return 1 - 6 * (d ** 2).sum() / (n * (n ** 2 - 1))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run", help="override the checkpoint folder from the config")
    args = parser.parse_args()
    cfg = yaml.safe_load(open(args.config))

    pool = build_pool(cfg["pool"], Path(args.run or cfg["run"]))
    ends = play_pool(pool, EVAL_SEEDS)
    again = play_pool(pool, EVAL_SEEDS)  # rerun the whole exam: must be bit-for-bit identical
    assert all(np.array_equal(ends[n], again[n]) for n in ends), "scores changed on rerun: something is not seeded"

    out = Path(cfg["scores_dir"]); out.mkdir(parents=True, exist_ok=True)
    per_seed = pd.DataFrame([{"name": n, "seed": s, "final_distance_m": d}
                             for n in ends for s, d in zip(EVAL_SEEDS, ends[n])])
    per_seed.to_csv(out / "real_scores_per_seed.csv", index=False)
    table = rank_table(ends)
    table.to_csv(out / "real_scores.csv", index=False)

    print(table.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"exam played twice on {len(EVAL_SEEDS)} eval seeds: identical | saved {out / 'real_scores.csv'}")
    screen = Path(cfg["out_dir"]) / "pool_screen.csv"
    if screen.exists():  # does the order from practice (training seeds) hold on the exam?
        rho = spearman(list(pd.read_csv(screen)["name"]), list(table["name"]))
        print(f"Spearman(practice order, exam order) = {rho:.3f}")


if __name__ == "__main__":
    main()
