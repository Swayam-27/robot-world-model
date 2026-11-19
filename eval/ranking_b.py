"""Step 4d: ranking B. Every brain plays the 50 eval seeds in the REAL simulator, but sees only the state reader's estimates.

Run:  python -m eval.ranking_b --config configs/ranking_b.yaml [--n-seeds N]     (reader v2: configs/ranking_b_v2.yaml)
Loop per step:  real sim -> frame -> reader(last few frames) -> 23 estimated numbers -> brain -> action -> real sim.
The score is still the TRUE final object-goal distance, so B vs A measures what seeing through the camera costs.
Sanity checks:
  1. plumbing: with a perfect reader (hands over the true state) this loop reproduces ranking A's numbers exactly;
  2. the brain really receives the reader's numbers (they differ from the true state);
  3. rerun of one brain gives identical numbers (deterministic, hard rule 4).
Saves ranking_b{tag}.csv and ranking_b{tag}_per_seed.csv in scores_dir and prints Spearman(A, B).
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from envs.constants import EVAL_SEEDS, OBJECT_XYZ, SUCCESS_DISTANCE
from envs.pusher import PusherEnv, object_goal_distance
from models.reader import FrameWindow, ReaderBundle
from policies.pool import build_pool, rank_table
from policies.score import spearman


def play_with_reader(policy, env: PusherEnv, reader: ReaderBundle | None, seed: int) -> tuple[float, float]:
    """One episode where the brain sees the reader's estimate (reader=None: the true state, for the plumbing check).

    Returns (true final object-goal distance in m, mean object position error of the reader in cm).
    """
    true_obs, done = env.reset(seed), False
    policy.reset(seed)
    window = FrameWindow(reader.n_frames if reader else 1)
    frames = window.reset(env.frame())                        # right after reset there is one frame: it fills the window
    obj_err = []
    while not done:
        seen = true_obs if reader is None else reader.read(frames)
        obj_err.append(np.linalg.norm(seen[OBJECT_XYZ] - true_obs[OBJECT_XYZ]) * 100)
        true_obs, _, done = env.step(policy.act(seen))
        frames = window.push(env.frame())
    return object_goal_distance(true_obs), float(np.mean(obj_err))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--n-seeds", type=int, default=len(EVAL_SEEDS), help="fewer seeds for a quick test only")
    args = parser.parse_args()
    cfg = yaml.safe_load(open(args.config))
    pool_cfg = yaml.safe_load(open(cfg["pool_config"]))
    seeds = EVAL_SEEDS[: args.n_seeds]
    out, tag = Path(cfg["scores_dir"]), cfg.get("tag", "")
    torch.set_num_threads(1)                                  # one small image at a time: threads only add overhead

    pool = build_pool(pool_cfg["pool"], Path(pool_cfg["run"]))
    reader = ReaderBundle.load(cfg["reader"])
    env = PusherEnv()

    # Check 1, plumbing: perfect reader -> must equal ranking A's per-seed numbers exactly.
    a_seed = pd.read_csv(out / "real_scores_per_seed.csv")
    probe = pool[0]
    expected = a_seed[a_seed["name"] == probe.name].set_index("seed")["final_distance_m"]
    got = [play_with_reader(probe, env, None, s)[0] for s in seeds[:3]]
    assert np.allclose(got, expected.loc[seeds[:3]].values, atol=1e-9), f"plumbing broken: {got} vs A {list(expected[:3])}"
    print(f"check 1 OK: with the true state this loop reproduces ranking A exactly ({probe.name}, 3 seeds)")

    # Check 2: the brain really gets the reader's numbers, not the true state.
    env.reset(seeds[0]); f0 = env.frame()
    est = reader.read([f0] * reader.n_frames)
    assert not np.allclose(est, env.state()), "reader output equals the true state: is the reader wired in?"
    print(f"check 2 OK: reader estimate differs from the true state (object off by "
          f"{np.linalg.norm(est[OBJECT_XYZ] - env.state()[OBJECT_XYZ]) * 100:.2f} cm at reset)")

    ends, obj_errs = {}, {}
    for p in pool:
        runs = [play_with_reader(p, env, reader, s) for s in seeds]
        ends[p.name] = np.array([r[0] for r in runs]); obj_errs[p.name] = np.mean([r[1] for r in runs])
        print(f"  {p.name:15s} end {ends[p.name].mean() * 100:5.1f} cm | reader object error {obj_errs[p.name]:.2f} cm")

    # Check 3: rerun one brain -> identical.
    again = np.array([play_with_reader(probe, env, reader, s)[0] for s in seeds])
    assert np.array_equal(again, ends[probe.name]), "ranking B changed on rerun: something is not deterministic"
    env.close()
    print(f"check 3 OK: rerun of {probe.name} is identical")

    table = rank_table(ends)
    table["success_rate"] = [(ends[n] < SUCCESS_DISTANCE).mean() for n in table["name"]]
    table["reader_object_err_cm"] = [obj_errs[n] for n in table["name"]]
    a = pd.read_csv(out / "real_scores.csv").set_index("name")
    if len(seeds) == len(EVAL_SEEDS):                         # compare like with like: A is on all 50 seeds
        table["rank_A"] = [int(a.loc[n, "rank"]) for n in table["name"]]
        table["end_mean_A_m"] = [a.loc[n, "end_mean_m"] for n in table["name"]]
        table["change_cm"] = (table["end_mean_m"] - table["end_mean_A_m"]) * 100
        table.to_csv(out / f"ranking_b{tag}.csv", index=False)
        pd.DataFrame([{"name": n, "seed": s, "final_distance_m": d} for n in ends for s, d in zip(seeds, ends[n])]
                     ).to_csv(out / f"ranking_b{tag}_per_seed.csv", index=False)
        print(f"saved {out / f'ranking_b{tag}.csv'} (reader {cfg['reader']}, {reader.n_frames} frames)")
    else:
        print(f"quick test on {len(seeds)} seeds: nothing saved, no comparison with A")
    print(table.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    if "rank_A" in table:
        rho = spearman(list(a.sort_values("rank").index), list(table["name"]))
        print(f"\nSpearman(A, B) = {rho:.3f}   (1.0 = seeing through the reader does not change the order)")


if __name__ == "__main__":
    main()
