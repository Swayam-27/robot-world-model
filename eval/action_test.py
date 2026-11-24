"""Step 6: does the dream OBEY the actions, or just draw plausible motion?  (laptop: simulator + reader, no dreaming)

Run:  python -m eval.action_test --config configs/dream_v1.yaml      (after eval/dream.py with swap_actions: true)
eval/dream.py dreamed every start frame twice: with its own recorded actions A, and with actions B taken from an
episode of ANOTHER brain. Here the simulator replays B from the same seed (all rendering stays on the laptop) and
reader v4 reads positions off every frame. If the dream obeys, dream B goes where real B goes, NOT where dream A went.
Reports at frames 10, 50, 100 for the fingertip: dream B vs true B (following error), the reader's own error on real B
frames (floor), and how far A -> B moved it in the dream and in reality, with the cosine between those two moves
(1 = same direction; only episodes where the true move is > 2 cm, since tiny moves have no clear direction).
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from envs.constants import FINGERTIP_XYZ
from envs.pusher import PusherEnv
from eval.dream_tracking import read_episodes
from models.reader import ReaderBundle

REPORT_FRAMES = [10, 50, 100]
MIN_MOVE_CM = 2.0


def simulate(seed: int, actions: np.ndarray, env: PusherEnv) -> tuple[np.ndarray, np.ndarray]:
    """Replay actions from a seed in the real simulator -> frames (101, 64, 64, 3), true states (101, 23)."""
    obs = env.reset(seed)
    frames, states = [env.frame()], [obs]
    for a in actions:
        obs, _, _ = env.step(a)
        frames.append(env.frame()); states.append(obs)
    return np.array(frames, dtype=np.uint8), np.array(states, dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--out-dir", help="override out_dir: the folder that holds dreams.npz")
    args = parser.parse_args()
    cfg = yaml.safe_load(open(args.config))
    out = Path(args.out_dir or cfg["out_dir"])
    a, b = np.load(out / "dreams.npz"), np.load(out / "dreams_swapped.npz")
    seeds, k, states_a, acts_b = a["seeds"], int(a["real_start_frames"]), a["states"], b["actions"]
    real_a, dream_a, dream_b = a["real"], a["dreamed"], b["dreamed"]

    env = PusherEnv()
    _, check = simulate(int(seeds[0]), a["actions"][0], env)   # sanity check: replaying A reproduces the recording
    assert np.abs(check - states_a[0]).max() < 1e-3, "replaying the recorded actions does not reproduce the episode"
    real_b, states_b = (np.stack(x) for x in zip(*(simulate(int(s), act, env) for s, act in zip(seeds, acts_b))))
    env.close()
    assert np.array_equal(real_b[:, 0], real_a[:, 0]), "same seed must give the same start frame"
    print(f"{len(seeds)} episodes | replay of A reproduces the recording | actions B from another brain")

    reader = ReaderBundle.load(cfg["reader"])
    tip = {n: read_episodes(reader, x)[:, :, FINGERTIP_XYZ] for n, x in
           (("dream_a", dream_a), ("dream_b", dream_b), ("real_b", real_b))}
    true_a, true_b = states_a[:, :, FINGERTIP_XYZ], states_b[:, :, FINGERTIP_XYZ]
    cm = lambda x, y, f: np.linalg.norm(x[:, f] - y[:, f], axis=-1) * 100
    rows = []
    for f in REPORT_FRAMES:
        move_d, move_t = tip["dream_b"][:, f] - tip["dream_a"][:, f], true_b[:, f] - true_a[:, f]
        size_d, size_t = np.linalg.norm(move_d, axis=1) * 100, np.linalg.norm(move_t, axis=1) * 100
        ok = size_t > MIN_MOVE_CM
        cos = (move_d * move_t).sum(1) / (np.linalg.norm(move_d, axis=1) * np.linalg.norm(move_t, axis=1) + 1e-9)
        rows.append({"frame": f, "following_error_cm": cm(tip["dream_b"], true_b, f).mean(),
                     "reader_floor_cm": cm(tip["real_b"], true_b, f).mean(), "dream_moved_cm": size_d.mean(),
                     "true_moved_cm": size_t.mean(), "cosine_mean": cos[ok].mean(),
                     "cosine_p10": np.percentile(cos[ok], 10), "n_cosine": int(ok.sum())})
    table = pd.DataFrame(rows)
    table.to_csv(out / "action_test.csv", index=False)
    print("fingertip, swapping actions A -> B:")
    print(table.round(2).to_string(index=False))
    print(f"saved {out / 'action_test.csv'}")


if __name__ == "__main__":
    main()
