"""Step 6: does the dream OBEY the actions, or just draw plausible motion?

Run:  python -m eval.action_test --config configs/action_test_v1.yaml     (after eval/dream.py wrote dreams.npz)
For each dreamed episode (start seed s, recorded actions A) we take actions B = the recorded actions of the NEXT
dreamed episode: realistic torques that send the arm somewhere else. Then:
  real B  = the simulator replays B from seed s (laptop, all rendering stays here)
  dream B = the world model dreams B from the same single start frame (same settings as dream A)
Reader v4 reads the fingertip and can off every frame. If the dream obeys the controls, dream B follows real B,
NOT dream A. Reports, at the chosen frames: dream B vs real B (the action-following error), dream B vs dream A
(how much the dream changed), real B vs real A (how much it SHOULD have changed).
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from envs.constants import FINGERTIP_XYZ, OBJECT_XYZ
from envs.pusher import PusherEnv
from eval.dream import dream
from eval.dream_tracking import read_episodes
from models.reader import ReaderBundle
from models.world_model import WorldModel

PARTS = {"fingertip": FINGERTIP_XYZ, "can": OBJECT_XYZ}


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
    cfg = yaml.safe_load(open(parser.parse_args().config))
    dcfg = yaml.safe_load(open(cfg["dream_config"]))
    out = Path(cfg["out_dir"])
    data = np.load(Path(dcfg["out_dir"]) / "dreams.npz")
    names, k = [str(n) for n in data["names"]], int(data["real_start_frames"])
    real_a, dream_a, states_a, acts_a = data["real"], data["dreamed"], data["states"], data["actions"]
    seeds = {f"{Path(d).name}/{r.file}": int(r.seed) for d in cfg["data_dirs"]
             for r in pd.read_csv(Path(d) / "episodes.csv").itertuples()}
    acts_b = np.roll(acts_a, -1, axis=0)                       # episode i gets episode i+1's actions
    print(f"{len(names)} episodes; actions B = the next episode's recorded actions")

    env = PusherEnv()                                          # sanity check: replaying A reproduces the recording
    f0, s0 = simulate(seeds[names[0]], acts_a[0], env)
    assert np.abs(s0 - states_a[0]).max() < 1e-3, "replaying the recorded actions does not reproduce the episode"
    real_b, states_b = zip(*(simulate(seeds[n], a, env) for n, a in zip(names, acts_b)))
    real_b, states_b = np.stack(real_b), np.stack(states_b)
    env.close()
    print(f"simulator replay of A reproduces the recording (max state diff {np.abs(s0 - states_a[0]).max():.1e}); "
          f"real B simulated")

    ck = torch.load(dcfg["checkpoint"], map_location="cpu", weights_only=False)
    model = WorldModel(ck["config"]); model.load_state_dict(ck["model"]); model.eval()
    torch.manual_seed(dcfg["seed"])
    start = torch.from_numpy(real_a[:, :k]).permute(0, 1, 4, 2, 3)
    assert np.array_equal(real_b[:, :k], real_a[:, :k]), "same seed must give the same start frame"
    dream_b = dream(model, start, torch.from_numpy(acts_b), dcfg["denoising_steps"], dcfg["aug_level"])
    dream_b = dream_b.permute(0, 1, 3, 4, 2).numpy()
    print("dream B done")

    reader = ReaderBundle.load(dcfg["reader"])
    est = {name: read_episodes(reader, x) for name, x in
           [("dream_a", dream_a), ("dream_b", dream_b), ("real_b", real_b)]}
    true = {"a": states_a, "b": states_b}
    rows = []
    for part, sl in PARTS.items():
        dist = lambda x, y: np.linalg.norm(x[:, :, sl] - y[:, :, sl], axis=-1) * 100      # cm, (N, T)
        pairs = {"dream B vs true B (following error)": dist(est["dream_b"], true["b"]),
                 "reader on real B vs true B (floor)": dist(est["real_b"], true["b"]),
                 "dream B vs dream A (dream changed)": dist(est["dream_b"], est["dream_a"]),
                 "true B vs true A (should change)": dist(true["b"], true["a"])}
        rows += [{"part": part, "comparison": c, **{f"frame {f}": v[:, f].mean() for f in cfg["report_frames"]}}
                 for c, v in pairs.items()]
    table = pd.DataFrame(rows)
    table.to_csv(out / "action_test.csv", index=False)
    print("mean distance in cm:")
    print(table.round(1).to_string(index=False))

    np.savez_compressed(out / "action_test.npz", dream_b=dream_b, real_b=real_b, states_b=states_b, actions_b=acts_b)
    import imageio.v2 as imageio                               # GIF: real A | dream A | real B | dream B
    gap = np.zeros_like(real_a[0][:, :, :4])
    for j in range(min(2, len(names))):
        side = np.concatenate([real_a[j], gap, dream_a[j], gap, gap, gap, real_b[j], gap, dream_b[j]], axis=2)
        imageio.mimsave(out / f"action_test_{j}.gif", [np.kron(f, np.ones((3, 3, 1), dtype=np.uint8)) for f in side],
                        duration=0.05, loop=0)
    print(f"saved {out / 'action_test.csv'} and 2 GIFs (real A | dream A || real B | dream B)")


if __name__ == "__main__":
    main()
