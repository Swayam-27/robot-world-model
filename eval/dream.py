"""Step 6: replay validation episodes inside the dream and measure how fast it drifts from reality.

Run:  python -m eval.dream --config configs/dream_v1.yaml [--limit 8]      (GPU if there is one; Kaggle: notebooks/step6_dream.ipynb)
The world model gets the first real_start_frames real frames of an episode, then draws every later frame from its
OWN earlier drawings plus the episode's recorded actions (actions are replayed, nobody reacts to the dream yet; that
is Step 7). Frame t+1 = model(frames t-3..t, actions t-3..t), with frame 0 repeated / zero actions before step 3,
exactly as in training. At every step: PSNR of dreamed vs real frame (higher = closer, identical = infinite),
next to a "frozen" baseline that just repeats the last real frame forever: a dream that is not clearly above the
frozen line has stopped moving the arm. Outputs: per-step CSV, drift chart, side-by-side GIFs (real | dream), and
dreams.npz (dreamed + real frames, true states) for eval/dream_tracking.py.
With swap_actions, every episode is also dreamed from the same start with the NEXT picked episode's actions (always
another brain, see pick_episodes): dreams_swapped.npz, for eval/action_test.py, which simulates the truth on the laptop.
"""
import argparse
import time
from pathlib import Path

import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

from envs.constants import ACTION_DIM, CONTEXT_FRAMES, EPISODE_STEPS  # noqa: E402
from models.world_model import WorldModel  # noqa: E402


def load_val_episodes(dirs: list[Path]) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], list[str], list[int]]:
    """All validation episodes -> frames (N, 101, 64, 64, 3) uint8, actions (N, 100, 7), true states (N, 101, 23),
    names 'folder/file', brain and seed of each episode."""
    frames, actions, states, names, brains, seeds = [], [], [], [], [], []
    for d in dirs:
        index = pd.read_csv(d / "episodes.csv")
        for f, brain, seed in index.loc[index["split"] == "val", ["file", "brain", "seed"]].itertuples(index=False):
            with np.load(d / f) as ep:
                frames.append(ep["frames"]); actions.append(ep["actions"].astype(np.float32))
                states.append(ep["states"].astype(np.float32))
            names.append(f"{d.name}/{f}"); brains.append(brain); seeds.append(int(seed))
    return np.stack(frames), np.stack(actions), np.stack(states), names, brains, seeds


def pick_episodes(brains: list[str], n: int, seed: int) -> np.ndarray:
    """n episode numbers in round-robin order over the brains (shuffled inside each brain), so that even the first
    few cover every brain. Sorting the picks put the 8 first tests all on sac_960k (DECISIONS 2026-11-24)."""
    rng = np.random.default_rng(seed)
    queues = [list(rng.permutation(np.flatnonzero(np.array(brains) == b))) for b in sorted(set(brains))]
    order = [q[i] for i in range(max(map(len, queues))) for q in queues if i < len(q)]
    return np.array(order[:n])


@torch.no_grad()
def dream(model: WorldModel, start: torch.Tensor, actions: torch.Tensor, steps: int, aug_level: float) -> torch.Tensor:
    """start (B, k, 3, 64, 64) uint8 real frames 0..k-1, actions (B, 100, 7) -> all 101 frames, frames k.. dreamed."""
    b, k = start.shape[:2]
    frames = list(start.unbind(1))                              # grows by one dreamed frame per step
    zero = torch.zeros(b, ACTION_DIM, device=start.device)
    for t in range(k - 1, EPISODE_STEPS):                       # draw frame t+1
        past = range(t - CONTEXT_FRAMES + 1, t + 1)             # t-3 .. t
        ctx = torch.stack([frames[max(j, 0)] for j in past], dim=1)
        act = torch.stack([actions[:, j] if j >= 0 else zero for j in past], dim=1)
        frames.append(model.sample(ctx, act, steps=steps, aug_level=aug_level))
    return torch.stack(frames, dim=1)


def dream_all(model: WorldModel, real: np.ndarray, actions: np.ndarray, k: int, cfg: dict, device: str) -> np.ndarray:
    """Dream every episode from its first k real frames, batch by batch -> (N, 101, 64, 64, 3) uint8.
    The seed is set here, so every setting (and the swapped actions) starts from the same random numbers."""
    torch.manual_seed(cfg["seed"])
    out, t0 = [], time.time()
    for i in range(0, len(real), cfg["batch"]):
        start = torch.from_numpy(real[i:i + cfg["batch"], :k]).permute(0, 1, 4, 2, 3).to(device)
        act = torch.from_numpy(actions[i:i + cfg["batch"]]).to(device)
        d = dream(model, start, act, cfg["denoising_steps"], cfg["aug_level"])
        out.append(d.permute(0, 1, 3, 4, 2).cpu().numpy())
        print(f"  dreamed {min(i + cfg['batch'], len(real))}/{len(real)} episodes, {time.time() - t0:.0f} s", flush=True)
    return np.concatenate(out)


def psnr(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per-frame PSNR in dB over the last 3 axes (H, W, C) of uint8 images; identical frames -> inf."""
    mse = ((a.astype(np.float32) - b.astype(np.float32)) ** 2).mean(axis=(-3, -2, -1))
    with np.errstate(divide="ignore"):
        return 10 * np.log10(255.0**2 / mse)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--limit", type=int, help="dream only this many episodes (quick test)")
    parser.add_argument("--checkpoint", help="override checkpoint")
    parser.add_argument("--data-dir", action="append", help="override data_dirs (repeatable; same order: main, dagger)")
    parser.add_argument("--out-dir", help="override out_dir (for a test run)")
    args = parser.parse_args()
    cfg = yaml.safe_load(open(args.config))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.out_dir or cfg["out_dir"]); out.mkdir(parents=True, exist_ok=True)

    ck = torch.load(args.checkpoint or cfg["checkpoint"], map_location="cpu", weights_only=False)
    assert ck["config"]["name"] == cfg["world_model"], f"config wants {cfg['world_model']}, checkpoint is {ck['config']['name']}"
    model = WorldModel(ck["config"]); model.load_state_dict(ck["model"]); model.to(device).eval()
    frames, actions, states, names, brains, seeds = load_val_episodes([Path(d) for d in args.data_dir or cfg["data_dirs"]])
    pick = pick_episodes(brains, cfg["n_episodes"], cfg["seed"])
    pick = pick[:args.limit] if args.limit else pick
    real, acts = frames[pick], actions[pick]
    k = cfg["real_start_frames"]
    print(f"{cfg['world_model']} on {device}, step {ck['step']:,} | dreaming {len(pick)} validation episodes from {k} real frame(s), "
          f"{cfg['denoising_steps']} denoising steps, aug_level {cfg['aug_level']}")

    dreamed = dream_all(model, real, acts, k, cfg, device)                     # (N, 101, 64, 64, 3)
    assert dreamed.shape == real.shape and np.array_equal(dreamed[:, :k], real[:, :k])

    frozen = np.repeat(real[:, k - 1:k], real.shape[1], axis=1)                # baseline: never moves
    steps = np.arange(k, EPISODE_STEPS + 1)
    p_dream, p_frozen = psnr(dreamed, real)[:, k:], psnr(frozen, real)[:, k:]
    table = pd.DataFrame({"frame": steps, "dream_psnr": p_dream.mean(0), "dream_psnr_p10": np.percentile(p_dream, 10, axis=0),
                          "frozen_psnr": p_frozen.mean(0)})
    table.to_csv(out / "drift.csv", index=False)
    for f in (k, 10, 25, 50, 75, 100):
        r = table.set_index("frame").loc[f]
        print(f"frame {f:3d}: dream PSNR {r.dream_psnr:5.1f} dB (worst 10% {r.dream_psnr_p10:5.1f}) | frozen {r.frozen_psnr:5.1f} dB")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(steps, table.dream_psnr, label="dream (mean)")
    ax.fill_between(steps, table.dream_psnr_p10, table.dream_psnr, alpha=0.2, label="dream (worst 10% of episodes)")
    ax.plot(steps, table.frozen_psnr, "--", color="gray", label="frozen: repeat the start frame")
    ax.set(xlabel="frame (steps after reset)", ylabel="PSNR vs real frame (dB, higher = closer)",
           title=f"Dream drift, {len(pick)} validation episodes")
    ax.legend(); ax.grid(alpha=0.3); fig.tight_layout(); fig.savefig(out / "drift.png", dpi=120)

    for j in range(min(cfg["gif_episodes"], len(pick))):                      # real | dream, 4x bigger
        side = np.concatenate([real[j], np.zeros_like(real[j][:, :, :4]), dreamed[j]], axis=2)
        imageio.mimsave(out / f"dream_{j}.gif", [np.kron(f, np.ones((4, 4, 1), dtype=np.uint8)) for f in side],
                        duration=0.05, loop=0)
    np.savez_compressed(out / "dreams.npz", dreamed=dreamed, real=real, states=states[pick], actions=acts,
                        names=np.array([names[i] for i in pick]),
                        brains=np.array([brains[i] for i in pick]), seeds=np.array([seeds[i] for i in pick]), real_start_frames=k)   # for eval/dream_tracking.py
    if cfg["swap_actions"]:
        partner = np.roll(np.arange(len(pick)), -1)                            # episode i gets episode i+1's actions
        assert all(brains[pick[i]] != brains[pick[j]] for i, j in enumerate(partner)), "partner must be another brain"
        np.savez_compressed(out / "dreams_swapped.npz", dreamed=dream_all(model, real, acts[partner], k, cfg, device),
                            actions=acts[partner], partner=partner)
    print(f"episodes: {', '.join(names[i] for i in pick[:cfg['gif_episodes']])} (GIFs: left real, right dream)")
    print(f"saved {out / 'dreams.npz'}, {out / 'drift.csv'}, {out / 'drift.png'}, {min(cfg['gif_episodes'], len(pick))} GIFs in {out}")


if __name__ == "__main__":
    main()
