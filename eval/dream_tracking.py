"""Step 6: where are the can and the fingertip in the dream? Read them off the frames with reader v4.

Run:  python -m eval.dream_tracking --config configs/dream_v1.yaml      (after eval/dream.py wrote dreams.npz)
The reader turns frames into positions. Run it on the REAL frames (its own error: the floor) and on the DREAM
frames, and compare both with the true states. Dream error far above the floor = the dream put the can or the arm
in the wrong place. Same actions are replayed in both, so a perfect world model would match
the real frames and sit on the floor. Reports errors at chosen frames and saves a chart of error vs frame.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

from envs.constants import FINGERTIP_XYZ, OBJECT_XYZ  # noqa: E402
from models.reader import ReaderBundle  # noqa: E402

PARTS = {"can": OBJECT_XYZ, "fingertip": FINGERTIP_XYZ}


@torch.no_grad()
def read_episodes(reader: ReaderBundle, frames: np.ndarray, batch: int = 512) -> np.ndarray:
    """frames (N, T, 64, 64, 3) uint8 -> reader estimates (N, T, 23), each from the last n_frames frames
    (frame 0 repeated at the start, exactly as the reader sees an episode live)."""
    n, t = frames.shape[:2]
    k = reader.n_frames
    idx = np.maximum(np.arange(t)[:, None] - np.arange(k - 1, -1, -1)[None], 0)     # (T, k): oldest first
    windows = frames[:, idx]                                                         # (N, T, k, 64, 64, 3)
    x = windows.transpose(0, 1, 2, 5, 3, 4).reshape(n * t, k * 3, *frames.shape[2:4])
    out = [reader.model(torch.from_numpy(np.ascontiguousarray(x[i:i + batch]))).numpy() for i in range(0, len(x), batch)]
    return (np.concatenate(out) * reader.std + reader.mean).reshape(n, t, -1)


def can_missing(frames: np.ndarray, bright: int) -> np.ndarray:
    """(N, T, 64, 64, 3) uint8 -> (N, T) bool: True where no pixel is as bright as the can in R, G and B.
    The reader ALWAYS outputs a can position, even for a can the dream lost, so its cm errors need this check.
    Real validation frames: 8.5 such pixels per frame near the can, 0.01 elsewhere. The arm hides the can in some
    real frames too, so the dream's rate is compared with the real one, not with zero."""
    return (frames.min(axis=-1) > bright).sum(axis=(2, 3)) == 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--out-dir", help="override out_dir: the folder that holds dreams.npz")
    args = parser.parse_args()
    cfg = yaml.safe_load(open(args.config))
    out = Path(args.out_dir or cfg["out_dir"])
    data = np.load(out / "dreams.npz")
    dreamed, real, states, k = data["dreamed"], data["real"], data["states"], int(data["real_start_frames"])
    reader = ReaderBundle.load(cfg["reader"])
    print(f"{len(real)} dreamed episodes | reader {cfg['reader']} ({reader.n_frames} frames)")

    est = {"real": read_episodes(reader, real), "dream": read_episodes(reader, dreamed),
           "frozen": np.repeat(states[:, :1], len(states[0]), axis=1)}   # yardstick: nothing ever moves from frame 0
    frames = np.arange(real.shape[1])
    rows, curves = [], {}
    for part, sl in PARTS.items():
        for src, e in est.items():
            err_cm = np.linalg.norm(e[:, :, sl] - states[:, :, sl], axis=-1) * 100            # (N, T)
            curves[(part, src)] = err_cm.mean(0)
            rows += [{"part": part, "frames_from": src, "frame": f, "mean_cm": err_cm[:, f].mean(),
                      "p90_cm": np.percentile(err_cm[:, f], 90)} for f in frames]
    table = pd.DataFrame(rows)
    table.to_csv(out / "tracking.csv", index=False)

    show = [f for f in (k, 10, 25, 50, 100) if f < len(frames)]
    wide = table[table.frame.isin(show)].pivot_table(index=["part", "frames_from"], columns="frame", values="mean_cm")
    print("mean position error in cm (real = reader on real frames, the floor; dream = reader on dream frames; "
          "frozen = assume nothing moves from frame 0, the yardstick a dream must beat):")
    print(wide.round(1).to_string())

    miss = {src: can_missing(x, cfg["can_bright"]) for src, x in (("real", real), ("dream", dreamed))}
    pd.DataFrame({"frame": frames, "real_missing": miss["real"].mean(0), "dream_missing": miss["dream"].mean(0)}).to_csv(
        out / "can_missing.csv", index=False)
    print("share of episodes with no can-bright pixel:")
    for f in show:
        print(f"  frame {f:3d}: real {miss['real'][:, f].mean():4.0%} | dream {miss['dream'][:, f].mean():4.0%}")
    gone = {src: int(m[:, -20:].all(1).sum()) for src, m in miss.items()}     # gone for good = absent in all last 20 frames
    print(f"can absent in ALL of the last 20 frames: real {gone['real']}, dream {gone['dream']} of {len(real)} episodes")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, part in zip(axes, PARTS):
        ax.plot(frames, curves[(part, "real")], color="gray", label="reader on REAL frames (floor)")
        ax.plot(frames, curves[(part, "dream")], color="C3", label="reader on DREAM frames")
        ax.plot(frames, curves[(part, "frozen")], color="gray", ls="--", label="frozen: assume nothing moves")
        ax.axvline(k - 0.5, color="k", lw=0.5, ls=":")
        ax.set(title=f"{part}: where the frames put it", xlabel="frame", ylabel="error vs true position (cm)")
        ax.grid(alpha=0.3); ax.legend()
    fig.suptitle(f"Dream tracking, {len(real)} validation episodes, recorded actions replayed")
    fig.tight_layout(); fig.savefig(out / "tracking.png", dpi=120)
    print(f"saved {out / 'tracking.csv'}, {out / 'tracking.png'}")


if __name__ == "__main__":
    main()
