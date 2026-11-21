"""Step 5: world-model training examples. One example = the last 4 frames + the last 4 actions -> the NEXT frame.

Run:  python -m models.world_model_data --config configs/world_model_v1.yaml [--limit 3]
Episode file format (data/record.py): frames (101, 64, 64, 3) uint8, frame t is seen BEFORE action t;
actions (100, 7). So action t turns frame t into frame t+1, and every episode gives 100 examples (t = 0..99):
    context = frames t-3 .. t  (oldest first)   actions = actions t-3 .. t   target = frame t+1
Before step 3 there is not enough history: frame 0 repeats and the missing actions are zeros (the arm had not been
pushed yet), so a dream can start straight after a reset. Frames stay uint8 here; scaling to [-1, 1] is the
model's job. Train/val split comes from each folder's episodes.csv (split by episode, never by frame).
Sanity checks: shapes and dtypes, target == frame t+1, padding at t = 0, an image grid of examples.
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
from torch.utils.data import Dataset  # noqa: E402

from envs.constants import ACTION_DIM, CONTEXT_FRAMES, EPISODE_STEPS, FRAME_SIZE  # noqa: E402


def load_episodes(data_dir: Path, split: str, limit: int | None) -> tuple[np.ndarray, np.ndarray]:
    """All episodes of one split -> frames (N, 101, 64, 64, 3) uint8 and actions (N, 100, 7) float32."""
    index = pd.read_csv(data_dir / "episodes.csv")
    files = index.loc[index["split"] == split, "file"].tolist()[:limit]
    frames, actions = [], []
    for f in files:
        with np.load(data_dir / f) as ep:
            frames.append(ep["frames"]); actions.append(ep["actions"].astype(np.float32))
    return np.stack(frames), np.stack(actions)


def load_split(dirs: list[Path], split: str, limit: int | None) -> tuple[np.ndarray, np.ndarray]:
    """The same split from several folders, joined (each folder keeps its own episode split)."""
    parts = [load_episodes(d, split, limit) for d in dirs]
    return np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts])


class NextFrameDataset(Dataset):
    """Item i = episode i // 100, step t = i % 100  ->  context (4, 3, 64, 64) uint8, actions (4, 7) float32,
    target (3, 64, 64) uint8."""

    def __init__(self, frames: np.ndarray, actions: np.ndarray, n_context: int = CONTEXT_FRAMES) -> None:
        self.frames, self.actions, self.n = frames, actions, n_context
        self.steps = actions.shape[1]                       # 100 transitions per episode

    def __len__(self) -> int:
        return len(self.frames) * self.steps

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ep, t = divmod(i, self.steps)
        past = range(t - self.n + 1, t + 1)                 # t-3, t-2, t-1, t
        ctx = np.stack([self.frames[ep, max(k, 0)] for k in past])                       # frame 0 repeats
        act = np.stack([self.actions[ep, k] if k >= 0 else np.zeros(ACTION_DIM, np.float32) for k in past])
        target = self.frames[ep, t + 1]
        return (torch.from_numpy(ctx).permute(0, 3, 1, 2), torch.from_numpy(act),
                torch.from_numpy(target).permute(2, 0, 1))


def save_grid(ds: NextFrameDataset, items: list[int], path: Path) -> None:
    """One row per example: the 4 context frames, then the target, titled with episode step and action size."""
    fig, axes = plt.subplots(len(items), ds.n + 1, figsize=(2 * (ds.n + 1), 2 * len(items)))
    for row, i in zip(axes, items):
        ctx, act, target = ds[i]
        t = i % ds.steps
        for k, ax in enumerate(row):
            img = ctx[k] if k < ds.n else target
            ax.imshow(img.permute(1, 2, 0).numpy()); ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"frame {max(t - ds.n + 1 + k, 0)}" if k < ds.n else f"TARGET {t + 1}", fontsize=8)
        row[0].set_ylabel(f"t={t}\n|a_t|={act[-1].norm():.2f}", fontsize=8)
    fig.suptitle("World-model examples: 4 context frames -> next frame")
    fig.tight_layout(); fig.savefig(path, dpi=100); plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-dir", help="override data.data_dir")
    parser.add_argument("--extra-data-dir", action="append", help="override data.extra_data_dirs (repeatable)")
    parser.add_argument("--limit", type=int, help="episodes per split per folder, for a quick test")
    parser.add_argument("--out", default="outputs/world_model_data_check.png")
    args = parser.parse_args()
    cfg = yaml.safe_load(open(args.config))["data"]
    assert cfg["context_frames"] == CONTEXT_FRAMES, "context_frames must match envs/constants.py"
    dirs = [Path(args.data_dir or cfg["data_dir"]), *map(Path, args.extra_data_dir or cfg.get("extra_data_dirs", []))]

    train_f, train_a = load_split(dirs, "train", args.limit)
    val_f, val_a = load_split(dirs, "val", args.limit)
    train, val = NextFrameDataset(train_f, train_a), NextFrameDataset(val_f, val_a)
    print(f"folders: {', '.join(map(str, dirs))}")
    print(f"train {len(train_f)} episodes -> {len(train):,} examples | val {len(val_f)} episodes -> {len(val):,} examples")
    print(f"RAM for frames: {(train_f.nbytes + val_f.nbytes) / 1e9:.2f} GB")

    ctx, act, target = train[0]                                                          # sanity check 1: shapes
    assert ctx.shape == (CONTEXT_FRAMES, 3, FRAME_SIZE, FRAME_SIZE) and ctx.dtype == torch.uint8
    assert act.shape == (CONTEXT_FRAMES, ACTION_DIM) and act.dtype == torch.float32
    assert target.shape == (3, FRAME_SIZE, FRAME_SIZE) and target.dtype == torch.uint8
    print(f"shapes OK | context {tuple(ctx.shape)} | actions {tuple(act.shape)} | target {tuple(target.shape)}")

    assert all(torch.equal(ctx[k], ctx[0]) for k in range(CONTEXT_FRAMES))               # check 2: padding at t=0
    assert torch.all(act[:-1] == 0) and torch.equal(act[-1], torch.from_numpy(train_a[0, 0]))
    rng = np.random.default_rng(0)                                                       # check 3: alignment
    for i in rng.choice(len(train), size=50, replace=False):
        ep, t = divmod(int(i), EPISODE_STEPS)
        c, a, y = train[int(i)]
        assert np.array_equal(y.permute(1, 2, 0).numpy(), train_f[ep, t + 1])            # target = frame t+1
        assert np.array_equal(c[-1].permute(1, 2, 0).numpy(), train_f[ep, t])            # newest context = frame t
        assert np.array_equal(a[-1].numpy(), train_a[ep, t])                             # newest action = action t
    changed = np.mean([(train_f[e, t + 1] != train_f[e, t]).any() for e in range(len(train_f)) for t in range(EPISODE_STEPS)])
    print(f"padding OK | alignment OK on 50 random examples | target differs from newest context in {changed:.0%} of steps")

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)                  # check 4: look at it
    save_grid(train, [0, 2, 3, 50, 99], out)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
