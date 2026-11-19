"""Step 4b: train the state reader on the recorded episodes, keep the best checkpoint, report errors per group.

Run:  python -m models.train_reader --config configs/reader.yaml [--data-dir DIR] [--limit N]
One example = the last n_frames frames up to t stacked -> state t, for t = 0..100 of every episode (before the episode
has enough frames, frame 0 repeats, exactly what the reader will see right after a reset). Train/val split comes from episodes.csv (split by episode, never by frame).
Sanity checks: parameter count, loss per epoch, per-group validation errors in real units, predicted-vs-true plot.
"""
import argparse
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import wandb  # noqa: E402
import yaml  # noqa: E402
from torch import nn  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402

from envs.constants import (FINGERTIP_XYZ, GOAL_XYZ, JOINT_ANGLES, JOINT_VELOCITIES, OBJECT_XYZ,  # noqa: E402
                            OBS_DIM, SUCCESS_DISTANCE)
from models.reader import Reader, stack_frames  # noqa: E402

GROUPS = {"joint_angles": JOINT_ANGLES, "joint_velocities": JOINT_VELOCITIES,
          "fingertip": FINGERTIP_XYZ, "object": OBJECT_XYZ, "goal": GOAL_XYZ}


def load_split(data_dir: Path, split: str, limit: int | None) -> tuple[np.ndarray, np.ndarray]:
    """All episodes of one split -> frames (N, 101, 64, 64, 3) uint8 and states (N, 101, 23) float32."""
    index = pd.read_csv(data_dir / "episodes.csv")
    files = index.loc[index["split"] == split, "file"].tolist()[:limit]
    frames, states = [], []
    for f in files:
        with np.load(data_dir / f) as ep:
            frames.append(ep["frames"]); states.append(ep["states"].astype(np.float32))
    return np.stack(frames), np.stack(states)


class PairDataset(Dataset):
    """Item i = episode i // 101, step t = i % 101  ->  (3 * n_frames, 64, 64) uint8 input, normalised (23,) target."""

    def __init__(self, frames: np.ndarray, states: np.ndarray, mean: np.ndarray, std: np.ndarray, n_frames: int) -> None:
        self.frames, self.targets, self.n_frames = frames, (states - mean) / std, n_frames
        self.steps = frames.shape[1]

    def __len__(self) -> int:
        return len(self.frames) * self.steps

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        ep, t = divmod(i, self.steps)
        x = stack_frames([self.frames[ep, max(t - k, 0)] for k in range(self.n_frames - 1, -1, -1)])  # oldest first
        return torch.from_numpy(x), torch.from_numpy(self.targets[ep, t])


def train_one_epoch(model: nn.Module, loader: DataLoader, loss_fn, optimizer, device: str) -> float:
    """One pass over the training pairs; returns the mean loss."""
    model.train()
    total, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        optimizer.step()
        total += loss.item() * len(y); n += len(y)
    return total / n


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: str) -> tuple[np.ndarray, np.ndarray]:
    """Normalised predictions and targets for every pair in the loader (in order)."""
    model.eval()
    preds, targets = [], []
    for x, y in loader:
        preds.append(model(x.to(device)).cpu().numpy()); targets.append(y.numpy())
    return np.concatenate(preds), np.concatenate(targets)


def group_errors(pred: np.ndarray, true: np.ndarray) -> pd.DataFrame:
    """Errors in real units: angles in degrees, velocities in rad/s, positions as 3D distance in cm."""
    rows = []
    for name, sl in GROUPS.items():
        diff = pred[:, sl] - true[:, sl]
        if name == "joint_angles":
            err, unit = np.degrees(np.abs(diff)).ravel(), "deg"
        elif name == "joint_velocities":
            err, unit = np.abs(diff).ravel(), "rad/s"
        else:
            err, unit = np.linalg.norm(diff, axis=1) * 100, "cm"
        rows.append({"group": name, "unit": unit, "mean": err.mean(), "p95": np.percentile(err, 95), "max": err.max()})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-dir", help="override data_dir, e.g. the dataset under /kaggle/input")
    parser.add_argument("--limit", type=int, help="episodes per split, for a quick test")
    parser.add_argument("--epochs", type=int, help="override epochs, for a quick test")
    parser.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], help="override wandb.mode")
    args = parser.parse_args()
    cfg = yaml.safe_load(open(args.config))
    epochs = args.epochs or cfg["epochs"]
    data_dir, out = Path(args.data_dir or cfg["data_dir"]), Path(cfg["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_f, train_s = load_split(data_dir, "train", args.limit)
    val_f, val_s = load_split(data_dir, "val", args.limit)
    mean = train_s.reshape(-1, OBS_DIM).mean(0)                                        # TRAINING data only
    std = np.maximum(train_s.reshape(-1, OBS_DIM).std(0), cfg["std_floor"])
    print(f"train {len(train_f)} episodes | val {len(val_f)} episodes | device {device}")

    workers = 0 if os.name == "nt" else cfg["num_workers"]                             # Windows: worker processes copy the data
    kw = dict(batch_size=cfg["batch_size"], num_workers=workers, pin_memory=device == "cuda")
    train_loader = DataLoader(PairDataset(train_f, train_s, mean, std, cfg["n_frames"]), shuffle=True, **kw)
    val_loader = DataLoader(PairDataset(val_f, val_s, mean, std, cfg["n_frames"]), **kw)
    print(f"{len(train_loader.dataset):,} training pairs | {len(val_loader.dataset):,} validation pairs")

    model = Reader(OBS_DIM, cfg["n_frames"]).to(device)
    x0, _ = next(iter(val_loader))
    assert model(x0.to(device)).shape == (len(x0), OBS_DIM)                            # sanity check 1
    n_params = sum(p.numel() for p in model.parameters())
    print(f"shape test OK | input {tuple(x0.shape)} {x0.dtype} | {n_params:,} parameters")

    mode = args.wandb_mode or cfg["wandb"]["mode"]
    if mode == "online" and not os.environ.get("WANDB_API_KEY"):
        print("no WANDB_API_KEY: logging to W&B offline (files in ./wandb, upload later with `wandb sync`)")
        mode = "offline"
    run = wandb.init(project=cfg["wandb"]["project"], job_type="reader", config={**cfg, "n_params": n_params}, mode=mode)

    loss_fn = nn.MSELoss()                                    # on NORMALISED states, so every number counts equally
    # AdamW = Adam + weight decay: every step shrinks all weights a little toward 0, so the network can't grow the
    # large, finely tuned weights it needs to memorise training photos. weight_decay 0 (v1, v2) is exactly plain Adam.
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg.get("weight_decay", 0.0))
    best_val, bad_epochs, log = float("inf"), 0, []
    ckpt_path = out / "reader_best.pt"
    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        pred, true = predict(model, val_loader, device)
        val_loss = float(((pred - true) ** 2).mean())
        errs = group_errors(pred * std + mean, true * std + mean).set_index("group")["mean"]
        log.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, **errs.to_dict()})
        run.log({"train/loss": train_loss, "val/loss": val_loss, **{f"val/{k}_mean": v for k, v in errs.items()}},
                step=epoch)
        print(f"epoch {epoch}: train {train_loss:.4f} | val {val_loss:.4f} | object {errs['object']:.2f} cm"
              f" | fingertip {errs['fingertip']:.2f} cm | velocities {errs['joint_velocities']:.3f} rad/s")
        if val_loss < best_val:
            best_val, bad_epochs = val_loss, 0
            torch.save({"model": model.state_dict(), "mean": mean, "std": std, "n_out": OBS_DIM, "n_frames": cfg["n_frames"],
                        "epoch": epoch, "val_loss": val_loss, "config": cfg}, ckpt_path)
        else:
            bad_epochs += 1
            if bad_epochs >= cfg["patience"]:
                print(f"no improvement for {cfg['patience']} epochs: stopping")
                break
    pd.DataFrame(log).to_csv(out / "train_log.csv", index=False)

    # Final report from the BEST checkpoint, reloaded from disk (also proves the file is complete).
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    pred, true = predict(model, val_loader, device)
    pred, true = pred * std + mean, true * std + mean
    table = group_errors(pred, true)
    table.to_csv(out / "val_errors.csv", index=False)
    print(f"\nbest epoch {ckpt['epoch']} (val loss {ckpt['val_loss']:.4f}); validation errors in real units:")
    print(table.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    obj = table.set_index("group").loc["object"]
    print(f"object error: mean {obj['mean']:.2f} cm, p95 {obj['p95']:.2f} cm vs success threshold "
          f"{SUCCESS_DISTANCE * 100:.0f} cm")

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))                                   # sanity check 2
    pick = np.random.default_rng(0).choice(len(true), size=min(3000, len(true)), replace=False)
    for ax, (label, col) in zip(axes, [("object x", OBJECT_XYZ.start), ("object y", OBJECT_XYZ.start + 1),
                                       ("joint 0 velocity", JOINT_VELOCITIES.start)]):
        ax.scatter(true[pick, col], pred[pick, col], s=2, alpha=0.3)
        lo, hi = true[pick, col].min(), true[pick, col].max()
        ax.plot([lo, hi], [lo, hi], "r--", lw=1); ax.set(title=label, xlabel="true", ylabel="reader")
    fig.suptitle("Reader on validation episodes: dots on the red line = perfect")
    fig.tight_layout(); fig.savefig(out / "reader_pred_vs_true.png", dpi=120)
    run.log({"val/pred_vs_true": wandb.Image(str(out / "reader_pred_vs_true.png")),
             "val/errors": wandb.Table(dataframe=table)})
    run.summary.update({f"final/{r.group}_mean": r.mean for r in table.itertuples()})
    run.finish()
    print(f"saved {ckpt_path}, {out / 'val_errors.csv'}, {out / 'reader_pred_vs_true.png'}")


if __name__ == "__main__":
    main()
