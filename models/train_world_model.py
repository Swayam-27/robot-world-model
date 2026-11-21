"""Step 5: train the world model (models/world_model.py) on stored episodes. Runs on a Kaggle GPU.

Run:  python -m models.train_world_model --config configs/world_model_v1.yaml [--data-dir DIR] [--extra-data-dir DIR]
      [--resume last.pt] [--limit N --max-steps N --batch-size N --wandb-mode disabled]   (quick laptop test)
Stops at training.max_steps or training.max_hours, whichever comes first (Kaggle sessions end at 12 h).
Every eval_every steps, on FIXED validation examples (same examples and same noise each time, so numbers compare):
  val loss, and real one-step predictions drawn with sample(): pixel error vs the true next frame, on the whole
  image and on the MOVING pixels only (arm and can), each next to the "copy the newest context frame" baseline,
  plus a grid (context | real | predicted | error).
Checkpoints: last.pt every checkpoint_every_minutes and at the end (resume with --resume), best.pt = lowest
error on the moving pixels. All written fully, then renamed, so a killed session never leaves a broken file.
"""
import argparse
import os
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import wandb  # noqa: E402
import yaml  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from models.world_model import WorldModel  # noqa: E402
from models.world_model_data import NextFrameDataset, load_split  # noqa: E402


def forever(loader: DataLoader):
    """Endless stream of batches: a new shuffle each pass through the data."""
    while True:
        yield from loader


def fixed_val_batch(ds: NextFrameDataset, n: int, seed: int = 0) -> list[torch.Tensor]:
    """n validation examples picked once, spread over episodes and steps (incl. mid-episode, where contact happens)."""
    idx = np.random.default_rng(seed).choice(len(ds), size=min(n, len(ds)), replace=False)
    return [torch.stack(x) for x in zip(*(ds[int(i)] for i in sorted(idx)))]


@torch.no_grad()
def evaluate(model: WorldModel, val: list[torch.Tensor], steps: int, device: str, batch: int) -> dict:
    """Val loss with fixed noise, and one-step pixel error of real samples vs the copy-last-frame baseline."""
    model.eval()
    with torch.random.fork_rng(devices=[torch.cuda.current_device()] if device == "cuda" else []):
        torch.manual_seed(1234)                                # same sigma/noise draws every evaluation
        losses, preds = [], []
        for i in range(0, len(val[0]), batch):
            ctx, act, tgt = (x[i:i + batch].to(device) for x in val)
            losses.append(model.loss(ctx, act, tgt).item() * len(tgt))
            preds.append(model.sample(ctx, act, steps=steps).cpu())
    model.train()
    ctx, _, tgt = val
    pred, last = torch.cat(preds).float(), ctx[:, -1].float()
    err, copy = (pred - tgt.float()).abs(), (last - tgt.float()).abs()       # 0-255 scale
    # Almost every pixel is still background, so the whole-image error flatters everyone. The moving pixels (the
    # arm and the can, where frame t+1 differs from frame t by > 10 levels) are where the model must be right.
    moving = (copy.amax(dim=1, keepdim=True) > 10).expand_as(err)
    return {"val/loss": sum(losses) / len(tgt), "val/pixel_err": err.mean().item(),
            "val/copy_last_err": copy.mean().item(), "val/moving_err": err[moving].mean().item(),
            "val/copy_last_moving_err": copy[moving].mean().item(), "val/moving_share": moving.float().mean().item(),
            "pred": pred.to(torch.uint8)}


def save_grid(val: list[torch.Tensor], pred: torch.Tensor, path: Path, n: int = 8) -> None:
    """Rows: newest context frame | real next frame | predicted next frame | error x4 (black = perfect)."""
    ctx, _, tgt = val
    rows = [("newest context", ctx[:n, -1]), ("real next", tgt[:n]), ("predicted", pred[:n]),
            ("|error| x4", ((pred[:n].float() - tgt[:n].float()).abs() * 4).clamp(0, 255).to(torch.uint8))]
    fig, axes = plt.subplots(len(rows), n, figsize=(1.6 * n, 1.7 * len(rows)))
    for r, (name, imgs) in enumerate(rows):
        for c in range(n):
            ax = axes[r, c]; ax.imshow(imgs[c].permute(1, 2, 0).numpy()); ax.set_xticks([]); ax.set_yticks([])
        axes[r, 0].set_ylabel(name, fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=90); plt.close(fig)


def save_ckpt(path: Path, **state) -> None:
    """Write to a temp file, then rename: never a half-written checkpoint."""
    tmp = path.with_suffix(".tmp"); torch.save(state, tmp); tmp.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-dir", help="override data.data_dir, e.g. the dataset under /kaggle/input")
    parser.add_argument("--extra-data-dir", action="append", help="override data.extra_data_dirs (repeatable)")
    parser.add_argument("--resume", help="last.pt from an earlier session: continue its step count and optimizer")
    parser.add_argument("--limit", type=int, help="episodes per split per folder (quick test)")
    parser.add_argument("--max-steps", type=int, help="override training.max_steps (quick test)")
    parser.add_argument("--batch-size", type=int, help="override training.batch_size (quick test)")
    parser.add_argument("--eval-every", type=int, help="override training.eval_every (quick test)")
    parser.add_argument("--wandb-mode", choices=["online", "offline", "disabled"])
    args = parser.parse_args()
    cfg = yaml.safe_load(open(args.config))
    tr = cfg["training"]
    max_steps, batch = args.max_steps or tr["max_steps"], args.batch_size or tr["batch_size"]
    eval_every = args.eval_every or tr["eval_every"]
    out = Path(tr["out_dir"]); out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp = tr["mixed_precision"] and device == "cuda"           # fp16 autocast: the T4 has no bf16

    dirs = [Path(args.data_dir or cfg["data"]["data_dir"]),
            *map(Path, args.extra_data_dir or cfg["data"].get("extra_data_dirs", []))]
    train_ds = NextFrameDataset(*load_split(dirs, "train", args.limit))
    val_ds = NextFrameDataset(*load_split(dirs, "val", args.limit))
    print(f"data: {', '.join(map(str, dirs))}\ntrain {len(train_ds):,} examples | val {len(val_ds):,} | device {device}")
    workers = 0 if os.name == "nt" else tr["num_workers"]
    loader = DataLoader(train_ds, batch_size=batch, shuffle=True, drop_last=True, num_workers=workers,
                        pin_memory=device == "cuda", persistent_workers=workers > 0)
    val = fixed_val_batch(val_ds, tr["val_examples"])

    model = WorldModel(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=tr["lr"], weight_decay=tr["weight_decay"], eps=1e-8)
    warmup = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / tr["lr_warmup_steps"]))
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    step, best, log = 0, float("inf"), []
    if args.resume:
        ck = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); warmup.load_state_dict(ck["sched"])
        scaler.load_state_dict(ck["scaler"]); step, best, log = ck["step"], ck["best"], ck["log"]
        print(f"resumed from {args.resume} at step {step:,} (best moving-pixel error {best:.2f})")
    print(f"{n_params:,} parameters | batch {batch} | mixed precision {amp} | until step {max_steps:,} "
          f"or {tr['max_hours']} h")

    mode = args.wandb_mode or cfg["logging"]["wandb_mode"]
    if mode == "online" and not os.environ.get("WANDB_API_KEY"):
        print("no WANDB_API_KEY: logging to W&B offline"); mode = "offline"
    run = wandb.init(project=cfg["logging"]["project"], job_type="world_model", mode=mode,
                     config={**cfg, "n_params": n_params, "resumed_at": step})

    def checkpoint(name: str) -> None:
        save_ckpt(out / name, model=model.state_dict(), opt=opt.state_dict(), sched=warmup.state_dict(),
                  scaler=scaler.state_dict(), step=step, best=best, log=log, config=cfg)

    t_start = t_ckpt = t_log = time.time()
    model.train(); running = []
    for ctx, act, tgt in forever(loader):
        if step >= max_steps or time.time() - t_start > tr["max_hours"] * 3600:
            break
        ctx, act, tgt = (x.to(device, non_blocking=True) for x in (ctx, act, tgt))
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            loss = model.loss(ctx, act, tgt)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)                                   # clip the real gradients, not the scaled ones
        torch.nn.utils.clip_grad_norm_(model.parameters(), tr["max_grad_norm"])
        scaler.step(opt); scaler.update(); warmup.step()
        step += 1; running.append(loss.item())

        if step % tr["log_every"] == 0:
            speed = tr["log_every"] / (time.time() - t_log); t_log = time.time()
            run.log({"train/loss": np.mean(running), "train/steps_per_s": speed, "lr": opt.param_groups[0]["lr"]},
                    step=step)
            print(f"step {step:,}: loss {np.mean(running):.4f} | {speed:.1f} steps/s", flush=True); running = []
        if step % eval_every == 0 or step == max_steps:
            ev = evaluate(model, val, cfg["sampling"]["denoising_steps"], device, batch)
            pred = ev.pop("pred")
            log.append({"step": step, **ev})
            grid = out / f"samples_{step:07d}.png"; save_grid(val, pred, grid)
            run.log({**ev, "val/samples": wandb.Image(str(grid))}, step=step)
            print(f"  eval: val loss {ev['val/loss']:.4f} | pixel error (of 255): all {ev['val/pixel_err']:.2f} "
                  f"(copy-last {ev['val/copy_last_err']:.2f}) | moving {ev['val/moving_err']:.1f} "
                  f"(copy-last {ev['val/copy_last_moving_err']:.1f}, {ev['val/moving_share']:.1%} of pixels)", flush=True)
            if ev["val/moving_err"] < best:                    # best = sharpest on the arm and the can
                best = ev["val/moving_err"]; checkpoint("best.pt")
        if time.time() - t_ckpt > tr["checkpoint_every_minutes"] * 60:
            checkpoint("last.pt"); t_ckpt = time.time(); print(f"  saved last.pt at step {step:,}", flush=True)

    checkpoint("last.pt")
    pd.DataFrame(log).to_csv(out / "eval_log.csv", index=False)
    run.summary.update({"final_step": step, "best_pixel_err": best}); run.finish()
    print(f"stopped at step {step:,} after {(time.time() - t_start) / 3600:.2f} h | best moving-pixel error "
          f"{best:.2f} | saved {out / 'last.pt'}, {out / 'best.pt'}, {out / 'eval_log.csv'}")


if __name__ == "__main__":
    main()
