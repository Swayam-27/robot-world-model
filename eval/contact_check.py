"""Step 5 done-check: are the world model's one-step predictions right when the arm pushes the can?

Run:  python -m eval.contact_check --config configs/contact_check_v1.yaml
Uses the TRUE states stored with every validation episode to find contact steps (the can moves more than contact_mm
between frame t and t+1, which only happens while it is being pushed) and still steps (the can does not move).
For n_examples of each, the model draws frame t+1 (3 sampling steps, fixed seed) and we measure the error on the
moving pixels, next to "copy the last frame". If contact is much worse than still, the model has not learned
pushing yet (BUILD_GUIDE Step 5 'Watch out'). Also saves a grid of the contact steps where the can moved most.
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

from envs.constants import OBJECT_XYZ  # noqa: E402
from models.world_model import WorldModel  # noqa: E402
from models.world_model_data import NextFrameDataset  # noqa: E402


def load_val(dirs: list[Path]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Validation episodes of all folders -> frames (N, 101, 64, 64, 3), actions (N, 100, 7), states (N, 101, 23)."""
    frames, actions, states = [], [], []
    for d in dirs:
        index = pd.read_csv(d / "episodes.csv")
        for f in index.loc[index["split"] == "val", "file"]:
            with np.load(d / f) as ep:
                frames.append(ep["frames"]); actions.append(ep["actions"].astype(np.float32))
                states.append(ep["states"].astype(np.float32))
    return np.stack(frames), np.stack(actions), np.stack(states)


@torch.no_grad()
def predict(model: WorldModel, ds: NextFrameDataset, items: np.ndarray, batch: int) -> tuple[torch.Tensor, ...]:
    """Context, target and the model's drawn next frame for the given dataset items (all uint8)."""
    ctx, act, tgt = (torch.stack(x) for x in zip(*(ds[int(i)] for i in items)))
    pred = torch.cat([model.sample(ctx[i:i + batch], act[i:i + batch], steps=3) for i in range(0, len(items), batch)])
    return ctx[:, -1], tgt, pred


def moving_errors(last: torch.Tensor, tgt: torch.Tensor, pred: torch.Tensor) -> dict:
    """Error on the pixels that change between frame t and t+1 (> 10 levels), for the model and copy-last-frame."""
    copy, err = (last.float() - tgt.float()).abs(), (pred.float() - tgt.float()).abs()
    moving = (copy.amax(dim=1, keepdim=True) > 10).expand_as(err)
    return {"model_moving_err": err[moving].mean().item(), "copy_last_moving_err": copy[moving].mean().item(),
            "model_all_err": err.mean().item(), "moving_share": moving.float().mean().item()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    cfg = yaml.safe_load(open(parser.parse_args().config))
    torch.manual_seed(cfg["seed"])

    ck = torch.load(cfg["checkpoint"], map_location="cpu", weights_only=False)
    model = WorldModel(ck["config"]); model.load_state_dict(ck["model"]); model.eval()
    print(f"world model from {cfg['checkpoint']} (step {ck['step']:,})")

    frames, actions, states = load_val([Path(d) for d in cfg["data_dirs"]])
    ds = NextFrameDataset(frames, actions)
    can = states[:, :, OBJECT_XYZ]
    moved_mm = np.linalg.norm(can[:, 1:] - can[:, :-1], axis=-1).ravel() * 1000      # item i = ep * 100 + t
    contact, still = np.flatnonzero(moved_mm > cfg["contact_mm"]), np.flatnonzero(moved_mm < cfg["still_mm"])
    print(f"{len(frames)} validation episodes, {len(ds):,} steps: contact {len(contact):,} ({len(contact) / len(ds):.1%}), "
          f"still {len(still):,} ({len(still) / len(ds):.1%})")

    rng = np.random.default_rng(cfg["seed"])
    rows, results = [], {}
    for name, pool in [("contact", contact), ("still", still)]:
        items = np.sort(rng.choice(pool, size=min(cfg["n_examples"], len(pool)), replace=False))
        results[name] = (items, *predict(model, ds, items, cfg["batch"]))
        m = moving_errors(*results[name][1:])
        rows.append({"group": name, "n": len(items), "can_moved_mm_mean": moved_mm[items].mean(), **m})
        print(f"{name:8s}: moving-pixel error {m['model_moving_err']:.1f} of 255 (copy-last {m['copy_last_moving_err']:.1f}) "
              f"| whole image {m['model_all_err']:.2f} | can moved {moved_mm[items].mean():.2f} mm on average")
    table = pd.DataFrame(rows)
    Path(cfg["out_csv"]).parent.mkdir(parents=True, exist_ok=True); table.to_csv(cfg["out_csv"], index=False)
    ratio = table.set_index("group")["model_moving_err"]
    print(f"contact / still moving-pixel error: {ratio['contact'] / ratio['still']:.2f}x")

    items, last, tgt, pred = results["contact"]                                     # grid: biggest pushes
    top = np.argsort(-moved_mm[items])[:8]
    grid = [("newest context", last), ("real next", tgt), ("predicted", pred),
            ("|error| x4", ((pred.float() - tgt.float()).abs() * 4).clamp(0, 255).to(torch.uint8))]
    fig, axes = plt.subplots(4, len(top), figsize=(1.6 * len(top), 7))
    for r, (label, imgs) in enumerate(grid):
        for c, k in enumerate(top):
            ax = axes[r, c]; ax.imshow(imgs[k].permute(1, 2, 0).numpy()); ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(f"can {moved_mm[items[k]]:.1f} mm", fontsize=8)
        axes[r, 0].set_ylabel(label, fontsize=8)
    fig.suptitle("Contact steps where the can moved most: one-step predictions")
    fig.tight_layout(); fig.savefig(cfg["out_png"], dpi=90)
    print(f"saved {cfg['out_csv']}, {cfg['out_png']}")


if __name__ == "__main__":
    main()
