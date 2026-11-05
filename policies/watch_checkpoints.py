"""Make GIFs + final distances for several checkpoints of one SAC run (Step 2 sanity check).

Run:  python -m policies.watch_checkpoints --run outputs/sac_pusher
Each checkpoint is played in its own process with OSMesa rendering: on Kaggle CPU, EGL rendering
crashes once PyTorch is loaded in the same process (segfault / "free(): invalid size", 2026-11-05).
Needs:  apt-get install -y libosmesa6
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

DEFAULT_CKPTS = ["ckpts/sac_20000_steps", "ckpts/sac_100000_steps", "ckpts/sac_200000_steps", "sac_final"]
NOISE = ("absl", "cudart", "Gym has been", "Please upgrade", "Users of this", "migration guide")


def watch(ckpt: Path) -> bool:
    """Run watch_policy on one checkpoint with OSMesa; print its result line(s). True if it worked."""
    env = {**os.environ, "MUJOCO_GL": "osmesa", "PYOPENGL_PLATFORM": "osmesa"}
    proc = subprocess.run([sys.executable, "-m", "policies.watch_policy", "--ckpt", str(ckpt)],
                          env=env, capture_output=True, text=True)
    lines = [l for l in (proc.stdout + proc.stderr).splitlines() if l.strip() and not l.startswith(NOISE)]
    print("\n".join(lines) if lines else f"{ckpt.name}: no output")
    if proc.returncode != 0:
        print(f"  -> FAILED with exit code {proc.returncode}")
    return proc.returncode == 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, help="folder of a training run, e.g. outputs/sac_pusher")
    parser.add_argument("--ckpts", nargs="*", default=DEFAULT_CKPTS, help="paths inside --run, without .zip")
    args = parser.parse_args()
    results = [watch(Path(args.run) / f"{c}.zip") for c in args.ckpts]
    print(f"{sum(results)}/{len(results)} checkpoints rendered")


if __name__ == "__main__":
    main()
