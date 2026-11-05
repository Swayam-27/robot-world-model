"""Step 2a: train SAC on Pusher's 23-number state, saving checkpoints along the way.

Run (Kaggle CPU):  python -m policies.train_sac --config configs/sac_pusher.yaml
Outputs in out_dir: ckpts/sac_<steps>_steps.zip, sac_final.zip, monitor.csv, learning_curve.png
"""
import argparse
from pathlib import Path

import yaml
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor

from envs.pusher import make_train_env
from policies.plot_curve import plot_learning_curve


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    cfg = yaml.safe_load(open(parser.parse_args().config))

    out = Path(cfg["out_dir"])
    (out / "ckpts").mkdir(parents=True, exist_ok=True)

    # Monitor writes one row per episode (reward r, length l, time t) to monitor.csv
    env = Monitor(make_train_env(rng_seed=cfg["seed"]), filename=str(out / "monitor.csv"))  # SB3 adds ".monitor.csv" unless the name already ends with it
    model = SAC("MlpPolicy", env, seed=cfg["seed"], verbose=1, **cfg["sac"])

    checkpoints = CheckpointCallback(save_freq=cfg["checkpoint_every"],
                                     save_path=str(out / "ckpts"), name_prefix="sac")
    model.learn(total_timesteps=cfg["total_timesteps"], callback=checkpoints, log_interval=20)
    model.save(out / "sac_final")

    plot_learning_curve(out / "monitor.csv", out / "learning_curve.png")
    n_ckpts = len(list((out / "ckpts").glob("*.zip")))
    print(f"done: {n_ckpts} checkpoints in {out / 'ckpts'}, curve at {out / 'learning_curve.png'}")


if __name__ == "__main__":
    main()
