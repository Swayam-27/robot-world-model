"""Plot SAC's learning curve (episode reward over training) from SB3's monitor log.

Run:  python -m policies.plot_curve --monitor outputs/sac_pusher/monitor.csv
Saves learning_curve.png next to the monitor file.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # no screen on Kaggle: draw plots straight to files
import matplotlib.pyplot as plt
import pandas as pd


def plot_learning_curve(monitor_csv: Path, out_png: Path, window: int = 20) -> None:
    """Plot episode reward (raw + rolling mean) against training steps."""
    df = pd.read_csv(monitor_csv, skiprows=1)  # first line is a JSON comment
    steps = df["l"].cumsum()                     # l = episode length -> total steps so far
    plt.figure(figsize=(7, 4))
    plt.plot(steps, df["r"], alpha=0.25, label="episode reward")
    plt.plot(steps, df["r"].rolling(window).mean(), label=f"mean of last {window}")
    plt.xlabel("training steps"); plt.ylabel("episode reward (higher = better)")
    plt.title("SAC on Pusher-v5"); plt.legend(); plt.tight_layout()
    plt.savefig(out_png, dpi=120); plt.close()
    print(f"{len(df)} episodes, last-{window} mean reward {df['r'].tail(window).mean():.1f} | saved {out_png}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--monitor", required=True)
    monitor = Path(parser.parse_args().monitor)
    plot_learning_curve(monitor, monitor.parent / "learning_curve.png")
