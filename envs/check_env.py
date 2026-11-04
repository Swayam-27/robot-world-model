"""Step 1 sanity check: shapes, seed repeatability, and a GIF of one random episode.

Run from the project folder:  python -m envs.check_env
"""
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from envs.constants import EPISODE_STEPS, TRAIN_SEEDS
from envs.pusher import PusherEnv, object_goal_distance

OUT_DIR = Path("outputs")
OUT_DIR.mkdir(exist_ok=True)

env = PusherEnv()
seed = TRAIN_SEEDS[0]  # a training seed; eval seeds are never used for checks like this

# 1) Same seed -> same first frame and state
obs_a = env.reset(seed)
frame_a = env.frame()
obs_b = env.reset(seed)
frame_b = env.frame()
print("same seed, same state:", np.array_equal(obs_a, obs_b))
print("same seed, same frame:", np.array_equal(frame_a, frame_b))

# 2) Different seed -> different start (can/goal move)
env.reset(TRAIN_SEEDS[1])
print("different seed, different frame:", not np.array_equal(frame_a, env.frame()))

# 3) One random episode -> GIF
obs = env.reset(seed)
frames = [env.frame()]
steps, done = 0, False
while not done:
    obs, reward, done = env.step(env.sample_action())
    frames.append(env.frame())
    steps += 1

print("state shape:", obs.shape, "| frame shape:", frames[0].shape, frames[0].dtype)
print(f"steps: {steps} (expected {EPISODE_STEPS}) | frames: {len(frames)} (expected {EPISODE_STEPS + 1})")
print(f"object-goal distance: start {object_goal_distance(obs_a):.3f} m, end {object_goal_distance(obs):.3f} m")

# Save the GIF small (64x64) and an enlarged copy (256x256) that is easier to look at
imageio.mimsave(OUT_DIR / "random_episode_64.gif", frames, duration=0.05, loop=0)
big = [np.kron(f, np.ones((4, 4, 1), dtype=np.uint8)) for f in frames]
imageio.mimsave(OUT_DIR / "random_episode_256.gif", big, duration=0.05, loop=0)
print("saved:", OUT_DIR / "random_episode_64.gif", "and", OUT_DIR / "random_episode_256.gif")
env.close()
