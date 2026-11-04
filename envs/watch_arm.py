"""Open a 3D window and watch the Pusher arm move randomly (laptop only, needs a screen).

The window starts from the project camera (CAMERA_CONFIG, what the world model sees),
but stays free: drag to rotate, scroll to zoom, right-drag to pan.

Run from the project folder:
    .venv\\Scripts\\python.exe -m envs.watch_arm
"""
import gymnasium as gym
import numpy as np

from envs.constants import CAMERA_CONFIG, ENV_ID

camera = {**CAMERA_CONFIG, "lookat": np.array(CAMERA_CONFIG["lookat"])}
env = gym.make(ENV_ID, render_mode="human", default_camera_config=camera)  # "human" = live window
obs, _ = env.reset(seed=0)                        # seed fixes where the can and goal start

for step in range(1000):                          # 1000 steps = 10 episodes of 100 (~50 s)
    action = env.action_space.sample()            # random torques: the "floor" policy
    obs, reward, terminated, truncated, _ = env.step(action)
    if terminated or truncated:                   # episode over after 100 steps
        obs, _ = env.reset()

env.close()
