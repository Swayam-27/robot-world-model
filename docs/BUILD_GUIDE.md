# Step-by-Step Build Guide

Every step of the robot arm world model, in build order: the goal, exactly what to do, the idea behind it, how you know it's done, and what usually goes wrong. Track progress in `PROGRESS.md`.

- Every step has five parts: **Goal**, **What you do**, **The idea**, **Done when**, **Watch out**.
- Claude Code writes code, but don't move on until you can explain the current step's code line by line.
- Commit at the end of every step with a clear message.

---

## Step 0 · Get ready

*This week, about 4 hours.*

**Goal:** write a PyTorch training loop yourself, and have a clean repo.

**What you do**

1. **Learn PyTorch by porting something you know.** Rebuild the custom CNN from Real vs AI Faces in PyTorch on a small subset: tensors, `nn.Module`, `DataLoader`, optimizer, `loss.backward()`, `torch.save`.
2. **Create the repo** from this folder and push it to GitHub.
3. **Set up Kaggle.** Verify your phone number (internet in notebooks), turn on the GPU, clone the repo in the first cell (see `notebooks/README.md`).
4. **Set up experiment tracking.** Free Weights & Biases account, or TensorBoard.

**The idea:** notebooks run things; the repo holds the code. Every experiment is a config file plus a commit, so any README number can be reproduced.

**Done when:** your PyTorch CNN trains, saves and reloads a checkpoint, and logs a loss curve.

**Watch out:** don't skip this. Every later step assumes you can read and change a training loop.

---

## Step 1 · Make the real world run

*Nov 3 to 9, about 3 hours.*

**Goal:** a simulator that produces a 64x64 camera frame at every step, from start positions you can repeat exactly.

**What you do**

0. **Watch it first, on your laptop** (no GPU needed, small install). Run this, and a 3D window opens with the arm moving randomly; rotate the camera with the mouse:

```python
import gymnasium as gym
env = gym.make("Pusher-v5", render_mode="human")
obs, _ = env.reset(seed=0)
for _ in range(500):
    obs, reward, term, trunc, _ = env.step(env.action_space.sample())
    if term or trunc:
        obs, _ = env.reset()
env.close()
```

1. **Install:** `pip install "gymnasium[mujoco]" stable-baselines3`. Pusher-v5 needs MuJoCo 2.3.3 or newer.
2. **Headless rendering** before importing anything: `os.environ["MUJOCO_GL"] = "egl"`. If EGL fails, use `"osmesa"` (slower, CPU).
3. **Create the environment with small frames:**

```python
import gymnasium as gym
env = gym.make("Pusher-v5", render_mode="rgb_array", width=64, height=64)
obs, info = env.reset(seed=0)
frame = env.render()  # numpy array, 64 x 64 x 3
```

4. **Check the camera.** Save a GIF of a random episode. Fingertip, cylinder and goal must be visible in every frame. If not, adjust `default_camera_config` in `gym.make`.
5. **Learn the observation** (23 numbers; indices in `envs/constants.py`). Write `object_goal_distance(obs)`.
6. **Use the 50 evaluation seeds** in `envs/constants.py`. Every policy is always scored from these; they never appear in training data.

**The idea:** MuJoCo is "reality". Everything the world model does is judged against it, so it must be repeatable: same seed, same start.

**Done when:** the GIF clearly shows all three objects, and resetting with the same seed gives the same first frame.

**Watch out:** rendering errors on Kaggle are the most common blocker. Solve once and keep the fix as a function in `envs/`. Wrap MuJoCo behind a small interface (`reset(seed)`, `step(action)`, `frame()`, `state()`) so a real arm can replace it in phase 2.

---

## Step 2 · Train the policies you'll rank

*Nov 3 to 9, about 4 hours, mostly waiting.*

**Goal:** about 10 policies from clearly bad to good, each scored in the real simulator.

**What you do**

1. **Train SAC on the 23-number observation, on CPU,** saving checkpoints:

```python
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback
env = gym.make("Pusher-v5")
model = SAC("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=300_000,
            callback=CheckpointCallback(save_freq=20_000, save_path="ckpts/"))
```

Run it in a Kaggle CPU session to save GPU quota. 300,000 steps is a starting point; train longer if the last checkpoint can't push the object onto the goal.

To watch a trained brain, replace the random action with `model.predict(obs, deterministic=True)[0]` in the laptop viewer from Step 1. A pretrained Pusher policy on Hugging Face (see `RESOURCES.md`) shows what "good" looks like.

2. **Build the pool (~10):** about 6 checkpoints across training, 2 or 3 noisy copies of the best (Gaussian action noise at a few levels), and a random policy.
3. **Score every policy** on the 50 eval seeds, 100 steps each: mean final object-to-goal distance and success rate. Choose the success threshold by looking at frames where the object clearly sits on the goal.
4. **Save** `real_scores.csv`. This is **ranking A**, the ground truth.

**The idea:** you're writing an exam for the world model. If all ten policies scored about the same, any ranking would be noise. A wide spread matters more than one excellent policy.

**Done when:** scores are clearly spread out, and scoring twice gives the same ranking.

**Watch out:** score with `deterministic=True`; give noisy copies a fixed noise seed per start position. If two policies are nearly tied, replace one.

---

## Step 3 · Record the dataset

*Nov 10 to 16, about 3 hours.*

**Goal:** training data for the world model and the state reader: frames, actions and true state from many behaviours.

**What you do**

1. **Record episodes** from every policy plus pure random actions, on training seeds only (`TRAIN_SEEDS`):

```python
obs, _ = env.reset(seed=seed)
frames, states, actions = [env.render()], [obs], []
for t in range(100):
    a = act(policy, obs)  # SB3 predict, noisy copy or random
    obs, r, term, trunc, info = env.step(a)
    frames.append(env.render()); states.append(obs); actions.append(a)
np.savez_compressed(f"ep_{seed}.npz", frames=np.array(frames, dtype=np.uint8),
                    states=np.array(states), actions=np.array(actions))
```

2. **About 2,000 episodes** (200,000 frames): roughly a third random or noisy, two thirds from policies of every skill level. ~12 KB per frame, ~2.5 GB before compression.
3. **Split by episode,** 90% train / 10% validation.
4. **Upload as a private Kaggle dataset.**

**The idea:** a world model only simulates what it has seen. If it only watched the best policy, it would never learn what happens when the arm misses, and it would flatter bad policies.

**Done when:** a script loads a random episode and plays it back as a GIF, with each action and state aligned to its frame.

**Watch out:** frame *t* + action *t* gives frame *t+1*: 101 frames, 100 actions per episode.

**Worth doing:** leave 2 of the 10 policies out of the data entirely, to test in Step 8 whether the dream ranks unseen policies correctly.

---

## Step 4 · Build the state reader

*Nov 10 to 16, about 3 hours.*

**Goal:** a small network that reads the 23-number observation from camera frames.

**What you do**

1. **Input/output.** Last 2 frames stacked (6 channels), because one image can't show velocity. Output the 23 numbers, normalised with training-state mean and std.
2. **Model.** Small CNN: a few conv blocks, global pooling, dense head with 23 outputs. MSE loss on normalised values.
3. **Train and check.** ~30 minutes on GPU. Measure validation error per group: joint angles, velocities, fingertip, object, goal.
4. **Crucial test.** Run each policy in the real simulator but feed it the reader's estimates; score on the 50 eval seeds. This is **ranking B**.

**The idea:** in the dream there is no true state, only pictures, so the reader is the policy's eyes. Ranking B shows what the eyes alone cost, before the dream is involved.

**Done when:** object and goal errors are small compared with the success threshold, and reader-driven policies still mostly succeed in the real simulator.

**Watch out:** velocities are hardest. If policies fall apart, try 3 stacked frames or 96x96 input. Never tune on eval seeds.

---

## Step 5 · Train the world model

*First version Nov 10 to 16, refined Dec 1 to 7. The heart of the project.*

**Goal:** a model that takes the last few frames plus actions and draws the next frame.

**The idea first.** A diffusion model learns to remove noise. In training, take a real next frame, bury it in noise of some strength, and ask the network to recover the clean frame, at every noise strength. To generate, start from pure noise and clean it up in a few steps. A world model is the same with extra inputs: past frames and the action, so the only clean frame consistent with them is the correct next frame.

**What you do**

1. **Start from DIAMOND's code** (github.com/eloialonso/diamond). Small U-Net denoiser trained with EDM, which keeps sampling stable with few steps; the Atari version has 4.4M parameters and fits a free T4. Run its Atari demo once, then read the world-model code until you understand every input.
2. **Adapt the inputs.**
   - **Context:** last 4 frames stacked as channels next to the noisy frame (15 input channels).
   - **Actions:** Pusher has 7 continuous torques, Atari has discrete buttons. Replace the action lookup with a small MLP turning the last 4 actions (28 numbers) into a vector, added to the noise-level embedding that conditions every block.
3. **Training loop** (simplified):

```python
x0  = next_frame                 # (B, 3, 64, 64), scaled to [-1, 1]
ctx = past_frames                # (B, 12, 64, 64): the last 4 frames
aug = torch.rand(B) * 0.7        # noise augmentation strength
ctx = ctx + aug.view(-1, 1, 1, 1) * torch.randn_like(ctx)
sigma = sample_noise_level(B)    # EDM: log-normal noise levels
x_noisy = x0 + sigma.view(-1, 1, 1, 1) * torch.randn_like(x0)
pred = model(x_noisy, sigma, ctx, past_actions, aug)
loss = edm_weighted_mse(pred, x0, sigma)
```

4. **Starting settings:** batch 64, AdamW, lr 1e-4, mixed precision, checkpoint every 30 minutes, log a predicted-vs-real grid every few thousand steps.
5. **Sample with 3 to 5 denoising steps**; compare quality vs speed.

**Why noise augmentation:** in training past frames are perfect; when playing they are the model's own slightly wrong outputs. Noising the context in training (and telling the model how much) teaches it not to trust context blindly. GameNGen found this essential and capped it at 0.7.

**Done when:** one-step predictions on validation episodes are hard to tell from real frames by eye, including contact frames.

**Watch out:** the background is learned almost immediately. The real test is the object moving at contact. If loss falls but contact frames look wrong, you need more contact data (Step 9).

---

## Step 6 · Make the dream last 100 steps

*Dec 1 to 7, about 8 hours.*

**Goal:** generate a full 100-step episode from the model's own outputs and measure degradation.

**What you do**

1. **Replay a recorded episode in the dream:**

```python
ctx = list(real_frames[:4])
for t in range(4, 101):
    nxt = sample(world_model, ctx[-4:], actions[t-4:t], steps=4)
    ctx.append(nxt)
```

2. **Measure drift.** At each step compare dreamed vs real frame with PSNR (higher better) and LPIPS (lower better; `pip install lpips`). Average over 100 validation episodes; plot against step.
3. **Prove noise augmentation matters.** Repeat with a model trained without it; plot both curves together. First README figure.
4. **Check action-following.** Replay one start with recorded actions, then with different actions. Read the fingertip from dream frames with the state reader and compare with the real simulator under the same actions. Report error at steps 10, 50, 100.

**The idea:** one frame is easy; the dream must hold up when every input is its own output. Action-following shows it obeys controls instead of producing plausible motion.

**Done when:** drift chart with/without augmentation, fingertip errors, and side-by-side GIFs.

**Watch out:** blur is the typical failure. If the object vanishes mid-episode, note the step and keep the example for the failure gallery.

---

## Step 7 · Run policies inside the dream

*Dec 8 to 14, about 8 hours.*

**Goal:** let every policy act inside the dream, step by step, and score it.

**What you do**

1. **Same start for every policy.** For each eval seed, reset the real sim and take 3 zero-torque steps, giving 4 real starting frames.
2. **Closed loop.** The policy only sees the dream, through the reader:

```python
ctx, acts = warmup_frames(seed), [zero_action] * 4
for t in range(4, 100):
    obs_hat = reader(ctx[-2:])                      # the policy's eyes
    a, _ = policy.predict(obs_hat, deterministic=True)
    acts.append(a)
    ctx.append(sample(world_model, ctx[-4:], acts[-4:], steps=4))
```

3. **Score.** Read object and goal from the last 5 dream frames with the reader, average the distance, apply the same success threshold.
4. **Batch** all 50 seeds of one policy as one GPU batch.
5. **Save** `dream_scores.csv` (**ranking C**) plus dream videos for a few policies.

**The idea:** this is what 1X does with its humanoid: the policy plays inside the learned simulator, which predicts how well it will do in reality.

**Done when:** `dream_scores.csv` covers all ten policies, and real vs dream videos of the same seed tell the same story.

**Watch out:** fix the dream's random seed per eval seed so reruns are identical; keep the warm-up identical for every policy.

---

## Step 8 · The headline experiment

*Dec 15 to 21, about 6 hours.*

**Goal:** one number that answers the question, and an honest account of where it fails. (Rankings A, B, C are defined in `PROJECT.md`.)

**What you do**

1. **Make it fair.** Rescore A and B with the same 3-step warm-up as Step 7, so all three play the same 96 policy steps.
2. **Rank correlations** A-C (headline), A-B, B-C:

```python
from scipy.stats import spearmanr
rho, p = spearmanr(real["success"], dream["success"])
```

3. **Confidence range.** Resample the 50 seeds 1,000 times, recompute, report the middle 95%.
4. **Absolute gap too.** A dream can rank perfectly and still be over-optimistic.
5. **Held-out policies:** does the dream place them correctly?
6. **Failure analysis.** For the biggest disagreements, watch real vs dream videos side by side and sort problems: contact, object drift/vanishing, blur, reader misreading.
7. **Headline figure:** real success vs dream success for 10 policies, diagonal drawn, correlation in the title.

**The idea:** ranking is what robotics teams need: "which checkpoint do we deploy?" Ranking B separates the two error sources.

**Done when:** figure, correlation with range, top 3 failure types with example frames.

**Watch out:** don't tune on eval seeds after seeing results. A low correlation is still a result if you can explain it.

---

## Step 9 · Fix one thing and re-measure

*Dec 22 to 28, about 8 hours.*

**Goal:** diagnose a failure and fix it with evidence.

**What you do**

1. **Pick the biggest failure type** from Step 8.
2. **One fix aimed at it:**

| Failure | Fix to try |
| --- | --- |
| Contact looks wrong | More episodes with contact, or oversample contact frames |
| Quality decays over long episodes | Fine-tune on own rollouts (context from generated frames, predict real next frame): the idea behind Self Forcing |
| Object vanishes when the arm hides it | Longer context, 8 frames instead of 4 |
| Reader misreads dream frames | Fine-tune the reader on dream frames from Step 6 replays |

3. **Retrain, rerun Steps 6 to 8 on the same seeds,** report before/after: correlation, drift curve, failure count.

**The idea:** one well-measured improvement beats five untested tweaks.

**Done when:** the README has a before/after table.

**Watch out:** change one thing at a time.

---

## Step 10 · Demo, README and resume

*Dec 29 to Jan 4, about 8 hours.*

**Goal:** someone who has never met you understands the project in 40 seconds and can watch it work.

**What you do**

1. **Demo** on Hugging Face Spaces (free CPU) with Gradio: pick a policy and start position, watch real and dreamed episodes side by side (precomputed videos), next to the ranking figure. If fast enough on CPU, add a tab to drive the arm in the dream with sliders.
2. **README order:** problem in one line; headline figure; method in three sentences with the pipeline diagram; drift chart; failure gallery; before/after table; how to reproduce; limitations (one task, simulator not real robot, small scale); related work (DIAMOND, GameNGen, 1X, GigaWorld-1).
3. **Ship:** pin the repo, add the resume entry with real numbers, LinkedIn post with the side-by-side GIF.

**Resume line to start from:** Built a diffusion world model of a 7-joint robot arm from camera frames and used it to evaluate RL policies without the simulator; dream rankings matched real rankings with Spearman ρ = [X] across 10 policies, and [your fix] raised it to [Y].

**Done when:** a friend reads the README cold and can explain the project back.

**Watch out:** lead with the result, not installation. Be upfront about limitations.
