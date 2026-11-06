# Concepts, in plain words

For revising before interviews, and for Claude Code to explain things at the right level.

## The project in one line

I taught a neural network to imagine how a robot arm's world behaves just by watching videos of it, then used that imagination to test which robot brains are good.

## The 5 steps

1. **Record videos:** 10 robot brains of different skill play in MuJoCo; save every frame and action.
2. **Train the world model:** a diffusion model learns to draw the next frame from past frames plus the action.
3. **Train the eyes:** the state reader learns to read positions from frames.
4. **Test in the dream:** each brain plays inside the world model, seeing only dream frames through the eyes.
5. **Compare:** does the dream rank the brains in the same order as MuJoCo? One number answers it: the Spearman correlation.

## The big ideas

**MuJoCo** is not a website. It's free physics-simulation software from Google DeepMind, installed with pip: a game engine for robots. It calculates real physics (forces, contact, friction). In this project it plays the role of "reality".

**Pusher-v5** is one ready-made MuJoCo task: a 7-joint arm (shoulder, elbow, forearm, wrist) on a table must push a can-like cylinder onto a goal spot. One attempt (an episode) is 100 steps.

**A policy is the robot's brain.** It runs a loop many times a second: sense (read the 23-number state), decide (output 7 joint torques), act (motors move), repeat. Inside it is a small neural network. It learns by **reinforcement learning**: trial and error with a score (reward for hand near can and can near goal, a small penalty for wasted effort).

**The policy decides what to do; the world model predicts what happens.** "Twist the elbow" is the policy. "If you twist the elbow, the arm swings this way and the can slides" is the world model.

**A world model is a learned simulator.** It watches videos and learns: given the last few frames and the action, what will the next frame look like? Feed its own predictions back in and it generates a whole episode with no MuJoCo running. That generated video is the **dream**.

**A diffusion model** is the kind of network the world model is. It learns to clean up noisy images: take a real frame, add random noise, train the network to recover the clean frame. Start from pure noise and clean it step by step, and it creates a new, realistic image (like a sculptor chipping a block of stone). To make it a world model, it also gets the past frames and the action as clues, so the only clean image that fits is the correct next frame.

**The state reader is the robot's eyes in the dream.** The dream only has pictures, but policies need numbers. A small CNN, trained from scratch on recorded frames (MuJoCo gives the correct 23 numbers with every frame for free), reads positions and speeds from the last 2 frames. It is a regression task: 23 outputs, mean squared error.

**The floor** is the random policy: an arm twisting randomly, like guessing every answer on a test. Anything better than random has learned something.

**Why rank 10 brains?** Like a driving school checking its simulator: it's only useful if students who drive well in it also drive well on the road. If the dream ranks the brains in the same order as MuJoCo, the dream is a trustworthy simulator.

**Why it matters:** companies like 1X test humanoid robot brains inside a learned world model because real-world testing takes days or weeks. This project is a small, fully measurable version of that.

## Glossary

| Term | Meaning |
| --- | --- |
| Episode | One attempt, 100 steps |
| Step | One moment: the brain acts, the world updates once |
| Frame | One 64x64 camera picture |
| State / observation | 23 numbers: joint angles (7), joint speeds (7), fingertip, can and goal positions (3 each) |
| Action | 7 numbers: torque on each joint, in [-2, 2] |
| Seed | A number fixing where the can and goal start, so a test repeats exactly |
| Reinforcement learning | Learning by trial and error with rewards |
| SAC | The RL algorithm used, ready-made in Stable-Baselines3 |
| Reward | The score the simulator gives every step. Pusher: minus can-to-goal distance, minus half the hand-to-can distance, minus a little for torque used. Always negative; closer to 0 is better |
| Actor | The part of SAC that picks actions: the policy itself |
| Critic | The part of SAC that guesses how much future reward a state + action will earn; the actor learns to pick what the critic likes. SAC uses two and trusts the more pessimistic one |
| Replay buffer | A big store of past steps (state, action, reward, next state); SAC learns from random handfuls of it |
| Off-policy | Can learn from old experience in the replay buffer, not only from the latest attempts. Makes SAC efficient enough for a CPU |
| Entropy bonus | Extra reward for staying a bit random, so the policy keeps exploring (the "soft" in Soft Actor-Critic) |
| Deterministic action | Using the policy's best-guess action instead of a random sample; makes every scoring run repeatable |
| Smoke test | A tiny, fast run that only checks the code works end to end before the long run |
| Checkpoint | A saved copy of a network mid-training; early ones clumsy, late ones skilled |
| Policy pool | The ~10 brains we rank: checkpoints from one SAC run, noisy copies of the best, and a random brain. Must be spread out, or any ranking is noise |
| Action noise ("shaky hands") | Random Gaussian jitter added to a good brain's torques; more noise = worse brain. Seeded by the start position, so it is repeatable |
| Screening | Trying candidate brains on practice (training) seeds to choose the pool, before the sealed exam |
| Paired comparison | Compare two brains on the *same* seeds, seed by seed. Easy and hard start positions cancel out, so smaller real gaps show up |
| Standard error (SE) | How much a mean would wobble if you repeated the test with new seeds: std / sqrt(n). A gap smaller than about 2 SE could be luck (a near-tie) |
| Context | The past frames the world model sees (the last 4) |
| U-Net | The network shape inside the diffusion model: a CNN that shrinks the image to understand it, then grows it back to draw |
| DIAMOND | A published open-source diffusion world model whose code we adapt |
| EDM | DIAMOND's recipe for adding and removing noise; good frames in a few steps |
| Autoregressive rollout | Predict a frame, feed it back as context, predict the next, repeat |
| Drift | Small mistakes piling up over a long dream until it turns blurry or wrong |
| Noise augmentation | The fix for drift: deliberately mess up past frames in training so the model doesn't trust them blindly |
| Action-following | Checking the dream obeys the actions rather than producing plausible motion |
| PSNR / LPIPS | Scores for how close a dreamed frame looks to the real one |
| Rankings A, B, C | Brains ranked in MuJoCo (A), in MuJoCo with the reader's eyes (B), in the dream (C) |
| Spearman correlation | One number from -1 to 1 for how well two rankings agree; 1 = identical order. The headline result |
| Sim-to-real gap | A model trained in simulation failing on a real robot because reality looks and behaves differently |
| Rendering | Turning the 3D scene (arm, table, can) into a flat picture, like taking a photo of it |
| Headless rendering | Rendering on a computer with no screen (like Kaggle): the picture is drawn into memory instead of a window. EGL uses the GPU driver, OSMesa does it in software on the CPU |
| Resolution | How many pixels a picture has. 64x64 = 4,096 pixels x 3 colours = 12,288 numbers per frame. Small on purpose: the world model must draw every pixel, 100 times per dream |

## Ways robots learn to move (context for interviews)

A world model isn't required to make a robot move. Common routes:
- **Train in a hand-built simulator, then transfer** (e.g. Humanoid-Gym trains humanoid walking in Isaac Gym).
- **Imitation learning:** a person moves a leader arm, the robot records it, a network learns to copy (most SO-101 projects).
- **World models** help in three ways: *test* brains before deployment (1X, this project), *train* brains inside the dream (DIAMOND, General Intuition), or *plan* by imagining a few moves and picking the best.

After training, a robot moves by running the policy in the sense-decide-act loop: in simulation `env.step(action)`; on a real robot a motor driver turns the action into signals for the servos, typically 10 to 50 times a second.

## Explaining it in 30 seconds

> Testing robot software on real robots is slow and expensive. Companies like 1X now train a neural network to imagine the robot's world, and test in that imagination first. I built a small version: my model watches videos of a simulated robot arm pushing a can, and learns to predict what the camera will see next for any move. Then I let 10 robot brains of different skill play inside its imagination. If it ranks them in the same order as the real simulator, it's a trustworthy simulator. It reached a correlation of [X].

Pattern: problem, idea, result; details only if asked.
