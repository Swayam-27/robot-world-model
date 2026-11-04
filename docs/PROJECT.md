# Project overview

## The problem

Robots learn and get tested inside simulators, but hand-building a simulator that matches the real world is slow, and testing every new robot policy on real hardware takes days. The humanoid company 1X answers this with a world model learned from robot data that predicts how a policy will perform before it touches a real robot.

This project builds that idea at small scale. A world model watches a robot arm push an object, learns to predict what the camera sees next, and then ranks 10 robot policies from best to worst. MuJoCo plays the role of "reality", so every prediction can be checked exactly.

**Headline question:** does the world model rank the policies the same way the real simulator does? One number answers it: the Spearman rank correlation between the two rankings.

## Components

| Part | What it does | Tools |
| --- | --- | --- |
| The "real world" | Pusher-v5: a 7-joint arm pushes a cylinder to a goal; 100 steps per episode, rendered as 64x64 frames | Gymnasium + MuJoCo, `MUJOCO_GL=egl` |
| Policies to rank | About 10 policies of different skill: SAC checkpoints along one run, noisy copies of the best, a random policy | Stable-Baselines3 (SAC), CPU |
| Dataset | Frames, actions and true states from every policy plus random actions; about 2,000 episodes | NumPy `.npz`, private Kaggle dataset |
| State reader | Small CNN: last 2 frames in, the 23-number observation out. It is the policy's "eyes" in the dream | PyTorch |
| World model | Small diffusion U-Net (DIAMOND-style, EDM): last 4 frames + last 4 actions in, next frame out, with noise augmentation on the context | PyTorch |
| Dream evaluator | Policy acts on the reader's estimate of dream frames; world model draws the next frame; 100 steps; score = final object-to-goal distance | Python |
| Demo | Real and dreamed episodes side by side, plus the ranking figure | Gradio on Hugging Face Spaces |

## The three rankings

| Ranking | Where policies run | What the policy sees | Comparing with A shows |
| --- | --- | --- | --- |
| A | Real simulator | True state | Ground truth |
| B | Real simulator | State reader on real frames | What the reader alone costs |
| C | Dream (world model) | State reader on dream frames | What reader + dream cost |

Headline: Spearman(A, C) with a bootstrap 95% range over seeds. Also report the absolute gap in success rate.

## Pipeline

```
MuJoCo Pusher ──> policies + random ──> dataset (frames, actions, states)
                                              │
                         ┌────────────────────┴──────────────┐
                    state reader                         world model
                         │                                    │
real sim + true state ─> ranking A                            │
real sim + reader ─────> ranking B                            │
dream loop: frames ─> reader ─> policy ─> action ─> world model ─> next frame
                                                   └──> ranking C
```

## Observation layout (Pusher-v5, 23 numbers)

- 0 to 6: joint angles
- 7 to 13: joint velocities
- 14 to 16: fingertip x, y, z
- 17 to 19: object x, y, z
- 20 to 22: goal x, y, z

Actions: 7 joint torques in [-2, 2].

## Results to produce

1. Drift chart: PSNR and LPIPS against rollout step, with and without noise augmentation
2. Action-following error (fingertip position) at steps 10, 50, 100
3. Headline scatter: real success vs dream success, 10 policies, Spearman in the title
4. Failure gallery: the top 3 failure types with example frames
5. Before/after table for one targeted fix

## After January (stretch goals)

- **Train inside the dream:** train a policy only in the world model, then test it in MuJoCo.
- **Generalisation:** goal positions, object shapes or camera angles the model never saw.
- **Real-time control:** distil to a few denoising steps and drive the arm live in the browser.
- **Phase 2, a real arm:** build an open-source SO-101 arm (LeRobot), film it with a webcam, record frames and motor commands, and train the same world model on real data. The trained sim model will not transfer directly (the sim-to-real gap); the pipeline does. This is why the data source sits behind an interface in `envs/`.

## Related work

DIAMOND (diffusion world model, NeurIPS 2024, github.com/eloialonso/diamond), GameNGen (noise augmentation, DOOM), 1X World Model (policy evaluation for humanoids), GigaWorld-1 (world models as robot policy evaluators, 2026), Self Forcing (exposure bias).
