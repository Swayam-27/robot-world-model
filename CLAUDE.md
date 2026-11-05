# Robot Arm World Model

A diffusion world model learns to simulate a 7-joint robot arm (MuJoCo Pusher-v5) from camera frames alone. It is then used to rank 10 robot policies without running the simulator. The headline result is the Spearman rank correlation between the dream ranking and the real ranking.

Owner: Swayam, 3rd-year CS student. This is his flagship portfolio project for ML/AI internship applications in January 2027.

## Read these first

- @docs/PROJECT.md: problem statement, architecture, the three rankings, hard rules
- @docs/PROGRESS.md: what is done, what is next, the session log
- @docs/DECISIONS.md: design choices already made and why

Learning links (repos, courses) are in `docs/RESOURCES.md`; suggest the relevant one when I hit a new concept.

Plain-language explanations of every concept and term are in `docs/CONCEPTS.md`. Explain things at that level, and add new terms there as they come up.

`docs/EXPLAIN.md` tells the whole project as a simple, fun story (everyday picture -> real term -> where in the repo). When a step finishes or a big new idea appears, add or update its chapter there.

The full step-by-step instructions are in `docs/BUILD_GUIDE.md`. Read only the section for the current step, not the whole file.

## How to work with me

- **I am learning.** I know Python, TensorFlow/Keras, scikit-learn and Git. PyTorch, diffusion models and RL are new to me.
- **One step at a time.** Work only on the current step in PROGRESS.md. Don't jump ahead or build future steps "while we're here".
- **Explain as you go.** Before writing non-trivial code, say in 2 to 3 sentences what it will do and why. After writing it, point out the lines I must understand for interviews.
- **Small, runnable pieces.** Prefer a 40-line script I can run and check over a 400-line framework. Every new script gets a quick sanity check (a shape print, a saved GIF, a plot).
- **Ask before big choices.** If something would change a decision in DECISIONS.md, stop and ask me.
- **Never fake results.** Never invent or hard-code metrics, scores or plots. If something can't run here, say so and give me the command to run on Kaggle.

## Environment

- Code runs on **free Kaggle GPUs (T4/P100, 16 GB)**, about 30 GPU hours a week, sessions up to 12 hours. Simulator, SAC and data recording run on CPU to save GPU quota.
- Headless rendering: set `os.environ["MUJOCO_GL"] = "egl"` before any MuJoCo import (fallback: `"osmesa"`).
- Notebooks in `notebooks/` only clone the repo and call scripts. Logic lives in `.py` files.
- Save checkpoints every 20 to 30 minutes; Kaggle sessions can die.

## Code conventions

- PyTorch only (no TensorFlow).
- Every experiment is driven by a YAML file in `configs/`; scripts take `--config`.
- Frames are 64x64 RGB, stored as uint8; scale to [-1, 1] only inside the model code.
- Use the constants in `envs/constants.py` (seeds, observation indices). Never hard-code them elsewhere.
- Log metrics and sample image grids to Weights & Biases (or TensorBoard).
- Type hints and a short docstring on every public function.
- **Keep the data source swappable.** Recording, training and evaluation code must not call MuJoCo directly; go through a small interface in `envs/` (`reset(seed)`, `step(action)`, `frame()`, `state()`). Phase 2 may replace MuJoCo with a real SO-101 arm and webcam, and only that interface should change.

## Hard rules (never break)

1. **Evaluation seeds** (`EVAL_SEEDS` in `envs/constants.py`) never appear in any training data, and models are never tuned on them.
2. **Split by episode**, never by frame.
3. **Change one thing at a time** in experiments, and record each change in DECISIONS.md.
4. Policies are scored with `deterministic=True`; noisy policies use a fixed noise seed per start position.

## Repo layout

```
envs/      simulator, rendering, camera, constants
policies/  SAC training, policy pool, real scores
data/      episode recording scripts (data itself lives on Kaggle)
models/    state reader, world model
eval/      dream rollouts, drift, action-following, rankings
demo/      Gradio app for Hugging Face Spaces
configs/   one YAML per experiment
notebooks/ thin Kaggle notebooks
docs/      project docs (keep PROGRESS.md and DECISIONS.md up to date)
```

## Session routine

- Start: `/start-session` reads progress and proposes today's plan.
- End: `/end-session` updates PROGRESS.md and DECISIONS.md and suggests a commit.
- Interview prep: `/explain <file>` walks me through code line by line.
