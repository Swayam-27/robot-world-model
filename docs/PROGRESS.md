# Progress

**Current step:** Step 2 · Policies
**Target:** resume-ready on 2027-01-04. Exams (assumed) Nov 17 to 30: no project work.

Tick boxes as work is finished. Step details are in `docs/BUILD_GUIDE.md`.

## Step 0 · Get ready (this week)
- [ ] Port the Real vs AI Faces custom CNN to PyTorch (train, save, reload, log a loss curve) · moved to just before Step 4 (see DECISIONS.md)
- [x] Create the GitHub repo from this folder and push it
- [ ] Kaggle: phone verified, GPU on, first cell clones the repo · phone + clone done; GPU not needed until Step 4
- [ ] Weights & Biases (or TensorBoard) logging works

## Step 1 · Real world runs (Nov 3 to 9)
- [x] Pusher-v5 renders 64x64 frames headless on Kaggle
- [x] GIF of a random episode shows fingertip, object and goal in every frame
- [x] `object_goal_distance(obs)` helper written and tested
- [x] Same seed gives the same first frame

## Step 2 · Policies (Nov 3 to 9)
- [x] SAC trained on CPU with checkpoints · 1M steps, 50 checkpoints; best ~0.054 m from goal at 960k (train seeds)
- [x] Pool of about 10 policies (checkpoints, noisy copies, random) · 10 brains, 0.054 to 0.241 m on training seeds, no near-ties
- [ ] Success threshold chosen by looking at frames
- [x] `real_scores.csv` (ranking A) on the 50 eval seeds; ranking is stable when rerun · identical on rerun; 0.058 (sac_960k) to 0.245 m (random), no near-ties; same order as on training seeds (Spearman 1.0)

## Step 3 · Dataset (Nov 10 to 16)
- [ ] About 2,000 episodes recorded on training seeds only
- [ ] 2 policies held out of the data entirely (optional, recommended)
- [ ] Split 90/10 by episode, uploaded as private Kaggle dataset
- [ ] Playback GIF confirms frames, actions and states line up (101 frames, 100 actions)

## Step 4 · State reader (Nov 10 to 16)
- [ ] Reader trained; per-group validation errors recorded
- [ ] Ranking B computed (policies on reader estimates in the real sim)

## Exams · Nov 17 to 30 · pause

## Step 5 · World model (Dec 1 to 7)
- [ ] DIAMOND Atari demo run and world-model code read
- [ ] Inputs adapted: 4 context frames, continuous actions via MLP
- [ ] Trained with noise augmentation; one-step predictions look right at contact

## Step 6 · 100-step dream (Dec 1 to 7)
- [ ] Autoregressive replay of validation episodes
- [ ] Drift chart with and without noise augmentation
- [ ] Action-following errors at steps 10, 50, 100
- [ ] Side-by-side GIFs

## Step 7 · Policies in the dream (Dec 8 to 14)
- [ ] Closed loop: dream frame -> reader -> policy -> action -> next frame
- [ ] `dream_scores.csv` (ranking C), deterministic on rerun

## Step 8 · Headline experiment (Dec 15 to 21)
- [ ] A, B rescored with the same warm-up as C
- [ ] Spearman A-C, A-B, B-C with bootstrap ranges
- [ ] Absolute gap reported; held-out policies checked
- [ ] Failure analysis: top 3 failure types
- [ ] Headline scatter figure

## Step 9 · Fix one thing (Dec 22 to 28)
- [ ] One targeted fix chosen and logged in DECISIONS.md
- [ ] Steps 6 to 8 rerun; before/after table

## Step 10 · Ship (Dec 29 to Jan 4)
- [ ] Gradio demo on Hugging Face Spaces
- [ ] README written (result first)
- [ ] Repo pinned, resume line added, LinkedIn post

---

## Session log

Newest first. One entry per session: date, what got done, what's next, blockers.

<!-- /end-session adds entries here -->

- **2026-11-04** · Step 0 repo pushed; Step 1 done. `PusherEnv` behind the `ArmEnv` interface, fixed camera C, `envs/check_env.py` passes locally and on Kaggle CPU (headless; both EGL and OSMesa work, EGL is the default), random-episode GIF looks right. Next: Step 2, SAC on Kaggle CPU. Blockers: none.
