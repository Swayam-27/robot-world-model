# Progress

**Current step:** Step 6 · 100-step dream (Step 5 done 2026-11-21)
**Target:** resume-ready on 2027-01-04. Exams start Nov 30 (no project work during exams).

Tick boxes as work is finished. Step details are in `docs/BUILD_GUIDE.md`.

## Step 0 · Get ready (this week)
- [x] Port the Real vs AI Faces custom CNN to PyTorch (train, save, reload, log a loss curve) · Step 4a, `warmup/faces_cnn_torch.py` on Kaggle T4: 312,289 params, best val acc 0.858 at epoch 4 (val loss 0.311), reload matches exactly; loss printed per epoch (W&B curve still to do)
- [x] Create the GitHub repo from this folder and push it
- [x] Kaggle: phone verified, GPU on, first cell clones the repo · GPU T4 x2 used for readers v1-v4
- [x] Weights & Biases (or TensorBoard) logging works · reader runs logged from Kaggle via a Kaggle secret (project robot-world-model)

## Step 1 · Real world runs (Nov 3 to 9)
- [x] Pusher-v5 renders 64x64 frames headless on Kaggle
- [x] GIF of a random episode shows fingertip, object and goal in every frame
- [x] `object_goal_distance(obs)` helper written and tested
- [x] Same seed gives the same first frame

## Step 2 · Policies (Nov 3 to 9)
- [x] SAC trained on CPU with checkpoints · 1M steps, 50 checkpoints; best ~0.054 m from goal at 960k (train seeds)
- [x] Pool of about 10 policies (checkpoints, noisy copies, random) · 10 brains, 0.054 to 0.241 m on training seeds, no near-ties
- [x] Success threshold chosen by looking at frames · 3 cm (`SUCCESS_DISTANCE`); eval success: sac_960k 30%, sac_600k 24%, others 0%
- [x] `real_scores.csv` (ranking A) on the 50 eval seeds; ranking is stable when rerun · identical on rerun; 0.058 (sac_960k) to 0.245 m (random), no near-ties; same order as on training seeds (Spearman 1.0)

## Step 3 · Dataset (Nov 10 to 16)
- [x] About 2,000 episodes recorded on training seeds only · 2,000 episodes, 209 MB, recorded on the laptop (`data/record.py`)
- [x] 2 policies held out of the data entirely (optional, recommended) · sac_400k and sac_960k_n0.8
- [x] Split 90/10 by episode, uploaded as private Kaggle dataset · 1,800 / 200 balanced per brain; on Kaggle as `swayam006/robot-world-model-dataset`, all 2,000 files load; replay with pinned mujoco 3.17.0 matches physics within 0.3 mm (renderer differs ~3.6% of pixels, see DECISIONS)
- [x] Playback GIF confirms frames, actions and states line up (101 frames, 100 actions) · all 2,000 files checked; replay of 2 episodes per brain matches exactly (`data/playback.py`)

## Step 4 · State reader (Nov 10 to 16)
- [x] Reader trained; per-group validation errors recorded · v1 (2 frames, Kaggle T4, best epoch 6 of 11, overfits after ~epoch 3): mean error object 1.79 cm (p95 4.45), goal 0.006 cm, fingertip 3.89 cm, joint angles 9.4 deg, joint velocities 0.53 rad/s (`outputs/reader/val_errors.csv`, W&B run light-moon-1)
- [x] Ranking B computed (policies on reader estimates in the real sim) · v1 done but **policies fall apart**: sac_960k 5.8 -> 20.4 cm, every brain 0% success, Spearman(A, B) 0.745 with 8 of 9 neighbour gaps near-ties (`outputs/scores/ranking_b.csv`). Diagnosis on 50 training seeds (sac_960k, one group from the reader at a time): velocities +4.4 cm, object +2.9, joint angles +2.7, fingertip +0.9, all together +13.0 -> no single culprit, errors add up; reader v2 needed
  - v2 (3 frames, W&B vital-surf-2): no better on validation (velocities 0.54 rad/s, best epoch 4) and **worse** in closed loop: diagnosis all-from-reader +17.2 cm (v1 +13.0); ranking B v2 sac_960k 21.4 cm, Spearman 0.855 but all near-ties (`ranking_b_v2.csv`). Rejected, v1 stays the base
  - v3 = v1 + weight decay 0.05 (W&B rural-disco-3): val loss 0.2455, speeds fixed in closed loop but angles/can worse; diagnosis all-from-reader +16.2 cm; ranking B v3 good brains 17.7-22.4 cm, Spearman 0.855, near-ties (`ranking_b_v3.csv`). Not good enough
  - v4 = v1 + DAgger data: 1,050 episodes recorded through reader v1 (`outputs/dataset_dagger`, 117 MB, replay check 14/14). Trained on Kaggle (W&B northern-water-4). Diagnosis all-from-reader +10.8 cm (best yet). **Ranking B v4: Spearman(A, B) = 1.000**, held-out brains placed correctly, 4 of 9 gaps clear; sac_960k 14.6 cm (`outputs/scores/ranking_b_v4.csv`). **Reader v4 = the project's reader**

## Exams · from Nov 30 · pause

## Step 5 · World model (Dec 1 to 7)
- [x] DIAMOND world-model code read (Atari demo skipped, see DECISIONS) · U-Net copied to `models/diamond_blocks.py` (4.26M params at 64 channels); dataset `models/world_model_data.py` checked on 3 episodes per split per folder
- [x] Inputs adapted: 4 context frames, continuous actions via MLP · `models/world_model.py` (4.48M params), plus context-noise level as a hint; self-test overfits one batch and samples uint8 frames
- [x] Trained with noise augmentation; one-step predictions look right at contact
  - v1 first run (Kaggle T4, 3 h, 32k steps, W&B azure-butterfly-5): moving-pixel error 12.9 of 255 vs copy-last-frame 30.5; val loss 0.0033, no overfitting, still improving; grids sharp at the can. Contact check (`eval/contact_check.py`): on steps where the can moves > 1 mm, moving-pixel error 16.1 vs copy-last 28.0 (still steps 11.7): 1.37x harder, but predictions look the same as the real frames by eye. Longer training (resume) later if the dreams need it

## Step 6 · 100-step dream (Dec 1 to 7)
- [ ] Autoregressive replay of validation episodes
  - first test (8 episodes, laptop): sharp frames but the dream drifts: fingertip 4.7 cm off at frame 10, 37.8 at 100; can 7.6 cm at 25, 25.8 at 100 (reader on real frames ~1 cm). No-augmentation ablation config ready for Kaggle. Dream-time aug_level 0.3: fingertip 27.8 cm at frame 100 (was 37.8; frozen yardstick 66.1), can 23.1 (was 25.8) but **worse than a can that never moves (19.6)**: pushing is the main problem
- [ ] Drift chart with and without noise augmentation
- [ ] Action-following errors at steps 10, 50, 100
  - first test (8 episodes, laptop): fingertip following error 5.1 / 25.9 / 39.1 cm at 10 / 50 / 100; the dream reacts to changed actions in the right direction early (cosine 0.97 at frame 10), so the problem is drift, not ignored actions
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

- **2026-11-21** · Step 4 done. Reader v1 broke the brains in closed loop (ranking B Spearman 0.745, all near-ties); diagnosis on training seeds showed errors in every group compounding. v2 (3 frames) and v3 (weight decay) only moved the error around. v4 = DAgger round 1 (1,050 episodes recorded through reader v1, true state saved) gave diagnosis +10.8 cm and **ranking B Spearman 1.000** (4 of 9 gaps clear). Results in `results/`. EXPLAIN chapter 5 (diffusion) written. Next: Step 5, read DIAMOND's denoiser/U-Net code, decide how to bring it in, write the world-model dataset. Blockers: laptop memory (long background jobs get stopped; run long jobs in your own terminal).

- **2026-11-04** · Step 0 repo pushed; Step 1 done. `PusherEnv` behind the `ArmEnv` interface, fixed camera C, `envs/check_env.py` passes locally and on Kaggle CPU (headless; both EGL and OSMesa work, EGL is the default), random-episode GIF looks right. Next: Step 2, SAC on Kaggle CPU. Blockers: none.
