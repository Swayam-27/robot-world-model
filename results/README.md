# Results

Small result files copied from `outputs/` (which is gitignored) so the numbers behind the project live in git. Every file is produced by a script; never edit by hand.

| File | Made by | What it is |
| --- | --- | --- |
| `checkpoint_measure_1m.csv/.png` | `policies/measure_checkpoints.py` | Pushing skill of all 50 SAC checkpoints (1M-step run), 20 training seeds |
| `pool_screen.csv` | `policies/pool.py` | The 10 pool brains on 50 training seeds, with near-tie check |
| `pass_mark_sheet.png` | `policies/pass_mark_sheet.py` | Final frames by distance, used to choose the 3 cm pass mark |
| `real_scores.csv` | `policies/score.py` | **Ranking A**: the 10 brains on the 50 eval seeds (mean distance, rank, success rate) |
| `real_scores_per_seed.csv` | `policies/score.py` | Every brain x every eval seed |
| `episodes.csv` | `data/record.py` | Index of the 2,000-episode dataset (brain, seed, split, final distance) |
| `episodes_dagger.csv` | `data/record_dagger.py` | Index of the 1,050 DAgger episodes (brains playing through reader v1; brain, seed, split, final distance, reader can error) |
| `reader_v{1..4}_train_log.csv` | `models/train_reader.py` | Train/val loss and per-group errors per epoch for each state reader |
| `reader_v{1..4}_val_errors.csv` | `models/train_reader.py` | Best checkpoint's validation errors per group, in real units |
| `reader_v4_pred_vs_true.png` | `models/train_reader.py` | Reader v4 predictions vs truth on validation (can x, can y, joint 0 speed) |
| `reader_diagnosis{,_v2,_v3,_v4}.csv` | `eval/reader_diagnosis.py` | sac_960k on 50 TRAINING seeds, one group of the reader's numbers at a time (v1 has no suffix) |
| `ranking_b{,_v2,_v3,_v4}.csv` | `eval/ranking_b.py` | **Ranking B** per reader: the 10 brains on the 50 eval seeds seeing the reader's estimates, with rank A and change. v4 (DAgger): Spearman(A, B) = 1.000 |
| `ranking_b{,_v2,_v3,_v4}_per_seed.csv` | `eval/ranking_b.py` | Every brain x every eval seed, per reader |
| `world_model_v1_eval_log.csv` | `models/train_world_model.py` | World model v1 (Kaggle T4, 3 h, 32,202 steps): val loss and one-step pixel errors every 2,000 steps, whole image and moving pixels, vs copy-last-frame |
| `world_model_v1_samples_step{02000,32000}.png` | `models/train_world_model.py` | Same 8 validation examples at step 2,000 and 32,000: newest context, real next frame, prediction, error x4 |
| `world_model_v1_contact_check.csv/.png` | `eval/contact_check.py` | World model v1 one-step error on 128 validation steps where the can moved > 1 mm (contact) vs 128 where it did not, next to copy-last-frame; grid of the biggest pushes |
