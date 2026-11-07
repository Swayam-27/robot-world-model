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
