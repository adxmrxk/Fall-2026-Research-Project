# Fall 2026 Research: Box-Prompt Strategies for SAM 2 Pancreas Segmentation

# Last Updated 2026-11-23

## Introduction

This project looks at how much manual input SAM 2 needs to segment the pancreas in 3D CT scans. SAM 2 is a promptable segmentation model: you give it a bounding box, and it outlines the object in that frame. For a 3D volume, the question is how many boxes you have to draw and where to place them.

## Dataset

- **Medical Segmentation Decathlon (MSD) Task07, Pancreas.** A set of 281 CT scans with expert pancreas masks.
- **Sample:** 150 scans, chosen at random with seed 44 (`random.Random(44)` applied to the sorted list of 281 scan names). The list is in `seed44_prompt_strategy_test/sample_seed44_n150.txt`.
- **Preprocessing:** CT intensities windowed to -100 to 240 HU. Each volume is saved as a sequence of JPEG frames for SAM 2.

## Method

SAM 2 video predictor (`sam2.1_hiera_small`) with box prompts. Scoring follows the supervisor's rules:

- Dice is computed over the **pancreas slices only** (slices with a non-empty ground-truth mask), per scan, using the whole volume: `2 × overlap / (predicted + ground truth)`.
- Propagation runs across the organ's extent, from the first to the last pancreas slice. Slices with no pancreas get no box or prediction.
- The various experiments we conduct fall into one of the two categories:
  1. **Interval re-prompting:** a new box is placed at a fixed spacing, such as every 5th, 10th, 15th, 20th, etc pancreas slice, whatever the mask looks like.
  2. **Confidence-based re-prompting:** a new box is placed only when SAM 2's confidence in a slice falls below a set threshold (below 0.8, 0.65, 0.9, etc).

### Experiments

- Starting from 2026-11-11 the experiments that I have ran are:

1. **Baseline: box on every pancreas slice.** Each slice is segmented from its own box. This is the upper bound on prompt effort.
2. **Every 10: re-prompt every 10th pancreas slice.** One box is placed on the largest pancreas slice. The mask is propagated forward and backward. A new box is placed every 10th pancreas slice, and the propagation restarts from there.
3. **Every 5: re-prompt every 5th pancreas slice.** The same method as Every 10, with a new box every 5th pancreas slice.
4. **Every 15: re-prompt every 15th pancreas slice.** The same method as Every 10, with a new box every 15th pancreas slice.
5. **Every 20: re-prompt every 20th pancreas slice.** The same method as Every 10, with a new box every 20th pancreas slice.

The runner is `seed44_prompt_strategy_test/runner.py`.

## Results

Dice is the average volume Dice per scan. Boxes are the average number of box prompts per scan. Each run covers the same 150 scans, split into three parts of 50.

### Baseline (150 scans, complete)

| Part    | Scans   | Avg Dice   | Avg boxes per scan |
| ------- | ------- | ---------- | ------------------ |
| 1       | 50      | 0.7675     | 31.28              |
| 2       | 50      | 0.7578     | 33.46              |
| 3       | 50      | 0.7370     | 31.14              |
| **All** | **150** | **0.7541** | **31.9600**        |

### Every 5 (150 scans, complete)

| Part    | Scans   | Avg Dice   | Avg boxes per scan |
| ------- | ------- | ---------- | ------------------ |
| 1       | 50      | 0.7296     | 6.30               |
| 2       | 50      | 0.6882     | 6.70               |
| 3       | 50      | 0.6776     | 6.16               |
| **All** | **150** | **0.6985** | **6.3867**         |

Compared with the baseline on the same 150 scans, every-5 uses 6.3867 boxes per scan versus 31.9600 for the baseline, and loses 0.0556 Dice (0.7541 vs 0.6985).

### Every 10 (150 scans, complete)

| Part    | Scans   | Avg Dice   | Avg boxes per scan |
| ------- | ------- | ---------- | ------------------ |
| 1       | 50      | 0.6714     | 3.04               |
| 2       | 50      | 0.6413     | 3.32               |
| 3       | 50      | 0.6118     | 2.98               |
| **All** | **150** | **0.6415** | **3.1133**         |

Compared with the baseline on the same 150 scans, every-10 uses 3.1133 boxes per scan versus 31.9600 for the baseline, and loses 0.1126 Dice (0.7541 vs 0.6415).

**What the detail shows:** Most of the loss comes from stretches of slices where the mask collapses after a box is placed. In several scans, the slices right after a re-prompt drop to near zero Dice, even when the box itself looks correct. Examples are `pancreas_186` (slices 14–19) and `pancreas_265` (slices 38–46). This is the main thing to investigate next.

### Every 15 (150 scans, complete)

| Part    | Scans   | Avg Dice   | Avg boxes per scan |
| ------- | ------- | ---------- | ------------------ |
| 1       | 50      | 0.6330     | 2.08               |
| 2       | 50      | 0.6012     | 2.14               |
| 3       | 50      | 0.5616     | 2.06               |
| **All** | **150** | **0.5986** | **2.0933**         |

Compared with the baseline on the same 150 scans, every-15 uses 2.0933 boxes per scan versus 31.9600 for the baseline and 3.1133 for every-10. It loses 0.1555 Dice (0.7541 vs 0.5986).

### Every 20 (150 scans, complete)

| Part    | Scans   | Avg Dice   | Avg boxes per scan |
| ------- | ------- | ---------- | ------------------ |
| 1       | 50      | 0.6095     | 1.70               |
| 2       | 50      | 0.5792     | 1.84               |
| 3       | 50      | 0.5345     | 1.68               |
| **All** | **150** | **0.5744** | **1.7400**         |

Compared with the baseline on the same 150 scans, every-20 uses 1.7400 boxes per scan versus 31.9600 for the baseline and 2.0933 for every-15. It loses 0.1797 Dice (0.7541 vs 0.5744).

### Summary of interval re-prompting (150 scans each)

| Run      | Avg boxes per scan | Avg Dice   | Dice lost vs baseline |
| -------- | ------------------ | ---------- | --------------------- |
| Baseline | 31.9600            | 0.7541     | —                     |
| Every 5  | 6.3867             | 0.6985     | 0.0556                |
| Every 10 | 3.1133             | 0.6415     | 0.1126                |
| Every 15 | 2.0933             | 0.5986     | 0.1555                |
| Every 20 | 1.7400             | 0.5744     | 0.1797                |

### Next experiments (planned)

Each experiment will run on the same 150 scans, split into the same three parts.

#### Interval re-prompting

| Interval | Re-prompt every     | Status   |
| -------- | ------------------- | -------- |
| Every 5  | 5th pancreas slice  | complete |
| Every 10 | 10th pancreas slice | complete |
| Every 15 | 15th pancreas slice | complete |
| Every 20 | 20th pancreas slice | complete |

#### Confidence-based re-prompting

| Threshold | Role                                  | Status  |
| --------- | ------------------------------------- | ------- |
| 0.5       | fewest re-prompts                     | planned |
| 0.65      | low variation                         | planned |
| 0.8       | **baseline** (tested on pancreas_001) | planned |
| 0.85      | high variation                        | planned |
| 0.9       | most re-prompts                       | planned |


## Repository layout

- `README.md`: this overview.
- `seed44_prompt_strategy_test/runner.py`: runs the box-prompt strategies on SAM 2 (the experiment code).
- `seed44_prompt_strategy_test/kaggle/runner.py`: the same runner, set up for Kaggle GPUs. It can split the 150 scans into three parts (`PART`), so each run fits in one Kaggle session.
- `seed44_prompt_strategy_test/pick_scans.py`: picks the random sample of scans with seed 44.
- `seed44_prompt_strategy_test/sample_seed44_n150.txt`: the 150 scans used in every run.
- `pilot/pilot_confidence.py`: the setup code (model loading, data loading, Dice) that `runner.py` reuses.
- `background-papers/`: reading notes on the papers behind the project.
- `results/`: the output CSVs from the experiment runs, with one subfolder per run (`baseline_every_slice/`, `every_10/`, `fixed_5/`, `fixed_15/`, `fixed_20/`). Each subfolder has per-slice run files (`run_*` / `every_slice_*` / `every_10_slices_*`) and per-scan summary files (`summary_*`).
