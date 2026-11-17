# Fall 2026 Research: Box-Prompt Strategies for SAM 2 Pancreas Segmentation

# 2026-11-17

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
  1. **Interval re-prompting:** a new box is placed at a fixed spacing, such as every 10th, 15th, 20th, etc pancreas slice, whatever the mask looks like.
  2. **Confidence-based re-prompting:** a new box is placed only when SAM 2's confidence in a slice falls below a set threshold (below 0.8, 0.65, 0.9, etc).

### Experiments

- Starting from 2026-11-11 the experiments that I have ran are:

1. **Baseline: box on every pancreas slice.** Each slice is segmented from its own box. This is the upper bound on prompt effort.
2. **Every 10: re-prompt every 10th pancreas slice.** One box is placed on the largest pancreas slice. The mask is propagated forward and backward. A new box is placed every 10th pancreas slice, and the propagation restarts from there.

The runner is `seed44_prompt_strategy_test/runner.py`.

## Results

### Baseline (150 scans, complete)

| Part    | Scans   | Avg Dice       | Avg boxes per scan |
| ------- | ------- | -------------- | ------------------ |
| 1       | 50      | 0.768          | 31.3               |
| 2       | 50      | 0.758          | about 33           |
| 3       | 50      | 0.737          | 31.1               |
| **All** | **150** | **about 0.75** | **about 32**       |

### Every 10 (work in progress, about 66% complete)

Parts 1 and 2 are done (100 of 150 scans). Part 3 is still to run.

| Part | Scans | Avg Dice | Avg boxes per scan |
| ---- | ----- | -------- | ------------------ |
| 1    | 50    | 0.671    | 3.0                |
| 2    | 50    | 0.641    | 3.3                |
| 3    | 50    | pending  | pending            |

Compared with the baseline on the same scans, every-10 uses about a tenth of the boxes but loses roughly 0.10 Dice.

**What the detail shows:** Most of the loss comes from stretches of slices where the mask collapses after a box is placed. In several scans, the slices right after a re-prompt drop to near zero Dice, even when the box itself looks correct. Examples are `pancreas_186` (slices 14–19) and `pancreas_265` (slices 38–46). This is the main thing to investigate next.

### Next experiments (planned)

Each experiment will run on the same 150 scans, split into the same three parts.

#### Interval re-prompting

| Interval | Re-prompt every     | Status  |
| -------- | ------------------- | ------- |
| Every 5  | 5th pancreas slice  | planned |
| Every 15 | 15th pancreas slice | planned |
| Every 20 | 20th pancreas slice | planned |

- Including the baseline and currently running every 10th slice experiment, there will be 5 total variations.

#### Confidence-based re-prompting

| Threshold | Role                                  | Status  |
| --------- | ------------------------------------- | ------- |
| 0.5       | fewest re-prompts                     | planned |
| 0.65      | low variation                         | planned |
| 0.8       | **baseline** (tested on pancreas_001) | planned |
| 0.85      | high variation                        | planned |
| 0.9       | most re-prompts                       | planned |

## Question for the professor

- Just to confirm, we will use the same 150 scans as the interval runs for the confidence threshhold run?

## Repository layout

- `README.md`: this overview.
- `seed44_prompt_strategy_test/runner.py`: runs the box-prompt strategies on SAM 2 (the experiment code).
- `seed44_prompt_strategy_test/kaggle/runner.py`: the same runner, set up for Kaggle GPUs. It can split the 150 scans into three parts (`PART`), so each run fits in one Kaggle session.
- `seed44_prompt_strategy_test/pick_scans.py`: picks the random sample of scans with seed 44.
- `seed44_prompt_strategy_test/sample_seed44_n150.txt`: the 150 scans used in every run.
- `pilot/pilot_confidence.py`: the setup code (model loading, data loading, Dice) that `runner.py` reuses.
- `background-papers/`: reading notes on the papers behind the project.
