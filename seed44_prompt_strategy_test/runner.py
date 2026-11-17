"""
One runner for the three prompting strategies, on the scans listed in sample_seed44.txt.

Usage:
    python runner.py every_slice            # box on every pancreas slice (best case)
    python runner.py fixed <N>              # new box every N slices (next pancreas slice after each mark)
    python runner.py confidence <threshold> # new box when SAM 2 confidence drops below threshold

Env var LIMIT=<n> runs only the first n scans from the list (for quick tests).
Writes run_<tag>.csv (one row per slice) and summary_<tag>.csv (one row per scan).
Prompts = number of boxes given to SAM 2 for that scan.
"""
import os
import sys
import csv

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PILOT = r"C:\Users\AdamR\Desktop\Fall 2026 Research\pilot"  # pilot_confidence.py and frames live here
SEED = 44

# Reuse the setup, loaders, and metrics from the first pilot (same trick as trigger_pilot.py).
sys.path.insert(0, PILOT)
saved_argv = list(sys.argv)
sys.argv = [saved_argv[0], "1"]
import pilot_confidence as pc
sys.argv = saved_argv

MODE = sys.argv[1]
PARAM = float(sys.argv[2]) if len(sys.argv) > 2 else None
if MODE not in ("every_slice", "fixed", "confidence"):
    sys.exit("MODE must be every_slice, fixed, or confidence")
if MODE != "every_slice" and PARAM is None:
    sys.exit("fixed needs N and confidence needs a threshold")
TAG = MODE if PARAM is None else f"{MODE}_{PARAM:g}"


def box_of(gt, z):
    ys, xs = np.nonzero(gt[:, :, z].T)
    return np.array([xs.min(), ys.min(), xs.max(), ys.max()], dtype=np.float32)


def run_every_slice(frames_dir, gt):
    """One fresh prompt per pancreas slice. Each slice is segmented from its own box only."""
    results = {}
    zs = np.nonzero(gt.sum(axis=(0, 1)))[0]
    if os.environ.get("MAX_SLICES"):
        zs = zs[: int(os.environ["MAX_SLICES"])]
    for z in zs:
        pc.iou_buffer.clear()
        with pc.torch.inference_mode():
            state = pc.predictor.init_state(video_path=frames_dir)
            _f, _ids, masks = pc.predictor.add_new_points_or_box(
                inference_state=state, frame_idx=int(z), obj_id=1, box=box_of(gt, z)
            )
        iou = pc.iou_buffer[-1] if pc.iou_buffer else float("nan")
        results[int(z)] = ((masks[0, 0] > 0).cpu().numpy(), iou)
    return results, len(results)


def pancreas_steps(has, start, f):
    """Pancreas slices passed going from start to f (f counts, start does not)."""
    if f > start:
        return int(has[start + 1:f + 1].sum())
    return int(has[f:start].sum())


def propagate_direction(frames_dir, gt, start, reverse):
    """Propagate from `start` one way. Re-prompt (restart with a new box) as the mode says.
    Returns (results, reprompts)."""
    results = {}
    reprompts = 0
    has = gt.sum(axis=(0, 1)) > 0  # which slices contain pancreas
    pancreas_idx = np.nonzero(has)[0]
    first, last = int(pancreas_idx[0]), int(pancreas_idx[-1])  # propagation stays inside the organ's extent
    next_mark = PARAM if MODE == "fixed" else None  # next every-N mark, counted in pancreas slices from start
    seg_start = start
    while True:
        broke_at = None
        pc.iou_buffer.clear()
        with pc.torch.inference_mode():
            state = pc.predictor.init_state(video_path=frames_dir)
            pc.predictor.add_new_points_or_box(
                inference_state=state, frame_idx=seg_start, obj_id=1, box=box_of(gt, seg_start)
            )
            pc.iou_buffer.clear()
            for f, _ids, masks in pc.predictor.propagate_in_video(
                state, start_frame_idx=seg_start, reverse=reverse
            ):
                iou = pc.iou_buffer[-1] if pc.iou_buffer else float("nan")
                pc.iou_buffer.clear()
                if (reverse and f < first) or (not reverse and f > last):
                    break  # past the last/first pancreas slice
                if f != seg_start and gt[:, :, f].sum() > 0:
                    if MODE == "confidence" and iou < PARAM:
                        broke_at = f
                        break
                    if MODE == "fixed" and pancreas_steps(has, start, f) >= next_mark:
                        broke_at = f
                        next_mark += PARAM
                        break
                results[f] = ((masks[0, 0] > 0).cpu().numpy(), iou)
        if broke_at is None:
            break
        reprompts += 1
        seg_start = broke_at
    return results, reprompts


def main():
    sample_name = os.environ.get("SAMPLE_FILE", f"sample_seed{SEED}.txt")
    listed = [l.strip() for l in open(os.path.join(HERE, sample_name)) if l.strip()]
    if os.environ.get("LIMIT"):
        listed = listed[: int(os.environ["LIMIT"])]
    suffix = ""
    if os.environ.get("PART"):  # e.g. PART=2/3 runs only the second third of the list
        i, n = (int(x) for x in os.environ["PART"].split("/"))
        listed = np.array_split(np.array(listed), n)[i - 1].tolist()
        suffix = f"_part{i}of{n}"

    rows, summary = [], []
    for case in listed:
        img, gt = pc.load_volume(case + ".nii.gz")
        prompt_slice = int(np.argmax(gt.sum(axis=(0, 1))))
        frames_dir = os.path.join(pc.OUT, "frames", case)
        if not os.path.isdir(frames_dir):
            pc.write_frames(img, frames_dir)

        print(f"[{case}] {TAG}: running", flush=True)
        if MODE == "every_slice":
            merged, prompts = run_every_slice(frames_dir, gt)
        else:
            fwd, rp_f = propagate_direction(frames_dir, gt, prompt_slice, reverse=False)
            bwd, rp_b = propagate_direction(frames_dir, gt, prompt_slice, reverse=True)
            merged = {**bwd, **fwd}
            prompts = 1 + rp_f + rp_b  # the first box is shared by both directions

        overlap = pred_total = 0
        for z in sorted(merged):
            pred, iou = merged[z]
            gt_slice = gt[:, :, z].T
            if gt_slice.sum() == 0:
                continue  # score only pancreas slices (Zia's rule)
            overlap += np.logical_and(pred, gt_slice).sum()
            pred_total += pred.sum()
            rows.append({"case": case, "slice": z, "conf": iou, "dice": pc.dice(pred, gt_slice),
                         "prompts_for_case": prompts})
        # Dice over the pancreas slices (prof's method): 2 * overlap / (all predicted + all ground truth)
        denom = pred_total + gt.sum()
        volume_d = float(2 * overlap / denom) if denom > 0 else float("nan")
        summary.append({"case": case, "prompts": prompts, "volume_dice": volume_d})
        print(f"[{case}] boxes used: {prompts}, volume dice {volume_d:.3f}", flush=True)

    run_path = os.path.join(HERE, f"run_{TAG}{suffix}.csv")
    with open(run_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    sum_path = os.path.join(HERE, f"summary_{TAG}{suffix}.csv")
    with open(sum_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)

    avg_prompts = np.mean([s["prompts"] for s in summary])
    avg_dice = np.mean([s["volume_dice"] for s in summary])
    print(f"SUMMARY {TAG}: {len(summary)} scans, avg boxes per scan {avg_prompts:.1f}, "
          f"avg volume dice {avg_dice:.3f}")
    print("Wrote", run_path, "and", sum_path)


if __name__ == "__main__":
    main()
