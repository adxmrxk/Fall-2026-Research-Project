"""
Pilot: does SAM 2's predicted IoU track the true per-slice Dice on MSD Task07 Pancreas?

For each CT volume:
  1. Pick the slice with the largest pancreas (label 1) and draw a box around it
     from the ground truth.
  2. Propagate SAM 2 forward and backward from that slice.
  3. For every slice, record conf (SAM 2 predicted IoU) and dice (true Dice vs GT).
Then report the Spearman correlation between conf and dice.

Usage:
    python pilot_confidence.py [num_cases]
"""
import os
import sys
import csv

import numpy as np
import nibabel as nib
from PIL import Image

REPO = r"C:\Users\AdamR\Desktop\Fall 2026 Research\sam2_repo"
DATA = r"C:\Users\AdamR\Downloads\Task07_Pancreas\Task07_Pancreas"
OUT = r"C:\Users\AdamR\Desktop\Fall 2026 Research\pilot"
CONFIG = "configs/sam2.1/sam2.1_hiera_s.yaml"
CKPT = os.path.join(REPO, "checkpoints", "sam2.1_hiera_small.pt")

# CT window for pancreas (HU): clip to this range, then scale to 0-255
HU_MIN, HU_MAX = -100, 240

NUM_CASES = int(sys.argv[1]) if len(sys.argv) > 1 else 3

# SAM2_CODE picks which copy of the SAM 2 code to run (plain repo by default,
# or the gated copy). RUN_TAG names the output file so runs don't overwrite each other.
CODE = os.environ.get("SAM2_CODE", REPO)
TAG = os.environ.get("RUN_TAG", "baseline")
sys.path.insert(0, CODE)
os.chdir(REPO)

import torch
from sam2.build_sam import build_sam2_video_predictor

# Record SAM 2's predicted IoU ("ious") by wrapping the mask decoder.
# The decoder runs once per frame, right before that frame is yielded.
iou_buffer = []
predictor = build_sam2_video_predictor(CONFIG, CKPT, device="cpu")
_orig_decoder_forward = predictor.sam_mask_decoder.forward


def _recording_forward(*args, **kwargs):
    out = _orig_decoder_forward(*args, **kwargs)
    iou_buffer.append(out[1].detach().float().max().item())  # out[1] = ious
    return out


predictor.sam_mask_decoder.forward = _recording_forward


def load_volume(name):
    img = nib.load(os.path.join(DATA, "imagesTr", name)).get_fdata()
    lab = nib.load(os.path.join(DATA, "labelsTr", name)).get_fdata()
    return img, (lab == 1)  # pancreas only


def write_frames(img, frames_dir):
    os.makedirs(frames_dir, exist_ok=True)
    scaled = np.clip(img, HU_MIN, HU_MAX)
    scaled = ((scaled - HU_MIN) / (HU_MAX - HU_MIN) * 255).astype(np.uint8)
    for z in range(img.shape[2]):
        # rows = y, cols = x, so the frame matches the (H, W) predicted masks
        Image.fromarray(scaled[:, :, z].T).convert("RGB").save(
            os.path.join(frames_dir, f"{z:05d}.jpg"), quality=95
        )


def run_direction(frames_dir, prompt_slice, box, reverse):
    """Propagate from prompt_slice in one direction. Returns {slice: (pred_mask, iou)}."""
    results = {}
    iou_buffer.clear()
    with torch.inference_mode():
        state = predictor.init_state(video_path=frames_dir)
        predictor.add_new_points_or_box(
            inference_state=state,
            frame_idx=prompt_slice,
            obj_id=1,
            box=np.array(box, dtype=np.float32),
        )
        iou_buffer.clear()  # ignore the decoder call made for the prompt itself
        for frame_idx, _obj_ids, masks in predictor.propagate_in_video(
            state, start_frame_idx=prompt_slice, reverse=reverse
        ):
            iou = iou_buffer[-1] if iou_buffer else float("nan")
            iou_buffer.clear()
            pred = (masks[0, 0] > 0).cpu().numpy()
            results[frame_idx] = (pred, iou)
    return results


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    keep = ~(np.isnan(x) | np.isnan(y))
    x, y = x[keep], y[keep]
    if len(x) < 3:
        return float("nan")
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    return float(np.corrcoef(rx, ry)[0, 1])


def dice(pred, gt):
    denom = pred.sum() + gt.sum()
    if gt.sum() == 0:
        return float("nan")  # undefined when there is no ground-truth pancreas
    return 2.0 * np.logical_and(pred, gt).sum() / denom


def main():
    os.makedirs(OUT, exist_ok=True)
    names = sorted(os.listdir(os.path.join(DATA, "imagesTr")))
    names = [n for n in names if n.endswith(".nii.gz") and not n.startswith("._")]
    # CASES (e.g. "pancreas_006,pancreas_007") picks specific scans; otherwise take the first NUM_CASES
    if os.environ.get("CASES"):
        wanted = os.environ["CASES"].split(",")
        names = [n for n in names if n.replace(".nii.gz", "") in wanted]
    else:
        names = names[:NUM_CASES]

    rows = []
    for name in names:
        case = name.replace(".nii.gz", "")
        print(f"[{case}] loading", flush=True)
        img, gt = load_volume(name)
        areas = gt.sum(axis=(0, 1))
        prompt_slice = int(np.argmax(areas))
        ys, xs = np.nonzero(gt[:, :, prompt_slice].T)
        box = [xs.min(), ys.min(), xs.max(), ys.max()]

        frames_dir = os.path.join(OUT, "frames", case)
        write_frames(img, frames_dir)

        print(f"[{case}] propagating forward from slice {prompt_slice}", flush=True)
        fwd = run_direction(frames_dir, prompt_slice, box, reverse=False)
        print(f"[{case}] propagating backward from slice {prompt_slice}", flush=True)
        bwd = run_direction(frames_dir, prompt_slice, box, reverse=True)

        merged = {**bwd, **fwd}  # forward wins on the shared prompt slice
        for z in sorted(merged):
            pred, iou = merged[z]
            gt_slice = gt[:, :, z].T
            rows.append({
                "case": case,
                "slice": z,
                "is_prompt": int(z == prompt_slice),
                "conf": iou,
                "dice": dice(pred, gt_slice),
                "gt_area": int(gt_slice.sum()),
                "pred_area": int(pred.sum()),
            })
        print(f"[{case}] done: {len(merged)} slices", flush=True)

    csv_path = os.path.join(OUT, f"pilot_results_{TAG}.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {csv_path}")

    conf = [r["conf"] for r in rows]
    dsc = [r["dice"] for r in rows]
    print(f"Pooled Spearman(conf, dice) = {spearman(conf, dsc):.3f}  "
          f"(over {sum(1 for d in dsc if not np.isnan(d))} slices with GT pancreas)")


if __name__ == "__main__":
    main()
