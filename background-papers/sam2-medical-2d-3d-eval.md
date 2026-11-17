# SAM 2: An Application to 2D and 3D Medical Images

**Authors:** Haoyu Dong, Hanxue Gu, Yaqian Chen, Jichen Yang, Yuwen Chen, Maciej A. Mazurowski (Duke University)
**arXiv:** [2408.00756](https://arxiv.org/abs/2408.00756) · Submitted 1 August 2024 · 20 pages, 13 figures
**Type:** Large-scale empirical evaluation (not a new method)

---

## TL;DR

SAM 2 came out and people immediately noticed that a 3D medical scan (CT/MRI) is basically a stack of 2D slices, which looks a lot like a video. So: can you prompt one slice and let SAM 2's video memory propagate the mask through the whole volume?

This paper tests that across 21 medical datasets. The answer is **no, not yet**. 2D slice-by-slice works about as well as the original SAM, but 3D propagation collapses — the best average result is about **0.19 IoU**. The paper's value is that it explains *why* it fails and catalogues exactly which design choices matter.

## Why this paper is on your reading list

This is the **motivation/gap paper** for anything involving SAM 2 in medicine. It ends by explicitly stating that "further fine-tuning and redesigning of the memory strategy are needed" — which is the door your PEFT project walks through. Expect to cite this in your literature review as evidence that out-of-the-box SAM 2 is insufficient.

---

## 1. Motivation

SAM was 2D-only, which is a real limitation in medicine where CT/MRI/PET are volumetric. SAM 2 added video, and video ≈ a sequence of related frames ≈ a stack of slices. The research questions:

- Does SAM 2 beat SAM on single 2D medical slices?
- Can SAM 2's memory propagate a mask through a 3D volume?
- What controls whether propagation succeeds or fails?

## 2. Datasets

21 datasets spanning surgical video, CT, MRI, PET, X-ray, and ultrasound. The enumerated tables cover:

**2D (6):** X-ray chest, X-ray hip, ultrasound breast, ultrasound kidney, ultrasound muscle, ultrasound ovarian tumour
**3D (12):** MRI spine/heart/prostate/brain (BraTS)/breast; CT colon/hepatic-vessel/pancreas/spleen/liver (LiTS)/organ; PET whole-body

> Note: the abstract says 21 datasets; the tables I could extract enumerate 18 (the remainder are the surgical video sets). Worth confirming when you read it.

## 3. The two evaluation settings

### Setting 1 — Single-frame 2D
Every slice gets its own prompt, no propagation. Four prompting modes:

| Mode | Prompt |
|---|---|
| P1 | One point at centre of largest connected region |
| P2 | One point per region (max 3) |
| P3 | One box around largest region |
| P4 | One box per region (max 3) |

### Setting 2 — Multi-frame 3D
Prompt a few slices, propagate to the rest. Three axes of variation:

- **Which slice to annotate (F-mode):** slice with largest object / centre slice / 3 uniformly spaced slices
- **Propagation direction (D-mode):** front-to-end, or bidirectional from the annotated slice
- **Which predicted mask to keep:** SAM 2's highest-confidence output, or simply the first output channel

## 4. Results

### 2D is fine
SAM 2 ≈ SAM on single slices. Box prompts beat point prompts consistently — same as in natural images. Range is wide and dataset-dependent: **0.907 IoU** (X-ray hip, ilium) down to **0.277 IoU** (MRI spine, gray matter).

### 3D falls apart

| Slice choice | Propagation | Avg DSC |
|---|---|---|
| Largest slice | Front-to-end | ~6% |
| Centre slice | Front-to-end | ~14% |
| Largest slice | Bidirectional | ~8% |
| Centre slice | Bidirectional | ~19% |

Best overall 3D result: **0.19 IoU**.

Crucially, the *prompted* slice itself scores ~41% IoU — consistent with the 2D results. So the model isn't failing at segmentation; it's failing at **propagation**. Errors accumulate slice by slice.

## 5. What actually affects propagation

1. **Annotate the centre slice, not the largest** — worth roughly 8% DSC.
2. **Bidirectional beats front-to-end** by ~2–5% IoU. Propagating outward from the prompt means adjacent (most similar) slices enter the memory bank first.
3. **Take the first output channel, not the highest-confidence one.** This is counterintuitive but worth up to 17% DSC. SAM 2's confidence scores are miscalibrated on medical data.
4. **More prompts barely help.** 1 point vs 3 points: negligible. 3 annotated slices vs 1 centre slice: ~2%.

## 6. Why it fails — the two structural reasons

1. **Memory window mismatch.** SAM 2 was trained with a memory window of **8 frames**. Medical volumes have tens to hundreds of slices. Memory attention stops being useful well before the volume ends.
2. **Slices aren't video frames.** Consecutive video frames are nearly identical (30 fps). Consecutive CT/MRI slices can change a lot — anatomy appears and disappears. The temporal-smoothness assumption baked into SAM 2 doesn't hold.

## 7. Limitations and stated future work

The authors are blunt: **"direct usage of SAM 2 for 3D medical imaging segmentation tasks is not yet feasible. Further fine-tuning and redesigning of the memory strategy are needed."**

They don't propose a fix — that's left open, and it's part of the space your course is working in.

---

## Questions to hold while reading

- If propagation is the bottleneck and not per-slice segmentation, does adapting the *image encoder* with LoRA even address the real problem? (Possible argument for adapting the memory module instead — that would be a novel angle.)
- Their "first channel beats highest confidence" finding suggests SAM 2's IoU-prediction head is badly calibrated on medical images. Could that head be a cheap PEFT target?
- How much of the 3D failure would disappear if you just prompted every 10th slice? (The BraTS benchmarking paper effectively tests this — see [brats-benchmarking-prompt-quality.md](brats-benchmarking-prompt-quality.md).)
