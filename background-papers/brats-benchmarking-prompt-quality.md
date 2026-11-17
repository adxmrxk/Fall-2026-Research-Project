# AI-Driven MRI-based Brain Tumour Segmentation Benchmarking

**Authors:** Connor Ludwig, Khashayar Namdar, Farzad Khalvati (University of Toronto)
**arXiv:** [2506.20786](https://arxiv.org/abs/2506.20786) · Submitted 25 June 2025
**Code:** https://github.com/IMICSLab/Brain_Tumor_Segmentation
**Type:** Benchmark / comparative evaluation

---

## TL;DR

Six models, one dataset, three grades of prompt quality, measured both zero-shot and after fine-tuning. The headline: **SAM and SAM 2 can beat nnU-Net — but only when handed near-perfect bounding boxes, which nobody has in a real clinic.** nnU-Net needs no prompt at all, and that's why it still wins in practice.

This is the most *directly useful* paper of the four for you: it hands you baseline numbers, a fine-tuning protocol, and compute costs.

## Why this paper is on your reading list

1. **It gives you the numbers your project will be compared against.** If you LoRA-adapt SAM 2 on brain tumours, these are your baselines.
2. **It fine-tunes SAM 2 fully — and reports how long that took.** Your PEFT angle has an obvious framing: *can LoRA match these fine-tuning gains at a fraction of the trainable parameters and compute?* That's a legitimate, well-scoped research question, and this paper is what you'd measure against.

---

## 1. The research gap

Lots of promptable models exist, but evaluations use different datasets, prompting styles, and metrics, so you can't compare them. Also, segmentation challenges withhold test masks — which means promptable models **can't even compete**, because there's no ground truth to generate a prompt from. So the comparison had never been done cleanly.

## 2. Models and datasets

**Models:** SAM, SAM 2 (image mode), SAM 2 (video mode), MedSAM, SAM-Med3D, nnU-Net

**Datasets:** BraTS 2023 adult glioma (1,251 patients) and BraTS 2023 pediatrics (99 patients). Each patient has 4 MRI sequences (T1, T1Gd, T2, FLAIR).

**The data-leakage problem — a nice piece of methodological care.** MedSAM, SAM-Med3D and nnU-Net were all trained on BraTS adult data, so adult-glioma results are inflated for them. The pediatrics dataset was *not* in anyone's training set, so it acts as the clean comparison. **Trust the pediatrics tables.**

## 3. Preprocessing

- Volumes sliced axially → 155 slices of 240×240
- Only the **FLAIR** sequence used (SAM-Med3D takes one channel; matched across models for fairness), replicated into 3 identical channels and upscaled to 1024×1024
- nnU-Net is the exception: it got all 4 sequences at native resolution because it was pretrained that way. The authors flag this as an advantage nnU-Net has.
- **Dice computed on the reassembled 3D volume**, not per-slice — important for fair comparison

## 4. Prompt quality — how they degraded prompts

This is the clever part of the design.

**Boxes:**

| Quality | Construction |
|---|---|
| High | Tightest box that fully contains the tumour |
| Medium | +10% size, shifted randomly up to 10% of height/width |
| Low | +30% size, shifted randomly up to 20% |

**Points:** sampled randomly from the ground-truth mask — 10 points (high), 5 (medium), 1 (low).

**SAM 2 video mode:** prompt the first slice containing tumour, then every 10th slice after — so ceil(N/10) prompts instead of N.

## 5. Zero-shot results

### Adult glioma (leakage affects MedSAM, SAM-Med3D, nnU-Net)

| Model | Box High | Box Med | Box Low | 10 pts | 5 pts | 1 pt | Unprompted |
|---|---|---|---|---|---|---|---|
| SAM | 0.869 | 0.783 | 0.688 | 0.605 | 0.453 | 0.295 | — |
| SAM 2 | **0.893** | **0.876** | **0.759** | 0.474 | 0.393 | 0.285 | — |
| SAM 2 Video | 0.835 | 0.792 | 0.685 | 0.459 | 0.330 | 0.166 | — |
| MedSAM | 0.685 | 0.654 | 0.606 | — | — | 0.050 | — |
| SAM-Med3D | — | — | — | **0.815** | **0.793** | **0.730** | — |
| nnU-Net | — | — | — | — | — | — | **0.958** |

### Pediatrics (clean — no leakage)

| Model | Box High | Box Med | Box Low | 10 pts | 5 pts | 1 pt | Unprompted |
|---|---|---|---|---|---|---|---|
| SAM | 0.877 | 0.859 | **0.797** | **0.749** | 0.586 | 0.345 | — |
| SAM 2 | **0.886** | **0.862** | 0.777 | 0.623 | 0.474 | 0.322 | — |
| SAM 2 Video | 0.767 | 0.743 | 0.674 | 0.562 | 0.416 | 0.231 | — |
| MedSAM | 0.572 | 0.633 | 0.618 | — | — | 0.115 | — |
| SAM-Med3D | — | — | — | 0.814 | **0.795** | **0.723** | — |
| nnU-Net | — | — | — | — | — | — | 0.860 |

**What to read from this:**

- Once leakage is removed, nnU-Net drops 0.958 → 0.860, and SAM 2 with a good box (0.886) actually **beats it**.
- SAM 2 beats SAM on boxes; **SAM beats SAM 2 on points**. Interesting and left unexplained.
- SAM 2 video mode is consistently *worse* than image mode — consistent with the propagation failure documented in [sam2-medical-2d-3d-eval.md](sam2-medical-2d-3d-eval.md).
- MedSAM does badly, contradicting its own paper. The authors show its masks are circular blobs ignoring actual anatomy — they suspect it overfit to organ-like shapes and lost generality.

## 6. Fine-tuning experiments

**Protocol:** 4-fold cross-validation, 50% train / 25% val / 25% test. Early stopping on validation Dice, best checkpoint per fold. Each training slice randomly got either a single point or a perfect box.

### Fine-tuned on pediatrics

| Model | Box High | Box Med | Box Low | 10 pts | 5 pts | 1 pt |
|---|---|---|---|---|---|---|
| SAM | **0.894** | 0.866 | 0.777 | 0.734 | 0.717 | 0.670 |
| SAM 2 | 0.893 | **0.881** | **0.837** | **0.775** | **0.751** | **0.686** |
| SAM 2 Video | 0.782 | 0.772 | 0.737 | 0.568 | 0.543 | 0.511 |
| MedSAM | 0.626 | 0.664 | 0.626 | — | — | 0.309 |
| SAM-Med3D | — | — | — | 0.814 | 0.802 | 0.737 |

**The key finding: point prompts improve enormously, boxes barely move.**

SAM 2 single-point went 0.322 → 0.686. SAM single-point 0.345 → 0.670. Meanwhile SAM 2 high-quality box went 0.886 → 0.893 (nothing).

Two explanations offered: points started from a much worse baseline, *and* fine-tuning teaches the model to resolve ambiguity. Their Figure 5 shows it concretely — before fine-tuning, a point placed in the tumour core segments only the core; afterwards, the model understands the task is the whole tumour.

Both fine-tuned SAM and SAM 2 with high/medium boxes now exceed nnU-Net's 0.860.

## 7. Compute costs (very relevant to you)

**Inference, per patient (L4 GPU):**
SAM-Med3D 0.80s · **SAM 2 7.2s** · SAM 2 video 19s · nnU-Net 22s · SAM 72s · MedSAM 120s

**Training, per epoch (A100):**
SAM-Med3D 1.5 min · **SAM 2 7.5 min** · MedSAM 10 min · SAM 25 min

Total training: SAM 16h · MedSAM 5.3h · **SAM 2 4.7h** · SAM-Med3D 0.65h

> SAM 2 full fine-tuning took **4.7 hours on an A100**. Kaggle's P100/T4 is slower than an A100, and you get ~30 h/week. Full fine-tuning is borderline; **LoRA should be substantially cheaper — and quantifying exactly how much cheaper is a perfectly good contribution.**

## 8. Conclusions and limitations

The practical verdict: **nnU-Net remains dominant** — not because it segments better, but because promptable models need many accurate prompts per patient and nnU-Net needs none. A clinician isn't drawing 60 tight boxes per scan.

Other findings:

- Boxes beat points everywhere (less ambiguity)
- Every model is highly prompt-sensitive; ~0.1 Dice between high and low box quality
- Lower-quality prompts benefit most from fine-tuning

Limitations worth noting: nnU-Net was not fine-tuned (compute limits), it received 4 channels while others got 1, and validation Dice for the 2D models was computed per-slice rather than per-volume.

---

## Questions to hold while reading

- The prompt-quality ladder is a clean, reusable experimental design. Could you run the same ladder but vary **LoRA rank** instead of prompt quality?
- Fine-tuning helps points far more than boxes. Does that hold for LoRA too, or does full fine-tuning have an advantage there?
- Their code is public — worth reading before you write your own training loop.
