# Background Papers — CP493 Foundation Models for BioSeg

Pre-reading summaries for the PEFT (parameter-efficient adaptation) sub-topic. Each document summarises one paper section by section, with a "why this is on your reading list" note and open questions.

**These are study aids, not substitutes.** They're meant to make the first real read faster, and to help you spot which papers deserve a slow read versus a skim.

---

## Suggested reading order

| # | Paper | File | Why here |
|---|---|---|---|
| 1 | **Segment Anything (SAM)** | [sam-original.md](sam-original.md) | The origin. Architecture vocabulary everything else assumes. |
| 2 | **SAM 2** | [sam2-original.md](sam2-original.md) | The model you're adapting. Memory bank + Hiera encoder. |
| 3 | **SAM 2 on 2D/3D Medical Images** | [sam2-medical-2d-3d-eval.md](sam2-medical-2d-3d-eval.md) | Shows SAM 2 fails on 3D medical propagation. Your motivation. |
| 4 | **BraTS Benchmarking** | [brats-benchmarking-prompt-quality.md](brats-benchmarking-prompt-quality.md) | Baseline numbers, prompt-quality protocol, compute costs. |
| 5 | **SAMed** | [samed-lora-customized-sam.md](samed-lora-customized-sam.md) | **The core PEFT paper.** LoRA on SAM, with critical ablations. |
| 6 | **Cheap Lunch (Few Exemplars)** | [cheap-lunch-few-exemplars-lora.md](cheap-lunch-few-exemplars-lora.md) | Few-shot LoRA. Closest template to a small project. |

Papers 3–6 are the four your prof assigned; 1–2 are the originals they build on. Reading 1–2 first makes the other four much faster.

If you're short on time: **read 5 properly**, skim 4 for the numbers, and use 1–2 as reference.

---

## The story these six tell together

1. SAM made segmentation promptable and general (paper 1)
2. SAM 2 added video via a streaming memory, and got faster (paper 2)
3. But on 3D medical volumes, propagation collapses — best ~0.19 IoU — because memory was trained on ~8 frames and slices aren't video frames (paper 3)
4. Head-to-head, SAM/SAM 2 can beat nnU-Net *only* with near-perfect prompts; nnU-Net still wins practically because it needs none. Fine-tuning helps point prompts enormously (paper 4)
5. LoRA on SAM's frozen encoder reaches competitive Dice while training ~5% of parameters — and the ablations show rank and placement matter enormously (paper 5)
6. With data synthesis plus LoRA, ~1% of labels suffices — and the *data* trick contributed more than the PEFT method (paper 6)

**The gap they leave open:** papers 5 and 6 apply LoRA to **SAM**, not **SAM 2**. Paper 3 says SAM 2 needs fine-tuning but doesn't do it. Paper 4 fully fine-tunes SAM 2 but doesn't try PEFT. Nobody in this set has done **LoRA on SAM 2 for medical segmentation** — which is roughly where your project sits.

---

## Numbers worth remembering

| Fact | Value | Source |
|---|---|---|
| SAMed trainable parameters | 18.81M = **5.25%** of SAM | Paper 5 |
| Cheap Lunch trainable parameters | 6.32M = **1.77%** of SAM ViT-B | Paper 6 |
| LoRA rank used by both | **4** (rank 16 collapsed to 69.03 DSC) | Papers 5, 6 |
| LoRA placement | **q and v projections only** (adding k and output → 50.93 DSC) | Paper 5 |
| Warmup contribution | 56.54 → 75.29 DSC | Paper 5 |
| SAM 2 full fine-tune cost | **4.7 h on an A100** | Paper 4 |
| SAM 2 memory bank size | ~6 recent frames | Paper 2 |
| SAM 2 3D medical propagation | **~0.19 IoU** (best) | Paper 3 |
| nnU-Net on clean pediatrics data | 0.860 Dice, unprompted | Paper 4 |

---

## Open questions across the set

These came out of the summaries and are candidate directions for the Week 4 research proposal — worth raising in a supervisory meeting rather than committing to unilaterally:

1. **Port SAMed's recipe to SAM 2.** Hiera is hierarchical and multi-scale; SAM's plain-ViT recipe doesn't transfer unchanged. Which stages should get LoRA?
2. **Adapt the memory module, not the encoder.** Paper 3 shows *propagation* is what breaks on 3D volumes, not per-slice segmentation. Everyone adapts the encoder. Nobody adapts memory attention.
3. **Revisit the rank-4 default.** Both PEFT papers use rank 4 and paper 5 reports rank 16 collapsing. Is that a real property or an artifact of a fixed learning rate? Apparently unchecked.
4. **LoRA vs full fine-tuning at matched cost.** Paper 4 gives full fine-tuning numbers and timings. Direct comparison is straightforward and genuinely useful.
5. **Report HD95, not just Dice.** Paper 6 beat SAMed on HD95 by 34 points with nearly identical Dice — Dice hides catastrophic boundary errors.

---

## Two things to confirm with Dr. Zia Ud Din

1. **The syllabus says the practical work is "inference-only," but PEFT requires training.** Which does he intend — training your own adapters, or evaluating existing adapted checkpoints? This changes scope and compute substantially.
2. **Which dataset.** The Week 4 proposal must name datasets and baselines. BraTS is well-covered by these papers (with published baselines to compare against), but 3D NIfTI preprocessing is a real time cost; a 2D RGB dataset like Kvasir-SEG is far cheaper to get moving on.
