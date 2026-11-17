# Cheap Lunch for Medical Image Segmentation by Fine-tuning SAM on Few Exemplars

**Authors:** Weijia Feng, Lingting Zhu, Lequan Yu (University of Hong Kong)
**arXiv:** [2308.14133](https://arxiv.org/abs/2308.14133) · 27 August 2023
**Venue:** MICCAI BrainLes workshop 2023 · 10 pages, 3 figures
**Type:** Method paper — few-shot PEFT

---

## TL;DR

Hospitals can't label thousands of scans. So: can you adapt SAM with a *handful* of labelled examples? This paper combines two cheap tricks — **synthesising extra training data from the few exemplars you have** (data level) and **LoRA** (model level) — and shows you can hit competitive Dice with as little as 1% of the labels.

The interesting result is that the **data synthesis mattered more than the LoRA did**.

## Why this paper is on your reading list

It's the closest thing to a **direct template for a small-scale project**, which is what you have. It also demonstrates something methodologically important: when you're comparing adaptation methods, a data-level trick can dominate the model-level one. That's a warning about attributing your gains to the wrong cause — and an easy ablation to run.

---

## 1. Motivation — why few-shot matters

SAM's zero-shot ability is impressive but insufficient in medicine, so you fine-tune. But fine-tuning needs labelled data, and in hospitals labelling is "time-consuming, labor-intensive, and expensive" — a radiologist tracing organ boundaries by hand.

The research question: **can a foundation model be adapted effectively with a minimal number of labelled exemplars?**

## 2. Method part 1 — exemplar-guided synthesis

**Important: there is no generative model here.** Despite "synthesis" sounding like diffusion or GANs, this is classical image manipulation:

1. Crop the organ/tumour region out of your few labelled exemplars
2. Apply transformations: **blur, intensity variation, scale, flip, rotation**
3. Paste the transformed crop onto a background image that contains no target structure
4. The pasted location gives you a free ground-truth mask

For multi-organ tasks (Synapse) it's done **category-wise** — each organ transformed separately, then merged onto a shared background.

Cheap, simple, and — per the ablations — the biggest single contributor to performance.

## 3. Method part 2 — LoRA fine-tuning

**Where it's applied:** the image encoder (ViT-Base) **and** the mask decoder with point-based prompt embedding.

> This differs from [SAMed](samed-lora-customized-sam.md), which trains the mask decoder fully rather than LoRA-ing it.

**Configuration:**

- Rank **r = 4**
- **6.32M trainable parameters = 1.77% of SAM ViT-B**
- All pretrained weights frozen; only the low-rank matrices train
- **Loss:** `L = λ₁·CE + λ₂·Dice` with λ₁ = 1, λ₂ = 0.8
- **Optimizer:** AdamW, lr = 0.001, with warmup and exponential decay
- **Training detail:** each iteration samples a random point from one randomly-selected class

## 4. Datasets and how "few" was defined

**BraTS 2018** — 285 MRI scans, 80/20 split, whole-tumour segmentation on the FLAIR sequence. Exemplars randomly drawn at **0.5%, 1%, and 3%** of the labelled data.

**Synapse** — 30 CT scans (18 train / 12 test), 8 organs. Exemplars chosen by organ coverage: **9, 18, or 36** exemplars, selected from the volumes showing the most organs.

Note the different selection strategies — random for the single-class task, coverage-driven for multi-organ. Sensible, and worth copying.

## 5. Results

**BraTS, 1% exemplars (≈150 slices):**

| Method | DSC | HD95 |
|---|---|---|
| This paper | **83.4** | **10.03** |
| SAMed | 82.5 | 43.99 |

Dice is barely different, but **HD95 is dramatically better** (10.03 vs 43.99). Hausdorff distance measures worst-case boundary error — so SAMed was producing occasional wild boundary excursions that Dice largely hides. **Good reminder to report more than Dice.**

**Synapse, 18 exemplars:** DSC 83.04, HD95 16.84 — an **8.5% DSC improvement over zero-shot SAM**.

## 6. The ablation that matters

**BraTS at 0.5% exemplars:**

| Setup | DSC |
|---|---|
| Without synthesis | 68.14 |
| With synthesis | **82.78** |

Nearly **15 Dice points from data synthesis alone.** The authors conclude synthesis "plays a vital role."

Read that carefully: in the very-low-data regime, the data-level trick contributed more than the choice of adaptation method. If you're benchmarking PEFT variants, you need to hold data augmentation fixed or you'll be measuring the wrong thing.

## 7. Limitations

The authors are upfront: the method **requires point prompts at test time**, so "this method is not an automatic solution to medical image segmentation." Prompts matter more for multi-organ than for single-tumour tasks.

Future work they suggest: combining small labelled exemplar sets with large *unlabelled* datasets — i.e. semi-supervised adaptation, which they call an open problem.

---

## Questions to hold while reading

- Their synthesis is pure copy-paste augmentation. Would a modern generative model beat it — or is that overkill for the gain?
- They LoRA the mask decoder; SAMed trains it fully. **Nobody has cleanly compared those two choices at matched parameter budgets.** That's a tractable, well-scoped experiment.
- The HD95 gap versus SAMed is large and under-discussed. Why does SAMed produce such bad worst-case boundaries? Reproducing and explaining that would be a solid contribution.
- Both this and SAMed use **rank 4**. Is that genuinely optimal for medical data, or has everyone just copied the original LoRA paper's default?
