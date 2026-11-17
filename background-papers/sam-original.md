# Segment Anything (SAM)

**Authors:** Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C. Berg, Wan-Yen Lo, Piotr Dollár, Ross Girshick (Meta AI Research, FAIR)
**arXiv:** [2304.02643](https://arxiv.org/abs/2304.02643) · ICCV 2023
**Type:** Foundation model — the origin paper

---

## TL;DR

Meta set out to build a *foundation model for segmentation* — one model that segments anything, in any image, without task-specific training. They did it by co-designing three things: a **task** (promptable segmentation), a **model** (SAM), and a **data engine** that bootstrapped the largest segmentation dataset ever made (1.1 billion masks).

The core idea: don't train a model per dataset. Train one model that takes a *prompt* — a click, a box — and returns a valid mask, then transfer it zero-shot to new domains.

## Why this paper is on your reading list

Everything downstream is a modification of this. To understand where LoRA attaches in [SAMed](samed-lora-customized-sam.md), you need to know the image encoder is a ViT with standard attention blocks containing q/k/v projections. To understand why medical adaptation is needed at all, you need SAM's own limitations section.

---

## 1. The three components

The paper is structured around a task, a model, and a dataset — each enabling the others.

### Task: promptable segmentation

Given **any** prompt, return a **valid** mask. "Valid" means: even if the prompt is ambiguous, output something reasonable.

Ambiguity is the subtle part. Click on a person's shirt — do you mean the shirt, the person, or the crowd? SAM's answer:

- Predict **3 masks** per prompt (roughly: whole, part, subpart)
- During training, backpropagate **only the lowest-loss** prediction ("minimum loss" trick), so the heads specialise rather than averaging into mush
- Predict a **confidence score** (estimated IoU) to rank them

### Model: SAM

Three parts, designed so the expensive one runs only once:

| Component | Design |
|---|---|
| **Image encoder** | MAE-pretrained **ViT-H/16**. Takes 1024×1024, outputs a 64×64 embedding. Heavy — runs **once per image**. |
| **Prompt encoder** | *Sparse* (points/boxes → positional encodings + learned embeddings; text → CLIP encoder). *Dense* (masks → convolutions, added elementwise to the image embedding). |
| **Mask decoder** | Lightweight two-way transformer: token self-attention, bidirectional cross-attention (tokens ↔ image), point-wise MLPs, transposed-conv upsampling. |

**The key engineering decision:** encode the image once, then each new prompt costs only ~50 ms — fast enough to run in a **web browser on CPU**. That's what makes interactive clicking feel instant.

### Data engine: three stages

There was no web-scale segmentation dataset, so they built one with the model in the loop.

| Stage | Method | Output |
|---|---|---|
| **1. Assisted-manual** | Annotators click with SAM assisting; model retrained 6×. Time per mask fell 34s → 14s. | 4.3M masks / 120k images |
| **2. Semi-automatic** | SAM auto-fills confident masks; annotators only label what's missing, boosting diversity. | +5.9M masks (10.2M total) |
| **3. Fully automatic** | Prompt SAM with a 32×32 grid of points, filter by confidence and stability, dedupe with NMS. | **1.1B masks / 11M images** |

A genuinely elegant bootstrap: each better model makes annotation cheaper, which makes the next model better.

## 2. SA-1B dataset

| Metric | Value |
|---|---|
| Images | 11 million (licensed, privacy-protected) |
| Masks | 1.1 billion |
| Automatically generated | 99.1% |
| Avg masks per image | ~100 |
| vs Open Images | 11× images, 400× masks |

**Quality check:** 94% of automatic masks reach >90% IoU against professional corrections; 97% exceed 75%. That's *better than typical inter-annotator agreement* (85–91%) in earlier datasets.

**Geography:** Europe 49.8%, Asia 36.2%, with acknowledged underrepresentation of Africa and low-income countries.

## 3. Zero-shot transfer results

| Task | Result |
|---|---|
| **Single-point mask** | Beats RITM on 16/23 datasets; human raters score it 7–9/10 |
| **Edge detection** (BSDS500) | Competitive with HED; high recall (0.928), lower precision |
| **Object proposals** (LVIS) | AR@1000 59.3 vs ViTDet's 63.0; **wins on medium/large/rare objects** |
| **Instance segmentation** | COCO 46.5 AP vs ViTDet 51.0 — but **humans rate SAM's masks higher** |
| **Text-to-mask** | Proof of concept only |

That instance-segmentation result is worth pausing on: SAM scores *lower* on the metric but *higher* with human judges. The authors argue ViTDet exploits dataset-specific annotation biases that AP rewards. A useful reminder that benchmark numbers and actual quality can diverge.

### Ablations

- Each data-engine stage helps; automatic-only training nearly matches all three combined
- **1M images (~10% of SA-1B) gives comparable results** — scale saturates earlier than expected
- ViT-H clearly beats ViT-B; ViT-L → ViT-H gains only ~0.5 mIoU

## 4. Limitations (SAM's own list)

1. Misses fine structures and small disconnected components
2. Boundaries less crisp than zoom-in-based methods
3. Dedicated interactive segmenters beat it when given many points
4. Image encoder too heavy for true end-to-end real-time
5. Text prompting is exploratory, not robust
6. **No clear way to do semantic or panoptic segmentation** ← precisely what SAMed fixes
7. **Domain-specific tools outperform it in narrow domains** ← precisely why medical adaptation exists

---

## Terms you'll need

- **ViT-H/16** — Vision Transformer, "Huge", 16×16 patches. The `/16` is patch size.
- **MAE pretraining** — Masked Autoencoder: hide image patches, train to reconstruct. Self-supervised, no labels.
- **Zero-shot transfer** — applying to a new task/domain with no training on it.
- **mIoU / AR@1000 / AP** — mean intersection-over-union; average recall at 1000 proposals; average precision.

## Questions to hold while reading

- SAM decides boundaries largely by **intensity variance**. SAMed argues that's exactly why it fails on medical images, where boundaries are defined anatomically rather than by contrast. Do you buy that explanation?
- The multi-mask ambiguity design is central to SAM — yet SAMed throws it away for deterministic per-class output. What's lost?
- Data-engine performance saturated at ~1M images. Does that suggest medical adaptation needs far less data than people assume?
