# SAMed: Customized Segment Anything Model for Medical Image Segmentation

**Authors:** Kaidong Zhang, Dong Liu (USTC)
**arXiv:** [2304.13785v2](https://arxiv.org/abs/2304.13785v2) · v1 26 Apr 2023, v2 17 Oct 2023 · Technical report, 14 pages
**Code:** https://github.com/hitachinsk/SAMed
**Type:** Method paper — **this is the core PEFT paper of your reading list**

---

## TL;DR

Take SAM, freeze the image encoder, bolt LoRA onto only the **query and value** projections of its attention blocks, and train that plus the prompt encoder's default embedding and the mask decoder. Result: 81.88 Dice on Synapse multi-organ CT, competitive with fully-trained specialist architectures, while updating only **5.25%** of the parameters.

**If you read only one paper before starting your project, read this one.** It is the template.

## Why this paper is on your reading list

This is the canonical "LoRA applied to SAM for medical imaging" paper. Everything you build will either follow it or deliberately deviate from it. Its **ablation studies are the most valuable part** — they tell you which design choices work and, more importantly, which ones fail catastrophically. That knowledge is worth weeks of your own trial and error.

---

## 1. Motivation — what's wrong with vanilla SAM here

Two distinct problems:

1. **SAM segments by intensity variance.** That's a reasonable heuristic for natural images, where objects have visible edges. Medical images need decisions based on *anatomical or pathological structure*, which doesn't reduce to local contrast.
2. **SAM can't do semantic segmentation.** It outputs "a mask," not "this mask is the liver." Medical work usually needs labelled organ classes.

SAMed addresses both.

## 2. Architecture — what's frozen, what's trained

| Component | Treatment |
|---|---|
| Image encoder (ViT) | **Frozen**, with LoRA bypasses added |
| Prompt encoder | Trainable (the default embedding only) |
| Mask decoder | Trainable (all parameters) |

### Where the LoRA goes

Applied to the **q and v projection layers** in the transformer blocks — *not* k, *not* the output projection:

```
Q̂ = W_q · F + B_q · A_q · F
V̂ = W_v · F + B_v · A_v · F
```

`W` stays frozen; `A` and `B` are the small trainable low-rank matrices with rank `r`.

**Rank = 4.**

**Parameter count:** 18.81M trainable = **5.25% of the 358M original**. (They also tried LoRA on both encoder and decoder, giving a smaller 6.32M — but it performed worse.)

## 3. How it does semantic segmentation without prompts

Neat trick worth understanding. SAM already has a "default embedding" it uses when no prompt is given. SAMed **keeps that embedding and makes it trainable.** So at inference you provide no prompt at all — the learned default embedding stands in for one.

The mask decoder then predicts **k masks, one per class** (including background), and the final prediction is:

```
Ŝ = argmax(Softmax(Ŝ_l, dim=-1), dim=-1)
```

This deliberately abandons SAM's multi-mask ambiguity handling — in medical semantic segmentation you want one deterministic answer per class, not three plausible alternatives.

> Contrast with [cheap-lunch-few-exemplars-lora.md](cheap-lunch-few-exemplars-lora.md), which *keeps* point prompts at test time. That's a real design fork you'll have to pick a side on.

## 4. Training details (copy these as your starting point)

- **Loss:** `L = 0.2·CrossEntropy + 0.8·Dice`
- **Warmup:** 250 iterations at lr 0.005, then exponential decay
- **Optimizer:** AdamW, β₁=0.9, β₂=0.999, weight decay 0.1
- **Duration:** 200 epochs scheduled (18,600 iters); early stopped at 160 epochs (14,880 iters)
- **Resolution:** 224×224 CT slices upsampled to 512×512 in; logits out at 128×128

## 5. Results — Synapse multi-organ CT

**Overall: DSC 81.88, HD 20.64**

| Organ | SAMed | DAE-Former (SOTA) |
|---|---|---|
| Aorta | 87.77 | 88.96 |
| Gallbladder | 69.11 | 72.30 |
| Kidney (L) | 80.45 | 86.08 |
| Kidney (R) | 79.95 | 80.88 |
| Liver | 94.80 | 94.98 |
| **Pancreas** | **72.17** | 65.12 |
| Spleen | 88.72 | 91.94 |
| **Stomach** | **82.06** | 79.19 |

SAMed wins on pancreas and stomach; overall it's "on par with" rather than beating SOTA (DAE-Former 82.43). The authors are honest about this.

## 6. Ablations — the most important section

### LoRA rank

| Rank | DSC |
|---|---|
| 1 | 78.26 (underfits) |
| **4** | **81.88** |
| 16 | 69.03 (**catastrophic**) |

Bigger rank is *not* better. This is counterintuitive and worth internalising — more trainable capacity actively hurt.

### Which projections get LoRA

| Config | DSC |
|---|---|
| Q only | 73.76 |
| **Q + V** | **81.88** |
| Q + K + V + O | 50.93 (**collapse**) |

Adapting everything destroys the model.

### Training strategies

| Config | DSC |
|---|---|
| Neither | 56.54 |
| + Warmup | 75.29 |
| + Warmup + AdamW | 81.88 |

**Warmup alone is worth ~19 Dice points.** Skip it and you'll conclude LoRA doesn't work, when actually your optimizer just diverged early. This is exactly the kind of silent failure that eats weeks.

## 7. Deployment argument

Because only ~5% of parameters change, SAMed ships as a **plugin**: keep the original frozen SAM, distribute only the small delta. Storage and deployment cost are near-zero on top of a SAM install you already have. This is the practical selling point of PEFT in general.

## 8. Limitations

- 81.88 DSC is competitive but not best-in-class
- **Only evaluated on Synapse** — no cross-dataset generalization evidence
- No discussion of inference speed
- No 3D volume handling

---

## Questions to hold while reading

- The rank-16 collapse and the Q+K+V+O collapse are strange. Is that a real property of LoRA-on-SAM, or an artifact of their fixed learning rate? **Nobody appears to have checked. That could be your project.**
- SAMed is built on **SAM**, not SAM 2. Porting it to SAM 2's Hiera encoder is non-trivial (hierarchical, multi-scale, different attention structure) — and is a legitimate contribution on its own.
- They train the whole mask decoder. Would LoRA-ing the decoder instead be cheaper at equal accuracy? (Cheap Lunch tries this and gets 6.32M params — compare their numbers.)
