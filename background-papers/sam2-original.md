# SAM 2: Segment Anything in Images and Videos

**Authors:** Nikhila Ravi, Valentin Gabeur, Yuan-Ting Hu, Ronghang Hu, Chaitanya Ryali, et al. (Meta FAIR)
**arXiv:** [2408.00714](https://arxiv.org/abs/2408.00714) · v2, 28 October 2024
**Type:** Foundation model — **this is the model you're adapting**

---

## TL;DR

SAM 2 extends SAM from images to video by adding a **streaming memory**. Prompt an object on any frame, and the model tracks it through the rest of the video, remembering what it looks like. Images are handled as single-frame videos with an empty memory, so one model does both.

It's also **6× faster** than SAM on images and slightly more accurate.

## Why this paper is on your reading list

You are attaching LoRA to this architecture. You need to know which parts exist, which are heavy, and which are plausible adaptation targets. In particular: the **Hiera image encoder** is hierarchical and multi-scale, structurally different from SAM's plain ViT — which means SAMed's recipe doesn't port over unchanged. That's both a complication and an opportunity.

---

## 1. The task: Promptable Visual Segmentation (PVS)

> "Given an input video, the model can be interactively prompted with different types of inputs (including clicks, boxes, or masks) on any frame in the video, with the goal of segmenting (and tracking) a valid object throughout the video."

The workflow: prompt on any frame → immediate mask → automatic propagation to all other frames → optionally add corrective prompts on any frame to refine.

PVS generalises both SAM's static task and semi-supervised Video Object Segmentation (which only allows first-frame prompts).

## 2. Architecture

### Image encoder — Hiera
MAE-pretrained **Hiera**, a *hierarchical* ViT. Produces multi-scale features via an FPN: stride-16 and stride-32 feed memory attention, while stride-4 and stride-8 are reserved for high-resolution mask decoding. Runs once per frame.

> **This is the big structural difference from SAM.** SAM used a single-scale plain ViT-H. Hiera is hierarchical and multi-scale, so "apply LoRA to the q and v projections" needs rethinking — which blocks, at which scales?

### Memory attention
Stacked transformer blocks that condition the current frame on past frames and prompts: self-attention, then cross-attention into memory, then MLP. Uses **2D Rotary Positional Embedding (RoPE)**. Temporal position embeddings encode recency for unprompted frames (but not for prompted ones — prompts are treated as timeless).

### Memory bank
Three kinds of stored state:

- **Recent frames queue** — spatial feature maps from up to *N* recent unprompted frames (FIFO)
- **Prompted frames queue** — up to *M* explicitly prompted frames
- **Object pointers** — lightweight semantic vectors from the mask decoder's output tokens

### Memory encoder
Downsamples the predicted mask with convolutions, fuses it elementwise with the frame embedding, applies light conv layers. Output goes into the memory bank.

### Prompt encoder and mask decoder
Same design as SAM: sparse prompts via positional encodings, mask prompts via convolutions, "two-way" transformer decoder, multiple candidate masks for ambiguity.

**New: an occlusion head** predicting whether the object is visible in the current frame at all.

### Streaming
Frames processed one at a time as they arrive; each prediction updates memory. Images are just single-frame videos with an empty memory bank — which is why SAM 2 is a strict generalisation of SAM rather than a separate model.

## 3. Occlusion and reappearance

- The occlusion head detects when the object has left the frame
- Occluded frames get **special learned embeddings**, so the model distinguishes "temporarily hidden" from "gone"
- If tracking is lost, a single click on a later frame recovers it using stored memory
- **42.5%** of SA-V masklets involve disappearance/reappearance, so this is trained for, not incidental

## 4. SA-V data engine

Same bootstrap philosophy as SAM, three phases:

| Phase | Method | Speed | Yield |
|---|---|---|---|
| **1. SAM per-frame** | Manual segmentation every frame at 6 FPS | 37.8 s/frame | 16K masklets |
| **2. SAM + SAM 2 Mask** | Mask first frame, propagate, refine, re-propagate | 7.4 s/frame (**5.1×**) | 63.5K masklets |
| **3. Full SAM 2** | Clicks + memory; only occasional corrections | 4.5 s/frame (**8.4×**) | 197K masklets |

### SA-V dataset

| Metric | Value |
|---|---|
| Videos | 50.9K (196 hours, 47 countries) |
| Manual masklets | 190.9K |
| Automatic masklets | 451.7K |
| Total masks | **35.5M** (53× the largest prior VOS dataset) |
| Small objects | >88% have normalized area < 0.1 |
| Disappearance rate | 42.5% |

## 5. Results

**Interactive video segmentation:** better accuracy with **3× fewer interactions** than SAM+XMem++ / SAM+Cutie.

**Semi-supervised VOS (17 datasets):**

| Clicks | SAM 2 | SAM + XMem++ |
|---|---|---|
| 1 | **64.7** | 56.9 |
| 3 | **75.3** | 68.4 |
| 5 | **77.6** | 70.6 |

**vs state of the art:** MOSE 76.6–77.9 J&F (Cutie 71.7) · DAVIS 2017 90.2–90.7 (Cutie 88.1) · LVOS 78.0 (DEVA 55.9) · SA-V val 76.8–77.9 (prior ~60)

**Image segmentation:** 58.9 mIoU 1-click vs SAM's 58.1 across 23 datasets — rising to **61.9** when video data is included in training.

**Speed:**

| Model | FPS |
|---|---|
| SAM | 21.7 |
| SAM 2 (Hiera-B+) | **130.1** (6×) |
| SAM 2 (Hiera-L) | 30.2 |

## 6. Ablations worth knowing

- **Resolution:** 1024² clearly better than 512², at 4.5× the cost
- **Memory size:** N = 4–8 recent frames barely differs; **6 is the default**. 64-dim memory channels suffice; 256 doesn't help.
- **Encoder scaling:** Hiera-S 70.9 → **B+ 73.0** → L 75.0 J&F. Large is +5% accuracy at 0.6× speed.
- **Object pointers:** +3.3% on SA-V, +4.6% on LVOS — a real contribution
- **GRU-based recurrent memory:** tried, didn't help
- Data quality vs quantity: filtering to the 50K most-edited masklets gives 66.2 vs 63.7 for random 50K — but the full 190K gives 69.9. **Quantity still wins.**

> **Directly relevant to you:** the memory bank holds only ~6 recent frames. That's the architectural fact behind the 3D medical propagation failure documented in [sam2-medical-2d-3d-eval.md](sam2-medical-2d-3d-eval.md).

## 7. Limitations

- Fails across **shot changes**; loses track in crowded scenes, after long occlusions, or in long videos
- Struggles with **thin, fast-moving** structures (hair, spider legs)
- Confuses **visually similar** objects (identical juggling balls)
- **Each object is tracked independently** — no inter-object communication, so crowded scenes are inefficient
- Suggests explicit motion modelling as future work

---

## Terms you'll need

- **J&F** — standard VOS metric: region similarity (Jaccard) averaged with boundary accuracy (F-measure).
- **Masklet** — a mask tracked across frames; one object's full spatiotemporal segmentation.
- **VOS** — Video Object Segmentation.
- **RoPE** — Rotary Positional Embedding, encoding position by rotating query/key vectors.
- **FPN** — Feature Pyramid Network, combining features at multiple resolutions.

## Questions to hold while reading

- Where would you attach LoRA in Hiera? All stages, or only the deepest? Nobody has settled this for SAM 2.
- Memory attention is a *separate* transformer from the image encoder. Adapting **it** rather than the encoder would be unusual — and is arguably the better-motivated target for 3D medical volumes, given that propagation is where things break.
- Object pointers gave +3.3%. They're small. Are they a cheap high-leverage adaptation target?
- Since images are single-frame videos with empty memory, does adapting for 2D medical slices leave the memory machinery entirely untouched — and is that a wasted opportunity or a sensible simplification?
