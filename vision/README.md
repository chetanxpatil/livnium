# vision/ — pure vector collapse over pixels

`img_collapse_pure.py` — the noun_collapse_pure.py discipline applied to MS COCO
Captions: **every attractor is a pixel of a fixed-size image**. The wells table
is an S×S grid of pixel-position wells; an image is encoded by collapsing a
start state through its own pixels in raster order, intensity gating each pull.
Readout is pure geometry: cosine to caption-noun wells, CE against nouns from
the image's own captions. No MLP, no conv, no SVD, no readout matrix.

Why S=64 default: the average COCO image is 574×483 (~523² pixels of
information). 512² would be information parity, 224² parameter parity with the
50k-noun text model — but the collapse is one sequential step per pixel, so
prove it at 64² (4,096 attractors) first and scale via `--size`.

```bash
# train (from repo root; needs nltk wordnet for the noun filter)
python3 vision/img_collapse_pure.py --train

# top-8 caption nouns for an image
python3 vision/img_collapse_pure.py --probe images/coco/val2017/000000179765.jpg

# where does a noun live on the pixel grid?
python3 vision/img_collapse_pure.py --render dog --render-out dog_map.png
```

Data: `images/coco/` (val2017 + caption annotations, downloaded 2026-07-05).
Images are resized once to S×S grayscale uint8 and cached
(`vision/model/img_cache_{S}.pt`); backprop memory through the S² sequential
steps is bounded by gradient checkpointing (`--ckpt-chunk`).

Smoke-tested 2026-07-05 (CPU, S=16, dim=64, 25 steps): loss 4.16 → 4.01,
probe + render modes working. Not yet trained at full scale.
