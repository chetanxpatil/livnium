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

---

`pixel_color_group.py` — group-aware level 1: each pixel gets its exact color
AND its color family (neutral / warm / cool), with soft blend percentages.
Hierarchy by construction: `color well = W_group[group(c)] + W_offset[c]`,
so same-group colors cluster in a cone around their group well and one
collapsed h reads out at both grains by cosine alone. Warm-starts from
pixel_color_pure.pt (step 1 IS the old model); saves materialized color
wells, so the checkpoint is a drop-in replacement for every
pixel_color_pure.pt reader — downstream models inherit group-clustered
color geometry with zero code change. `--probe R G B` prints color top-3 +
group blend ("cool 0.81 neutral 0.12 warm 0.07"). Edit GROUPS, retrain in
minutes (synthetic data). Written 2026-07-07, not yet trained.

---

`img_fovea.py` — foveal level 2: position is polar, relative to a focal
center, instead of 4,096 unrelated raster wells. Each pixel's well is

    W_ring[radius_bin] * W_angle[angle_bin] * sqrt(D)  +  ray @ W_ray

with `ray = norm([dx, dy, focal])` — "brown, slightly below center, at this
distance", not "pixel 2087 is brown". 32+64+3 ≈ 99 wells replace 4,096, and
pixels close in radius/angle share wells, so 2D adjacency exists by
construction. Collapse physics, pure-geometry decode, and the capacity
question are unchanged from img_from_state_pure, so recon accuracy is
directly comparable (free wells ~0.56, positional prior ~0.38). Extras:
`--scan spiral` (center-out, default), a learnable foveal falloff scalar
(starts uniform), `--rgb` raw channel wells as in img_from_state_rgb.

Binding lesson (first run, 2026-07-07): ADDITIVE ring+angle addresses are so
correlated (every pixel in a ring shares half its address) that H collapses
onto the mean pull and decode paints each image one flat color. Hence
`--bind mul` default — elementwise-product keys — plus `--pos-init smooth`
(random wells gaussian-blurred along the index, sigma 1.5 bins): neighbor
cos ~0.9, mean |cos| between distinct addresses 0.067 vs free wells' 0.062.
Adjacency and addressing at once. `--bind add --pos-init fourier` kept for
the ablation.

```bash
python3 vision/img_fovea.py --selftest                   # 1-min mechanics gate
python3 vision/img_fovea.py --train --addr-reg 0.1 --device mps
python3 vision/img_fovea.py --recon 0 1 2                # side-by-side PNGs
python3 vision/img_fovea.py --inspect                    # address drift check

# group-aware level 1: recon also reports neutral/warm/cool shares,
# whole image and fovea vs periphery
python3 vision/img_fovea.py --train --addr-reg 0.1 \
    --l1 vision/model/pixel_color_group.pt \
    --out vision/model/img_fovea_group.pt --device mps
python3 vision/img_fovea.py --recon 0 1 2 --out vision/model/img_fovea_group.pt
```

Cleaned 2026-07-07: recon/inspect now rebuild geometry from the CHECKPOINT
config (CLI geometry flags can't silently mismatch), level 1 is pluggable
via --l1 (path saved in the checkpoint), decode no longer hardcodes 13
colors, dead fourier init removed (the add-binding failure is structural,
not init-dependent).

Trained 2026-07-07 (S=64, dim=256, 4k steps, mps): recon acc ~0.52 avg
(imgs 0/1/2: 0.56/0.28/0.51) vs free wells 0.56 and prior 0.38 — with
28,931 numbers instead of ~1.05M position parameters. Selftest verdict:
mechanism stores (structured maps 0.50 vs prior 0.20); the live finding is
that CE training RE-CORRELATES the addresses (mean|cos| 0.27 -> 0.37),
eroding mul binding's orthogonality — likely what caps full scale below
free wells. Hence `--addr-reg` (try 0.1) and `--inspect` to measure drift
on any checkpoint. `--selftest` is the 1-minute no-images mechanics gate.

Cross-verified 2026-07-07 (`vision/test_levels.py`, numpy-only — reads the
.pt checkpoints without torch and re-implements both forwards): L1 anchors
13/13, 20k random RGBs 0.970 vs nearest-anchor; L2 numpy recon accs match
the torch run to 3 decimals (0.560/0.278/0.510) — no implementation bug at
either level. BUT the trained L2 checkpoint's addresses sit at mean|cos|
0.291 (free-well ref 0.062) and its predictions use only 1–5 of 13 colors
per image: the 0.51 on img 2 is a one-color prediction matching brown's
51% share. Recon acc alone flatters; watch uniq-colors. Address drift is
THE bottleneck -> retrain with --addr-reg.
