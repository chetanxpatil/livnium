---
language: en
license: other
tags:
  - nli
  - text-classification
  - attractor-dynamics
  - geometry
  - snli
  - pytorch
datasets:
  - snli
metrics:
  - accuracy
model-index:
  - name: Livnium
    results:
      - task:
          type: natural-language-inference
          name: Natural Language Inference
        dataset:
          name: SNLI
          type: snli
        metrics:
          - type: accuracy
            value: 0.8206
            name: Dev Accuracy (v2 — Joint BERT)
          - type: accuracy
            value: 0.7632
            name: Dev Accuracy (v1 — BoW encoder)
          - type: latency
            value: 0.4
            name: ms per batch (32 samples, CPU, head only)
          - type: throughput
            value: 85335
            name: Samples per second (CPU, head only)
---

# Livnium — A Dynamical Classification Head with Anchor-Guided Basin Settling

NLI classifier on SNLI where inference is not a single forward pass — it is a sequence of geometry-aware state updates before the final readout.

📄 **Paper (PDF):** [Livnium.pdf](https://github.com/chetanxpatil/livnium/blob/main/Livnium.pdf)
📝 **LaTeX source:** [livnium_paper.tex](https://github.com/chetanxpatil/livnium/blob/main/livnium_paper.tex)
🌐 **Zenodo preprint:** [zenodo.org/records/19058910](https://zenodo.org/records/19058910)
🤗 **Model on HuggingFace:** [chetanxpatil/livnium-snli](https://huggingface.co/chetanxpatil/livnium-snli)

---

## Version History

### v1.0 — Paper Release *(stable, tagged)*
> Matches the paper: *"Iterative Attractor Dynamics for Classification"*

- Encoder: pretrained bag-of-words (256-dim, frozen)
- Accuracy: **76.32%** dev (E: 87.5%, C: 81.2%, N: 70.9%)
- System: attractor dynamics head + basin field + Lyapunov analysis
- `--encoder-type pretrained`

### v2.0 — Joint BERT + Livnium-Native *(in development, on `main`)*

- Joint BERT bi-encoder: **82.06%** dev — BERT fine-tunes alongside attractor dynamics (+5.74pp vs v1)
- Cross-encoder BERT: `[CLS] premise [SEP] hypothesis [SEP]` — fixes role-reversal failures
- **Livnium-native encoder** (`--encoder-type livnium`): small transformer (~3M params) trained end-to-end in Livnium attractor space — no BERT at inference
- Geometry extraction: trained anchors define a principled semantic coordinate system (32-dim from 768)
- See [`book/`](book/) for the full design rationale

---

## Results — Full Accuracy Journey

| Stage | Encoder | Accuracy | Params |
|---|---|---|---|
| v1 Legacy | Random BoW (256-dim) | ~56% | ~2M |
| v1 Pretrained | Livnium BoW (256-dim, frozen) | **76.32%** | ~2M |
| v2 Frozen BERT | BERT bi-encoder (frozen) | ~61% | 110M + 2M |
| v2 Joint BERT | BERT bi-encoder (fine-tuned) | **82.06%** | 110M + 2M |
| v2 Cross-BERT | BERT cross-encoder (fine-tuned) | in progress | 110M + 2M |
| v2 Livnium-native | Small transformer (32-dim) | in progress | ~3M total |

> **Goal of Livnium-native:** match or exceed cross-encoder accuracy with 33× fewer parameters and full interpretability — no BERT at inference.

---

## The Core Idea

Most classifiers do: `h → linear layer → logits`. One step, no dynamics.

Livnium does: `h₀ → L steps of geometry-aware state evolution → logits`.

At each collapse step `t = 0…L-1`:

```
h_{t+1} = h_t
         + δ_θ(h_t)                              ← learned residual (MLP)
         - s_y · D(h_t, A_y) · n̂(h_t, A_y)      ← anchor force
         - β · B(h_t) · n̂(h_t, A_N)              ← neutral boundary force
```

```
D(h, A)  = 0.38 − cos(h, A)               ← divergence from equilibrium ring
n̂(h, A) = (h − A) / ‖h − A‖              ← Euclidean radial direction
B(h)     = 1 − |cos(h,A_E) − cos(h,A_C)|  ← E–C boundary proximity
```

Three learned anchor vectors `A_E`, `A_C`, `A_N` define the label geometry. The attractor is a **ring** at `cos(h, A_y) = 0.38`, not the anchor itself. At inference all three anchors compete — whichever basin wins determines the label.

**Key properties:**
- Force magnitudes are cosine-based; directions are Euclidean radial — geometrically inconsistent by design (measured: 135.2° ± 2.5° from true gradient). Not gradient descent — discrete-time attractor dynamics.
- Local contraction proven: `V(h) = (0.38 − cos(h, A_y))²` decreases at every step when the learned residual is small (Lyapunov argument).
- Head-only inference: **0.4 ms / batch (32 samples)** on CPU — 85,335 samples/sec.

---

## Directory Structure

```
livnium/
├── book/                              ← design rationale (Pages 1–6)
│   ├── page_1_what_is_livnium.md
│   ├── page_2_the_physics.md
│   ├── page_3_collapse_engine.md
│   ├── page_4_basin_field.md
│   ├── page_5_encoder_story.md        ← full accuracy journey
│   └── page_6_livnium_native_representation.md  ← v2 native encoder design
├── pretrained/
│   ├── collapse4/
│   │   └── quantum_embeddings_final.pt  ← v1 pretrained BoW embeddings
│   ├── bert-joint/                      ← v2 joint BERT checkpoint (82.06%)
│   └── livnium-native/                  ← v2 native encoder (in training)
├── training_logs/                     ← saved training runs
└── system/
    └── snli/
        ├── model/
        │   ├── train.py               ← unified training script (all encoder types)
        │   ├── eval.py                ← evaluation
        │   ├── infer.py               ← interactive / batch inference
        │   ├── speed_test.py          ← latency benchmark
        │   ├── extract_livnium_basis.py  ← geometry extractor (v2)
        │   ├── run_specialists.sh     ← 3-specialist pipeline
        │   ├── core/                  ← VectorCollapseEngine, BasinField, physics_laws
        │   ├── tasks/snli/            ← all encoders + heads
        │   │   ├── encoding_snli.py   ← SNLIEncoder, BERTSNLIEncoder,
        │   │   │                         CrossEncoderBERTSNLIEncoder,
        │   │   │                         LivniumNativeEncoder
        │   │   └── head_snli.py       ← SNLIHead, BinaryHead, LinearSNLIHead
        │   ├── text/                  ← vocab-based text encoders
        │   └── utils/                 ← vocab helpers
        └── embed/                     ← v1 pretrained embedding module
```

---

## How to Train

### v1 — Pretrained BoW (matches the paper)

```bash
pip install torch
cd system/snli/model/

python3 train.py \
  --snli-train ../../../data/snli/snli_1.0_train.jsonl \
  --snli-dev   ../../../data/snli/snli_1.0_dev.jsonl \
  --encoder-type pretrained \
  --embed-ckpt ../../../pretrained/collapse4/quantum_embeddings_final.pt \
  --output-dir ../../../runs/my_run \
  --dim 256 --num-layers 6 \
  --epochs 10 --batch-size 32 --lr 0.001 \
  --lambda-fn 0.05 --lambda-rep 0.1 --margin-rep 0.3
```

### v2 — Joint BERT (bi-encoder, 82.06%)

```bash
pip install torch transformers
cd system/snli/model/

python3 train.py \
  --snli-train ../../../data/snli/snli_1.0_train.jsonl \
  --snli-dev   ../../../data/snli/snli_1.0_dev.jsonl \
  --encoder-type bert \
  --bert-model bert-base-uncased \
  --output-dir ../../../pretrained/bert-joint \
  --epochs 5 --batch-size 32 --lr 1e-3 --bert-lr 2e-5 \
  --lambda-rep 0.1 --margin-rep 0.3 --lambda-fn 0.05 \
  --disable-dynamic-basins
```

### v2 — Cross-Encoder BERT (fixes role-reversal)

```bash
python3 train.py \
  --snli-train ../../../data/snli/snli_1.0_train.jsonl \
  --snli-dev   ../../../data/snli/snli_1.0_dev.jsonl \
  --encoder-type crossbert \
  --bert-model bert-base-uncased \
  --output-dir ../../../pretrained/bert-cross \
  --epochs 10 --batch-size 32 --lr 1e-3 --bert-lr 2e-5 \
  --lambda-rep 0.1 --margin-rep 0.3 --lambda-fn 0.05 \
  --disable-dynamic-basins
```

### v2 — Livnium-Native (no BERT at inference)

**Step 1 — Extract the geometry from a trained checkpoint:**

```bash
python3 extract_livnium_basis.py \
  --checkpoint ../../../pretrained/bert-joint/best_model.pt \
  --snli-train ../../../data/snli/snli_1.0_train.jsonl \
  --encoder-type bert --basis-dim 32 \
  --output ../../../pretrained/livnium_basis.pt
```

This extracts the trained anchor geometry (E/N/C positions), runs PCA on 5,000 collapse trajectories, and saves a 768×32 projection matrix. Result: **71.2% of BERT's h0 variance lives in 32 dims**.

**Step 2 — Train the small native encoder:**

```bash
python3 train.py \
  --snli-train ../../../data/snli/snli_1.0_train.jsonl \
  --snli-dev   ../../../data/snli/snli_1.0_dev.jsonl \
  --encoder-type livnium \
  --livnium-dim 32 --livnium-layers 2 --livnium-nhead 4 \
  --livnium-cross-encoder \
  --livnium-basis ../../../pretrained/livnium_basis.pt \
  --output-dir ../../../pretrained/livnium-native \
  --epochs 20 --batch-size 64 --lr 1e-3 \
  --lambda-rep 0.1 --margin-rep 0.3 --lambda-fn 0.05 \
  --disable-dynamic-basins
```

---

## How to Eval

```bash
cd system/snli/model/

python3 eval.py \
  --model-dir ../../../pretrained/bert-joint \
  --snli-test ../../../data/snli/snli_1.0_dev.jsonl \
  --batch-size 256
```

## How to Run Inference

```bash
# Interactive mode
python3 infer.py --model-dir ../../../pretrained/bert-joint --interactive

# Single pair
python3 infer.py --model-dir ../../../pretrained/bert-joint \
  --premise "A man is sleeping." \
  --hypothesis "A man is awake."

# Batch (JSONL)
python3 infer.py --model-dir ../../../pretrained/bert-joint --file pairs.jsonl
```

---

## What's Novel

**1. Classification as attractor dynamics.**
The state `h` moves through space across `L` steps before the classifier reads it. The label is determined by which basin the state *settles into*, not by a direct projection.

**2. Measured geometric inconsistency.**
Force magnitude is cosine-based; force direction is Euclidean radial. The true cosine gradient is tangential on the hypersphere. Measured angle between the two: **135.2° ± 2.5°** (n=1,000). The system runs non-conservative physical forces, not gradient descent.

**3. Attractor ring, not attractor point.**
Equilibrium is `cos(h, A_y) = 0.38` — a ring on the hypersphere. The system settles to a proximity zone, not the anchor itself.

**4. Proven local contraction.**
`V(h) = (0.38 − cos(h, A_y))²` is a Lyapunov function that decreases at every step when the residual MLP is small (proven analytically, verified empirically).

**5. Livnium-native representation (v2).**
The trained model defines a principled semantic coordinate system: E/N/C anchors are nearly orthogonal (cos(E,C)=−0.56, cos(E,N)=−0.77, cos(C,N)=−0.09). A small transformer can learn to navigate this geometry directly — no BERT required at inference. See [`book/page_6`](book/page_6_livnium_native_representation.md).

---

## Data

**SNLI:**
```bash
wget https://nlp.stanford.edu/projects/snli/snli_1.0.zip
unzip snli_1.0.zip -d data/snli/
```

**WikiText-103** (for v1 pretrained embeddings only):
```bash
wget https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-v1.zip
unzip wikitext-103-v1.zip -d data/wikitext/
```

---

## Lyapunov Analysis

Define `V(h) = D(h, A_y)² = (0.38 − cos(h, A_y))²`

`V = 0` at the attractor ring. When `δ_θ = 0`, `V` decreases at every step (mean `ΔV = −0.00131`):

```
∇_h cos · n̂ = −(r · sin²θ) / (α · ‖h − A‖)  ≤ 0
```

**Livnium is a locally-contracting pseudo-gradient flow.** Global convergence is not proven — the learned residual can dominate at large scale, and finite step sizes can overshoot.

---

## Citation

```bibtex
@misc{patil2026livnium,
  author       = {Patil, Chetan},
  title        = {Iterative Attractor Dynamics for Classification: A Dynamical Classification Head with Anchor-Guided Basin Settling},
  year         = {2026},
  publisher    = {Zenodo},
  howpublished = {\url{https://zenodo.org/records/19058910}},
  note         = {Code: \url{https://github.com/chetanxpatil/livnium}. Model: \url{https://huggingface.co/chetanxpatil/livnium-snli}}
}
```

For questions or collaboration: [GitHub](https://github.com/chetanxpatil) · [HuggingFace](https://huggingface.co/chetanxpatil)
