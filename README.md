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
            value: 0.7632
            name: Dev Accuracy
          - type: accuracy
            value: 0.875
            name: Entailment Accuracy
          - type: accuracy
            value: 0.812
            name: Contradiction Accuracy
          - type: accuracy
            value: 0.709
            name: Neutral Accuracy
          - type: latency
            value: 0.4
            name: ms per batch (32 samples, CPU)
          - type: throughput
            value: 85335
            name: Samples per second (CPU)
---

# Livnium — Energy-Guided Attractor Network (EGAN) for NLI

NLI classifier on SNLI where inference is not a single forward pass — it is a sequence of geometry-aware state updates before the final readout.

🤗 **Model on HuggingFace:** [chetanxpatil/livnium-snli](https://huggingface.co/chetanxpatil/livnium-snli)

## Directory Structure

```
livnium/
├── pretrained/
│   └── collapse4/
│       └── quantum_embeddings_final.pt  ← pre-trained word embeddings
├── runs/
│   └── triple_crown_slow/
│       ├── best_model.pt                ← ⭐ BEST MODEL: 76.32% dev acc
│       └── test_errors.jsonl            ← misclassified examples
└── system/
    └── snli/
        ├── model/                        ← main model: train + eval
        │   ├── train.py                  ← TRAIN
        │   ├── eval.py                   ← EVAL
        │   ├── core/                     ← VectorCollapseEngine, BasinField, physics_laws
        │   ├── tasks/snli/               ← PretrainedSNLIEncoder, SNLIHead
        │   ├── text/                     ← text encoders (vocab-based)
        │   └── utils/                    ← vocab helpers
        └── embed/                        ← pretrained embedding module
            ├── text_encoder.py           ← PretrainedTextEncoder
            ├── collapse_engine.py        ← lightweight collapse for embed module
            └── basin_field.py            ← basin field dynamics
```

---

## The Update Rule

At each collapse step `t = 0…L-1`:

```
h_{t+1} = h_t
         + δ_θ(h_t)                              ← learned residual
         - s_y · D(h_t, A_y) · n̂(h_t, A_y)      ← anchor force (training: correct label only)
         - β · B(h_t) · n̂(h_t, A_N)              ← neutral boundary force
```

```
D(h, A)  = 0.38 − cos(h, A)               ← divergence from equilibrium cosine
n̂(h, A) = (h − A) / ‖h − A‖              ← Euclidean radial direction
B(h)     = 1 − |cos(h,A_E) − cos(h,A_C)|  ← E–C boundary proximity
```

Three learned anchor vectors `A_E`, `A_C`, `A_N` define the label geometry. The attractor is a ring at `cos(h, A_y) = 0.38`, not the anchor itself.

At inference all three anchors compete simultaneously — whichever basin has the strongest geometric pull wins.

Force magnitudes are cosine-based; force directions are Euclidean radial. These are geometrically inconsistent (true cosine gradient is tangential). Correct description: **discrete-time attractor dynamics with anchor-directed forces. Energy-like, not exact gradient flow.**

---

## Data

**SNLI** (for training/eval the classifier):
```bash
wget https://nlp.stanford.edu/projects/snli/snli_1.0.zip
unzip snli_1.0.zip -d data/snli/
```
Or via HuggingFace: `datasets.load_dataset("snli")`
Homepage: https://nlp.stanford.edu/projects/snli/

**WikiText-103** (for training the pretrained embeddings):
```bash
wget https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-v1.zip
unzip wikitext-103-v1.zip -d data/wikitext/
```
Or via HuggingFace: `datasets.load_dataset("wikitext", "wikitext-103-v1")`
Or via Kaggle: https://www.kaggle.com/datasets/vadimkurochkin/wikitext-103
Homepage: https://huggingface.co/datasets/Salesforce/wikitext

---

## How to Train

```bash
cd system/snli/model/

python3 train.py \
  --snli-train ../../../data/snli/snli_1.0_train.jsonl \
  --snli-dev   ../../../data/snli/snli_1.0_dev.jsonl \
  --encoder-type pretrained \
  --embed-ckpt ../../../pretrained/collapse4/quantum_embeddings_final.pt \
  --output-dir ../../../runs/my_run \
  --dim 256 --num-layers 6 \
  --epochs 10 --batch-size 32 --lr 0.001 \
  --lambda-traj 0.1 --lambda-fn 0.15 --lambda-rep 0.1 --margin-rep 0.3 \
  --adaptive-metric
```

## How to Eval

```bash
cd system/snli/model/

python3 eval.py \
  --model-dir ../../../runs/triple_crown_slow \
  --snli-test ../../../data/snli/snli_1.0_dev.jsonl \
  --batch-size 256
```

---

## What's Novel

Most classifiers do: `h → linear layer → logits`. One step, no dynamics.

Livnium does: `h₀ → L steps of geometry-aware state evolution → logits`. The final state `h_L` is dynamically shaped before readout — it isn't just a linear projection of `h₀`.

The specific things that are different:

**1. Classification as attractor dynamics, not a lookup.**
The state `h` moves through space across `L` steps under anchor forces before the classifier reads it. The label isn't computed from the raw embedding — it's read from where the state *settled*.

**2. The force geometry is deliberately inconsistent — and that's measured.**
Force magnitudes follow cosine divergence `D(h, A) = 0.38 − cos(h, A)`. Force directions are Euclidean radial `n̂ = (h − A) / ‖h − A‖`. These are not the same thing — the true gradient of a cosine energy is tangential on the sphere, not radial. The mean angle between these two directions is **135.2° ± 2.5°** (measured, n=1000). This means the system is running explicit physical forces, not gradient descent on the written energy.

**3. The attractor is a ring, not a point.**
The equilibrium condition is `cos(h, A_y) = 0.38`, which defines a ring on the hypersphere — not the anchor itself. The system settles to a *proximity zone*, not a target location. Standard energy minimisation would push to the anchor; this stops at the ring.

**4. Proven local contraction.**
`V(h) = (0.38 − cos(h, A_y))²` is a Lyapunov function that decreases at every step when `δ_θ = 0` (proven analytically, confirmed empirically on 5000 samples). Livnium is a provably locally-contracting pseudo-gradient flow. Most residual classifiers have no such stability guarantee.

**5. Inference is a single unsupervised collapse.**
Training uses `s_y · D(h, A_y)` — only the correct anchor pulls. At inference, all three anchors compete with no label. The label is implicit in which basin wins. Cost: 1× forward pass through a small MLP, 428× faster than BERT on CPU.

**What it isn't:** global convergence is not proven (finite step size + learned residual `δ_θ` can escape the basin). The geometric inconsistency is not fixed. It isn't yet competitive with fine-tuned transformers on accuracy. Whether iterated attractor dynamics outperform a standard deep residual block at equivalent parameter count is an open question.

---

## Results — SNLI NLI Classification

### Accuracy (SNLI dev set)

| Class | Accuracy |
|-------|----------|
| **Overall** | **76.32%** |
| Entailment | 87.5% |
| Contradiction | 81.2% |
| Neutral | 70.9% |

### Model Config

| Parameter | Value |
|-----------|-------|
| Dim | 256 |
| Collapse layers | 6 |
| Encoder | Pretrained bag-of-words embeddings (frozen) |
| Parameters | ~2M |

### Speed vs BERT (CPU, batch size 32)

| Model | ms / batch | Samples / sec | Full SNLI train (549k) |
|-------|------------|---------------|------------------------|
| **Livnium** | **0.4 ms** | **85,335 / sec** | **~6 sec** | (at inference)
| BERT-base | 171 ms | 187 / sec | ~49 min | (at inference)

**428× faster than BERT-base on CPU.**

---

## Lyapunov Analysis

Define `V(h) = D(h, A_y)² = (0.38 − cos(h, A_y))²`

`V = 0` at the attractor ring. When `δ_θ = 0`, `V` decreases at every step (mean `ΔV = −0.00131`). Analytically:

```
∇_h cos · n̂ = −(β · sin²θ) / (α · ‖h − A‖)  ≤ 0
```

**Livnium is a provably locally-contracting pseudo-gradient flow.**

See `runs/livnium_collapse_equation.md` for the full derivation and empirical direction mismatch analysis (135.2° ± 2.5° between Euclidean and cosine gradients).

---

## Citation

If you use this work in your research, please cite:

```bibtex
@misc{patil2026livnium,
  author       = {Patil, Chetan},
  title        = {Livnium: Energy-Guided Attractor Network (EGAN) for Natural Language Inference},
  year         = {2026},
  publisher    = {GitHub},
  howpublished = {\url{https://github.com/chetanxpatil/livnium}},
  note         = {Model available at \url{https://huggingface.co/chetanxpatil/livnium-snli}}
}
```

For questions or collaboration: [GitHub](https://github.com/chetanxpatil) · [HuggingFace](https://huggingface.co/chetanxpatil)
