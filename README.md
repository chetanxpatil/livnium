# Livnium — Geometry-Aware Attractor Dynamics for NLI

NLI classifier on SNLI where inference is not a single forward pass — it is a sequence of geometry-aware state updates before the final readout.

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

## How to Train

```bash
cd system/snli/model/

python train.py \
  --snli-train ../../../data/snli/snli_1.0_train.jsonl \
  --snli-dev   ../../../data/snli/snli_1.0_dev.jsonl \
  --encoder-type pretrained \
  --quantum-ckpt ../../../pretrained/collapse4/quantum_embeddings_final.pt \
  --output-dir ../../../runs/my_run \
  --dim 256 --num-layers 6 \
  --epochs 10 --batch-size 32 --lr 0.001 \
  --lambda-traj 0.1 --lambda-fn 0.15 --lambda-rep 0.1 --margin-rep 0.3 \
  --adaptive-metric
```

## How to Eval

```bash
cd system/snli/model/

python eval.py \
  --model-dir ../../../runs/triple_crown_slow \
  --snli-test ../../../data/snli/snli_1.0_dev.jsonl \
  --batch-size 256
```

---

## Results — SNLI NLI Classification

### Best Model: triple_crown_slow
| Metric | Value |
|--------|-------|
| **Dev Accuracy** | **76.32%** |
| Entailment | 80.2% |
| Contradiction | 77.6% |
| Neutral | 70.9% |
| Dim | 256 |
| Collapse layers | 6 |
| Encoder | Pretrained embeddings (frozen) |

### Speed

| Model | ms/batch (32) | Samples/sec | Time on SNLI train (549k) |
|-------|---------------|-------------|---------------------------|
| Livnium | 0.4 ms | 85,335/sec | ~6 sec |
| BERT-base | 171 ms | 187/sec | ~49 min |

---

## Lyapunov Analysis

Define `V(h) = D(h, A_y)² = (0.38 − cos(h, A_y))²`

`V = 0` at the attractor ring. When `δ_θ = 0`, `V` decreases at every step (mean `ΔV = −0.00131`). Analytically:

```
∇_h cos · n̂ = −(β · sin²θ) / (α · ‖h − A‖)  ≤ 0
```

**Livnium is a provably locally-contracting pseudo-gradient flow.**

See `runs/livnium_collapse_equation.md` for the full derivation and empirical direction mismatch analysis (135.2° ± 2.5° between Euclidean and cosine gradients).
