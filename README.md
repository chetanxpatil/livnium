# Livnium — Geometry-Aware State Evolution for NLI

## Directory Structure

```
livnium-sacred/
├── runs/
│   └── triple_crown_slow_20260314_114951/
│       └── best_model.pt                       ← ⭐ BEST MODEL: 76.32% dev acc
├── quantum-embeddings/
│   └── collapse4/quantum_embeddings_final.pt   ← pre-trained quantum embeddings
└── system/
    └── snli/
        ├── nova_v3/
        │   ├── train_snli_vector.py             ← TRAIN
        │   ├── test_snli_vector.py              ← TEST
        │   ├── core/                            ← VectorCollapseEngine, BasinField
        │   ├── tasks/snli/                      ← QuantumSNLIEncoder, SNLIHead
        │   ├── text/                            ← text encoders
        │   └── utils/                           ← vocab helpers
        └── quantum_embed/
            ├── text_encoder_quantum.py
            ├── vector_collapse.py
            └── basin_field.py
```

---

## How to Train

```bash
cd /path/to/livnium-sacred

python3 system/snli/nova_v3/train_snli_vector.py \
  --snli-train data/snli/snli_1.0_train.jsonl \
  --snli-dev   data/snli/snli_1.0_dev.jsonl \
  --encoder-type quantum \
  --quantum-ckpt quantum-embeddings/collapse4/quantum_embeddings_final.pt \
  --output-dir runs/my_run \
  --dim 256 --num-layers 6 \
  --epochs 10 --batch-size 32 --lr 0.001 \
  --lambda-traj 0.1 --lambda-fn 0.15 --lambda-rep 0.1 --margin-rep 0.3 \
  --adaptive-metric
```

## How to Test

```bash
cd /path/to/livnium-sacred

python3 system/snli/nova_v3/test_snli_vector.py \
  --model-dir runs/triple_crown_slow_20260314_114951 \
  --snli-test data/snli/snli_1.0_dev.jsonl \
  --batch-size 256
```

---

## Results — SNLI NLI Classification

### Best Model: triple_crown_slow_20260314_114951

**Checkpoint:** `runs/triple_crown_slow_20260314_114951/best_model.pt`

| Metric | Value |
|--------|-------|
| **Dev Accuracy** | **76.32%** |
| Entailment | 80.2% |
| Contradiction | 77.6% |
| Neutral | 70.9% |
| Dim | 256 |
| Collapse layers | 6 |
| Encoder | Quantum embeddings (pre-trained) |

**Architecture:**
- Encoder: QuantumTextEncoder (pre-trained, frozen)
- Initial state: `h₀ = v_p − v_h` (premise − hypothesis difference)
- Inference: single collapse pass, all three E/C/N anchors compete simultaneously
- Classification head: SNLIHead takes `(h_final, v_p, v_h)`
- Dynamics: discrete-time attractor with learned residual + anchor forces + adaptive metric

### Physics & Theory

The collapse update rule:

```
h_{t+1} = h_t + δ_θ(h_t) − s_y · D(h_t, A_y) · n̂(h_t, A_y) − β · B(h_t) · n̂(h_t, A_N)
```

Where:
- `D(h, A) = BARRIER − cos(h, A)` — divergence from anchor (BARRIER = 0.38)
- `n̂(h, A) = (h − A) / ‖h − A‖` — Euclidean radial direction
- `B(h) = 1 − |cos(h, A_E) − cos(h, A_C)|` — E-C boundary proximity

See `runs/livnium_collapse_equation.md` for full derivation, Lyapunov stability analysis, and empirical direction mismatch (135.2° ± 2.5° between Euclidean and cosine gradients).

**Collapse = "don't search, settle"**
The difference vector settles into the nearest E/N/C basin in a single local pass — no global pairwise comparison needed.
