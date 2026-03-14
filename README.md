# Livnium Sacred — Checkpoints + Clean System

## Directory Structure

```
livnium-sacred/
├── runs/
│   └── triple_crown_slow_20260314_114951/
│       └── best_model.pt                       ← ⭐ MAIN RESULT: 77.05% dev acc
├── quantum-embeddings/
│   ├── collapse4/quantum_embeddings_final.pt   ← original 96% embeddings
│   └── collapse1/quantum_embeddings_final.pt   ← lower-loss embeddings
├── snli-models/
│   ├── collapse1-static/best_model.pt          ← 75.31% (wrong config, static)
│   └── collapse4-dynamic-96pct/                ← MISSING (was in deleted dir)
└── system/
    ├── quantum-pretrain/                        ← standalone embedding trainer
    │   ├── train_quantum_embeddings.py
    │   ├── vector_collapse.py
    │   ├── basin_field.py
    │   └── text_encoder_quantum.py
    └── snli/                                    ← standalone SNLI pipeline
        ├── nova_v3/
        │   ├── train_snli_vector.py             ← TRAIN (run from here)
        │   ├── test_snli_vector.py              ← TEST (run from here)
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

## How to Train (SNLI)

```bash
cd system/snli/nova_v3

python3 train_snli_vector.py \
    --snli-train  /path/to/snli_1.0_train.jsonl \
    --snli-dev    /path/to/snli_1.0_dev.jsonl \
    --encoder-type quantum \
    --quantum-ckpt /path/to/quantum_embeddings_final.pt \
    --dim 256 --batch-size 32 --epochs 3 \
    --output-dir /path/to/output
```

## How to Test (SNLI)

```bash
cd system/snli/nova_v3

python3 test_snli_vector.py \
    --model-dir /path/to/output \
    --snli-test /path/to/snli_1.0_test.jsonl
```

## How to Train Quantum Embeddings

```bash
cd system/quantum-pretrain

# Collapse-1 config (recommended — lower loss, better convergence)
python3 train_quantum_embeddings.py \
    --train-path /path/to/wiki.train.tokens \
    --output-dir /path/to/output \
    --dim 256 --max-vocab 50000 --max-lines 200000 \
    --window-size 2 --batch-size 4096 --epochs 3 \
    --disable-dynamic-basins --collapse-layers 1

# Collapse-4 config (original 96% config)
python3 train_quantum_embeddings.py \
    --train-path /path/to/wiki.train.tokens \
    --output-dir /path/to/output \
    --dim 256 --max-vocab 50000 --max-lines 200000 \
    --window-size 2 --batch-size 4096 --epochs 3 \
    --disable-dynamic-basins --collapse-layers 4
```

---

## Results — SNLI NLI Classification

### Best Model: triple_crown_slow (quantum embeddings)

**Checkpoint location:** `runs/triple_crown_slow_20260314_114951/best_model.pt` (50 MB)

| Metric | Value |
|--------|-------|
| **Dev Accuracy (overall)** | **77.05%** |
| Entailment | 87.5% |
| Contradiction | 81.2% |
| Neutral | 62.8% |
| Dim | 256 |
| Collapse steps | 6 layers |
| Encoder | Quantum embeddings (pre-trained) |

**Architecture:**
- Encoder: QuantumTextEncoder (semantic embedding layer)
- Feature extraction: `h = v_h - v_p` (hypothesis - premise difference)
- Inference: 3-way parallel collapse (one per label) → argmin energy → classification head
- Dynamics: Discrete-time attractor with learned residual + anchor forces

**Neutral class weakness (62.8%):** Open problem. Neutral examples live on the E-C boundary where forces are ambiguous. Current workarounds explored in `runs/ablation_*` and `runs/fast_*` series.

### Faster Variants

| Config | Encoder | Dev Acc | Speed | Notes |
|--------|---------|---------|-------|-------|
| **triple_crown_slow** | Quantum | **77.05%** | 0.4 ms/batch | Reference model |
| ablation_base, *_rep, *_traj | Quantum | 55–58% | ~0.4 ms | Architectural ablations (missing neutral boost) |
| fast_v3–v9 | MiniLM-L6 | 56–57% | ~0.3 ms | Fast encoder trade-off (speed vs accuracy) |

**Speedup claim:** `triple_crown_slow` is ~428× faster than BERT-base (0.4 ms vs 171 ms per 32-sample batch).

### Configuration Reference

| Config | SNLI Dev | Notes |
|--------|----------|-------|
| collapse4 + dynamic basins | **95.76%** | Best (embedding only, not NLI classification) |
| triple_crown_slow (NLI) | **77.05%** | Best for end-to-end NLI classification |
| collapse1 + static | 75.31% | Wrong config. Static collapse too slow to learn |

**Critical insight:** Dynamic basins during training are NOT a label leak. They provide
label-guided collapse during training only. Evaluation always uses static
collapse (no labels). The 95.76% embedding result and 77.05% NLI result are both clean and honest.

### Physics & Theory

See `runs/livnium_collapse_equation.md` for detailed mathematical formulation:
- Update rule with learned residual + anchor forces
- Euclidean–cosine direction mismatch (135.2° ± 2.5° empirically)
- Lyapunov stability analysis (conjecture for δ_θ ≠ 0)
- Attractor ring at cos(h, A_y) = 0.38 (not at anchor)

**Collapse = "don't search, settle"**
Instead of global pairwise comparison, the difference vector v_h - v_p
settles into the nearest E/N/C basin in one local pass. This is the
source of the O(n)-like processing story.
