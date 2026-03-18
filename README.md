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

### v2.0 — Equation of Motion + Livnium-Native *(stable, tagged)*

- Joint BERT bi-encoder: **82.06%** dev — BERT fine-tunes alongside attractor dynamics (+5.74pp vs v1)
- **Empirical equation of motion discovered** — trained collapse dynamics reduce to gradient descent on a simple anchor-based energy (see below)
- **Livnium-native encoder** (`--encoder-type livnium`): 772K params, **~80%+ dev**, **5.3× faster** than BERT pipeline, **142× fewer encoder params**
- **Basin stability result**: correct predictions are in deeper basins than wrong ones (3.9× flip ratio, 2× entropy gap)
- Geometry extraction: trained anchors define a principled semantic coordinate system (32-dim from 768)
- See [`book/`](book/) for the full design rationale

---

## Results — Full Accuracy Journey

| Stage | Encoder | Accuracy | Params |
|---|---|---|---|
| v1 Legacy | Random BoW (256-dim) | ~56% | ~2M |
| v1 Pretrained | Livnium BoW (256-dim, frozen) | **76.32%** | ~2M |
| ~~v2 Frozen BERT~~ | ~~BERT bi-encoder (frozen)~~ | ~~\~61%~~ | ~~deprecated — BERT geometry doesn't align with attractor space without joint training~~ |
| v2 Joint BERT | BERT bi-encoder (fine-tuned) | **82.06%** | 110M + 2M |
| v2 Cross-BERT | BERT cross-encoder (fine-tuned) | in progress | 110M + 2M |
| v2 Livnium-native | Small transformer (32-dim) | **~80%+** (epoch 9/20) | 772K encoder |

> **Livnium-native result:** 142× fewer encoder params, 5.3× faster full pipeline, 9.3× faster encoder-only vs BERT-joint. No BERT at inference.

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

## The Three Laws of Livnium (v2)

Livnium v2 establishes three fundamental laws that fully specify the system's behavior from input representation through final prediction:

**Law 1 — Relational State Formation**
```
h₀ = v_h − v_p
```
The initial state is the hypothesis embedding minus the premise embedding. This encodes the *direction of semantic change* from premise to hypothesis. Entailment should pull `h₀` toward the entailment anchor; contradiction pulls it away; neutral is orthogonal. *Design choice — the weakest law, but necessary to ground the others.*

**Law 2 — Energy Landscape**
```
V(h) = −logsumexp( β · cos(h, A_E),  β · cos(h, A_C),  β · cos(h, A_N) )
```
The energy at any point `h` in semantic space is the negative log-partition function of Boltzmann-weighted cosine similarities to the three class anchors. The system assigns lower energy to points near an anchor — defining three semantic basins. *Empirically confirmed: robustly validated across β ∈ {1…50}, Δ within ±1%.*

**Law 3 — Collapse Dynamics**
```
h_{t+1} = h_t − α · ∇V(h_t)
```
The state evolves by gradient descent on the energy landscape over `L` steps. The trained MLP `δ_θ` was found empirically to be approximating this gradient — it can be replaced by the three-line analytic formula with no accuracy loss (and slight improvement: +0.30%, neutral recall +1.3pp).

*This is the key discovery of v2: gradient descent on a logsumexp cosine energy is not a design assumption — it was recovered empirically from a trained system.*

**Paper-ready framing:**
> *"We propose a three-part formulation of NLI as a dynamical system: relational state formation (Law 1), an energy landscape over semantic anchors (Law 2), and gradient-based collapse dynamics (Law 3). Laws 2 and 3 were not designed — they were discovered by analyzing what the trained system was doing."*

---

## Empirical Equation of Motion (v2 Discovery)

Replacing the trained MLP residual `δ_θ(h_t)` with an analytical gradient shows the collapse dynamics are well-approximated by a clean geometric law:

```
h_{t+1} = h_t − α · ∇V(h_t)

V(h) = −logsumexp( β · cos(h, A_E),  β · cos(h, A_C),  β · cos(h, A_N) )
```

**Empirical result on SNLI dev (2000 samples):**

| Mode | Accuracy | N-recall | Notes |
|---|---|---|---|
| Full (δ_θ + anchor forces) | 82.65% | 70.16% | trained system |
| Anchor forces only (no δ_θ) | 82.15% | 66.77% | −0.50% |
| **Grad-V (β=1.0, α=0.2)** | **82.95%** | **71.49%** | **+0.30% over full** |

The analytical gradient descent not only matches the trained dynamics — it slightly **outperforms** them, particularly on the neutral class (+1.3pp recall). The result is robust across β ∈ {1…50} (Δ within ±1% across the entire sweep).

**Interpretation:** The trained MLP `δ_θ` was approximating gradient descent on this energy landscape but introducing small distortions. The clean analytical gradient recovers and slightly exceeds the learned behavior. `V(h)` is the Boltzmann log-partition function over anchor similarities — a natural measure of semantic basin proximity.

> **Candidate Livnium equation of motion:** `h_{t+1} = h_t − α∇V(h_t)`, `V(h) = −logsumexp(β·cos(h, anchors))`, β=1.0, α=0.2.
> *Confirmed empirically. Requires multi-seed and trajectory-level validation for full claim.*

---

## Basin Stability Results (v2)

Correct predictions sit in deeper attractor basins than wrong ones. Measured by perturbing `h_0` with Gaussian noise (σ=0.3) across 20 trials:

| Group | Flip rate | Entropy |
|---|---|---|
| Correct predictions (n=1653) | 0.0017 ± 0.025 | 0.293 |
| Wrong predictions (n=347) | 0.0068 ± 0.048 | 0.604 |
| **Ratio** | **3.93×** | **2.06×** |

Per-class stability of errors: Entail errors flip **7.18×** more than correct Entail predictions, Neutral errors **3.17×**, Contra errors **1.82×**. Contradiction errors are the most systematically stable wrong predictions — indicating a geometric bias in that class, not boundary ambiguity.

**Interpretation:** Correctness correlates with basin depth. The model is more uncertain where it's wrong. This makes stability an interpretable confidence proxy — no calibration layer needed.

---

## Speed: BERT-joint vs Livnium-native (v2)

Measured on Apple M-series MPS, batch=32:

| Metric | BERT-joint | Livnium-native | Ratio |
|---|---|---|---|
| Encoder params | 109.5M | 772K | **142× fewer** |
| Full pipeline | 716 ex/s | 5,139 ex/s | **7.2× faster** |
| Encoder-only | — | — | **9.3× faster** |
| Accuracy | 82.06% | ~80%+ (epoch 9/20) | −~2% |
| Collapse share of pipeline | 5% | 47% | dynamics are a full participant |

Pipeline breakdown at batch=32: BERT spends 95% of time in the encoder; native encoder spends 47% in the Livnium collapse — encoder and dynamics are balanced partners.

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

**2. Measured geometric inconsistency (v1).**
Force magnitude is cosine-based; force direction is Euclidean radial. The true cosine gradient is tangential on the hypersphere. Measured angle between the two: **135.2° ± 2.5°** (n=1,000). The system runs non-conservative physical forces, not gradient descent.

**3. Attractor ring, not attractor point.**
Equilibrium is `cos(h, A_y) = 0.38` — a ring on the hypersphere. The system settles to a proximity zone, not the anchor itself.

**4. Proven local contraction.**
`V(h) = (0.38 − cos(h, A_y))²` is a Lyapunov function that decreases at every step when the residual MLP is small (proven analytically, verified empirically).

**5. Livnium-native representation (v2).**
The trained model defines a principled semantic coordinate system: E/N/C anchors are nearly orthogonal (cos(E,C)=−0.56, cos(E,N)=−0.77, cos(C,N)=−0.09). A small transformer can learn to navigate this geometry directly — no BERT required at inference. See [`book/page_6`](book/page_6_livnium_native_representation.md).

**6. Empirical equation of motion discovery (v2).**
The trained collapse dynamics are well-approximated by gradient descent on a logsumexp cosine-anchor energy. Replacing the 1.2M-FLOP learned MLP with three lines of analytic math matches accuracy — and slightly exceeds it (+0.3%, neutral recall +1.3pp). The system naturally converges to a gradient-flow regime during training. The MLP was learning an approximation to a law that already existed in the geometry.

**7. Stability as a correctness signal (v2).**
Wrong predictions are 3.9× more likely to flip under input perturbation, and carry 2× higher softmax entropy. Correctness correlates with basin depth. No calibration layer needed — the dynamics themselves encode confidence. Contradiction errors are the exception (1.82× ratio), suggesting systematic geometric bias rather than boundary ambiguity — a concrete target for v3.

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
