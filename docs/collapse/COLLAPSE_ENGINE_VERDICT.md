# docs/collapse/COLLAPSE_ENGINE_VERDICT.md
**Livnium v0.1: Forensic Audit & Post-Mortem**
**Date:** June 18, 2026
**Status:** Architecture Audited, Patched, and Re-wired — pending one real training run

## 1. Overview
The `cortex_v2` architecture consists of two primary halves: the geometric **Lattice & MPS
Simulator** and the **PyTorch Thermodynamic Collapse Engine**.

An in-depth forensic audit revealed that while the Lattice math was remarkably solid (`O(1)`
scaling and verified permutation properties), the MPS swap network contained a norm-collapse
bug, and the PyTorch Collapse Engine had completely failed to train due to architectural
disconnections and unstable dynamics.

All identified issues have been patched. This document details the critical flaws found in the
v1 pipeline and outlines the tripwire playbook for future training runs.

---

## 2. The Disconnections (The "Dead Weight" Bugs)
The original PyTorch engine was functionally dead during both training and inference, and
**neither failure raised an error** — that is the dangerous part.

* **The Optimizer Disconnect (Training):** The `collapse_engine` module was instantiated
  separately from the embedding model and was never passed to `torch.optim.Adam`. It remained
  frozen at its random initialization for the entire training run (confirmed: `max|Δanchor|`
  and `max|Δmlp.weight|` were both `0.000000` between epoch 1 and the final checkpoint). The
  embeddings were doing 100% of the work.
* **The Load-Path Disconnect (Inference):** In `text_encoder_collapse.py`, the engine was only
  restored if `use_dynamic_basins` was True. For static-collapse checkpoints, the engine was
  silently dropped, and `collapse_sentence` fell back to raw mean-pooled embeddings without
  throwing an error.

**Fix:** The optimizer is now built **after** the engine and explicitly tracks
`list(model.parameters()) + list(collapse_engine.parameters())` (and prints the tracked
parameter count at startup). The encoder load-path now restores the engine based on its
presence in the checkpoint, independent of basin mode; `collapse_sentence` runs the static
three-anchor collapse when basins are off; and the final checkpoint now also persists
`collapse_config` so geometry (num_layers, strengths) is rebuilt correctly.

---

## 3. The Physics Fixes (Stable Dynamics)
Even if the engine had been connected to the optimizer, its mathematical dynamics were
inherently unstable.

* **The Overpowering MLP (the detonator):** The residual MLP's random initialization was
  `≈37×` stronger (‖delta‖≈4.3) than the thermodynamic force (‖force‖≈0.04). *This* is what
  blasted every state against the `||h|| = 10` clamp on step one (Jacobian spectral radius
  `ρ = 1.24`, `ρ ≈ 1.1e4` near the origin) — the "cooling" process was an explosive,
  random coordinate-bomb.
    * **Fix:** The final `Linear` layer of the MLP is zero-initialized. The residual block
      starts as the identity (`Δ = 0`), so the thermodynamic force shapes the early manifold
      and the MLP grows only as the loss finds it useful.
* **The Bad Compass — `0.38 - align` vs `1 - align`:** Distinct from the explosion above, the
  divergence law was simply pointing the wrong way. Its force was tiny (‖·‖≈0.04), but its
  equilibrium sat at `align = 0.38` — a *shell ~68° off the anchor* — and it became
  **repulsive** for `align > 0.38`. It never pulled vectors home.
    * **Fix:** Updated the law to `1 - align`: non-negative everywhere, zero only at perfect
      alignment, making the anchor a true point-attractor. Verified in NumPy: a state climbs
      from align `0.43 → 0.996`, and the one-step Jacobian near the anchor now **contracts
      (`ρ = 0.82`)**. Mirrored in `basin_field.py` for the dynamic path.
* **Dynamic Basin Thrashing:** Spawning a basin at `count = 0` while aggressively pruning
  basins with `count < 10` created a structural race condition where newborn basins were
  deleted before they could accumulate mass. (Currently mitigated by relying on the patched
  static mode; the dynamic path was never engaged in the trained checkpoints.)

---

## 4. The Lattice & MPS Simulator
The core `cortex_v2` state-vector bridge was mostly intact, but required a critical fix for deep
entanglement scenarios.

* **Norm Collapse in Swap Network:** The MPS SVD `_split` assumed local isometric tensors
  (`||θ||² = 1`). During leftward sweeps in the long-range CNOT swap network over
  already-entangled bonds, this assumption failed, aggressively suppressing amplitude (e.g.
  `CNOT(0,3)` drove the state norm to 0.25). Adjacent gates and GHZ masked it, so the
  original selftest never caught it.
    * **Fix:** A relative renormalization rescale (`sqrt(total/kept)`) preserves unitarity
      locally without requiring full canonicalization sweeps. Verified: all long-range CNOTs
      now hold norm 1.0 and match the exact statevector to `1e-9`. Guarded by
      `research/archive/cortex-v2/test_regressions.py`; the original `selftest.py` still passes 13/13.
* **Hash Randomization:** Replaced Python's process-dependent `hash()` with an MD5 digest so
  `word_to_rotation` is genuinely deterministic across parallel launches.

---

## 5. The Tripwire Playbook (What to Run and Watch)
The architecture is now mathematically sound and correctly wired. A tripwire is installed to
prevent silent regressions.

**How to run the next training epoch** — always enable the verify flag:

```bash
cd collapse_retrain
python train_collapse_embeddings.py \
    --train-path <wikitext-103 train.txt> \
    --output-dir model_collapse_v2 \
    --collapse-layers 4 \
    --verify-engine-trained \
    --sep-threshold 0.5
```

**What to watch, in order:**

1. **Startup:** `[init] optimizer tracking N params across M tensors`. `M` must exceed an
   embeddings-only run (the engine adds 3 anchors + 4 MLP tensors). If not, the engine is
   still detached — stop here.
2. **`[verify] epoch 1` — `max|Δanchor| > 1e-6`:** guarantees the engine is connected to the
   optimizer. If this fails, the optimizer bug has returned.
3. **Separation line — `max pairwise cos(E,C / E,N / C,N) < threshold`:** guarantees the
   Entailment (E), Contradiction (C), and Neutral (N) anchors are separating into distinct
   geometric gravity wells (trending toward 0 or negative, not toward +1).

### ⚠️ Critical Warning for the Next Run
If the tripwire fails at Epoch 1 because the anchors are **clumping** together (e.g.
`cos ≥ 0.5`) while still moving, the mechanics are fine and the **loss function is the
culprit**: `livnium_energy_loss` rewards low energy *at* each label's anchor but contains no
term that pushes the three anchors apart, so the cheapest solution is to collapse them onto one
point. **Do not adjust hyperparameters.** Rewrite the loss to explicitly penalize anchor
overlap before proceeding, e.g.

```python
# margin m (e.g. 0.0); lambda_sep weights the separation pressure
sep_loss = (relu(cos(E, C) - m) + relu(cos(E, N) - m) + relu(cos(C, N) - m))
loss = energy_loss + lambda_sep * sep_loss
```

(or add a cross-entropy head over the three anchor logits, which rewards separation
implicitly). Until a separation pressure exists in the objective, fixing the mechanics only
guarantees the engine *can* learn geometry — not that it *will*.

---

## 6. Status — CONFIRMED ON HARDWARE (Apple M5 / MPS)

The skip-gram objective never sees an NLI label, so the E/C/N anchors cannot specialize under
it (proven forward-only: the static collapse is label-blind and `anchor_E` / `anchor_C` are
interchangeable). A label-supervised path (`--task nli`) was added: the pair vector
`pool(premise) − pool(hypothesis)` is collapsed and classified by cosine to the three anchors
(cross-entropy on the gold label), plus an explicit anchor-separation penalty.

### Reproduce (do NOT commit the run outputs)

```bash
cd collapse_retrain
python3 train_collapse_embeddings.py \
    --task nli \
    --nli-path ../data/snli_1.0_train.jsonl \
    --output-dir model_nli_v1 \
    --verify-engine-trained --sep-threshold 0.5 \
    --lambda-sep 1.0 --temp 0.1
```

Defaults used above: `--dim 256`, `--epochs 3`, `--batch-size 512`, `--lr 3e-4`,
`--collapse-layers 4`, `--max-lines 200000`. On an M5 each epoch is ~6 s (195 batches).
The `model_nli_v1/` checkpoints (~50 MB each) are run artifacts and are git-ignored — re-run
the command to regenerate them.

### Observed result (3 epochs, SNLI, 200k examples, vocab 35,320)

| epoch | train_acc | max\|Δanchor\| | cos(E,C) | cos(E,N) | cos(C,N) |
|------:|----------:|---------------:|---------:|---------:|---------:|
| 1 | 0.534 | 0.072 | +0.062 | +0.107 | +0.000 |
| 2 | 0.638 | 0.142 | −0.002 | +0.039 | −0.000 |
| 3 | 0.664 | 0.177 | −0.000 | −0.004 | −0.001 |

Accuracy rises well above the 3-class chance baseline (0.33), anchors move every epoch
(optimizer is connected), and E/N/C drive to near-orthogonal — the separation tripwire passes
at every epoch. **Test 1 (the Geometry Crucible) is passed.** Next: evaluate the saved
checkpoint on `data/snli_1.0_test.jsonl` for held-out accuracy (Test 2).

The offline NumPy verifications (MPS norm, divergence-law fixed point/contraction, loss
gradient behavior) remain valid and were the predictors of the above.

**Files touched:** `research/archive/cortex-v2/mps.py`, `research/archive/cortex-v2/lattice.py`, `research/archive/cortex-v2/test_regressions.py`
(new), `research/nli/supervised-collapse/train_collapse_embeddings.py`, `research/nli/supervised-collapse/vector_collapse.py`,
`research/nli/supervised-collapse/basin_field.py`, `research/nli/supervised-collapse/text_encoder_collapse.py`.
