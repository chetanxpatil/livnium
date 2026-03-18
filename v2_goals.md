# Livnium v2 — Goals

## What v1 Proved

The attractor dynamics head works. A state vector can collapse toward a semantic basin across L steps under geometric forces, with a provable local contraction guarantee. The physics is real.

What v1 did not prove: whether the approach can compete with standard baselines when the encoder is not a bottleneck.

v2 answers that question — and then goes further.

---

## Goal 1 — Break the Encoder Ceiling (BERT bootstrap)

**Status: done**

v1 was limited by the bag-of-words encoder. Mean pooling destroys word order, negation, and cross-sentence structure. The attractor head cannot recover information the encoder threw away.

v2 plugs in BERT as a bootstrap encoder and trains it jointly with the collapse dynamics:

| Encoder | Accuracy |
|---|---|
| Pretrained BoW (v1) | 76.32% |
| Frozen BERT | ~61% (wrong — BERT geometry doesn't align with attractor space without joint training) |
| Joint BERT bi-encoder | **82.06%** ✓ |
| Joint BERT cross-encoder | in progress |

**Key finding:** BERT must train *alongside* the collapse engine — not just provide frozen features. When BERT fine-tunes against the attractor objective, it reshapes its geometry to align with the basin structure.

---

## Goal 2 — Fix Role-Reversal (Cross-Encoder)

**Status: in progress**

The bi-encoder architecture encodes premise and hypothesis independently. BERT never sees both sentences together. This means "a man bites a dog" and "a dog bites a man" produce nearly the same `h0 = v_h - v_p` because the role structure is lost before the collapse engine sees anything.

**Fix:** Cross-encoder architecture — feed `[CLS] premise [SEP] hypothesis [SEP]` as a single sequence. BERT's attention heads can now attend across sentences. Role structure, negation, quantifier mismatches — all visible inside the transformer.

This is how SOTA NLI models (>90%) work. The collapse engine gets a genuinely relational `h0` instead of a reconstructed one.

**Target:** 84%+ dev accuracy, elimination of E↔C role-reversal errors.

---

## Goal 3 — Livnium-Native Encoder (no BERT at inference)

**Status: in progress**

The central v2 insight:

> BERT discovers the geometry. Livnium defines how it behaves. Next step: make Livnium generate it.

BERT was always a bootstrap. Once the attractor dynamics have converged and the anchor geometry is stable, BERT's 768-dim space can be projected into a compact **Livnium basis** — a coordinate system where every axis has a physical meaning:

- cos(E, C) = −0.561 → entailment and contradiction are nearly antipodal
- cos(E, N) = −0.771 → entailment and neutral are well-separated
- cos(C, N) = −0.087 → contradiction and neutral are nearly orthogonal

**71.2% of BERT's h0 variance lives in 32 dimensions.** The other 736 dims are noise for this task.

### The Pipeline

```
Stage 1 (done):    words → BERT (joint) → collapse → label
Stage 2 (done):    extract Livnium basis (768 → 32 projection)
Stage 3 (active):  words → LivniumNativeEncoder → collapse → label
```

`LivniumNativeEncoder` is a small transformer (~3M params total) trained end-to-end against the attractor objective. No BERT at inference. The loss is the same — cross-entropy + contrastive repulsion + false-neutral penalty. The geometry is seeded from the extracted basis so training starts with meaningful anchor positions.

**Target:** match cross-encoder accuracy at 33× fewer parameters.

---

## Goal 4 — Interpretable Trajectories

**Status: not started**

With a Livnium-native encoder operating in 32-dim space, collapse trajectories become inspectable:

- **Visualise** E/N/C clusters by projecting to the first 3 basis directions
- **Watch** a vector collapse step-by-step toward its basin
- **Measure** uncertainty by how slowly a vector collapses (slow collapse = ambiguous example)
- **Explain** predictions by showing which anchor "won" the competition and at which step

This is what interpretability means in Livnium — not attention maps or SHAP values, but physical trajectory inspection.

---

## Goal 5 — Adversarial Sanity Suite

**Status: not started**

A minimal test suite for structural failures that accuracy metrics hide:

| Test | What it checks |
|---|---|
| Role reversal | "A bites B" vs "B bites A" — should flip E↔C |
| Negation flip | "is sleeping" vs "is not sleeping" — should flip E↔C |
| Quantifier swap | "some" vs "all" — neutral should become contradiction |
| Synonym entailment | "dog" / "canine" — should be entailment |
| Unrelated neutral | Completely unrelated sentences — should be neutral, not E or C |

Target: 100% on all 5 categories. A model at 82% overall that fails role reversal is not ready.

---

## Goal 6 — Livnium-Native Tokenizer (research frontier)

**Status: theoretical**

Every tokenizer today (BPE, WordPiece, SentencePiece) is designed for language modeling — units are chosen to minimise prediction loss on text.

Livnium's question is different: **what is the smallest unit of language that has a stable attractor address?**

A Livnium tokenizer would be trained so that its units have stable, predictable positions in the attractor space — not so that they compress text efficiently. The "vocabulary" would be semantic primitives, not frequency-based subwords.

This is the long horizon. No implementation timeline.

---

## Summary

| Goal | Status | Target |
|---|---|---|
| Joint BERT training | ✅ done | 82.06% ✓ |
| Cross-encoder (role-reversal fix) | 🔄 in progress | 84%+ |
| Livnium-native encoder | 🔄 in progress (training) | match cross-encoder, ~3M params |
| Interpretable trajectories | ⏳ not started | visualise E/N/C in 3D |
| Adversarial sanity suite | ⏳ not started | 100% on 5 structural tests |
| Livnium-native tokenizer | 💡 research frontier | — |

---

*v2 is not about making a better SNLI classifier. It is about proving that a physics-based representation system can learn to navigate semantic space without borrowing a language model to do the heavy lifting.*
