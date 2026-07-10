# Benchmarks — the honest record

The mathematics in this repo is proven. The separate question — *does the
geometry help a machine reason?* — was tested directly. This file reports the
result, including where it failed. That honesty is deliberate.

## Setup

- Task: 3-class natural language inference (entailment / neutral / contradiction).
- Test set: **ANLI** (Adversarial NLI), chosen because it has no hypothesis
  artifacts. On the easier SNLI, a model reading *only the hypothesis* scores
  ~65% by exploiting crowd-worker word habits; on ANLI that drops to ~37–41%,
  so you can only win by actual reasoning.
- Classifier head: logistic regression on extracted features, same split and
  preprocessing for every method.

## Boring baselines (the bar to beat), ANLI

| method | accuracy |
|---|---|
| majority class | ~33% |
| hypothesis-only bag-of-words | ~37–41% |
| Count / Hash / TF-IDF bag-of-words | ~35–39% |
| random-encoder (same dim) control | ~34% |
| shuffled-label sanity check | ~33% |

## Livnium geometry, ANLI (converged logistic regression)

| condition | accuracy | beats bar? |
|---|---|---|
| lexical (word overlap + negation) | ~29–34% | no |
| warp + fracture + cosine + overlap | ~29–35% | no |
| full geometry combo (random basis) | ~31–34% | no |
| full geometry combo (PCA-semantic) | ~29–33% | no |

Increasing training data 3k → 30k did not move the geometry off chance.

## Conclusion

On artifact-free data, the Livnium geometry performs **at chance** and does not
reach the bag-of-words bar. The SNLI numbers that once looked like wins (~76%)
were the dataset leaking the label through hypothesis wording, not reasoning by
the geometry.

## Why (the structural reason)

A useful representation requires **lossy** compression — discard the irrelevant
(spelling, surface form), keep the meaningful. Livnium's defining properties —
reversibility and conservation — guarantee that **nothing is ever discarded**.
A system that cannot forget cannot abstract. The cube is a faithful, lossless
container; meaning needs a lossy, data-shaped lens. They are different tools.

The math stands. The "beats standard methods at NLI" claim does not. Both are
recorded here on purpose.

---

## Update (Jun 2026) — the cleaner char-vs-word run

A later, simpler run made the structural point even clearer (full table in
[`../docs/results/nli.md`](../docs/results/nli.md)):

- **Char-level** Livnium (each *letter* → a cube symbol): **43%** on SNLI — above
  chance, below word-counting. It only sees spelling.
- **Word-level** Livnium (each *word* → its own lattice cell): **60%** on SNLI —
  jumps to match bag-of-words, because that is exactly what it has become.
- **Geometry alone** (the lattice shape, no word identity): **38%** — ≈chance.
- **Supervised Collapse Model (v1)** (learned embeddings + 4-layer attractor collapse): **68.92%** on SNLI — clearing the hypothesis-only baseline by warping word embeddings to learned attractors. A post-hoc frozen-embedding probe scores 64.06% with a linear head, 68.92% with collapse and 70.13% with an MLP; because the embeddings were originally optimized for collapse, a matched end-to-end multi-seed ablation is still required before attributing the gain to the collapse dynamics.
- **ANLI**: all of the above sit at chance (~33%), like every word-counting method. (The Supervised Collapse model has not been evaluated on ANLI due to adversarial complexity and vocabulary bounds).

The accuracy tracks *word identity*, not the geometry. Confirmed by a shuffled-label
control that collapses to chance every time.
