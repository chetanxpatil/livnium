# SNLI Baselines and the Cross-Interaction Target (CollapseNLI)

Reference note: where CollapseNLI sits on SNLI, what the real baselines are, and
the one ingredient that separates it from the simple models. Numbers verified
against the official Stanford SNLI leaderboard (see Sources).

## Verified landscape (SNLI test accuracy)

| model | test acc | uses pretrained embeddings? | trained on |
|---|---:|---|---|
| chance / majority | ~34% | — | — |
| Unlexicalized features (Bowman '15) | 50.4% | no | SNLI only |
| **CollapseNLI (full recipe)** | **~69%** | **no** | **SNLI only** |
| + unigram/bigram + cross features (Bowman '15) | 78.2% | no | SNLI only |
| 100D LSTM encoder (Bowman '15) | 77.6% | yes (embeddings) | SNLI |
| Decomposable Attention (Parikh '16) | 86.3% | yes (GloVe) | SNLI |
| ESIM (Chen '17) | 88.0% | yes (GloVe) | SNLI |
| BERT / RoBERTa fine-tuned | 90–93% | yes (massive pretraining) | pretrain + SNLI |

Key facts this corrects:

- There is **no published model at ~69%** on the leaderboard; 69% sits in the
  gap between the unlexicalized-feature floor (50%) and every real model (77%+).
- The **fair benchmark** for "from scratch, SNLI only, no pretrained embeddings"
  is **not** 69% — it is **78.2%**, achieved by a plain lexical *feature*
  classifier (Bowman '15). CollapseNLI is currently ~9 points below it on equal
  footing.
- "Lexicalized" is not a data type to add — CollapseNLI is already word-based
  (that is why it is at 69%, not 50%). The gap to 78% is a *mechanism*, below.

## The missing mechanism: cross-sentence word interaction

The 50% → 78% jump in Bowman '15 came from adding **lexical features**, the
decisive ones being **cross-unigram / cross-bigram** features:

- *unigram / bigram*: indicators for words / adjacent word pairs in a sentence
  (bag-of-words). A feature template, not a model.
- *cross-unigram*: an indicator for each pair *(premise word, hypothesis word)*
  (Bowman restricts to pairs sharing a POS tag). This hard-codes
  premise-word × hypothesis-word co-occurrence — an explicit cross-sentence
  interaction.
- *cross-bigram*: the same for word pairs.

These are **feature-engineering methods** fed to a linear classifier — the
hand-built analog of attention. They inject the one thing CollapseNLI lacks:
premise and hypothesis words interacting *before* the decision.

CollapseNLI mean-pools each sentence into a single vector and compares the two
vectors (`meanpool → (u−v) → collapse`), which discards word-to-word interaction.
Both Bowman's cross-features (hand-built) and Parikh's Decomposable Attention
(learned) keep it — and both land far above CollapseNLI. That is the diagnosis,
confirmed from two directions.

## Targets and next experiment

Target ladder (honest, fair-footing first):

1. **Match 78.2%** — beat the SNLI-only, no-embeddings lexical feature baseline.
   This is the fair fight; passing it shows the architecture earns its keep
   against the simplest thing that works.
2. Then approach the encoder range (77–86%) — likely needs cross-sentence
   alignment and/or pretrained word vectors.

Next experiment (the lever, not more data):

- **Let premise and hypothesis words interact before the collapse.** Either
  (a) add cross-unigram features (a sparse premise×hypothesis word-pair layer)
  to CollapseNLI, or (b) add a lightweight alignment step (Decomposable-Attention
  style, ~380k params) so each hypothesis word attends to premise words, *then*
  collapse. Measure whether CollapseNLI moves off the 69% floor toward 78%+.
- Separately, an isolation test: warm-start CollapseNLI with GloVe ("world
  words") and re-measure, to split "weak vocabulary" from "weak mechanism."

## Result: alignment bolted on (measured)

Controlled comparison, full SNLI train, 8 epochs, no pretrained embeddings,
identical except the cross-sentence alignment step (`--align`):

| variant | dev acc | test acc | params |
|---|---:|---:|---|
| CollapseNLI (mean-pool, baseline) | 66.1% | — | 5.25M |
| CollapseNLI + alignment (before collapse) | **74.7%** | **74.4%** | 5.52M |

**+8.6 dev points from alignment alone** — the diagnosis confirmed. The aligned
model also learns far faster (reaches the baseline's *final* 66% in epoch 1). Dev
and test agree to within 0.2 points (74.66% / 74.43% on the official leak-free
splits via `eval_snli.py`), so there is no dev overfitting. Per-class (test):
entailment 73.6%, contradiction 80.7%, neutral 69.1% (neutral still hardest, as
expected). The ~260k extra params are two small FFNs; the collapse engine is
unchanged — alignment just supplies per-word correspondence before the pooling
instead of after.

Placement: 74.4% (test) clears the hypothesis-only floor (~67%) comfortably and
sits ~3.5 points below the simplest SNLI-only baselines (LSTM 77.6%, lexical
features 78.2%). The dev curve plateaued near 74–75, so closing the last ~3.5 points likely
needs another ingredient — GloVe warm-start ("world words"), intra-sentence
attention (Parikh's +0.5), or running the collapse to a true fixed point — not
just more epochs.

## Honest framing (for presentation)

> CollapseNLI reaches ~69% on SNLI trained on SNLI alone, with no pretrained
> embeddings — well above the bag-of-words feature floor (~50%), below the
> published encoders. The fair same-footing baseline is a lexical feature
> classifier at 78.2% (also SNLI-only, no embeddings); CollapseNLI is ~9 points
> under it. The gap is a known mechanism — cross-sentence word interaction
> (hand-built cross-features, or learned attention) — which CollapseNLI's
> sentence pooling currently discards. The novelty is an attention-free
> architecture built and trained from scratch; the roadmap is to add cross-
> sentence interaction and test whether it closes the gap.

## Sources

- SNLI leaderboard — Stanford NLP: <https://nlp.stanford.edu/projects/snli/>
- Natural Language Inference — NLP-progress:
  <https://nlpprogress.com/english/natural_language_inference.html>
- Bowman et al. 2015, "A large annotated corpus for learning natural language
  inference" (SNLI; feature baselines).
- Parikh et al. 2016, "A Decomposable Attention Model for Natural Language
  Inference" (cross-sentence attention, ~86%).
