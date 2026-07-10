<!--
  This is the measured head-to-head: Livnium and a standard learned representation
  (GloVe) against the boring baselines (majority, bag-of-words, hypothesis-only), on
  the same split, with a shuffled-label kill-test every time. See README.md in this
  folder for how to reproduce. Short story: on SNLI, char-level Livnium = 43%,
  word-level Livnium = 60% (matching bag-of-words), the geometry alone = 38% (≈chance);
  on ANLI everything sits at chance. Meaning lives in the words, not the geometry.
  The Supervised Collapse Model (v1) reaches 68.92% on SNLI, clearing the hypothesis-only
  baseline by warping word embeddings to learned attractors.
-->

# Results — Livnium vs the boring baselines (measured)

*Run 2026-06-13 17:08. GloVe-wiki-gigaword-100, mean-pooled, pair features [u, v, |u-v|, u*v], logistic regression. Same split per task, seed 42.*

## SNLI (does learned beat word-counting?)

Train subsample = 50,000 (same for all models), test = 9,824.

| model | test acc % |
|---|---|
| majority | 34.3 |
| full BoW (prem+hyp) | 59.4 |
| **GloVe-100 avg + logreg (learned)** | **60.7** |
| hypothesis-only BoW | 61.5 |
| **Supervised Collapse Model (v1)** | **68.92** |
| shuffled-label control | 33.0 |

Margin: GloVe learned beats full BoW by **+1.3 pts**, hyp-only by **-0.8 pts**. Supervised Collapse (v1) clears the hypothesis-only artifact by **+7.42 pts**.

GloVe confusion matrix (rows=true, cols=pred):

```
          entail neutral  contra
 entail     2181     648     539
neutral      650    1886     683
 contra      665     677    1895
```

## ANLI (the hard, artifact-free task — honest run)

Train = R1+R2+R3 (30,000), tested per round. Bar = hypothesis-only (~38-41%).

| round | maj | hypOnly | Count | Hash | TFIDF | **GloVe** | shufLbl |
|---|---|---|---|---|---|---|---|
| R1 | 33.3 | 39.6 | 34.7 | 41.3 | 39.0 | **34.2** | 34.4 |
| R2 | 33.3 | 36.9 | 34.5 | 37.8 | 36.6 | **36.7** | 34.3 |
| R3 | 33.5 | 36.7 | 34.5 | 36.8 | 35.8 | **34.4** | 33.3 |

## Verdict (measured)

- **SNLI:** learned GloVe = **60.7%** vs full BoW 59.4% vs hyp-only 61.5%. Learned edged full BoW by +1.3 but did **not** clear the hypothesis-only artifact (the stronger bar). Mean-pooling can't model premise→hypothesis alignment, which is exactly what SNLI rewards — so the literal "learned > BoW" gate barely passed, the real bar did not. Crossing the hyp-only artifact on SNLI needs interaction modeling (Rung 3), not a better static embedding.

- **ANLI:** GloVe minus hyp-only bar: R1 -5.4, R2 -0.2, R3 -2.2. Does NOT clear the hyp-only bar on all rounds — honest near-chance, as expected on ANLI.

- **Kill-test:** shuffled-label controls collapse to chance (SNLI 33.0%, ANLI ~34%) → pipeline has no leakage.

---

# Livnium's turn — the geometric encoder in the same fair harness

*Run 2026-06-13 17:18. Livnium base-27 -> N=3 lattice encoder (no learned parameters), pair features [u,v,|u-v|,u*v], same logreg, same split, seed 42.*

## SNLI

| model | test acc % |
|---|---|
| majority | 34.3 |
| full BoW (prem+hyp) | 59.4 |
| hypothesis-only BoW | 61.5 |
| GloVe-100 learned | 60.7 |
| **Livnium geometry (static)** | **43.2** |
| Livnium shuffled-label control | 33.5 |

Livnium confusion matrix (rows=true, cols=pred):

```
          entail neutral  contra
 entail     1621     927     820
neutral     1007    1301     911
 contra      921     995    1321
```

## ANLI (bar = hypothesis-only ~37-41%)

| round | hypOnly | best BoW | GloVe | **Livnium** | Livnium shufLbl |
|---|---|---|---|---|---|
| R1 | 39.6 | 41.3 | 34.2 | **33.0** | 33.3 |
| R2 | 36.9 | 37.8 | 36.7 | **32.2** | 33.3 |
| R3 | 36.7 | 36.8 | 34.4 | **32.0** | 33.5 |

## Verdict for Livnium (measured, same rules)

- **SNLI:** Livnium static geometry = **43.2%**. vs full BoW 59.4, hyp-only 61.5, GloVe 60.7. Below BoW. Below the hyp-only artifact.

- **ANLI:** Livnium by round R1 33.0 / R2 32.2 / R3 32.0; hyp-only bar 39.6/36.9/36.7. Does NOT clear the hyp-only bar on all rounds — near chance, like every static representation here.

- **Kill-test:** Livnium shuffled-label control collapses to chance (SNLI 33.5%, ANLI ~33%) -> the pipeline is honest; the score is the representation's, not leakage.

---

# Word-level Livnium — encode the word, not the letter

*Run 2026-06-13 21:29. Each word hashed onto the N=27 lattice (19,683 cells); per-cell occupancy for premise+hyp; same logreg/split/seed 42. Geometry-only = exposure-class fractions + mean coord of occupied cells.*

## SNLI

| model | test acc % |
|---|---|
| majority | 34.3 |
| char-level Livnium | 43.2 |
| Livnium geometry-only (word) | 38.0 |
| full BoW (prem+hyp) | 59.4 |
| GloVe-100 learned | 60.7 |
| hypothesis-only BoW | 61.5 |
| **word-level Livnium occupancy** | **59.9** |
| word-level shuffled-label control | 32.9 |

Word-level Livnium confusion matrix (rows=true, cols=pred):

```
          entail neutral  contra
 entail     2138     611     619
neutral      684    1907     628
 contra      749     651    1837
```

## ANLI (bar = hypothesis-only ~37-41%)

| round | hypOnly | best BoW | **word-Livnium** | shufLbl |
|---|---|---|---|---|
| R1 | 39.6 | 41.3 | **35.7** | 33.7 |
| R2 | 36.9 | 37.8 | **34.4** | 33.5 |
| R3 | 36.7 | 36.8 | **34.8** | 33.7 |

## Verdict — did moving to the word unit change the number?

- **SNLI:** char-level Livnium 43.2 -> **word-level 59.9**. Climbs +16.7 pts and reaches bag-of-words territory. Clears full BoW.

- **Geometry-only (word):** 38.0% — the lattice *shape* alone (no word identity) is near chance. The accuracy comes from WORD IDENTITY occupying cells, not the geometry.

- **ANLI:** word-Livnium R1 35.7 / R2 34.4 / R3 34.8; bar 39.6/36.9/36.7. Matches the Hash baseline — same word-counting ceiling.

- **Kill-test:** shuffled-label control at chance (SNLI 32.9%, ANLI ~34%). Honest.


**Reading:** once Livnium encodes the *word* (one lattice cell per word) instead of the *letter*, it stops being letter-geometry and becomes a lattice-indexed bag-of-words — and it recovers BoW-level accuracy. That is the lesson made concrete: the container (geometry) is fine; the accuracy lives in the units you put in it (words), and at that point you are doing word-counting, not reasoning. ANLI confirms it: no word-counting method, geometric or not, beats the bar there.

---

# Supervised Collapse Model (v1) — The Learned Attractor Benchmark

*Run 2026-06-18. 256-dimensional learned embeddings + 4-layer collapse engine trained end-to-end on SNLI (549,367 examples) with anchor-separation pressure. Checked on dev set periodically, evaluated once on test.*

## SNLI Test Set

| Model | test acc % |
|---|---|
| majority | 34.3 |
| char-level Livnium | 43.2 |
| GloVe-100 learned | 60.7 |
| hypothesis-only BoW | 61.5 |
| **Supervised Collapse (v1)** | **68.92** |

## Verdict & Ablation Results (Run 2026-06-18)

We executed the ablation script `collapse_retrain/ablate_nli.py` on the properly trained NLI checkpoint `nli_epoch20.pt` where the collapse engine optimizer connection was active. The frozen embeddings were evaluated on SNLI dev and test sets:

| Model Configuration | Dev Acc % | Test Acc % |
|---|---|---|
| **Full Collapse** (Warping + Cosine classification) | 69.62 | 68.92 |
| **Linear Head Probe** (Linear head on top of frozen `u - v`) | 63.76 | 64.06 |
| **MLP Head Probe** (2-layer MLP on top of frozen `u - v`) | 70.11 | 70.13 |
| **Random-Anchor Collapse** (Randomized anchors) | 32.96 | 32.44 |
| **Random-Embedding Collapse** (Random embeddings) | 35.81 | 34.73 |

### Verdict & Interpretation

1. **Collapse beats a linear readout — on frozen embeddings**: The **Full Collapse** model (68.92% test) beats the linear probe (64.06%) by **+4.86 points**. Important caveat: this is a **post-hoc probe on embeddings that were originally optimized for collapse**, so the comparison is biased toward the collapse head. It shows the collapse readout uses these embeddings better than a linear map does; it does **not** prove causality.
2. **Learned geometry is load-bearing**: Randomizing the anchors or the embeddings collapses accuracy to chance (~32.4% / ~34.7%), so the trained anchor/embedding geometry is necessary for this model's performance — necessary, but not shown to be uniquely sufficient.
3. **Comparable to an MLP, not better**: A 2-layer MLP probe on the same frozen embeddings scores **70.13%**, above Full Collapse (68.92%). Collapse offers a structured, inspectable attractor readout at comparable (slightly lower) accuracy — a representational trade-off, not a win.

**What is still required**: a matched **end-to-end multi-seed ablation** — train linear-head, MLP-head and collapse-head models from scratch under identical budgets and several seeds — before attributing any gain causally to the collapse dynamics.

## Note on ANLI and Supervised Collapse

As of June 2026, the Supervised Collapse Model is only trained and evaluated on **SNLI** (where it reaches 68.92% test accuracy; a post-hoc frozen-embedding probe favors collapse over a linear head by +4.86%, pending a matched end-to-end ablation). Evaluation on **ANLI** has not been performed because:
1. **Adversarial Complexity**: ANLI is designed to be adversarial and artifact-free, requiring deep semantic reasoning that is typically out of reach for non-attention mean-pooled embeddings (which discard word order and sentence interaction).
2. **Vocabulary Coverage**: The current vocabulary is constrained to SNLI (~50k words) and out-of-vocabulary words are mapped to `<unk>`. Running on ANLI would require training/fine-tuning embeddings on the ANLI training dataset to cover its broader vocabulary.


