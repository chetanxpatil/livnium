<!--
  This is the measured head-to-head: Livnium and a standard learned representation
  (GloVe) against the boring baselines (majority, bag-of-words, hypothesis-only), on
  the same split, with a shuffled-label kill-test every time. See README.md in this
  folder for how to reproduce. Short story: on SNLI, char-level Livnium = 43%,
  word-level Livnium = 60% (matching bag-of-words), the geometry alone = 38% (≈chance);
  on ANLI everything sits at chance. Meaning lives in the words, not the geometry.
-->

# Results — Livnium vs the boring baselines (measured)

*Run 2026-06-13 17:08. GloVe-wiki-gigaword-100, mean-pooled, pair features [u, v, |u-v|, u*v], logistic regression. Same split per task, seed 42.*

## SNLI (does learned beat word-counting?)

Train subsample = 50,000 (same for all models), test = 9,824.

| model | test acc % |
|---|---|
| majority | 34.3 |
| hypothesis-only BoW | 61.5 |
| full BoW (prem+hyp) | 59.4 |
| **GloVe-100 avg + logreg (learned)** | **60.7** |
| shuffled-label control | 33.0 |

Margin: learned beats full BoW by **+1.3 pts**, hyp-only by **-0.8 pts**.

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

