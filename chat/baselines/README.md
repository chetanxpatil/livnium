# Matched-baseline harness — collapse vs SGNS vs PPMI+SVD

The experiment that makes (or breaks) the noun-collapse claim: all three model
families on the **identical** corpus, preprocessing, vocabulary, noun targets,
±5 window, 256 dimensions and occurrence/token budget, across multiple seeds,
scored with tie-aware Spearman on SimLex-999, WordSim-353 and MEN.

## Run it

```bash
cd chat/baselines
./run_all.sh ~/path/to/enwiki-latest-pages-articles-multistream.xml.bz2
```

Resumable: every stage skips finished outputs; the collapse trainer resumes
mid-pass from its checkpoint. On macOS the script arms `caffeinate -i` so the
machine can't sleep mid-run. Deps: `torch`, `gensim`, `scipy`, `scikit-learn`
(`pip install "livnium-core[results,experimental]" gensim`).

## Matching guarantees

| Axis | How it is enforced |
|---|---|
| Corpus | Frozen ONCE to `work/corpus.txt` (same `clean()` as the published run); SHA-256 in `work/corpus_manifest.json`; every artifact carries the hash and `report.py` refuses to mix hashes |
| Preprocessing | Done once at freeze time; trainers only `.split()` |
| Vocabulary + noun targets | Built once (`build_vocab.py`, same rules as `noun_collapse_pure.py`); SGNS/PPMI corpora are restricted to this lexicon |
| Window | ±5 everywhere |
| Dimensions | 256 everywhere |
| Budget | Collapse: `--max-occ` occurrence budget (default one full pass). SGNS/PPMI: one pass over the same corpus by default; token budget recorded in each model's `.meta.json` |
| Seeds | ≥5 (`SEEDS="0 1 2 3 4"`); collapse: init + negative sampling; SGNS: init + sampling (workers=1 for determinism); PPMI: randomized SVD only (counts are deterministic) |
| Correlation | Tie-aware `scipy.stats.spearmanr`, nothing else |
| Coverage | Per-pair predictions saved to CSV; the table reports rho on each model's own coverage AND on the strict pair intersection covered by **all** families (equal coverage), with pair counts |
| v1 vs v2 | `collapse_v1` (legacy objective, no false-negative mask — the published checkpoint's setting) and `collapse_v2` (masked negatives) are separate families, never pooled |

## Outputs

- `work/models/*.npz` — `{words, vectors}` per model per seed, plus `.meta.json` lineage
- `work/results/*.pairs.csv` — raw per-pair predictions (the audit trail)
- `work/results/aggregate.{json,csv}` — raw + mean ± std per family per dataset
- `work/BASELINE_RESULTS.md` — the comparison table, generated, never hand-edited

## Reading the numbers

`simlex_nouns` (official SimLex, POS = N) is the row comparable to the
published ρ = 0.362; `simlex999` is the standard full set. The published
word2vec/GloVe numbers in the model card come from different corpora — the
only honest comparison is inside this table, where everything is matched.
