# Results & reproduction

This folder holds the **measured** head-to-head between Livnium and the standard
boring baselines, plus the exact scripts that produced it. The write-up is in
[`RESULTS.md`](RESULTS.md).

## The rule every number here had to pass

1. **Beat the dumbest competent baseline** (majority vote, bag-of-words,
   hypothesis-only) on the *same* split — or it doesn't count.
2. **No leakage.** A shuffled-label control is run every time; if the pipeline
   were cheating, that control would score above chance. It never does (~33%).
3. **Same everything.** Every model gets the same training data, the same
   classifier, the same preprocessing. Only the *representation* changes.

## What's in here

| file | what it does |
|---|---|
| `RESULTS.md` | the full table: char-level Livnium, word-level Livnium, GloVe, bag-of-words, all baselines, on SNLI and ANLI |
| `rung2_lib.py` | loads the data, runs the GloVe and bag-of-words baselines |
| `rung2_livnium.py` | the **character-level** Livnium encoder (base-27 → 3×3×3 lattice) |
| `rung2_livnium_word.py` | the **word-level** Livnium encoder (each word → one lattice cell) |

## Reproducing it

The datasets are **not** bundled (they're large and publicly available). To re-run:

1. **SNLI** — download from https://nlp.stanford.edu/projects/snli/ and place
   `snli_1.0_{train,dev,test}.jsonl` where the scripts expect them (see the path
   constants at the top of `rung2_lib.py`).
2. **ANLI** — download `facebook/anli` (parquet) from Hugging Face and place the
   `anli_{train,test}_r{1,2,3}.parquet` files in a `benchmarks/nli/anli_data/` folder.
3. Install deps: `pip install -r ../requirements.txt`
4. Run, e.g.:
   ```bash
   python rung2_livnium_word.py snli     # word-level Livnium on SNLI
   python rung2_livnium_word.py anli      # ...and on ANLI
   ```

Every run is seeded (`seed = 42`) so the numbers are reproducible. The scripts
were written to run in short, memory-light steps; adjust the train-subsample sizes
at the top if you have more compute.

> **Note on honesty:** these scripts are the actual record of how the numbers were
> produced, kept here so anyone can check them. They are research scripts, not a
> polished library — the polished, proven part of the project is `packages/livnium-core/src/livnium_core/`.
