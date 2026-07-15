# Benchmarks — grade A−

Controlled comparisons live here, outside training code.

- `embeddings/matched-corpus/` freezes one cleaned corpus and compares noun
  collapse, SGNS, and PPMI-SVD under the same input.
- `nli/` contains the character/word Livnium ladder and conventional baselines.

External SNLI files have one canonical location:

```text
benchmarks/nli/data/snli/snli_1.0_{train,dev,test}.jsonl
```

The shared path modules in `models/collapse-nli/` and
`research/language-probes/` resolve to this directory. Generated data and model
outputs remain gitignored.
