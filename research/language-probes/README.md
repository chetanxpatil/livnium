# Language probes — grade B

Standalone tests for where language signal comes from: flat lexical overlap,
ordered paths, all-pairs relations, learned gravity wells, or the nested ping
representation. These scripts are deliberately separate from trained model code.

They run on built-in synthetic/smoke samples when possible. For SNLI, place the
dataset in `benchmarks/nli/data/snli/`; `paths.py` is the single path definition.

Key files:

- `ordered_sentence_embed.py` — order-sensitive sentence descent.
- `relational_sentence_embed.py` — directed all-pairs word relations.
- `gravity_embed.py` / `supervised_gravity.py` — learned field experiments.
- `ping_embed_probe.py` — nested path geometry against a flat character baseline.
- `vocab_overlap.py` — diagnostic between noun and chat vocabularies.

These are representation probes, not production models.
