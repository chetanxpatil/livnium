# Infected Archive Nova-v3 and Quantum-Embed Audit

Updated: 2026-07-26

## Scope

Audited together:

- `/Users/chetanpatil/Desktop/test/lab/infected/Archive/nova_v3`
- `/Users/chetanpatil/Desktop/test/lab/infected/Archive/quantum_embed`

The roots occupy roughly 536 MB combined. Neither is tracked by the parent
`test` Git repository, and neither contains a nested Git history. They are a
mixed-time archive: most Nova source dates to December 2025, the surviving
checkpoint and some training source date to March 2026, and two core source
files date to June 2026. Current source therefore cannot automatically be
treated as the exact source that produced every artifact.

No source project was edited during this audit.

## What is actually present

`nova_v3` is mostly a full 483 MB SNLI data copy. Its train, development, and
test JSONL files have the same SHA-256 identities as the already audited
Sacred/Lab data:

| Split | SHA-256 |
|---|---|
| train | `ee95dfbc57800f7b1f62b7602ad2b176c2b983210435a49238e660324a01e963` |
| dev | `9c03faff70182ef086ebfeed2cffbabb5fcc6a84a8b3314decbbb5b01f07f4bf` |
| test | `1147550151ca8b16ddb31d8dc0e739a670a65f7c3e64aac5951eaeff103fb220` |

Many ambitious-looking model or run names are empty directories:

- `truth-legacy`
- `snli_legacy_basins`
- `snli_geom_basins`
- `truth`
- `truth-sanskrit`
- `snli_quantum_basins2`
- `snli_quantum_basins`
- `truth-geom`
- `runs/snli_quantum_v01`

The adjacent `quantum_embed/model_qe_v01` and
`quantum_embed/model_full_physics` directories are also empty. Empty names are
preserved as evidence of attempted branches, not counted as completed models.

## The sole checkpoint is a duplicate

Only one PyTorch weight survives:

`nova_v3/model/snli_quantum_collapse1/best_model.pt`

Its SHA-256 is:

`9a32ffcd1bd939a95cb8dfbcfd52453a750d1004404206c84a34ad7d81f78dff`

This is byte-for-byte identical to the already audited sacred
`collapse1-static` checkpoint. Its saved configuration disables dynamic basins
and points to the exact embedding backbone recovered under
`test/livnium-sacred-v2`.

A fresh deterministic, noise-free replay on all 9,824 valid SNLI test examples
gave:

- correct: 7,478
- accuracy: **76.1197%**
- confusion matrix, actual rows and predicted columns in E/C/N order:
  `[[2765,191,412],[260,2521,456],[566,461,2192]]`

This exactly confirms the previous collapse1-static replay. It is not an
additional independent result or a better model.

## The orphan calibrated-error artifact

`nova_v3/errors_calibrated.jsonl` is the most important unique artifact in these
roots:

- SHA-256:
  `b7df0cc406c6b03824ddc94548137397dcedf8553db83ff72cf063ea8b56add2`
- size: 711,795 bytes
- rows: 2,243, all unique
- every row matches a valid SNLI test example
- every row is a misclassification
- probability argmax agrees with the saved predicted class on all 2,243 rows

If, and only if, this file is the complete error-only output for the 9,824 valid
test examples, it implies:

- correct: 7,581
- accuracy: **77.1682%**
- confusion matrix, actual rows and predicted columns in E/C/N order:
  `[[2734,144,490],[207,2527,503],[412,487,2320]]`

The error confidence has mean `0.7161`, median `0.7065`, and 90th percentile
`0.9578`. Gold-class error counts are 899 neutral, 710 contradiction, and 634
entailment.

This file is not the error output of the surviving duplicate checkpoint. A
fresh collapse1-static replay has 2,346 errors; only 1,466 overlap, with 880
fresh-only and 777 calibrated-only errors (Jaccard `0.4694`). The calibrated
file predates the saved checkpoint and no matching model, calibration command,
seed, log, or complete prediction file survives.

Decision: preserve it as a **partial historical prediction artifact**. The
conditional 77.17% number may be cited only with its missing-provenance warning;
it cannot become the best valid Livnium checkpoint.

## Recovered leakage path

The archive's default Nova-v3 training and chat evaluators pass gold `labels`
into `collapse_dynamic` whenever dynamic basins are enabled. The training
defaults enable dynamic basins and default to the geometry encoder.

That interface injects the target class into held-out inference. It is an older
source stage of the same failure directly measured in `LAB_NOVA_SNLI_AUDIT.md`,
where proper inference scored 32.99% but gold-routed inference scored 90.58%.

The saved collapse1 checkpoint in this archive explicitly disables dynamic
basins, so its valid 76.12% replay is not invalidated by this code path.

The evaluators also print confusion headers in E/N/C order while their numeric
label mapping is E/C/N. Historical printed class reports from this source need
relabeling before use.

## Abandoned deterministic encoders

### Geometry encoder

`nova_v3/text/geom_encoder.py` creates a base-27 token signature and maps it to
three cube coordinates plus six derived features before an optional Transformer
and attention pooler.

The conversion uses only the signature modulo 27. Every character after the
first is discarded by the coordinate mapping, so words with the same first
character have identical base features. Direct checks include:

- `cat = car = c`
- `dog = door`
- `apple = angle`
- `abc = acb`

There are at most 27 lexical base types before learned contextual processing.

Decision: preserve the general idea of deterministic token geometry, but retire
this mapping as a lexical representation. No checkpoint survives.

### Sanskrit/phoneme encoder

`nova_v3/text/sanskrit_encoder.py` maps characters to a simplified phoneme
feature table, applies a learned projection, and mean-pools characters.

Because pooling ignores order, anagrams produce the same representation up to
floating-point summation noise (`abc/acb` and `cat/tac` differed by at most
approximately `2.98e-8` in a direct check). The character table also maps many
letters to the same simplified place/manner features.

Decision: preserve phonetic-feature encoding as a possible component, but retire
the claim that this bag-of-characters representation supplies sentence
semantics. No checkpoint survives.

## Adjacent quantum-embed root

The meaningful current source files in `quantum_embed` are exact copies of the
already preserved Sacred-v2 `code/quantum-pretrain` source. Its backup source is
also byte-identical to its current source. WikiText validation and test files
are exact copies of the Lab Nova-SNLI physics-embedding corpus files.

Important boundaries:

- no embedding weight or result artifact survives here;
- the optimizer trains only the word embedding table, not the collapse update
  or anchors;
- random seeds are not fixed;
- the evaluation script rebuilds vocabulary IDs from the test corpus instead of
  using the checkpoint vocabulary, so IDs can be assigned to the wrong rows;
- the analogy script hardcodes MPS and a nonexistent local model path.

Decision: this is a **source-only precursor/duplicate**, not separate evidence
of successful quantum embeddings. Its experiment idea and surviving one-epoch
artifact are already represented by `NS-01` in `IDEA_LEDGER.md`.

## Tests and preservation decision

Pytest found no collected tests across the two roots. Focused source checks were
therefore used for the encoder collisions, checkpoint replay, and artifact
consistency.

Preserve these roots because they contain:

- provenance for an older target-label leakage interface;
- the unique orphan calibrated-error artifact;
- two abandoned deterministic encoder ideas;
- empty branch names documenting attempted directions.

Do not count the copied data, copied source, or duplicate checkpoint as new
models. Do not delete or merge the source roots until the global recovery pass
is complete.

## Next lineage node

Audit one canonical copy of:

`/Users/chetanpatil/Desktop/test/lab/infected/python/clean-nova-livnium/archives-local/arch-archive/experiments/NLI-ALL`

It is a roughly 975 MB archive containing eight NLI generations (`nli`,
`nli_simple`, and `nli_v3` through `nli_v8`), saved `brain_state.pkl` files,
rules, physics documents, pattern files, and tests. No `.pt`, `.pth`, or
`.ckpt` model was found in the initial map.

Two near-mirrors also exist under `python/clean=noba=back` and
`workspace/clean-nova-livnium`. Meaningful source/data content matched in the
initial recursive comparison; observed differences were Finder metadata,
`__MACOSX` remnants, and generated Python bytecode caches. Audit the canonical
path once, then use hashes to classify mirror-only material.

Post-audit status, 2026-07-26: this next node is now incorporated in
`NLI_ALL_AUDIT.md`. All 186 meaningful files match across the three roots. The
simple generation is 40.71% on the full valid SNLI test split; runnable v3–v7
are near chance, v8 is broken, and the high debug/self-label/resubstitution
numbers are not valid NLI accuracy.
