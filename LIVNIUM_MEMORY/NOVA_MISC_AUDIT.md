# Nova-and-Misc Audit

Date: 2026-07-27

Status: incorporated as a mixed historical/diagnostic family.

This closes the final `_ORGANIZED/02_Experiments` family. It reconciles the
thirteen scripts against the Sacred, Nova, basin-memory, cube-embedding, and
NLI lineages instead of treating them as thirteen new projects.

No historical source, model, result, dataset, image, or state file was edited.

## 1. Inventory and preservation identity

The organized folder contains thirteen meaningful Python files. Every file is
an exact SHA-256 copy of the same-named file at the root of
`/Users/chetanpatil/Desktop/test`.

| File | SHA-256 | Role |
|---|---|---|
| `ablation_study.py` | `b3111e938249f9b4e8941f5f639d4c17347b7f92e031875eef3b736c1e1a4b37` | Cube-embedding ablation already incorporated |
| `evaluate_ablation.py` | `1b8fc8be308987102f120e4f14d3071c1fb29e8f92bbd75cf05314a9fb8d900f` | Sacred SNLI hidden-state ablations |
| `evaluate_all.py` | `0415eb4ee23c3b6950e3292ed689acb641fe4e6e3edf11fd3e98c68d423df284` | Static/dynamic checkpoint comparison |
| `evaluate_original.py` | `b043278ae0f7ea3b43a643ba2875fbc432508e93f814e3df7d361b521eacf6e2` | Label-blind static evaluation |
| `experiment_modes.py` | `88c80906d8772aaa4a8f4ed2ff96f4919982bb4ee703c167eb96aae6aafa99f4` | Basin-policy modes already incorporated |
| `nova_basin_store.py` | `07c535c5e81b9895e59c905120e0f292b2459a8b66c78524afa601efdb2e7d11` | Persistent basin/court prototype |
| `nova_improvements_demo.py` | `332cdcce3d6c84e4931cd0eb92a070a5fa68018172333a8db3d389da46876fdd` | Nova Memory v1/v2 comparison |
| `observer_vision_test.py` | `34b672fa71cefa0d96a3ccc872377da66f34bc43748765ae1b946dae8e3c2932` | Multistat versus observer features |
| `plot_landscape.py` | `ecc83425a43743707d0896abe4a9f4bc6eaa212817717f22e99f39b332ddaee4` | Chosen three-anchor potential plot |
| `test_basin_dynamics.py` | `6412e906b7d3fedb0cec039c5054c092343791a0888550e221130d0eaf347e53` | Spawn/merge/decay print diagnostic |
| `test_gradient_large.py` | `3a052df924013607d46648d282e4e80b3ad7643d3ab6faffed9e0a6db65444df` | Large angular-gradient sweep |
| `test_gradient_static.py` | `1002c872bb310b2674c32fa5491e04d5980da493dad3bed0367aac5244101cd3` | Static angular-gradient sweep |
| `train_swapped_head.py` | `3dd0b4dcd8d0d1e55ad483082925aeece473c38fe263986688ad2abb473a6eeb` | Fresh SNLI-head diagnostic |

These are preservation pairs, not independent replications.

## 2. Sacred SNLI diagnostics

### 2.1 What the evaluators actually do

- `evaluate_original.py` is the correct label-blind shape: encode premise and
  hypothesis, collapse without the target label, then classify.
- `evaluate_all.py` routes the dynamic checkpoint through
  `collapse_dynamic(..., labels, ...)`. The target labels therefore influence
  the representation before classification. That route cannot be used as an
  NLI result.
- `evaluate_ablation.py` proposes useful hidden-state controls: full state,
  `L=0`, all-zero, random, and constant hidden states, with noise toggled.

The ablation script is not replayable as preserved. A fresh run fails before
evaluation because it searches the stale path:

`/Users/chetanpatil/Desktop/test/quantum_retrain/model_collapse1/quantum_embeddings_final.pt`

The exact embedding artifact instead survives under Sacred and the repaired
collapse branch. The fallback path construction also reuses the wrong saved
path tail. This is path drift, not a missing checkpoint.

### 2.2 Gradient and swapped-head work

The two gradient scripts explore a smooth angular potential based on
log-sum-exp attraction to class anchors. Together they cover more than 130
development configurations. They hard-code a 75.93% baseline, save no selected
result, use no untouched final test split, and provide no matched multi-seed
comparison. Preserve the gradient formula as a candidate mechanism, not as a
measured improvement.

`train_swapped_head.py` trains a fresh classifier head for five epochs on
50,000 examples with fixed `beta=80` and `eta=2`. No resulting checkpoint or
result artifact was found. It remains an uncompleted diagnostic.

The provisional verdict on the remembered 95.76–96.07% model is unchanged:
**leaked/unusable unless the missing artifact and a label-blind protocol are
recovered**.

## 3. Nova Memory v1 to v2

Compared roots:

- v1: `/Users/chetanpatil/Desktop/test/nova-memory-main`
- v2: `/Users/chetanpatil/Desktop/test/livnium-sacred/nova-memory-v2`

Ignoring caches, only four files differ:

1. `ai/encoder.py`
2. `ai/evaluator.py`
3. `core/semantic.py`
4. `server/state_store.py`

Both roots contain fourteen script-style checks. Pytest collects none of them,
but executing every `tests/test_*.py` file directly produces:

- v1: **14 passed, 0 failed**
- v2: **14 passed, 0 failed**

The scripts verify useful maintenance mechanics—bundle indexing, growth,
consolidation, surgery/haircuts, quarantine, receipt determinism, deterministic
retrieval, tension calculation, and reasoning—but they are not a collected
pytest suite.

### 3.1 Improvements that survive

- v2 supports spaces and arbitrary text without immediately rejecting it.
- the alphabet is global rather than rebuilt per pair;
- the evaluator can calculate conservation from supplied state snapshots;
- the state store keeps a bounded live ledger and a recoverable archive
  sidecar.

### 3.2 Boundaries that remain

The v2 encoder folds arbitrary text, seeds a PRNG with MD5, and permutes the
same 27-symbol alphabet. It can distinguish nominated examples, but it is a
lossy deterministic hash-to-permutation—not reversible text encoding and not a
learned semantic representation.

The evaluator has three contract gaps:

- when no states are supplied, it still reports conservation as true;
- its “weighted median” is an ordinary median; and
- validity uses mean gain while the report foregrounds the median. Equal
  successive gains are also accepted despite “strictly increasing” prose.

The semantic cache hashes cells but not their weights even though the cached
embedding depends on both. Mutating weights under the same cells can return a
stale embedding.

The ledger count is also incorrect. With fifty entries and a live capacity of
ten, the demo reports a total of forty: it counts archived entries rather than
all fifty historical entries. Appending one archived item rewrites the entire
archive sidecar, so maintenance cost grows with history.

`nova_improvements_demo.py` expects v2 beside v1 at
`/Users/chetanpatil/Desktop/test/nova-memory-v2`; the preserved v2 is nested
under `livnium-sacred`. The “all four improvements verified” headline is
therefore both non-replayable without path repair and too strong given the
contracts above.

## 4. Basin and landscape diagnostics

`test_basin_dynamics.py` is a useful smoke diagnostic, not a scientific test:
it has no assertions and no fixed random seed. A fresh run demonstrated anchor
spawning and merged a near-duplicate anchor at cosine `0.9964`, but one of five
reported tensions increased.

`nova_basin_store.py` and `experiment_modes.py` were already tested in
`DEMOS_LINEAGE_AUDIT.md`. Their surviving value is persistence and prototype
bookkeeping. Their boundaries remain:

- live plus archived receipt totals do not reconcile;
- decay is attributed to every anchor rather than the mutated anchor;
- promotion metadata does not change scoring;
- promoted anchors are never quarantined; and
- the integrity hash covers centers, not every mutable field.

`plot_landscape.py` generated the exact preserved `energy_landscape.png`
(`6b07fb982840ee6d71450f0eac244519ee8a9c9736cceac3dd6257ea2a9ab401`,
188,308 bytes). The plot evaluates a chosen potential around
three saved anchors after a two-dimensional projection. It is an explanatory
visual, not training evidence, a theorem, or proof of an external record.

## 5. Observer/vision diagnostic

The script compares:

- flat mean;
- multistat `[mean, max, std]`; and
- observer `[R.mean, R.max, R.std, Om]`, with `Om=mean(E)` and `R=E-Om`.

The observer representation contains no new information:

- `R.mean = 0`;
- `E.max = R.max + Om`; and
- `E.std = R.std`.

It is therefore a reparameterization of multistat plus a zero block. This
independently confirms the Om/LO redundancy recorded in the cube-geometry
audit. The script also contains stale `/sessions/...` paths and has no saved
result artifact.

## 6. Final handling

| Material | Evidence status | Handling |
|---|---|---|
| Label-blind static evaluator | Reusable diagnostic shape | Preserve |
| Gold-label dynamic evaluator | Leaked | Retire as result; keep as warning |
| Hidden-state ablation design | Partial, path-broken | Repair only in a future controlled replay |
| Angular potential sweeps | Candidate mechanism | Open research, no result claim |
| Swapped-head trainer | Uncompleted | Historical |
| Nova v1/v2 maintenance mechanics | Verified script-level engineering | Preserve |
| Hash-permutation text encoder | Deterministic but nonsemantic | Narrow claim |
| Evaluator/cache/ledger contracts | Defective/incomplete | Record before reuse |
| Basin spawn/merge diagnostic | Smoke evidence | Preserve as diagnostic |
| Energy landscape | Explanatory visualization | Do not cite as proof |
| Observer features | Algebraically redundant | Retire as added-information claim |

The organized `Nova-and-Misc` row is now fully incorporated. No unreviewed P1
experiment family remains.
