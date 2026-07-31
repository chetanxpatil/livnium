# Sacred Livnium Vault Audit

Audited: 2026-07-26

## Scope

This audit compared:

- `/Users/chetanpatil/Desktop/livnium-sacred`
- `/Users/chetanpatil/Desktop/livnium-sacred copy`
- `/Users/chetanpatil/Desktop/test/checkpoints-sacred`
- `/Users/chetanpatil/Desktop/test/collapse_retrain`

The source roots were read only. No checkpoint, dataset, source file, or archive
was moved, renamed, deleted, or rewritten.

## Short conclusion

The sacred work is real and worth preserving, but its strongest recoverable result
is not the remembered 96.07%.

- The best of the three self-contained sacred-copy SNLI models is
  `collapse1-static`, a label-blind checkpoint that freshly scores
  **7,478/9,824 = 76.12%** on SNLI test with deterministic, noise-free
  evaluation.
- The saved error-analysis workbook records a nearby historical run at
  **76.01%**, ahead of `barrier-0` at 75.56% and
  `collapse4-dynamic-038` at 75.20%.
- The 95.76% dev / 96.07% test statement appears only in README files. No saved
  log, prediction file, workbook, or checkpoint supports it.
- The dynamic evaluator does pass gold labels into label-specific basin routing,
  contrary to the README's “evaluation always uses static collapse” statement.
  That is a genuine evaluation-design bug. On the surviving dynamic checkpoint,
  however, static, correct-label, and shuffled-label routing all score about
  75.3%, so the bug does not explain or recover a 96% result.
- The supposedly lost quantum embedding backbones are present in multiple exact
  copies and are loadable.
- `collapse_retrain` is a distinct June repair branch. Its surviving
  `nli_epoch23.pt` is label-blind and freshly scores **68.87%** on SNLI test, but
  its training log, seed, and earlier epochs are missing.

The preservation roles are therefore:

1. `livnium-sacred` — canonical artifact and experiment-history vault.
2. `livnium-sacred copy` — canonical self-contained replay bundle and source
   version museum.
3. `collapse_retrain` — separate repaired training branch.
4. `checkpoints-sacred` — policy/manifest placeholder, not an actual checkpoint
   store.

## Root inventory

| Root | Approx. size | Files | Checkpoint role |
|---|---:|---:|---|
| `Desktop/livnium-sacred` | 2.0 GB | 200 | 54 `.pt` files, all distinct within the root; logs, caches, workbook, scripts, figures, and historical source |
| `Desktop/livnium-sacred copy` | 785 MB | 5,981 | Five `.pt` files, all exact copies found elsewhere; raw data, clean replay code, and 5,278 `eyes/` before/after source snapshots |
| `test/checkpoints-sacred` | 4 KB | 1 | README only; no weights |
| `test/collapse_retrain` | 452 MB | 22 | Nine internally distinct `.pt` files and five unique current Python sources |

Neither sacred Desktop root contains a `.git` directory. The artifact vault's
`archive/misc/CHANGELOG.md` records historical commit identifiers, but the Git
objects themselves are absent. The `eyes/` session snapshots in the copy are
therefore meaningful source-version evidence rather than disposable noise.

## Checkpoint identity

Across the four audited roots there are 68 `.pt` files representing 63 distinct
SHA-256 values. The five duplicate pairs are:

1. sacred-copy collapse-1 final embedding =
   `collapse_retrain/model_collapse1/quantum_embeddings_final.pt`
2. sacred-copy collapse-4 final embedding =
   `collapse_retrain/model_full_physics/quantum_embeddings_final.pt`
3. sacred-copy `barrier-0/best_model.pt` = the vault's archived copy
4. sacred-copy `collapse4-dynamic-038/best_model.pt` = the vault's archived copy
5. sacred-copy `collapse1-static/best_model.pt` = the vault's archived copy

Important hashes:

| Artifact | SHA-256 prefix |
|---|---|
| collapse-1 final quantum embeddings | `37895313f9661d8d` |
| collapse-4 final quantum embeddings | `3acac66e89b29890` |
| collapse1-static SNLI checkpoint | `9a32ffcd1bd939a9` |
| barrier-0 SNLI checkpoint | `c5eeac327...` |
| collapse4-dynamic-038 SNLI checkpoint | `5d36c387...` |

The sacred archive's `triple_crown_slow` checkpoint is not byte-identical to the
one in `Desktop/livnium`; their SHA-256 prefixes are respectively
`bc5ffcb8...` and `1ca42de9...`. They must remain separate artifacts.

## Fresh replay of the three core sacred models

A temporary evaluator restored each checkpoint's saved barrier, disabled encoder
noise, and explicitly selected static, saved-label, or shuffled-label routing.
It used the sacred-copy code and raw SNLI splits.

### Deterministic results

| Model | Dev | Test | Interpretation |
|---|---:|---:|---|
| collapse1-static | 76.01% | **76.12%** | Best surviving sacred model; label-blind inference |
| barrier-0 | 74.90% | 75.60% | Saved barrier must be restored; stock evaluator defaults to 0.38 |
| collapse4-dynamic-038, stored labels | 75.32% | 75.32% | Gold-label routing is present in code |
| collapse4-dynamic-038, static | 75.30% | 75.32% | Essentially identical to stored-label routing |
| collapse4-dynamic-038, shuffled labels | 75.32% | 75.31% | Label routing has negligible effect in this saved state |

The `collapse1-static` checkpoint contains a broken absolute path to a former
quantum-embedding location. The exact required embedding file was recovered by
hash from both the sacred copy and `collapse_retrain`.

### Workbook evidence

The read-only workbook
`archive/misc/snli_error_analysis.xlsx` contains five sheets:

- 2,357 collapse1-static errors;
- 2,401 barrier-0 errors;
- 2,436 collapse4-dynamic-038 errors;
- a summary sheet;
- 1,554 cases all three models got wrong.

Its recorded test accuracies are 76.01%, 75.56%, and 75.20%. All five sheets
render correctly, the sampled rows are visually coherent, and no formula errors
were found. The small difference from the deterministic replay is consistent with
the old evaluator's stochastic encoder noise. The workbook is strong historical
evidence for the approximately 76% result and no evidence for 96%.

## Why the 96.07% claim is provisionally leaked/unusable

The README says:

- dynamic basins produced 95.76% dev / 96.07% test;
- dynamic basins were not a label leak;
- evaluation always used static collapse.

The audit found:

1. a search for `95.76` and `96.07` finds only the README;
2. the named `collapse4-dynamic-96pct` directory has no surviving checkpoint;
3. `test/checkpoints-sacred` contains no weights;
4. both the training evaluator and test script pass ground-truth labels to
   `collapse_dynamic` when dynamic evaluation is selected;
5. the surviving dynamic model replays at about 75.3%, whether routing is static,
   correct-label, or shuffled-label.

The honest conclusion is not that the 96% number was fabricated. It is that the
artifact chain needed to verify it was never preserved, the surviving evaluator
contains a real label-leak path, and the surviving code and model do not
reproduce it. By the 2026-07-26 recovery decision, the claim is provisionally
classified as leaked/unusable unless an independently identified prediction
artifact or checkpoint appears.

## `collapse_retrain` repair branch

This branch is not source-identical to either sacred Desktop root. Its five
current Python files repair several training problems:

- collapse-engine parameters are explicitly optimized;
- label-supervised anchor learning is separated from static inference;
- saved models can be evaluated without gold-label routing;
- `eval_nli.py` provides held-out evaluation.

Fresh evaluation of `model_nli_v1/nli_epoch23.pt` gives:

| Split | Accuracy |
|---|---:|
| SNLI dev | 69.76% |
| SNLI test | **68.87%** |

On test, correct predictions by gold class are 2,507 entailment, 1,934 neutral,
and 2,325 contradiction. The evaluator's own reference bars are majority 34.3%,
bag-of-words 59.4%, hypothesis-only 61.5%, and GloVe average 60.7%.

This is a genuine label-blind held-out checkpoint and clears those stated
reference bars. It is not the best sacred model, and its provenance is incomplete:
only epoch 23 survives, with no training log, seed record, or earlier SNLI
checkpoints. Preserve it as a measured repair artifact, not as a fully replicated
benchmark.

## Fast ablation history

The vault preserves many small cache-based checkpoints and their logs. Recorded
best dev values range from 54.44% to about 60.01%:

| Family | Best dev in saved log |
|---|---:|
| v5-fast | 59.68% |
| triple-crown merge | 58.26% |
| adaptive | 57.71% |
| trajectory | 57.54% |
| repulsion | 57.31% |
| rotation | 56.20% |
| locking | 55.31% |
| null | 54.44% |

The combined `run_all.log` briefly reaches 60.01% but ends after epoch 6.
`triple_crown_slow2.log` ends during the first epoch. These artifacts show that
adding adaptive metrics, rotation, repulsion, trajectory features, or null
variants did not produce a saved improvement over the approximately 76% full
pipeline. This is valuable negative ablation history.

## Corrected dynamics and exploratory experiments

The vault contains an unusually useful correction document,
`archive/runs/livnium_collapse_equation.md`. It correctly narrows the mechanism
to a discrete-time attractor update combining a learned residual with
anchor-directed radial forces whose magnitudes come from cosine divergence. It
explicitly withdraws the description “exact gradient descent on the written
energy.”

Saved follow-up artifacts further bound the theory:

- Jacobian spectral norm: mean 41.11, maximum 198.07, 0% below 1. The saved map
  is not generally contractive.
- The saved cosine-gradient formula test has 40.84° mean absolute error; both
  tested directions descend the chosen diagnostic on that sample, so the
  proposed analytic formula is not established.
- Neutral examples have the smallest mean entailment/contradiction anchor gap
  (0.222 versus 0.236 and 0.271), a useful diagnostic but not a theorem.
- In the dimensional-collapse experiment, Livnium beats the plain MLP only at
  dimensions 5 and 16 by 0.96 and 0.50 points. The MLP wins at dimensions 3, 64,
  and 256, including a 4.20-point win at 256.

The baseline, fusion, decoherence, and Collatz figures are preserved as
exploratory communication artifacts. Some use hard-coded or temporary result
inputs, and the Collatz chart itself acknowledges that the all-integer proof and
the convergence connection are still missing. None should be cited as a
mathematical proof or a replacement for a reproducible result table.

## Source lineage

- `livnium-sacred` has 57 distinct source/document hashes; 20 were confined to
  that root at inventory time. Several confined files are PIDs or exit codes, but
  the changelog and corrected collapse-equation document are genuinely valuable.
- `livnium-sacred copy` has 5,319 source/document rows but only 72 distinct
  hashes. Most rows are repeated `eyes/` snapshots. Nineteen hashes were confined
  to this root, mainly old source versions.
- The two sacred roots share 37 distinct source/document hashes.
- Neither root is a source duplicate of `collapse_retrain`.

The copy is therefore not “just another duplicate.” Its five checkpoints are
duplicates, but its self-contained data/code layout and source-version snapshots
provide replay and provenance value.

## Newly exposed continuation

During comparison, another related root was confirmed:

`/Users/chetanpatil/Desktop/test/livnium-sacred-v2`

It is about 888 MB and contains the same two quantum-embedding hashes and the same
collapse1-static checkpoint, plus unreviewed `torque-v1`,
`torque-256-memory`, and `torque-replace-256` checkpoint hashes. Two torque
checkpoints are duplicated between its `nova_eye` and `nova_v3` subtrees.

This continuation is now incorporated in `SACRED_V2_AUDIT.md`. Its best surviving
checkpoint is torque-memory at 76.52% dev and 76.42% test under static
label-blind inference. The named fixed forces are not load-bearing, failure
memory is observer-only, and the Nova Eye branch is unfinished.

## Preservation decision

- Preserve all four audited roots.
- Do not repopulate `checkpoints-sacred` or move weights during recovery; record
  paths and hashes first.
- Treat `collapse1-static` as the best replayable model in the audited sacred
  replay bundle.
- Keep 95.76% / 96.07% as historical memory but classify it provisionally as
  leaked/unusable unless a supporting artifact is later found.
- Preserve dynamic basins as an experimental mechanism, but fix the evaluation
  interface so inference cannot accept gold labels.
- Preserve the corrected equation, workbook, negative ablations, and `eyes/`
  snapshots. They are the real scientific memory of this branch.
