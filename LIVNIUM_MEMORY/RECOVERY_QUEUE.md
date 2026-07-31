# Recovery Queue

Updated: 2026-07-27

Recovery status: **complete for every named P0, P1, and P2 source in this
ledger**. Future discoveries are appended as new rows; they do not reopen old
rows by default.

Status values:

- **Discovered:** location is known.
- **Indexed:** file-level map exists.
- **Reviewed:** concepts/results were inspected.
- **Incorporated:** durable summary and evidence status are in this memory.
- **Duplicate:** exact hash or complete semantic duplicate was verified.
- **Historical:** preserved, but intentionally not active.

## P0 — review before any cleanup

| Source | Status | Why it matters |
|---|---|---|
| `test/_ORGANIZED/INDEX.md` and its organized view | Incorporated (2026-07-27) | `REALCORE_LEGACY_SNAPSHOTS_AUDIT.md`: 203 meaningful regular files; 202 exact root copies, one generated index; 27/28 links resolve |
| `test/lab/index/` and its 54 theory/index files | Incorporated (2026-07-27) | Every generated submap now routes to an audited lineage; legacy/pre-Core/Ramsey closure is in `REALCORE_LEGACY_SNAPSHOTS_AUDIT.md` |
| `Desktop/core` | Incorporated (2026-07-27) | Unique 106-test Dart Realcore remains separate; Python `learn/` quantum/Rule30 lineages and the February Core delta are now incorporated |
| `Desktop/uantum` | Incorporated (2026-07-26) | `UANTUM_AUDIT.md`: MPS copy superseded/buggy; dynamic alpha mixed; retrieval negative on stronger evaluation; Ramsey search has no witness |
| `Desktop/livnium` | Incorporated (2026-07-26) | `DESKTOP_LIVNIUM_AUDIT.md`: reproducible SNLI 0.7634 pilot; later runs regress; Nova mechanics verified but retention gap is policy-circular; TRL4 not established |
| `Desktop/livnium-sacred` | Incorporated (2026-07-26) | `SACRED_VAULT_AUDIT.md`: canonical 54-checkpoint artifact/history vault with logs, workbook, corrected theory, and negative ablations |
| `Desktop/livnium-sacred copy` | Incorporated (2026-07-26) | Self-contained replay and `eyes/` source-version museum; five weights duplicate vault/retrain artifacts but unique history remains |
| `test/checkpoints-sacred` and `test/collapse_retrain` | Incorporated (2026-07-26) | First is an empty policy placeholder; second is a distinct repaired branch with a freshly measured 68.87% label-blind test checkpoint |
| `test/livnium-sacred-v2` | Incorporated (2026-07-26) | `SACRED_V2_AUDIT.md`: torque-memory is 76.42% label-blind test; 96% is provisionally leaked/unusable; fixed forces and failure memory are not load-bearing; Nova Eye is unfinished |
| `test/lab/nova-snli` | Incorporated (2026-07-26) | `LAB_NOVA_SNLI_AUDIT.md`: one-epoch physics embedding; saved proper SNLI test 32.99%; gold-label routing inflates it to 90.58%; README 74.4% unsupported |
| `test/lab/infected/Archive/nova_v3` plus `quantum_embed` | Incorporated (2026-07-26) | `INFECTED_ARCHIVE_NOVA_AUDIT.md`: sole weight duplicates 76.12% collapse1-static; conditional 77.17% error artifact has no model/protocol; old evaluator leaks labels; deterministic encoders have first-letter/anagram collapse |
| `test/lab/infected/python/clean-nova-livnium/archives-local/arch-archive/experiments/NLI-ALL` | Incorporated (2026-07-26) | `NLI_ALL_AUDIT.md`: three semantic mirrors; simple 40.71% full test; v3–v7 near chance; v8 broken; debug/self-label/resubstitution scores retired; DTW semantic warp and lexical-memory lesson preserved |
| Remaining `arch-archive/experiments` siblings and Rule-30 continuation | Incorporated (2026-07-26) | `ARCHIVED_EXPERIMENTS_RULE30_AUDIT.md`: mirrors reconciled; unique K17 checkpoint has 21 violations; archived AES implementations fail correctness boundaries; quantum demos are classical; even-cube rotation claims corrected; Rule-30 99.7% prediction retired and exact causal density lookup preserved |
| `archives-public/semantics/packages` plus Livnium domain mindmap | Incorporated (2026-07-26) | `SEMANTICS_MINDMAP_AUDIT.md`: source mirrors reconciled; workspace is artifact-complete; embeddings/manifolds are effectively rank one with zero graduates; orphan SNLI classifier and hypothesis-only cache boundary recorded; 499-node mind-map preserved with tension inversion documented |
| `arch-archive/core-o`, `core-t`, and SAT/CSP/Max-Cut `benchmark` | Incorporated (2026-07-26) | `ARCHIVED_CORE_VARIANTS_BENCHMARK_AUDIT.md`: three exact mirror sets; spherical packing/gradient, tetrahedral A4/quantum/recursion/ledger, and shared-state solver failures recorded; datasets, baselines, and negative result artifacts preserved |
| Complete `arch-archive` roots and previously missed base Core, Core-C, market, language, O-A8/O-A9/O-A10, cache, and figures | Incorporated (2026-07-27) | `ARCH_ARCHIVE_ROOT_AUDIT.md`: three root variants reconciled; `clean=noba=back` is the oldest self-contained copy, workspace is artifact-complete, and the archive is not the oldest Livnium evidence overall; valid cube/state-vector/Core-C mechanisms and narrow hypotheses are preserved while completeness, market-alpha, physical-law, trained-brain, and evidence-figure claims are retired or narrowed |

## P1 — experiment families missing from the July repository

All of these are indexed under `test/_ORGANIZED/02_Experiments/` but return no
matching references in `lets_clean_it/livnium`.

| Family | Status | Examples |
|---|---|---|
| Sudoku | Incorporated (2026-07-26) | `SUDOKU_LINEAGE_AUDIT.md`: 26-file lineage duplicated at root; hybrid valid but learned-ordering advantage unproven; pure learner partial; tabular RL memorizes; policy RL fails; random-deletion difficulty invalid |
| Cube/Sokoban and adjacent cube geometry | Incorporated (2026-07-26) | `CUBE_GEOMETRY_LINEAGE_AUDIT.md`: 22 exact root/organized pairs; correct 24-rotation action and odd-cube partition; canonical 100% is transformed-input identity; learned locality partial; whole AE not equivariant; graph-denoising/Om-LO claims narrowed |
| Attractor dynamics | Incorporated (2026-07-26) | Five experiments audited in `IDEA_LEDGER.md`; preserve locality/placement/reversibility lessons, do not port wholesale |
| Governance/economy | Incorporated (2026-07-26) | `GOVERNANCE_ECONOMY_LINEAGE_AUDIT.md`: 40 exact root/organized pairs; valid median/incentive/projector/observability boundaries; election ignores information, judges clone one anomaly rule, anchors/sensors supply truth, and purge/silence/shared-fate narratives exceed implementation |
| Symmetry spectrum | Incorporated (2026-07-26) | `SYMMETRY_SPECTRUM_LINEAGE_AUDIT.md`: four exact root/organized pairs; exact P7³ 70-level spectrum preserved, with full 48-element symmetry, separable-sum collisions, disconnected random control, and irrep overstatement corrected |
| Holonomy/cube embeddings | Incorporated (2026-07-26) | `CUBE_EMBED_LINEAGE_AUDIT.md`: 37-file four-generation lineage plus exact session-summary/ablation mirrors; useful PPMI-SVD and negative ablation preserved, while Fourier/QR locality, sign preservation, independent 94D channels, true holonomy, default SimLex generalization, and SNLI advantage are retired or narrowed |
| Games | Incorporated (2026-07-26) | `GAMES_LINEAGE_AUDIT.md`: six exact root/organized pairs plus hidden 13-module chess project and three saved sliding memories; preserve verified chess transport and receipt history, while mate-basin, frozen tic-tac-toe, persistent-puzzle-memory, and learned-sorting claims are retired or narrowed |
| NLI-Language organized family | Incorporated (2026-07-26) | `NLI_LANGUAGE_LINEAGE_AUDIT.md`: 39 exact root/organized pairs, trimmed GloVe, neural checkpoint, and hidden 729 MB basin/receipt state preserved; standard context prediction survives, while character/word geometry, basin routing, fracture, and neural-language advantages fail direct controls |
| Demos | Incorporated (2026-07-26) | `DEMOS_LINEAGE_AUDIT.md`: six exact root/organized pairs plus saved ten-anchor bridge state; preserve standard base-27/prototype/persistence teaching and partial warm-stream lift, while chosen-seed learning, deep-puzzle, superior-policy, Karmic-law, court, receipt-total, and full-hash claims are narrowed or retired |
| Nova-and-Misc | Incorporated (2026-07-27) | `NOVA_MISC_AUDIT.md`: all 13 exact root/organized pairs reconciled; label-routed evaluator retired, gradient/head work bounded, Nova v1/v2 contracts audited, observer redundancy proved |

Review question for each family:

1. Is there a unique mechanism?
2. Is there a valid result artifact?
3. Was a baseline or kill-test run?
4. Is it already represented under a different name?
5. Should it be active research, historical evidence, or a reusable component?

## P2 — archives and snapshots

| Source | Status | Result |
|---|---|---|
| `Desktop/livnium-sacred.zip` | Incorporated (2026-07-27) | Earlier Sacred snapshot; 81 meaningful files exact and three meaningful Nova-v3 sources revised later |
| `test/livnium.core-0.0.1-multi-basin.zip` | Incorporated (2026-07-27) | Historical full bundle reconciled against 0.0.3 and audited lineages |
| `test/livnium.core-0.0.1.tar.gz` | Incorporated (2026-07-27) | Near-snapshot of live tree; two NLI files changed and one runner added later |
| `test/livnium.core-0.0.3.zip` | Incorporated (2026-07-27) | Follow-up full bundle; unique NLI/Rule30/Nova artifacts already assigned |
| `test/livnium-crux-main.zip` | Incorporated (2026-07-27) | Unique classical Dart/JS/docs release; fresh 32/32 tests pass |
| `test/GitNexus-main.zip` | Exact duplicate (2026-07-27) | All 360 regular files match extracted tree |
| `test/nova-memory-main.zip` | Exact duplicate (2026-07-27) | All 60 regular files match extracted Nova Memory v1 |
| Nested archives and Git worktrees under `test/lab/infected` | Incorporated (2026-07-27) | Two nested quantum releases indexed; six first-party and three third-party WikiExtractor Git roots recorded |

See `REALCORE_LEGACY_SNAPSHOTS_AUDIT.md` for entry counts, hashes, deltas, and
the temporary-extraction protocol.

## Known duplicate

The conversation exports in `core` and `lets_clean_it/livnium` are byte-for-byte
identical. Keep at least one recoverable copy; do not repeatedly parse both.

`test/lab/infected/realcore` is a source/document duplicate subset of
`Desktop/core`: same checked commit, with all 475 distinct infected-copy hashes
present in the Desktop copy.

## Completion condition

Recovery is complete when every P0, P1, and named P2 row is either:

- incorporated into the concept/claim ledger,
- marked historical with a reason,
- or verified as a duplicate with a content hash.

That condition is satisfied as of 2026-07-27. “Complete” means the named
machine-visible corpus is indexed and its claims are bounded; it does not mean
no forgotten offline or future file can ever be discovered.
