# Livnium Lineage Map

Updated: 2026-07-27

This file explains projects nested inside other projects. A deep path is not
automatically a later version, and two folders with different names are not
automatically different work.

## `lab/infected` boundary map

The February generated index counted 45,522 files and 5,095 code-like files in
`lab/infected`. The current audit found six first-party Git worktrees, one nested
third-party WikiExtractor checkout, and at least fifteen first-party package
roots. These overlap: a Git worktree can contain many archived packages.

### First-party Git worktrees

| Worktree | Branch / checked commit | Relationship |
|---|---|---|
| `lab/infected/realcore` | `main` / `588dfc89` | Same checked commit as `Desktop/core`; every one of its 475 distinct source/document hashes also exists in `Desktop/core` |
| `lab/infected/python/clean-nova` | `main` / `b6d9c07e` | Clean-Nova lineage |
| `lab/infected/python/clean-nova-livnium` | `cleanup/structure` / `f589b901` | Large Livnium archive containing ECW-BT, quantum, NLI, Ramsey, Rule30 and other predecessors |
| `lab/infected/python/clean-nova-livnium 2` | `structural-phase2.2-testing` / `44329824` | Separate Clean-Nova-Livnium history; do not infer sequence from the suffix alone |
| `lab/infected/python/nova-livnium` | `main` / `369ab2ae` | Nova-Livnium lineage |
| `lab/infected/workspace/clean-nova-livnium` | `main` / `f58be369` | Workspace snapshot of the large archive, close to but not identical with the `python/` copy |

`ecw-BT/wikipedia/wiki_extractor_src/.git` is a nested third-party extraction
tool, not a separate Livnium theory project.

### Additional package/application roots

- `LIVNIUM_CORE_COLLECTED/clean-nova-livnium`
- early standalone `ecw-BT`
- Flutter/Dart apps: `audio_bridge`, its nested `app`, `chat_crystal`,
  `iot_controller`, `legal_auditor`, and `snli`
- `realcore`
- workspace packages: `livnium_core`, its `memory` package, `livnium_nlp`,
  `livnium_semantic_field`, and `parallax_at_om`

This confirms the user's memory: there really are projects inside projects. No
cleanup should operate at the top-folder level without consulting this map and
the content-hash inventory.

## Quantum lineage

### Q1 — Legacy quantum-islands bundle

Primary inspected path:
`/Users/chetanpatil/Desktop/test/lab/infected/quantum`

- 64 inventoried source/document files, all with distinct hashes inside that
  folder.
- 58 of those hashes have copies elsewhere on the machine, accounting for 326
  additional copy rows in the inventory.
- Only six scripts were unique to this exact top-level bundle at inventory time:
  `circuit.py`, `conflict_resolver.py`, `grover_sat_solver.py`,
  `integration_example.py`, `p_vs_np_experiment.py`, and
  `policy_contradiction_detector.py`.
- Most of the simulator code reappears under legacy `quantum_2`,
  `pre_core_systems/quantum`, `realcore/learn`, release snapshots, and other
  Livnium roots.

Actual mechanisms:

1. a fixed three-qubit dense state-vector simulator;
2. exact two-qubit Bell-pair objects;
3. independent or pairwise “quantum islands”;
4. a geometric correlation graph which is explicitly not a global wavefunction;
5. a conventional exponentially sized dense/sparse state-vector experiment;
6. toy SAT, classification, and conflict-resolution applications.

Audit evidence:

- A direct rerun teleported 200 seeded random complex states with minimum fidelity
  `0.9999999999999996`.
- The exact GHZ state had only `|000>` and `|111>` support, each at probability
  0.5.
- Four collected pytest functions ran, but each returned data instead of asserting
  a result; pytest warned about all four. The full folder cannot collect because
  the classifier integration imports a missing `layers` package.
- The geometric simulator's own comparison deliberately produces states illegal
  for a true GHZ register. It is therefore a classical local-correlation model,
  not an approximate proof of scalable entanglement.

Verdict: the small exact simulator is **verified standard engineering**. The
islands and geometric graph remain reusable classical representations. The SAT,
P-vs-NP, “quantum conflict resolution,” and large-qubit language are historical
application experiments, not evidence of quantum advantage.

### Q2 — Organized Realcore quantum museum

Primary preservation path:
`/Users/chetanpatil/Desktop/core/learn`

This separates several earlier meanings of “quantum”:

- `core/quantum`: small tensor-product simulation plus custom non-unitary
  geometry coupling;
- `quantum/islands`: local classical/quantum-inspired units;
- `quantum/hierarchical`: recursively addressable geometric structures;
- `quantum/livnium_core`: a classical MPS/DMRG physics solver;
- `quantum_computer`: earlier hierarchical-geometry simulator;
- `quantum_core`: omcube capacity and cryptography experiments;
- `quantum_embed`: skip-gram-style language embeddings with collapse dynamics.

`Desktop/core` and `lab/infected/realcore` share the same checked Git commit, and
all inventoried source/document hashes in the infected copy are also present in
`Desktop/core`. Keep `Desktop/core` as the accessible preservation copy; treat the
infected Realcore tree as a verified duplicate subset unless future hashes change.

The February `learn/core` tree is a small revision of the later archived base
Core: three added files and two changed shared files. Its quantum-config
inheritance repair is useful, but its new `n_qubits=27` fixture makes the exact
state-vector capacity route exhaust memory. A bounded replay gives 254 passed,
25 failed, two skipped, and one deselected; the old semantic/temporal/recursive/
Moksha/dynamic failures remain.

The adjacent older `archives-local/archive` layer is now also reconciled. Exact
small-register simulation and a C clique validator survive; dual/trapped cubes
are inserted semantic state-machine rules; and the Ramsey geometry maps every
complete binary coloring to `(0,0,0)`. See
`REALCORE_LEGACY_SNAPSHOTS_AUDIT.md`.

### Q3 — MPS/cortex lineage

Canonical audited narrative:
`/Users/chetanpatil/Desktop/lets_clean_it/livnium/archive/cortex-v2`

This is the strongest honest record of the high-site-count work:

- long GHZ chains are efficient because GHZ has bond dimension 2;
- arbitrary global states still face bond-dimension/exponential limits;
- “addressable geometric cells” and “faithful entangled sites” are different
  capacities;
- the implementation and regression checks are useful, but the mechanism is
  textbook matrix-product-state simulation.

### Q4 — Standalone `Desktop/uantum`

`/Users/chetanpatil/Desktop/uantum` is not a duplicate of
`lab/infected/quantum`, but most of its contents reappear in
`/Users/chetanpatil/Desktop/livnium`: 49 of 56 distinct Python hashes have
external copies there. Its nested legacy simulator is already covered by Q1.

The truly relevant bridge material is:

- an older 12-test Cortex/MPS snapshot;
- dynamic-alpha fidelity experiments;
- GloVe/PCA semantic scoring and bounded-memory triage;
- a stochastic 43-vertex Ramsey search.

The `uantum` MPS copy fails adjacent and non-adjacent reverse CNOT directionality.
The later `Desktop/livnium` implementation fixes those cases and contains saved
mixed/negative benchmark artifacts. The Ramsey incremental counter is correct on
a focused small-graph check, but no 43-vertex zero-violation witness exists.

Verdict: **incorporated as a superseded/mixed bridge root**. See
`UANTUM_AUDIT.md` for exact evidence and decisions.

## ECW-BT lineage

### E1 — Early Level-0 trainer

Path: `/Users/chetanpatil/Desktop/test/lab/infected/ecw-BT`

Role: random unit word vectors trained from Wikipedia windows using mass-weighted
attraction, a 0.38 barrier, renormalization, and gravity pooling. The folder is
about 19 GB because it includes Wikipedia material. It is the early skeleton, not
the latest pipeline.

### E2 — SBERT-seeded pairwise pipeline

Most complete evidence bundle:
`/Users/chetanpatil/Desktop/test/lab/infected/python/clean-nova-livnium/archives-local/ecw-BT`

Role: a 50,000-word, 256-dimensional SBERT seed, 63 co-occurrence-pair shards,
pairwise CCD updates, repeated fusion back to the teacher, validation, and final
vectors. This bundle is about 22 GB and contains the latest inspected
`distill_pairwise.py` fix.

Copy relationships:

- 34 of its 35 inventoried source/document hashes occur elsewhere.
- Its only inventory-unique source file is the latest `scripts/distill_pairwise.py`.
- `workspace/clean-nova-livnium/.../ecw-BT` is almost the same 22 GB bundle, but
  lacks the final vector artifact and validation log and has an older distillation
  script.
- `test/git-final/ecw-BT` is a 4.8 GB trimmed copy with the same seed and frequency
  arrays, a numerically near-identical but byte-different output vector file, no
  pair shards, and an incomplete training log.

### Evidence audit

The saved student vector table is `(50000, 256)` with unit row norms. Relative to
its SBERT seed:

- mean row cosine: `0.99999908`;
- first-percentile cosine: `0.99999397`;
- mean row displacement: `0.00027669`.

The “accepted” validation checks teacher drift on ten words, considers any
nonempty neighbour list “sane,” and does not include analogy correctness in its
acceptance decision. Its two analogies score one correct and one incorrect.

Two implementation issues dominate interpretation:

1. the main pairwise trainer's positive and negative update signs are reversed
   relative to its own documented pull/repel law; and
2. after each of 63 shards, default fusion computes approximately
   `0.05 × student + 0.95 × seed`, erasing most accumulated movement.

The excellent neighbours are therefore inherited almost entirely from SBERT.
The checkpoint proves that the pipeline preserves its teacher and avoids total
collapse; it does not yet prove that CCD improves embeddings.

### ECW-BT decision

Preserve E2 as the canonical evidence bundle and E1 as historical origin. Do not
freeze or promote the current checkpoint as a Livnium result. A revival must fix
the force directions and anchoring schedule, then compare the same SBERT seed
before/after on standard intrinsic and downstream benchmarks over multiple seeds.

## Desktop Livnium lineage

Primary audited path:
`/Users/chetanpatil/Desktop/livnium`

- Git records only 24 files from a short 2026-03-15 to 2026-03-16 history. The
  project begins under quantum/Nova names and is renamed Energy-Guided Attractor
  Network.
- Cortex, results, Nova, experiments, scripts, and iDEX documents are untracked,
  so the folder is a composite working root rather than one complete Git lineage.
- The best reproduced saved model is
  `runs/triple_crown_slow/best_model.pt` at 0.7634 SNLI dev accuracy. All audited
  later alpha, memory, InferSent, and Lyapunov checkpoints are worse.
- Most embedded Nova content has external copies. For Nova Git history,
  `/Users/chetanpatil/Desktop/test/lab/nova/public` at commit `38287d7` is the
  stronger inspected lineage root.
- Thirty-five source/document hashes were confined to `Desktop/livnium` at
  inventory time. They must remain until both sacred roots are hash-compared.
- The iDEX documents belong to the proposal lineage, not the scientific evidence
  lineage.

See `DESKTOP_LIVNIUM_AUDIT.md` for the checkpoint, Nova, retrieval, and document
evidence audit.

## Sacred Livnium lineage

Detailed audit:
`SACRED_VAULT_AUDIT.md`.

### S1 — Desktop artifact/history vault

Path: `/Users/chetanpatil/Desktop/livnium-sacred`

- 54 internally distinct `.pt` artifacts.
- Preserves fast and slow ablation logs, cached SNLI features, the five-sheet
  error workbook, exploratory figures, scripts, a changelog, and the corrected
  collapse equation.
- No `.git` directory survives.

Role: **canonical sacred artifact and experiment-history vault**.

### S2 — Desktop self-contained replay and Eyes museum

Path: `/Users/chetanpatil/Desktop/livnium-sacred copy`

- Its five `.pt` files are exact duplicates of artifacts in S1 or S3.
- It includes raw SNLI data, exact collapse1/collapse4 embedding backbones,
  runnable quantum/SNLI source, and 5,278 `eyes/` source snapshots.
- The snapshots contain historical source versions not retained in S1.

Role: **canonical self-contained replay bundle and source-version provenance
museum**, not a disposable duplicate.

### S3 — June repaired retraining branch

Path: `/Users/chetanpatil/Desktop/test/collapse_retrain`

- Five current Python source hashes do not match S1 or S2.
- Two final embedding files are exact copies of S2's collapse1/collapse4
  embeddings.
- Its unique `nli_epoch23.pt` is a label-blind measured checkpoint at 68.87%
  SNLI test.

Role: **separate repaired branch**.

### S4 — Intended checkpoint policy directory

Path: `/Users/chetanpatil/Desktop/test/checkpoints-sacred`

This contains only a README describing a “nothing ever deleted” policy and
listing remembered lost artifacts. It contains no weights and is not the actual
vault.

Role: **historical manifest/policy placeholder**.

### S5 — Sacred-v2 torque and Nova Eye continuation

Path: `/Users/chetanpatil/Desktop/test/livnium-sacred-v2`

This 888 MB root repeats the collapse1/collapse4 embedding hashes and
collapse1-static model, then continues through torque-v1, basin replacement, and
failure-memory checkpoints. Torque-memory is the best freshly replayed surviving
sacred model at 76.52% dev and 76.42% test under deterministic label-blind
inference.

The fixed torque/anchor/axial forces are not load-bearing post hoc; the repeated
learned residual update is. Failure memory only records observations and cannot
affect training or inference.

The `nova_eye` subtree shares common code and two checkpoints byte-for-byte with
`nova_v3`, then adds raw-character, retina/glyph, emergent-basin, watcher,
WebSocket, and Flutter sources. No Eye checkpoint survives, and current trainer,
evaluation, and conservation bugs prevent treating it as a completed result.

Role: **canonical late sacred torque continuation plus incomplete Nova Eye
design archive**.

### S6 — Untracked lab Nova-SNLI physics predecessor

Path: `/Users/chetanpatil/Desktop/test/lab/nova-snli`

This roughly 812 MB root is untracked inside the parent `test` Git repository.
It contains a one-epoch 50,000×256 physics embedding history and the unique
`nova_v3/model/snli_physics/best_model.pt`.

The checkpoint is a chance-level model under proper static inference
(32.99% test), but rises to 90.58% when gold labels route dynamic basins. This
is the clearest recovered ancestral evidence for the target-label shortcut. Its
README's 74.4% result is unsupported.

Role: **canonical pre-Sacred physics-embedding and label-shortcut evidence**.

### S7 — Infected Archive Nova-v3 and quantum-embed history

Paths:

- `/Users/chetanpatil/Desktop/test/lab/infected/Archive/nova_v3`
- `/Users/chetanpatil/Desktop/test/lab/infected/Archive/quantum_embed`

These roots total roughly 536 MB and are not a coherent source snapshot: their
files span December 2025 through June 2026. Their only saved SNLI checkpoint is
an exact SHA-256 duplicate of sacred collapse1-static (`9a32ffcd…`) and freshly
replays at the same 76.12% label-blind test accuracy. Many named model/run
directories are empty.

Unique history survives in an orphan 2,243-row error file that conditionally
implies 77.17% test if complete, but its model and protocol are missing. The
archive source also preserves an older gold-label-routed evaluation path, a
geometry encoder whose lexical base retains only the first character, and a
mean-pooled phoneme encoder that collapses anagrams.

The adjacent quantum-embed source and corpus files duplicate the already
incorporated Sacred-v2/Lab physics-embedding lineage. It contains no weight or
result artifact, and its evaluator rebuilds incompatible test vocabulary IDs.

Role: **incorporated mixed-time history archive; preserve the unique error and
abandoned-source provenance, but do not count copied data/source/checkpoint as a
new model**. See `INFECTED_ARCHIVE_NOVA_AUDIT.md`.

### S8 — Nested NLI-ALL generation archive

Canonical path:

`/Users/chetanpatil/Desktop/test/lab/infected/python/clean-nova-livnium/archives-local/arch-archive/experiments/NLI-ALL`

This roughly 976 MB root contains eight visible NLI generations: `nli`,
`nli_simple`, and `nli_v3` through `nli_v8`. It preserves lexical
`brain_state.pkl` files, a v3 prototype head, supervised v4 rule artifacts, v5
pattern/geometry files, physics documents, source, and diagnostic/broken tests.
There is no PyTorch checkpoint.

The simple generation is the best honest saved member at 40.71% on the complete
valid SNLI test split. Runnable v3–v7 score only 33.85–36.06%; v8 has a source
indentation error. High archived numbers come from example prose, supervised
training-set resubstitution, gold-label debug paths, or prediction of
self-generated geometry labels.

The saved brains are supervised word-polarity dictionaries, not geometry
checkpoints. Only the simple brain materially changes current predictions.
Semantic warp is dynamic-time-warping-like alignment and remains a candidate
engineering component. Fracture dynamics does not identify negation: it fires
on 86.6% of a focused sample with chance-level contradiction precision. v7
reinforcement is discarded with each fresh classifier instance.

The `python/clean=noba=back` and workspace roots each contain the same 186
meaningful files as canonical. All differences are generated bytecode or macOS
metadata.

Role: **incorporated historical pure-vector/geometric NLI lineage; one semantic
project copied three times**. See `NLI_ALL_AUDIT.md`.

### S9 — Archived experiment siblings and displaced Rule-30 continuation

Canonical parent:

`/Users/chetanpatil/Desktop/test/lab/infected/python/clean-nova-livnium/archives-local/arch-archive/experiments`

Five small siblings beside NLI-ALL—`ramsey`, `quantum_core`,
`quantum_teleportation`, `crypto`, and `nxn_demo`—are meaningful source mirrors
across the canonical, `clean=noba=back`, and workspace roots. The workspace
alone preserves a K17/K4 Ramsey checkpoint; an independent recount confirms its
saved 21 violations, so it is near-state history rather than a witness.

The README-listed Rule-30 experiment moved into several other roots.
`Desktop/core/learn/rule30` is the organized 456 MB Realcore copy;
`clean=noba=back/experiments/rule30` is a compact old snapshot; and the 459 MB
standalone `lab/infected/rule30` is the most complete continuation because it
adds a causal Phase-9 dataset, two models, and independent-seed result files.

The sibling audit found no new AES, quantum, or Ramsey result. The useful
boundaries are:

- odd cubes have a unique center cell, but even cubes still preserve exposure
  and symbolic weight under all 24 proper rotations;
- teleportation is a duplicate small classical state-vector demo;
- quantum capacity counts independent/local registers and cells, not one global
  entangled state;
- both archived cipher families fail the correctness boundary needed for AES
  interpretation;
- Rule-30 Phase 1 contains a valid explicit 3-gram non-closure result, while its
  four exact formulas are generic cyclic de Bruijn flow identities;
- the later causal 99.6% target is next global density, exactly computable as a
  linear lookup on current pattern frequencies; autonomous rollout remains near
  chance with horizons around two or three steps.

Role: **incorporated compact application/history branch plus canonical Rule-30
causal continuation**. See `ARCHIVED_EXPERIMENTS_RULE30_AUDIT.md`.

### S10 — Semantics packages and Livnium domain mindmap

This node is indexed by:

- `THEORY_SUBMAP_python_clean-nova-livnium_archives-public_semantics_packages.md`
- `THEORY_SUBMAP_python_clean-nova-livnium_livnium_domains_mindmap.md`

The Python and workspace roots contain the same 101 meaningful non-checkpoint
source/document/config/log files, and their mind-map sources match. They are
mirrors, not independent projects. The workspace root is the complete artifact
copy because it alone preserves the current 49.7 MB emergent model,
`physics_small.pt`, 256 active manifold shards, and 1,071 extra per-word backup
files. All 15,310 per-word files shared with the Python root are byte-identical.

The semantic refactor is a distinct implementation generation:

- a real online 64D skip-gram/negative-sampling service;
- sharded per-word centroids plus mass/radius/noise/age policy state;
- clean domain/application/infrastructure package boundaries;
- 18D and 256D SNLI heads from incompatible sub-generations.

Its saved scientific state is negative evidence. Both 16,589-row embedding
banks have about 99.64% of centered variance in PC1 and effective rank about
1.04. The 16,381 active centroids are still more collapsed: 99.951% in PC1,
effective rank about 1.006. No saved word meets mass>2 and radius<0.2. The
surviving 256D SNLI cache reaches 49.5% with a fresh linear head, but
hypothesis-only reaches 48.8%; the saved 6D linear classifier matches neither
the cache nor current source. The claimed 15-test tree is absent and pytest
collects zero tests.

The mind-map is a separate useful application: it ingests paragraphs, embeds
them with MiniLM, retains cosine edges, greedily forms non-overlapping anchor
neighborhoods, template-narrates them, and exports JSON. Its surviving artifact
contains 499 nodes, 9,579 valid unique edges, and four basins. The “tension”
interpretation is inverted on the retained graph: with alignment>0.4 and
`tension=|0.38-alignment|`, higher similarity always creates higher tension,
while narration describes it as conflict.

Role: **incorporated semantic-learning prototype and conversation explorer;
preserve clean architecture, SGNS, sharding, input/output graph artifacts, and
collapse diagnostics; retire WordOracle/Jacobian/physics-accuracy and
mind-map-as-physics claims**. See `SEMANTICS_MINDMAP_AUDIT.md`.

### S11 — Archived spherical/simplex variants and task benchmark

This node is indexed by the remaining `arch-archive` submaps and exists as
byte-identical `core-o`, `core-t`, and `benchmark` subtrees in:

- `python/clean-nova-livnium`;
- `python/clean=noba=back`; and
- `workspace/clean-nova-livnium`.

The matched inventory is 30 meaningful Core-O files, 36 Core-T files, and
1,051 benchmark files. These are three preservation copies, not replications.
Only the `clean=noba=back/arch-archive` parent still places the required
historical `core` package beside the benchmark, making it the self-contained
replay root.

Core-O is the spherical variant: one core, tangent neighbor spheres, continuous
cap-derived `f`, `SW=9f`, SO(3) flow, a cap-sum packing test, and a soft
Hamiltonian. Its standard rotation/undo utilities survive, but the cap sum
admits 14 unit neighbors and never checks neighbor-to-neighbor distance. Even
the generated six-neighbor placement overlaps. The exposure formula tends to
0.5, and finite-difference checks show the reported force is not the gradient of
the logged potential inside overlaps.

Core-T is the five-node simplex variant. Its node schema and base-5 utilities
are clear, but two of the 12 alleged A4 matrices are not tetrahedral symmetries
and 50/144 products fall outside the table. System-level rotation is a no-op.
Bell-pair vectors are disconnected from local node states and measurements;
CNOT cannot pass through the single-node API; and the concurrence approximation
labels separable `|++>` maximally entangled. Five-way recursion allocates
97,655 nodes by depth 6 but indexes only the root globally. Basin reinforcement
changes total SW away from 108 and invalidates its own ledger.

The benchmark preserves 1,010 CNFs, six CSPs, ten GSET graphs, baselines,
plots, and saved result JSON. Its central flaw is shared candidate state:
SAT/CSP “basins” all store the same coordinate list and read the final one
global assignment. The saved SAT summary says 10/10 solved although only one
assignment satisfies all clauses; CSP says 6/6 although none of its saved
assignments satisfies all constraints. N-Queens diagonal constraints contribute
zero search tension. Max-Cut stops at step one, ignores GSET signed weights, and
loses badly to its own greedy baseline on G1 and G14.

Role: **incorporated alternative-core prototypes, reusable test/data harness,
and high-value negative evidence; preserve every source/data/result artifact,
but retire packing, A4, entangled-capacity, recursive-conservation, and solver
advantage claims**. See `ARCHIVED_CORE_VARIANTS_BENCHMARK_AUDIT.md`.

### S12 — Sudoku learning and search lineage

This node is preserved as 26 files under
`test/_ORGANIZED/02_Experiments/Sudoku` and as 26 same-named files at the
`test` root. Every pair matches byte-for-byte. The lineage contains five source
generations, nine JSON results, nine PNG figures, and three verdicts.

The original 9x9 MLP uses 27 row/column/box presence bits and a hard legality
mask. Its saved 85% easy result is real as an exact-source metric on generated
51-given boards, but its printed 58.61% “held-out style” number is training
resubstitution. Fresh cell accuracy is 53.22%. The claimed symmetry
augmentation is defined but never invoked.

The pure 9x9 successor removes legality and search, separates generated solution
boards, and exposes rollout failure. Its strongest preserved run uses
`py-sudoku`, 1,200/150/200 solution boards, reaches 43.36% unseen cell accuracy,
and exactly matches the generating completion on 62.5%, 14.5%, 1.5%, and 0% of
easy through expert boards. This is partial local prediction, not a general
solver.

The hybrid uses standard naked-single propagation, MRV, and backtracking, with
the MLP only ordering candidate digits. All independently replayed completions
are valid. The saved means reproduce exactly, including the expert reduction
from 43.24 to 25.69 candidate attempts, but the expert paired sign test is
p=0.755, the bootstrap interval crosses zero, and LCV averages 23.49. The source
counter is candidate attempts, not actual backtracks.

The tabular 4x4 branch keys Q-values by exact grid bytes. Full replay solves 3/3
training puzzles and 0/100 unseen puzzles. The 9x9 reward-policy branch records
zero training and test solves; its saved example changes no cell because a
wrong deterministic greedy action leaves the state unchanged and repeats.

Across both full-board generators, random clue deletion usually creates
multiple-solution puzzles and never establishes recognized difficulty. Fresh
50-puzzle checks range from 20–24% multiple at 51 givens to 100% at 23/26.

Role: **incorporated learning/search lineage and reusable negative controls;
keep the pure rollout question and hybrid injection scaffold, retire symmetry,
general difficulty, learned-search-advantage, and RL-generalization claims until
they pass unique hashed puzzle benchmarks**. See `SUDOKU_LINEAGE_AUDIT.md`.

### S13 — Cube/Sokoban and adjacent geometry lineage

This node consists of 22 files copied byte-for-byte between
`test/_ORGANIZED/02_Experiments/Cube-and-Geometry` and the `test` root. It
contains five empirical sources, one partition-counting script, five JSON
results, five PNG figures, and six verdict/theory documents.

CubeSokoban correctly generates the 24 proper cube rotations: they are unique
bijections with zero closure failures among 576 products. Canonicalization maps
every rotated random 5x5x5 occupancy template to one representative. That is
also why the 100% result is deterministic: all 720 canonical test rows are
byte-identical to a training row, and hash/one-template nearest-neighbor controls
also score 100%. Train and test share the same 40 world identities, and no
player, crate, goal, move, reachability, or Sokoban solution exists.

The directional decomposition is the exact odd-cube binomial partition
`(2m+1)^3 = 1+6m+12m²+8m³`. Its learned 2-D autoencoder improves fixed means
from 0.5956 to 0.6357 but trails PCA at 0.7181. A matched five-seed/eight-layout
audit places the directional layout above 30/40 random controls, while some
random layouts reach 0.6458.

The rotation experiment is negative and not fully equivariant: only encoder
weights are tied; independent encoder biases and the dense decoder remain free.
Whole-model parameters are 512 versus 548, not simply 13 versus 49. Tying is
worse than untied on both oriented digits and isotropic fields.

Geometry-direct is robust IRLS graph-Laplacian denoising. `I+5L` is full rank
343, so its generated fields are not a low-dimensional code. Clean smooth input
has 0.016 relative decode error while a clean checkerboard is distorted by
0.637. The alleged social median is exactly the naive report array. The Om/LO
SNLI diagnostic reaches 40.39% versus bag-of-words 62.44%, but all three LO
features are algebraic functions of Om norms/dot/cosine.

Role: **incorporated exact group/partition utilities, partial locality evidence,
negative equivariance result, standard graph denoiser, and semantic diagnostic;
retire learned-Sokoban, first-decisive-win, error-code/truth, and novel-LO
claims**. See `CUBE_GEOMETRY_LINEAGE_AUDIT.md`.

### S14 — Governance, economy, anchors, and structural selectors

This node consists of 40 files copied byte-for-byte between
`test/_ORGANIZED/02_Experiments/Rule-Economy-Governance` and the `test` root:
ten Python sources, ten JSON results, ten PNG figures, and ten verdict
documents.

The lineage begins with a 27-region energy/favor election economy. The archived
negative result is important: all 26 loud liars are elected, economy error
4.6818 exactly equals energy-only, and the same winner is stable after the first
round. Fresh inspection shows why this is structural: `run_region` never reads
its `obs` argument. Reversing every report leaves the winner and all trajectories
exactly unchanged. The rising reciprocal-affinity metric is feedback-driven
concentration, not independent evidence that agents chose cooperation.

The judge successor is a median-deviation filter with unelectable monitors.
Saved strength 0.5 removes all 20 loud outliers and reduces error to 0.06895.
Judge identity, evidence, history, and belief are absent; every judge applies the
same global anomaly vector plus independent noise. The result supports a useful
honest-majority outlier filter, not an independent judge community.

The majority/anchor sequence correctly demonstrates the median's 50%
coordinated-contamination boundary and catalogs oracle, patient-stake, and
in-tolerance structural attacks. One trusted value generated directly around
truth already reduces fresh 70%-cartel error from 4.9720 to 0.02716. The three
anchors are never integrated. The ideal oracle boundary is 16; equality is
tie-convention dependent and the source implementation chooses truth at exactly
16.

Deterrence has valid conditional expectation math. One-shot unlimited-stake
penalty is exactly `1/q`; the finite 40-round threshold at `q=.15` is 1.00911,
not simply one. In the saved `gain=20,q=.1` one-shot cap, stake 200 is exact
break-even. Detection remains exogenous, and reputation, identity reset, false
positives, and strategic adaptation are not modeled.

Shared fate directly defines `net=G-kappa*E*retained`; purge directly sets
majority detection to zero and deletion directly reforms liars; silence uses one
fixed oracle-known reliability mask with no rounds or staggering. These preserve
mechanism-design hypotheses and failure boundaries, not simulated equilibria or
observed deterrence.

Same-layer reality is sparse low-rank sensing with the exact ten-mode generating
basis. Forty random sensors are full rank with median condition 2.73, while the
chosen concentrated block has rank 6 and condition `1.29e15`. Reports are unused,
only 40 of 343 cells are sampled, and the corruption branch assigns fake values
from the wrong cell indices. The result is distributed observability under a
trusted channel.

The scalar rotation selector is an exact rank-21 orthogonal projector. Its saved
0.2485 symmetric-signal error matches the analytic projected-noise value 0.2474.
The equivariant vector projector is exact rank 42 of 1,029 and improves robust
decoding of large random off-subspace arrows, but cells occupy 21 orbits of
sizes 1/6/8/12/24 and coordinated in-subspace lies remain indistinguishable.

Role: **incorporated mechanism-design notebook, exact group projectors,
conditional robustness/incentive equations, distributed-sensing lesson, and
high-value capture evidence; preserve all ideas while retiring autonomous
governance, emergent cooperation, independent judges, strategic shared-fate,
public deterrence, staggered silence, oracle-free truth, and universal
error-correction claims**. See `GOVERNANCE_ECONOMY_LINEAGE_AUDIT.md`.

### S15 — Symmetry spectrum

This node consists of four files copied byte-for-byte between
`test/_ORGANIZED/02_Experiments/Symmetry` and the `test` root: one source, JSON,
PNG, and verdict.

The saved spectral result is exact. The 343-node/882-edge grid is the Cartesian
product `P7 □ P7 □ P7`; its analytic eigenvalues are sums of three path
eigenvalues and match numerical diagonalization to `3.55e-14`. They occupy 70
levels with the saved multiplicities 1/3/6/15/18.

The historical irrep explanation is incomplete. Axis-permutation orbits alone
produce 84 unordered mode triples with multiplicities 1/3/6. Exact path
identities `μ1+μ6=μ2+μ5=μ3+μ4=4` merge seven sets of three levels, giving
`84-14=70` and the six 15-fold plus one 18-fold collisions. Those larger
multiplicities are not individual proper-cube-group irrep dimensions.

All 24 proper and 24 improper signed-axis symmetries commute exactly with the
operator. The archived random graph's maximum multiplicity five is its five
connected components at eigenvalue zero. Ten connected same-node/edge random
controls have 343 distinct levels; a separable anisotropic-axis control also has
343 distinct sums.

Role: **incorporated exact separable-spectrum and full-cubic-symmetry artifact;
preserve analytic/unit-test value while narrowing irrep attribution and leaving
task, protection, compression, meaning, and learning advantage open**. See
`SYMMETRY_SPECTRUM_LINEAGE_AUDIT.md`.

### S16 — Cube-embedding and semantic-holonomy lineage

Primary root: `/Users/chetanpatil/Desktop/test/cube_embed`.

Adjacent exact preservation pairs:

- root `session_summary.md` and organized `00_START_HERE/session_summary.md`;
- root `ablation_study.py` and organized
  `02_Experiments/Nova-and-Misc/ablation_study.py`.

The 37-file package records four rapid generations: a 26-node graph-response
word operator, a 27-position angle field, PPMI-SVD/counter-fitting/spin lexical
grounding, and sequential sentence/SNLI features. No saved result log,
checkpoint, or official SimLex file survives.

The graph uses face-sharing cliques rather than its documented cubie incidence:
edge/corner degrees are 14/18 instead of 4/6. The raw cosine Fourier probe
matrix is rank 14; QR returns 27 orthonormal rows, eliminating the claimed
neighbor correlation and supplying 13 completion directions. All within-field
angles depend only on squared probe coefficients, so independent coefficient
sign flips leave the signature unchanged. The 27 neighbor and 13 loop entries
are reconstructed from the first 54 edge phases, and the unsigned,
orientation-blind loop sums are roughness statistics rather than holonomy.

The default built-in “SimLex” list has 131 rather than 150 pairs and only 15
score under the surviving default corpus. Counter-fitting, blend selection, and
spin training use the same target scores they report. At expanded N=100,
same-pair SVD+CF reaches rho 0.8459 but pair-held-out out-of-fold rho is 0.0337;
the cube channel is -0.0620 and the source layout ranks last of 21.

The archived three-seed ablation freshly reproduces the negative direction:
word-overlap+negation 48.92%, scalar SVD 40.77%, cube 35.82%, and cube+SVD
39.58% on selected dev. A new clean test-split ablation gives full cube 41.65%
versus direct 27D SVD sentence-pair features 47.85%.

Role: **incorporated historically valuable negative-result lineage; preserve
PPMI-SVD, evaluation decomposition, and scientific self-correction, while
retiring semantic-holonomy and task-advantage claims in the current
representation**. See `CUBE_EMBED_LINEAGE_AUDIT.md`.

### S17 — Games lineage

Primary organized folder:
`/Users/chetanpatil/Desktop/test/_ORGANIZED/02_Experiments/Games`.

Adjacent lineage:

- six same-named exact root copies;
- the hidden 13-module `/Users/chetanpatil/Desktop/test/evaluate_chess`
  project and its eight-test suite;
- three unique basin JSONs and three large receipt archives under
  `/Users/chetanpatil/Desktop/test/state/exp_sliding`.

The legacy chess file encodes board pieces and validates 1,000 random spatial
transitions, but copies metadata from the already-updated Python board. The
improved project explicitly transports side-to-move, castling, and en-passant
tokens. It passes eight unit tests, 1,000 random transitions, 22 adversarial
transitions, and a new 1,000-step continuous-state control with no state or
symbol-multiset failures. Python chess still supplies legality and move
semantics; move clocks and repetition history are incomplete.

The Level-2 “basin ranker” is a manually weighted linear score over mostly
Python-chess-derived features. Decoded and hybrid rankings are identical.
It reproduces 14/15 nominated handcrafted mates and 84/100 generated mates, but
the apparent handcrafted miss is another mate and the elementary
check-plus-fewest-legal-replies baseline reaches 100/100.

Tic-tac-toe preserves physical piece identity and demonstrates online
loss-state repulsion. The evaluation stream also trains the field. Across five
seeds, a frozen field versus the heuristic scores 810 wins, 729 losses, and 961
draws; against minimax it has no wins. A symbolic heuristic draws all 2,500
games against both heuristic and minimax opponents.

The 150 sliding starts advertised as 25 moves have mean exact depth 8.493 and
maximum 17. Exact search solves all under the 300-step budget. The source
memory modes solve only 1–9%, versus 74% for greedy Manhattan; persistence
worsens every mode. The state feature has 63,591 exact signatures for 181,440
reachable boards, and cosine distance cannot attract a nonzero state to the
all-zero solved feature.

The three receipt archives contain 620,327 chained archived entries plus 3,000
live entries, with zero adjacent receipt breaks. Their top-level JSON self-hash
does not verify the current serialized state because save order hashes the
previous hash field.

The sorting demo solves in 1,345 annealing steps while the exact arbitrary-swap
minimum is nine. Both sorting and one-off sliding are objective-supplied
SearchEngine demonstrations, not learned algorithms.

Role: **incorporated verified chess transport plus negative memory/search
evidence; preserve every state/receipt artifact, reuse the evaluation
boundaries, and retire game-intelligence or learned-planning advantage
language**. See `GAMES_LINEAGE_AUDIT.md`.

### S18 — NLI-Language honesty and compression ladder

Primary organized folder:
`/Users/chetanpatil/Desktop/test/_ORGANIZED/02_Experiments/NLI-Language`.

Adjacent lineage:

- 39 same-named root files, all byte-identical to the organized top-level
  copies;
- one organized-only pytest-rewritten CPython 3.11 bytecode file that embeds
  the surviving source timestamp and size;
- nine hidden basin JSONs and six receipt archives under
  `/Users/chetanpatil/Desktop/test/state/exp_snli`;
- the already incorporated SNLI/ANLI datasets, Nova tensor, semantics, cube,
  Om/LO, sacred, and NLI-ALL roots;
- an exact adjacent `session_summary.md` containing the only historical
  `nli_v8 × Nova` score record.

The folder is a six-generation self-correction ladder rather than an
independent breakthrough model. It moves from persistent 20D lexical basin
routing, through ANLI hypothesis-only/BoW gates and trimmed GloVe, into
character/word cube tests, adaptive context compression, and a small pure-NumPy
neural n-gram.

The strongest positive is conventional context modeling. A fresh fixed-corpus
replay gives online K4 ideal code length 1.781675 bits/char, versus bz2
1.800074, lzma 2.000211, and gzip 2.406544. A 64-bit message-length header
changes K4 only to 1.781839. Frozen order-3 prediction is 1.800547 on held-out
text; pruning 13,224 entries to 6,631 moves it only to 1.805276. The
probability model is normalized and self-adapting, but the source does not emit
or decode an arithmetic-coded bitstream. This is verified predictive modeling,
not yet a complete compressor implementation. Character surprise concentration
does not by itself establish where semantic meaning resides.

Every Livnium-specific language advantage fails a direct control. Character
exposure classes score 3.942856 bits/char, worse than the matched random mean
3.905738 and learned partition 3.752239. Word-level cube occupancy is literally
MD5 hashed BoW: it reaches 59.8738% SNLI with 39,366 columns, while direct BoW
reaches 60.1690% with 23,069; 14 geometry summaries add no lift. The saved
non-cheat basins score 41.60–42.27% on a balanced 1,500-example dev protocol,
where logistic regression on the exact same 20 inputs reaches 53.13%.
Cheat-mode 100% appends the true label as a one-hot input.

The hidden 729 MB state is preserved read-only. It contains three historical
12D states, three completed 20D non-cheat states, three partial 23D cheat states,
1,973,569 archived receipts, and 6,265 live receipts. All archive-to-live
receipt chains have zero adjacent breaks. The top-level JSON self-hashes do not
verify current canonical serialization because the save routine hashes before
replacing its own prior hash. The current `experiment_snli.py` deletes matching
state directories before retraining and must not be run against this archive.

Static mean GloVe reaches a saved 60.69% SNLI but remains below
hypothesis-only 61.48%; its ANLI results also stay below the stronger bars. The
small neural n-gram reaches 1.617459 held-out bits/char, while a count model
trained on the same bytes reaches 1.557914. Predictive score, ideal float16
parameter amortization, and the actual 2.84 MB optimizer checkpoint correspond
to 1.617459, 6.344796, and 77.407781 bits/char respectively and must remain
distinct payload claims.

The adjacent `nli_v8 × Nova` continuation gives a narrow dev-only observation:
DTW/distance summaries add about 1.6 points to a compact 13D lexical feature
set. Full features replay at 51.47%, below Count 55.73%, TF-IDF 56.65%, and
hypothesis-only 58.03%. Fracture fires on roughly 96%, adds no lift, and two of
four outputs are exact transforms of existing warp features. No official test
result exists, and process-random OOV hashing remains a reproducibility defect.

Role: **incorporated artifact-complete scientific self-correction and standard
context-prediction result; preserve trimmed vectors, checkpoint, basin/receipt
state, exact controls, and negative verdicts, while retiring Livnium
character/word geometry, basin-routing, fracture, and learned-language
advantage claims**. See `NLI_LANGUAGE_LINEAGE_AUDIT.md`.

### S19 — Demos, feedback, and persistent bridge

Primary organized folder:
`/Users/chetanpatil/Desktop/test/_ORGANIZED/02_Experiments/Demos`.

Adjacent lineage:

- six same-named exact archive-root copies;
- the saved ten-anchor `state/basin_memory.json` and its 1,557-line archive;
- exact root/organized `nova_basin_store.py`, owned by the final
  `Nova-and-Misc` row;
- previously incorporated Core base-27/rotation, Games tic-tac-toe/puzzle, and
  NLI/sliding Karmic-state continuations.

The folder is a chronological teaching layer, not six new projects. Its
base-27 demo correctly maps 27 glyphs to digits and shows that the 24 proper
cube rotations fix the center and permute outer symbols. Bare integer conversion
is not a fixed-width state codec: leading zero/core glyphs are lost, negative
integers return an empty string, and no string-to-state decoder exists.

The selected toy “random start” is already 100% accurate at epoch zero. Across
100 independent anchor starts, eight epochs raise mean accuracy from 43.32% to
95.91%, but 17 runs remain below 90% and the direct sign rule is perfect. The
swapped-anchor condition improves from 0% to only 54%. Preserve the prototype
mechanics and initialization controls, not a convergence advantage.

The inside-engine trace starts from a nominal 25-move shuffle that is exactly
depth three. Greedy Manhattan solves in three; the displayed annealer remains
unsolved after 5,001 proposals at Manhattan distance four. This is useful
Metropolis instrumentation and negative search evidence.

Feedback and Karmic game results are prequential. Across five 500-game streams,
naive pull scores 933/1470/97 W/L/D, naive both 465/1082/953, and Karmic
672/1293/535. Frozen Karmic fields score 1000/500/1000, while a direct symbolic
win/block/center/corner policy draws all 2,500 games with zero losses.
Freshness is inactive because `tick()` is never called. Losses write bad karma
to `X_Win`, while push reads `O_Win` bad karma, which remains zero; the claimed
adaptive reputation gate is therefore not operating.

The bridge has one partial positive result. Five matched future-stream pairs
give warm continuation 474/649/377 versus cold 279/810/411, a 13-point win-rate
lift and 10.73-point loss reduction in aggregate. One seed regresses, both
conditions continue learning on the reported games, and the saved frozen policy
still loses 838/2,500 versus zero for the symbolic control. This is task-relevant
prequential persistence, not held-out game-policy superiority.

The saved state has 1,000 live plus 1,557 archived receipts, although its
`ledger_total_count` and demo output call 1,557 the total. Operation counts sum
to 2,557. The adjacent receipt center-hash chain has zero breaks, but it excludes
authority, bad karma, support/harm counts, status, step, and ledger. All 1,108
decay hashes are unchanged despite metadata mutation. The wrapper charges each
nearest-anchor decay to every label anchor. Promotion has no scoring effect, and
an early return prevents promoted anchors from later being quarantined.

The state predates the surviving bridge, store, and Karmic source revisions.
Fresh final-source replay reproduces its headline behavior in temporary storage
but does not bind the historical bytes to final code. The demo's top-level
cleanup deletes the JSON/lock but leaves the archive, so it must not run against
the preserved root.

Role: **incorporated teaching and stream-persistence bridge; preserve base-27
digit presentation, prototype updates, transparent traces, paired continuation,
and bounded-log ideas, while narrowing fixed-width codec, learned convergence,
deep search, superior policy, adaptive law/court, total-count, and
full-mutation receipt claims**. See `DEMOS_LINEAGE_AUDIT.md`.

### S20 — Complete arch-archive root and oldest-copy boundary

The name `arch-archive` resolves to three 1.8 GB mirrors:

- `python/clean-nova-livnium/archives-local/arch-archive`, born
  2025-12-12 but missing the 136-file base `core`;
- `python/clean=noba=back/arch-archive`, born 2025-12-12 and the oldest
  self-contained copy;
- `workspace/clean-nova-livnium/archives-local/arch-archive`, born 2026-01-07
  and the artifact-complete copy because it alone keeps the K17/K4 Ramsey
  checkpoint already audited under S9.

After excluding bytecode and macOS metadata, the shared `brain`, `core-c`,
`language`, and `market-killer` branches are exact across all three roots; base
`core` is exact across the two copies that contain it. The archive name is not a
global chronology claim. The recovered control-group transcript records a
2025-02-14 room, and the conversation export begins 2025-03-03, both earlier
than the surviving arch-archive directory births.

The previously missed base Core has 106 Python files, 28 documents, and two
figures. A fresh full collection gives 252 passes, 25 failures, and six
collection errors. Standard odd-cube construction, exposure identities,
quarter-turn transformations, local gates/Born sampling, small exact
state-vector simulation, and modular switches survive. Completeness,
all-code-verified, globally entangled lattice, physical geometry coupling,
quantum speedup/hardware, K=0.38 laws, and label-inversion “semantic
consistency” do not.

Core-C passes 11/11 tests when imported through a valid package alias. It
preserves a standard center-plus-cycle representation, cyclic rotations and
inverse, and a structural-work ledger. It has no semantic task, encoder/decoder,
or discovered encoding base, so it is a clean structural prototype rather than
a semantic engine.

The market branch contains 503 OHLCV CSVs and six scripts but no provider,
download command, license, constituent policy, or held-out protocol. Its saved
NaN correlations come from an unmasked shifted target. With only that bug
corrected, Livnium tension has mean correlation -0.0109 with next absolute
return across 321 finite symbols, versus 0.2495 for current absolute return and
0.3536 for rolling volatility. Whole-history normalization leaks the future,
SPY is absent despite being the default, and the “euphoria” threshold is
unreachable because its maximum possible tension is 0.62 below 0.8.

`important.md` preserves three speculative layers. O-A8 contributes the useful
idea of transactional updates under a monotone declared objective, but its
example deletes nodes rather than implementing donor-backed promotion and
monotonic decrease proves neither correctness nor global convergence. O-A9 is a
multiscale boundary/circular-frame metaphor without a defined contraction,
topology, or physical-equivalence proof. O-A10 is a testable fixed-resource
information-density hypothesis, but exposure is not information and the fresh
reinforcement test fails.

The `brain` subtree is only an exact stock
`sentence-transformers/all-mpnet-base-v2` cache, not a Livnium-trained
checkpoint. The SNLI geometry images are orphan visual history without source
binding, while the efficiency chart is generated from hard-coded measurements
without protocol, hardware, or accuracy evidence. The layer-language note
preserves hollow/filled symbols, depth, alternation, and function output; later
exact parser copies honestly implement structure rather than word meaning.

Role: **incorporated historical root and chronology boundary; use
`clean=noba=back` as the oldest self-contained source copy and workspace as the
artifact-complete copy, preserve verified structural mechanisms, negative
results, hypotheses, notation, and visual history, while retiring completeness,
physical-law, semantic-engine, market-alpha, trained-brain, and unsupported
figure claims**. See `ARCH_ARCHIVE_ROOT_AUDIT.md`.

The complete directory/navigation boundary is recorded separately in
`ARCH_ARCHIVE_STRUCTURE_MAP.md`: one 98-directory canonical tree, every
first-party subbranch, counted repetitive corpora/caches, and the two exact
mirror overlays.

### S21 — Nova-and-Misc reconciliation

The thirteen remaining organized scripts are exact copies of same-named root
files, not separate projects. They bridge four existing lineages:

- Sacred static/dynamic evaluators and hidden-state ablations;
- cube-embedding ablation and angular-potential experiments;
- Nova Memory v1/v2 maintenance changes; and
- basin/observer visualization diagnostics.

The correct evaluator is label-blind; the dynamic comparison routes target
labels into collapse and is leaked. Gradient sweeps select only on development
conditions and save no final artifact. The swapped-head run has no checkpoint.

Nova v1 and v2 differ in four files. All fourteen script checks pass in each,
but pytest collects none. V2's bounded ledger and explicit conservation input
are useful, while its text representation remains a lossy hash-to-permutation,
its evaluator/cache contracts are incomplete, and its total ledger count is
wrong.

The landscape is a chosen three-anchor potential, not proof. Observer features
are exactly a reparameterization of mean/max/std plus a zero block.

Role: **incorporated as diagnostics and contract lessons; retain correct
label-blind evaluation, maintenance scaffolds, and candidate angular potential,
while retiring label-routed accuracy, semantic hash/permutation, proof-figure,
and added-observer-information claims**. See `NOVA_MISC_AUDIT.md`.

### S22 — Realcore legacy, snapshots, and Livnium Crux

The second older archive layer contains 231 meaningful canonical files plus
four preserved binary/archive extras and five later dual-cube revisions. Its
quantum-islands, pre-Core hierarchy, MPS, and quantum-computer content is the
historical source of Q1/Q2 rather than a new quantum implementation.

Every named P2 snapshot is indexed. GitNexus and Nova ZIPs exactly match their
extracted roots. Sacred is an earlier near-snapshot. The Core tarball is a
near-snapshot, and the two larger Core ZIPs share 1,691 exact meaningful files,
ten revisions, and already audited Nova/NLI/Rule30-only artifacts.

`livnium-crux-main.zip` is the unique survivor: a self-contained classical
Dart/JS/CLI/docs/visualizer package for base-27 codecs/arithmetic, cube moves,
couplers, Potts recall, and a 27-child tree. A fresh replay passes 32/32 tests;
analysis has no errors but reports 41 warnings/info items. It is a polished
release ancestor, not semantic, compression, or quantum proof.

Nine Git worktrees sit below `lab/infected`: six first-party Livnium/Nova roots
and three exact third-party WikiExtractor checkouts. The latter are corpus
dependencies, not secret Livnium projects.

Role: **incorporated as historical source, validated classical release, exact/
near-exact snapshots, and third-party boundaries**. See
`REALCORE_LEGACY_SNAPSHOTS_AUDIT.md`.

## Current lineage conclusion

- Best active preservation copy for Realcore/quantum: `Desktop/core`.
- Best honest quantum research narrative: July repo `archive/cortex-v2`.
- Best ECW-BT evidence bundle: the `python/clean-nova-livnium/archives-local`
  copy.
- `Desktop/uantum` and `Desktop/livnium` are incorporated; the latter preserves
  the corrected Cortex successor, mixed/negative results, and the best March
  SNLI checkpoint.
- The sacred artifact vault, replay copy, empty checkpoint-policy directory, and
  repaired retraining branch are incorporated without merging them.
- Sacred-v2 is incorporated: its torque-memory checkpoint is the current
  replayed sacred leader, the historical 96% model is provisionally
  leaked/unusable, and Nova Eye remains unfinished.
- Lab Nova-SNLI is incorporated: its 90.58% gold-routed result directly
  demonstrates the ancestral label shortcut, while proper inference is 32.99%.
- Infected Archive Nova-v3/quantum-embed is incorporated: its only model is the
  duplicate 76.12% collapse1-static artifact; its conditional 77.17% error file
  and abandoned encoders remain historical evidence.
- NLI-ALL is incorporated: its best honest archived generation is the simple
  40.71% classifier; later physics variants are near chance, broken, or backed
  by invalid target/evaluation paths; all three roots are semantic mirrors.
- S9 is incorporated: sibling mirrors, the unique Ramsey checkpoint, archived
  crypto/quantum/NxN applications, and the displaced Rule-30 lineage now have
  explicit evidence and preservation roles.
- S10 is incorporated: the semantic package mirrors, complete workspace
  artifacts, effective-rank collapse, zero-graduate boundary, SNLI cache/orphan
  classifier, and saved conversation mind-map now have explicit roles.
- S11 is incorporated: the spherical/simplex variants and task benchmark now
  have exact mirror boundaries, independent mathematical/software checks, and
  explicit preservation versus retirement decisions.
- S12 is incorporated: the two exact Sudoku preservation copies, five
  implementation generations, puzzle-identity failure, valid hybrid scaffold,
  partial pure learner, and failed RL generalization now have explicit roles.
- S13 is incorporated: all 22 Cube-and-Geometry files, exact group/partition
  mathematics, transformed-input identity, partial learned locality, broken
  whole-map equivariance, graph-denoising boundary, and redundant Om/LO
  features now have explicit roles.
- S14 is incorporated: all 40 governance/economy files, election/judge/anchor/
  incentive/purge/sensing/selector generations, exact duplicate boundary, and
  explicit oracle-versus-mechanism decisions now have preservation roles.
- S15 is incorporated: all four symmetry-spectrum files, exact P7³ spectral
  arithmetic, full 48-element commutation, product collisions, and comparator
  boundaries now have explicit roles.
- S16 is incorporated: the complete cube-embedding package, session memory,
  negative ablation, algebraic sign/redundancy failures, SimLex leakage
  boundary, and held-out SNLI comparison now have explicit roles.
- S17 is incorporated: six exact Games preservation pairs, the hidden improved
  chess project, saved sliding memories/receipts, continuous/frozen/exact-depth
  controls, and direct symbolic baselines now have explicit roles.
- S18 is incorporated: 39 exact NLI-Language preservation pairs, trimmed GloVe,
  neural checkpoint, hidden basin/receipt state, six-generation honesty ladder,
  adaptive context result, and Nova-DTW continuation now have explicit roles.
- S19 is incorporated: six exact Demos preservation pairs, the saved bridge
  state, base-27/prototype/puzzle/feedback/Karmic sequence, matched warm/cold
  continuation, frozen policy, court, and receipt contracts now have roles.
- S20 is incorporated: the three complete arch-archive roots, their true
  chronology, previously missed base Core/Core-C/market/language/O-A8 material,
  stock cache, and orphan figures now have explicit preservation and evidence
  boundaries.
- S21 is incorporated: all thirteen Nova-and-Misc scripts have exact
  preservation identity and explicit diagnostic, leakage, and contract roles.
- S22 is incorporated: the February Core delta, second legacy archive, all
  named snapshots, Livnium Crux, nested archive objects, and all deep Git roots
  have explicit roles.
- No named P0, P1, or P2 lineage remains unassigned.

No source project was moved, deleted, renamed, or merged during this audit.
