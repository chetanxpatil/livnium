# Claim Ledger

Updated: 2026-07-27

This ledger uses the July organized repository as the current evidence source.
Older documents remain historical evidence but do not override a later audit.

## Proven core mathematics

| Claim | Status | Evidence location |
|---|---|---|
| Base-27 codec over `0,a..z` is reversible and has exact carry | Proven | `lets_clean_it/livnium/packages/livnium-core` |
| Odd-cube exposure classes have closed-form counts | Proven | `packages/livnium-core/src/livnium_core/lattice.py` |
| Canonical symbolic weight is `SW = 9f` with a closed-form total | Proven | Core package and tests |
| The cube has 24 orientation-preserving rotations | Proven | Rotation package and tests |
| Rotations preserve exposure class, symbolic-weight total, and bijection | Proven | Rotation and move tests |
| Nested cubes have additive ledgers and a wreath-product description | Proven | Hierarchy implementation and tests |
| Current cyclic 3-gram frequencies do not determine their next Rule-30 frequencies | Proven | Explicit N=6 witness `001011` versus `001101`; same current count vector, different next vectors |
| The four archived Rule-30 3-gram formulas lie in cyclic de Bruijn flow plus normalization constraints | Proven/narrowed | Exact symbolic rank/span check; identities are real but not Rule-30-specific dynamical discoveries |

## Verified engineering

| Claim | Status | Boundary |
|---|---|---|
| Small exact state-vector simulation supports gates, Bell/GHZ states, and measurement | Verified engineering | Classical simulation; exponential within each register |
| The recovered three-qubit simulator performs the teleportation protocol correctly | Verified engineering | 200 seeded random complex states checked on 2026-07-26; classical fixed-size simulation |
| MPS can represent long GHZ chains with small bond dimension | Verified engineering | Textbook MPS behavior, not new quantum hardware |
| The Ramsey prototype's incremental K5 violation delta matches full recounts | Verified component | 100/100 random small-graph flips checked; this does not produce a 43-vertex witness |
| The archived Ramsey clique counter reproduces the exact R(3,3) small boundary | Verified component | Exhaustive all 32,768 K6 colorings: minimum two monochromatic triangles; K5 five-cycle has zero |
| Dart Realcore conserves its own chosen ledger across rotations and coupling | Verified engineering | Uses conventions that differ from canonical Python core |
| DynamicsLedger records collapse trajectories and prevents unsupported exact-energy claims | Verified engineering | Observability feature, not proof that every collapse rule is conservative |
| Livnium-O's global SO(3) flow preserves distances and tangency | Verified engineering | Standard Rodrigues/Euler rotations and history-backed undo pass 42 focused direct tests; this does not establish packing validity or new physics |
| The archived SAT/CSP/Max-Cut suite preserves independently checkable witnesses and baselines | Verified harness | DIMACS/CSP/GSET data, assignments/partitions, objective fields, CSP/SAT checkers, PySAT/python-constraint, and greedy Max-Cut survive; solver interpretation is separately retired |
| Nova produces deterministic receipts/state hashes and enforces archive-only maintenance in its tested small system | Verified engineering | All nine collected pytest tests and fifteen direct test scripts passed; toy scale and placeholder gold set |
| The semantics refactor implements online tabula-rasa SGNS and sharded per-word state | Verified engineering | Real 64D skip-gram/negative-sampling and persistence mechanisms; semantic quality and “physics” interpretation are separate questions |
| The saved conversation mind-map is a structurally valid graph artifact | Verified engineering | 499 unique nodes, 9,579 unique valid edges, four basins, and matched `chats.txt`/JSON hashes; graph usefulness is not proof of new physics |

## Measured and promising

| Claim | Status | Required caution |
|---|---|---|
| Supervised collapse NLI reached about 68.9% on SNLI | Measured | Matched end-to-end, multi-seed causal ablation remains pending |
| The saved March EGAN checkpoint reaches 76.34% on the complete 9,842-example SNLI dev split | Measured artifact | Reproduced from tracked code; no matched independently trained no-collapse/MLP baseline or multi-seed result |
| The sacred collapse1-static checkpoint reaches 76.12% on the complete 9,824-example SNLI test split | Measured artifact | Fresh deterministic, noise-free replay with a recovered exact embedding backbone; historical workbook records a nearby 76.01% run |
| The sacred-v2 torque-memory checkpoint reaches 76.52% dev and 76.42% test | Measured artifact | Fresh deterministic, noise-free, static label-blind replay; best surviving sacred checkpoint audited so far, but no matched seeds or independently trained causal ablations |
| The lab Nova-SNLI physics checkpoint reaches 32.99% proper test and 90.58% with gold-label routing | Measured shortcut artifact | Same checkpoint and complete split; the 57.59-point difference demonstrates target-label injection, not valid NLI performance |
| The repaired `collapse_retrain` epoch-23 checkpoint reaches 69.76% dev and 68.87% test | Measured artifact | Label-blind held-out replay; training log, seed, and preceding epochs are missing |
| Sacred cache-based ablations reach only about 54.44–60.01% dev in their saved logs | Historical measured artifacts | Several runs are incomplete and the family uses cached features; useful negative history, not a matched final benchmark |
| Noun-collapse embeddings reached SimLex noun ρ≈0.362 | Measured | Matched-corpus SGNS/PPMI comparison remains important |
| Exact-gradient collapse v2 descends the cosine potential it implements | Proven for that update law | Task advantage over simpler alternatives remains unproven |
| A compression experiment beat gzip on one structured-text protocol | Partial | Reproduce with fixed corpus, decoder accounting, and standard compressors |
| Grad-V approximated a trained update on one checkpoint | Partial | Requires multi-seed and cross-dataset replication |
| Local 3×3 pooling retained more variance than random pooling on one split of 7×7 digit images | Measured pilot | Standard locality prior; repeat across splits and compare with pyramid/wavelet/conv baselines |
| Directional 27-block pooling beat size-matched random pooling on scale-matched synthetic smooth fields | Partial | Axis-aligned generator; parity on white noise and reversal for over-smooth fields bound the result |
| Reversible rotations cannot change permutation-invariant coherence | Proven | General consequence of a rotation acting only as a permutation |
| The saved ECW-BT student checkpoint retains its SBERT teacher geometry | Artifact verified | Mean row cosine ≈0.999999; this is retention, not evidence that CCD improved semantics |
| Dynamic alpha can alter MPS compression at a fixed budget | Partial/mixed | Random run lowers internal truncation without output-fidelity gain; structured run trades one observable for another |
| GloVe/PCA cosine alpha separates facts from filler on the X-ray vocabulary | Diagnostic only | One hand-selected seed vocabulary and one document; seed-frequency IDF is nearly flat |
| The one-epoch lab physics embedding retains local context information | Diagnostic only | On 200k filtered WikiText test pairs, context beats random 54.97% versus 51.03% for a fixed row-shuffled control; six runnable hand-written analogies all miss and no matched skip-gram baseline exists |
| The infected Archive calibrated-error file conditionally implies 77.17% SNLI test | Partial historical artifact | Its 2,243 unique rows are internally consistent test errors, but completeness is assumed and the producing model, command, seed, calibration protocol, and full predictions are missing |
| NLI-ALL `nli_simple` reaches 40.71% on the complete valid SNLI test split | Measured historical baseline | Fresh static replay at `PYTHONHASHSEED=0`; saved 7,900-word polarity memory helps, but the model never predicts neutral and its within-word letter bag collapses anagrams |
| NLI-ALL v4's saved supervised rule tree reaches 46.0% on `features_test.csv` | Partial historical diagnostic | Only 100 rows with uncertain provenance; the saved 48.37% is resubstitution on its 13,988 training rows, and the default classifier does not load the tree |
| The workspace Ramsey K17/K4 checkpoint has 21 monochromatic K4 violations | Measured historical artifact | Restricted decode plus independent recount; 7 color-0 and 14 color-1 violations, about 99.1176% constraint satisfaction but not a valid coloring |
| Rule-30 causal Phase 9 reaches 99.6% logistic and 98.2% MLP accuracy on next-density threshold | Measured but analytically dominated | Label-blind and future-free, but `c_{t+1}` is exactly `f_t` dotted with the Rule-30 local lookup; exact baseline is effectively 100% |
| The saved semantics embeddings and active manifolds are effectively rank one | Measured negative artifact | Current/backup 64D PC1 variance ≈99.64% with effective rank ≈1.04; 16,381 active centroids have PC1 ≈99.951% and effective rank ≈1.006 |
| Zero of 16,381 active semantic manifolds meet the project's graduation rule | Measured negative artifact | Saved mass is at most 1.3 and radius at least ≈0.7546; criterion is mass>2 and radius<0.2 |
| The saved 256D semantic SNLI cache reaches 49.5% with a fresh linear head | Diagnostic only | Its next-1,000-row split is not an official independent test and hypothesis-only reaches 48.8%, accounting for nearly all linear signal |
| The archived N-Queens-8 basin sweep has a repeatable score distribution | Measured negative artifact | 50 runs: mean 22.34/29, median 19, range 19–28, zero valid solutions; diagonal constraints are absent from search tension, so the distribution does not validate CSP solving |
| The archived G1 Max-Cut sweep produces a one-step distribution | Measured negative artifact | All 50 runs stop at step one, mean cut 3,047 versus known 11,624; it measures initialized/shared-state variation, not converged optimization |

## Retired or narrowed

| Former claim | Current verdict |
|---|---|
| Geometry alone understands language or beats standard AI | Retired: fair NLI benchmarks put geometry-only systems near chance |
| SNLI numbers alone prove reasoning | Retired: hypothesis-only artifacts and leakage must be controlled |
| Livnium simulates hundreds/thousands of globally entangled qubits | Retired: many independent small registers or MPS states are not one global arbitrary state |
| Livnium created quantum internet or physical teleportation | Retired: simulations reproduce mathematical correlations only |
| R(5,5) ≥ 44 from the proposed construction | Falsified by independent clique checks; the `uantum` stochastic script has no replacement witness |
| Rule 30 near-perfect center-bit prediction | Retired: the 99.7–99.8% path predicts a thresholded global-density summary rather than a fixed spatial bit and reconstructs it from a representation containing its ingredients; later causal next-density is an exact local-rule lookup, while autonomous rollout remains near chance |
| The four Phase-1 Rule-30 identities are novel dynamical invariants | Narrowed: they are exact generic cyclic substring-flow/normalization identities; the distinct valid result is the explicit failure of 3-gram dynamic closure |
| Rule 30 is a discovered high-dimensional continuous rotation | Unsupported: one-step aggregate fits and same-state reconstruction do not establish a rotation law; five-step fit collapses and saved autonomous horizons are about two to three steps |
| Rule-30 shadow density near 0.5 proves trajectory reconstruction | Retired: marginal density can match while bitwise/field accuracy is near chance, perfect-grid rate is zero, and one saved generator emits only zeros |
| The archived AES-32 search demonstrates AES cryptanalysis | Retired: AES-32 is a custom broken four-byte toy whose decryption fails; the fixed disclosed key is recovered using classical pair/local brute force |
| The archived variable-round code measures AES-128 | Falsified: its third ShiftRows row is wrong and a standard 10-round known-answer vector fails |
| Even-dimensional cubes break rotation, exposure, or symbolic-weight invariants | Falsified: all 24 proper rotations on N=2,4,6 are bijective and preserve exposure and `SW=9f`; only the absence of a unique center cell is established |
| The archived quantum-core capacity represents millions of globally entangled qubits | Retired: counts are cells or independent/local small registers, and deeper runs sometimes skip quantum-state tests entirely |
| Archived teleportation/Bell demos prove physical quantum entanglement | Retired: they are duplicate classical dense state-vector simulations; sampled CHSH values above 2√2 are finite Monte Carlo noise |
| The K17 Ramsey checkpoint is a successful R(4,4) coloring | Falsified: independent recount finds 21 monochromatic K4 violations |
| Cortex retrieval superiority from a single toy paragraph | Retired: the mock contains the answer key; a 150-document evaluation has LIVNIUM-B below TF-IDF/YAKE and identical to Alpha-Only |
| Chord-directed collapse v1 has an exact global scalar potential | Withdrawn: v1 is non-conservative; v2 is the exact-gradient alternative |
| Universal geometry engine / new physics | Retired as a scientific claim; may remain a design metaphor |
| The recovered “shell funnel” implements literal concentric-shell collapse | Incorrect description: the code performs one-step pooling into 27 sign-based directional blocks |
| Semantic placement plus funnel proves that the cube filters meaning | Narrowed: placement and reconstruction use the same embedding, so it demonstrates self-quantization rather than independent semantics |
| Nested attractor depth shows a cube-specific conditioning advantage | Retired for this run: a random-graph operator accelerated more, and methods used different right-hand sides |
| Pairwise/geometric quantum islands are a scalable globally entangled register | Retired: scalability comes from independent/local states; the project's GHZ test produces globally illegal outcomes |
| Livnium's SAT/Grover demos establish P=NP or a realized quantum speedup | Retired: basis states are enumerated by classical simulation; no complexity breakthrough is demonstrated |
| ECW-BT's `accepted=true` proves CCD improved SBERT embeddings | Retired for the saved run: output is nearly identical to SBERT, validation lacks a seed baseline, and the trainer has force-sign/anchoring issues |
| `Desktop/uantum` Cortex is the best current MPS implementation | Retired: reverse-direction CNOT fails; later Desktop and July copies contain the fixes |
| The March SNLI run achieved 87.5% entailment and 81.2% contradiction accuracy | Not reproduced: audited values are 80.32% entailment, 77.55% contradiction, and 71.00% neutral |
| The March EGAN system has about 2M total parameters and the documented end-to-end speed | Unsupported/miscounted: the saved state contains about 13.0M parameter elements; timing claims need a complete protocol |
| Later alpha, memory, InferSent, and Lyapunov checkpoints improve the original March model | Retired for saved artifacts: all audited later checkpoints score below the original 0.7634 |
| Nova's `+0.5321` survival gap proves semantic causality or memory usefulness | Retired: the haircut policy explicitly archives lowest-alpha nodes, so the measured relationship is largely built into the policy |
| The local Nova package establishes TRL4 | Unsupported: self-defined thresholds, placeholder three-query gold set, toy scale, and no independent task benchmark |
| Synthetic alpha survival gaps generalize through the MPS governor | Retired for current artifacts: arXiv Mode A averages only +0.0068 and Mode B aggregates -0.0006 |
| The iDEX documents are accepted technical evidence | Narrowed: preserve them as dated proposal artifacts; their TRL and universality language exceeds the audited local evidence |
| Sacred collapse4 with dynamic basins achieved 95.76% dev / 96.07% test | Provisionally leaked/unusable by recovery decision: the evaluator exposes gold labels, the number exists only in README text, the named checkpoint is missing, no result artifact supports it, and every surviving related model replays near 76% or lower |
| Sacred dynamic-basin evaluation was always static and therefore could not leak labels | Retired for the saved code: both train and test evaluators can pass gold labels into label-specific routing; the surviving checkpoint is insensitive to shuffled/static routing, so this bug does not recover the 96% claim |
| The sacred radial/cosine collapse rule is exact gradient descent on its written energy | Withdrawn by the project's own corrected equation: magnitudes are cosine-based while directions are Euclidean radial, and the neutral rule omits gradient terms |
| The saved sacred collapse map is generally contractive | Falsified by the saved Jacobian experiment: mean spectral norm 41.11, maximum 198.07, and 0% of sampled norms below 1 |
| Sacred-v2's fixed torque/anchor/axial physics forces cause its saved SNLI accuracy | Narrowed/unsupported: zeroing every fixed force post hoc leaves predictions essentially unchanged, while bypassing or zeroing the learned repeated residual update drops test accuracy to roughly 59–66% |
| Sacred-v2 failure memory improved the torque-memory checkpoint | Retired for the saved implementation: it records failures but its difficulty boost is never used by the training loss, gradients, routing, or inference |
| Nova Eye trained a visual/raw-signal SNLI model | Unverified: `eye-v1` is empty, the only log stops before epoch results, the trainer uses clipped ordinal characters rather than the retina/glyph pipeline, and a statistics-key mismatch prevents checkpointing after epoch one |
| Nova Eye's emergent BasinField strictly conserves energy | Falsified for the current code: strengthening a lone basin creates energy and decay removes energy without redistribution |
| Lab Nova-v3 achieved 74.4% SNLI test | Retired for the surviving root: its only unique checkpoint saves 33.33% best dev and freshly replays at 32.99% proper test; no supporting log or prediction artifact exists |
| Lab Nova-v3's 90.58% gold-routed score is valid held-out NLI | Retired as leaked: the target label selects the basin immediately before classification; shuffled routing is 48.85% and static routing is 32.99% |
| The lab physics embedding jointly learned its collapse physics | Unsupported: the optimizer receives only the embedding table; collapse anchors and the randomly initialized learned update are excluded, and no matched standard embedding baseline exists |
| The lab Nova-v3 inference path uses a learned collapse update and a frozen encoder | Incorrect for the saved source: the main collapse has only three anchors, the nested embedding-collapse update is not called by sentence encoding, and the 50,000×256 table is fine-tuned |
| Lab Nova-v3's 52.3 MB size, 7,800+ CPU pairs/s, and 28-minute training are saved benchmark evidence | Unsupported: the complete checkpoint is 155.4 MB and no timing or successful training log survives |
| Infected Archive Nova-v3 contains a distinct improved SNLI checkpoint | Retired: its sole weight is byte-identical to sacred collapse1-static and freshly replays at the same 76.12% label-blind test accuracy |
| Infected Archive Nova-v3's dynamic evaluator is valid held-out inference | Retired: its default path can pass the gold target into `collapse_dynamic`; the saved static duplicate checkpoint is valid only because dynamic basins are disabled |
| The archive geometry encoder preserves word-level base-27 structure | Falsified for its base features: coordinate conversion keeps only the signature modulo 27, so tokens sharing their first character collide exactly |
| The archive Sanskrit encoder supplies ordered semantic representations | Retired for the saved mapping: character features are mean-pooled and therefore collapse anagrams up to floating-point summation noise |
| The adjacent infected `quantum_embed` root is a completed quantum-language result | Unsupported: no checkpoint or result survives; meaningful source duplicates Sacred-v2, and its evaluator rebuilds incompatible test vocabulary IDs |
| NLI-ALL's v3–v8 “physics” generations improve the simple classifier | Retired for surviving static artifacts: simple is 40.71% full test, while runnable v3–v7 range from 33.85% to 36.06%; v8 does not import |
| NLI-ALL v4 discovered an 85.23%-accurate unsupervised law | Retired: 85.23% is example prose; the saved tree is supervised, scores 48.37% on its own training rows and 46.0% on a separate 100-row file, and is unused by default inference |
| NLI-ALL v5's 82.32% geometry-alignment number is SNLI accuracy | Retired: it predicts labels generated by its own geometry teacher, whose agreement with SNLI is only 34.17% |
| NLI-ALL v5/v6/v8 100% debug accuracy validates the decision law | Retired as answer injection: debug paths receive and return the gold label and can overwrite forces from it |
| NLI-ALL fracture dynamics detects negation or contradiction | Falsified by focused replay: it fires on 86.6% of 1,000 examples and contradiction precision given fracture is 32.45%, essentially the base rate |
| NLI-ALL v7 learns persistent geometry shaping | Falsified for current source: reinforcement mutates a fresh per-example classifier after prediction, then the instance is discarded and no geometry state is saved |
| The three NLI-ALL roots are independent replications | Retired: all 186 meaningful files match after excluding generated bytecode and macOS metadata |
| The semantic workspace proves a rich 32D/64D physics of meaning | Retired for saved artifacts: nominal dimension is not effective dimension; all saved semantic states are dominated by one principal component |
| The current semantic generation validates a 100% WordOracle, Jacobian R²=0.905, or the Phase-20 bank/lambda/livnium/gravity/model scores | Unsupported historical prose: producing code, labels, probes, Jacobians, predictions, and sidecars are absent; current per-word backup files contain only names, dimensions, counts, and scalar histories |
| The surviving semantic package has 15/15 passing tests | Unsupported for the recovered tree: the documented `tests/` directory is absent and pytest collects zero tests |
| The saved semantic SNLI classifier validates the 256D feature cache | Falsified by artifact shape: the classifier is an orphan 6→3 projection, while the cache is 256D and current heads expect 18D or 256D with different state keys |
| `physics_small` is highly accurate because it aligns with a synthetic cluster center | Retired: the target center is freshly random and unrelated to the model; 1,000 trials give mean cosine 0.0066 and none above 0.8 |
| Passing E/N/C to the recovered `physics_small` audit creates class-conditional behavior | Falsified for its active configuration: dynamic basins are disabled and labeled, blind, and all three label outputs are exactly identical |
| Mind-map basin tension measures semantic conflict | Falsified on every retained edge: with alignment>0.4, `tension=|0.38-alignment|=alignment-0.38`, so higher similarity always produces higher tension |
| The mind-map's greedy basins emerge from a non-clustering physical law | Narrowed: they are disjoint anchor neighborhoods selected by cosine threshold and centrality; preserve as a graph heuristic |
| The Python and workspace semantics roots are independent replications | Retired: all 101 meaningful non-checkpoint files match; workspace is a strict artifact superset with the current model, physics-small, shards, and 1,071 extra word backups |
| Livnium-O's cap-sum constraint guarantees non-overlapping sphere packing | Falsified: unit weight admits 14 neighbors, no pairwise separation is checked, and generated configurations contain overlaps starting with the default six-neighbor case |
| Livnium-O reproduces the classical three-dimensional kissing number | Falsified: its formula admits 14, documentation also misstates about 6, and satellite-to-satellite exclusion is absent |
| Livnium-O exposure spans `[0,1]` and total SW is `9N` | Falsified for current formula: cap-derived `f` tends to 0.5; six unit neighbors sum to about 3.617 rather than 54 |
| Livnium-O's overlap dynamics is Hamiltonian and discovers gravity | Retired: the force differs from the numerical gradient by 33 in the probed overlap region, and “gravity discovery” refits the quadratic inserted by source |
| Livnium-T implements the 12-element tetrahedral rotation group A4 | Falsified: matrix IDs 3 and 7 do not map the tetrahedron to itself and 50/144 ordered products are outside the stored set |
| Livnium-T rotations create reversible system dynamics | Falsified for current system method: `apply_rotation` returns the same node objects in the same mapping |
| Livnium-T Bell pairs and recursive capacity establish multi-qubit entanglement | Retired: pair state is detached from local measurement, CNOT cannot use the API, and capacity scripts count independent two-amplitude allocations |
| Livnium-T conserves its 108-SW ledger during basin search | Falsified: one correct update to vertices 1 and 2 changes total SW to 108.2 and `verify_ledger()` fails |
| The archived SAT solver solved 10/10 test formulas | Retired: the summary counts any winner object; only 1/10 saved assignments satisfies every clause and there is no UNSAT proof path |
| The archived CSP solver solved all six test problems | Retired: no saved assignment satisfies all constraints; all five satisfiable cases are missed, and one invalid candidate does not prove the triangle UNSAT |
| The archived N-Queens basin objective includes diagonal constraints | Falsified: the universal constraint encoder has no `diagonal` branch, so those fields return zero tension |
| The archived Max-Cut solver is competitive on GSET | Falsified: G1 and G14 reach 26.37% and 31.43% of stored known values versus the included greedy baseline at 97.53% and 95.43%, with Livnium stopping at step one |
| Archived G11/G12 ratios are comparable to the literature targets | Invalid: the parser discards signed edge weights and optimizes unweighted crossing counts while the stored targets use the weighted objective |
| Core-O, Core-T, and benchmark copies in three roots are independent replications | Retired: 30, 36, and 1,051 meaningful files respectively match byte-for-byte across the roots |
| The original Sudoku solver uses symmetry augmentation and no hand-coded Sudoku logic | Falsified: `symmetry_variant` has zero call sites, while inference explicitly masks row/column/box-illegal digits |
| The original Sudoku “held-out style” score is held-out accuracy | Falsified: 58.61% is `model.score(X,Y)` on its 44,240 training examples; a fresh 20-board/80-mask replay is 53.22% |
| The original Sudoku solver solves easy puzzles 85% | Narrow measured artifact: 34/40 exact generating-completion matches on random 51-given boards with a legality mask; uniqueness and standard difficulty were not enforced |
| The pure Sudoku model learns some local structure on unseen generated boards | Measured partial: strongest saved 1,200/150/200 `py-sudoku` run has 43.36% test-cell accuracy and exact completion 62.5/14.5/1.5/0% from easy through expert |
| The hybrid Sudoku implementation is a complete working solver | Verified standard engineering on the sampled boards: independent replay validates every returned grid and givens; completeness comes from propagation/MRV/backtracking |
| The hybrid learned ordering gives 1.7x less expert search | Narrowed to one unstable mean: 43.24→25.69 candidate attempts reproduces, but paired p=0.755, bootstrap CI crosses zero, and LCV averages 23.49 |
| Hybrid “backtracks” are actual backtracks | Falsified label: the counter increments on every candidate attempt, including the successful path; independently counted failed branches are lower |
| The 4x4 tabular agent learned Sudoku from reward | Retired as generalization: full replay solves 3/3 memorized training puzzles and 0/100 unseen six-hole puzzles |
| The 4x4 RL action space includes erasure | Falsified: source declares `nA=64` and implements placement only despite an 80-action docstring |
| The 9x9 policy is episodic REINFORCE and learns Sudoku | Falsified: training uses immediate reward-baseline updates without `gamma`; all saved train/test solve rates are zero and the example changes no cell |
| Random clue count establishes Sudoku difficulty | Unsupported: puzzles are not checked for uniqueness or logical/search difficulty; fresh multiple-solution rates reach 56–70% at 40 givens and 100% at 23/26 |
| CubeSokoban's 100% is learned unseen-world/Sokoban generalization | Retired: all 720 canonical test rows exactly match a training row, hash/one-template 1NN also score 100%, train/test share 40 world identities, and no Sokoban mechanics exist |
| The CubeSokoban 24-rotation action is valid | Verified engineering: 24 unique bijections and zero closure failures among all 576 products |
| A plain model must see every exact cube angle | Falsified literally: the archived architecture trained on 23 rotations reaches 95% on the withheld view |
| The odd-cube directional counts close exactly | Proven elementary identity: `1+6m+12m²+8m³=(2m+1)³` |
| Learned directional filters beat fixed directional means | Measured on the saved digit split: 0.6357 versus 0.5956, while PCA reaches 0.7181 |
| Learned directional blocks decisively beat learned random blocks | Unsupported: matched fresh controls give 30/40 wins and random layouts as high as 0.6458 versus directional 0.6357 |
| The rotation-tied autoencoder is equivariant as a whole | Falsified structurally: encoder biases and dense decoder/bias are untied; only encoder weights satisfy the C4 relation |
| Rotation tying reduces the model from 49 to 13 parameters | Misleading partial count: those are encoder weights only; whole totals are 548 versus 512, a 6.57% reduction |
| Rotation tying helps on matching isotropic fields | Negative saved result: tied is about 0.011–0.014 worse than untied at every sampled size |
| Geometry-direct smooth fields form a low-dimensional error-correcting code | Falsified: `I+5L` is full rank 343 and the source-generated family spans all `R^343` |
| Geometry-direct protects honest reports exactly | Falsified generally: clean smooth error is 0.016 but a clean checkerboard is distorted by 0.637 |
| Geometry-direct beats a local social median | Not tested: the “social” array is byte-identical to the naive corrupted reports and computes no median |
| Only a social anchor can distinguish prior-compatible global alternatives | Overstated: external evidence is required, but trusted measurements, temporal constraints, redundancy, or another model are alternatives |
| Om/LO adds distinct local-observer information | Falsified algebraically: all three LO features are exact functions of the Om norms, dot product, and cosine |
| Om/LO 40.39% proves observer-relative semantic understanding | Diagnostic only: compact geometry features carry SNLI signal but trail bag-of-words by 22.05 points and lack hypothesis-only/lexical controls |

## Governance/economy lineage decisions

Detailed evidence: `GOVERNANCE_ECONOMY_LINEAGE_AUDIT.md`.

| Claim | Current verdict |
|---|---|
| The cube economy uses traded favor to clean information | Falsified structurally: `run_region` never reads observations; reversing every report preserves the winner and all trajectories exactly |
| Cooperation autonomously emerges in the favor economy | Unsupported: reciprocal affinity is installed by the update and concentrates around the first winner; the metric grows from 1.04 to 243.76 while all 26 loud liars remain elected |
| An unelectable judge community independently detects liars | Narrowed to a median-outlier filter: judge identity/evidence/history is unused and every judge repeats one global anomaly vector with injected noise |
| Report-only median consensus survives arbitrary coordinated majorities | Falsified at the standard boundary: coordinated contamination flips the median at 50%; extra truth information or assumptions are required |
| A few verified anchors defeat a 70% cartel | Oracle-conditional: one value sampled directly around truth already gives about 0.027 error; the experiment does not establish how honesty/truth is verified |
| The oracle holds iff `lambda > 16` at 70% cartel | Boundary correction: above 16 truth uniquely dominates; at exactly 16 the median is set-valued and the source tie-break chooses truth |
| Oracle, stake, and structure were tested as one robust stack | Unsupported proposal: the three components are tested separately and no correlated/joint attacker is simulated |
| A one-shot lie needs penalty approximately `1/q` | Proven for the stipulated risk-neutral process with unlimited stake and exogenous independent detection; exact threshold is `1/q` |
| Repeated play needs only penalty 1 because detection is almost sure | Narrowed: asymptotically yes, but at finite horizon 40 and `q=.15` exact threshold is 1.00911 with escape probability 0.001502 |
| Stake must always cover `penalty × worst-case gain` to deter | Overstated necessity: that is sufficient to implement the uncapped fine; in the saved one-shot `gain=20,q=.1` setup exact expected break-even stake is 200 |
| Shared fate closes majority capture strategically | Unsupported as an equilibrium: negative payoff is inserted directly as `G-kappa*E*retained`; agents, adaptation, hedging, timing, and exit are absent |
| Public ten-strike deletion was observed to deter liars | Retired for this implementation: deletion directly converts each remaining liar to honest with a stipulated visibility probability |
| Majority purge failure and anchor rescue emerge from detection | Hardcoded boundary: `q_eff` is set to zero above 50% without an anchor and restored with one |
| The silence experiment implements staggered three-round rest | Falsified structurally: it has no time, rounds, cooldown, or schedule and permanently masks oracle-known noisiest voters |
| Same-layer reality pins every cell and defeats a global lie without an oracle | Retired: 40/343 truth-centered sensors are passed to a known ten-mode decoder while reports are unused; the trusted channel is the oracle assumption |
| Distributed same-layer sensing beats the concentrated block | Verified for the constructed design: random 40-sensor matrices are rank 10 with median condition 2.73; the archived block is rank 6 with condition `1.29e15` |
| Same-layer corrupted ties use the fake value at each corrupted cell | Falsified by index mismatch: almost every selected row receives `Vfake` from a different cell, inflating fresh 20%/40% error |
| The natural selector nonlinearly cleans at every hierarchy level | Retired description: it is one fixed linear rank-21 orthogonal projection with no hierarchy |
| The scalar selector's 0.248 symmetric-data error is a novel learned effect | Narrowed to matched projection: analytic RMS projected white-noise error is 0.24744 because truth is generated in the same 21D subspace |
| The equivariant vector projector has about 42 valid degrees of freedom | Verified exactly: character trace gives rank 42 of 1,029; idempotence/self-adjoint errors are numerical zero |
| Every vector cell has 24 independent rotated copies and can be corrected arbitrarily | Falsified/overstated: cell orbits have sizes 1, 6, 8, 12, and 24, and coordinated equivariant in-subspace lies remain valid |

## Symmetry-spectrum lineage decisions

Detailed evidence: `SYMMETRY_SPECTRUM_LINEAGE_AUDIT.md`.

| Claim | Current verdict |
|---|---|
| The 7x7x7 cube grid has 70 distinct pull speeds among 343 modes | Proven for the archived operator/tolerance: analytic P7³ sums match numerical eigenvalues to `3.55e-14` and reproduce the exact histogram |
| The 15- and 18-fold levels are cube-group irreducible representations | Falsified attribution: proper octahedral irreps have dimensions 1/1/2/3/3; 15/18 merge several axis-mode permutation families through exact path-spectrum identities |
| Only the 24 proper cube rotations organize the spectrum | Narrowed: 24 improper signed-axis/reflection operations also commute; equal-axis Cartesian separability and arithmetic collisions are load-bearing |
| The random graph's maximum multiplicity five is an accidental shared speed | Falsified: it is the zero eigenvalue from five connected components |
| A connected random graph at the same node/edge count has a generic simple spectrum | Measured control: all ten fresh connected seeds have 343 distinct levels |
| Separability alone causes the 70-level pattern | Falsified by anisotropic control: generic unequal axis weights retain separable sums but produce 343 distinct levels |
| The spectrum uniquely fingerprints spatial cube locality | Unsupported: orthogonal conjugation preserves all eigenvalues while generally destroying coordinate locality; spectrum is not a complete invariant |
| Spectral degeneracy establishes a Livnium task advantage | Open: no task, model, perturbation, sample-efficiency, or downstream result is present |

## Cube-embedding and holonomy lineage decisions

Detailed evidence: `CUBE_EMBED_LINEAGE_AUDIT.md`.

| Claim | Current verdict |
|---|---|
| The v1 graph implements documented physical cubie incidence | Falsified: face-sharing cliques give edge/corner degrees 14/18 rather than 4/6 |
| The v1 WikiText agreement establishes semantics | Falsified: similarities collapse near one and all ten top-5 neighbor overlaps are zero |
| QR preserves locally correlated Fourier probes | Falsified: raw cosine matrix rank is 14 and the final 27 probes are mutually orthogonal for neighbors and non-neighbors |
| The field preserves semantic direction | Falsified exactly: its within-field measurements depend only on squared probe coefficients, producing independent sign ambiguity |
| The 94D signature contains 94 independent geometric channels | Falsified: the 27 neighbor and 13 loop values are deterministic functions of the 54 edge phases |
| The loop-winding block is holonomy | Retired terminology: it is an unsigned, orientation-reversal-invariant sum of ordinary edge angles with no transport connection or group composition |
| The transform is intrinsic to the underlying semantic space | Falsified: a shared semantic-basis rotation preserves raw cosine to `2.36e-16` but changes cube similarity by mean 0.289 and up to 0.946 |
| Character trigram vectors are deterministic across processes | Falsified unless `PYTHONHASHSEED` is externally fixed before startup |
| PPMI-SVD provides corpus semantic grounding | Verified engineering: it is the actual source of semantic signal, although the preserved default falls back to a 177K-token validation file and only 636 vectors |
| Default SimLex counter-fitting generalizes | Falsified by resubstitution: same-pair rho 0.8459 collapses to pair-held-out 0.0337; cube out-of-fold is -0.0620 |
| The chosen cube layout is semantically privileged | Negative measured control: its raw cube rho ranks last among the source plus 20 random layouts |
| SentenceField is order-sensitive | Preserved narrowly as deterministic recency/state blending; common alpha/beta scale cancels and the returned embedding is final state, not cumulative trajectory |
| The remembered 43.10% SNLI run is stable evidence | Retired by the archive's own later ablation, lacks a saved log/model, and does not fresh-replay |
| Cube geometry improves SNLI beyond direct semantic vectors | Falsified on held-out test: full cube 41.65% versus direct SVD sentence-pair 47.85% |
| Cube features contain no supervised signal | Too strong: cube-only reaches 40.40% versus 34.50% majority, but the transform is lossy and not advantageous |

## Games lineage decisions

Detailed evidence: `GAMES_LINEAGE_AUDIT.md`.

| Claim | Current verdict |
|---|---|
| Livnium can transport chess pieces and main position metadata with conserved symbols | Verified engineering: eight tests, 1,000 random moves, 22 adversarial moves, and a continuous 1,000-move control pass with zero symbol-multiset failures |
| The improved representation is a complete bijection over full chess state | Narrowed: halfmove/fullmove clocks are not updated or decoded and repetition history is absent |
| Livnium independently implements chess legality and move semantics | Retired: `apply_livnium_move` requires a Python-chess board, which supplies legality and special-move meaning |
| The random chess test covers captures, promotions, castling, and en passant | Falsified for the measured seed: 98 captures, nine promotions, one castle, and zero en-passant captures |
| The handcrafted mate ranker scores 14/15 top-1 | Reproduced for the nominated target; it scores any legal mate first in 15/15 because the single miss is an alternative mate |
| A Livnium basin learns to rank mate-in-one | Retired: the ranker is fixed manual weights over rule-derived features and contains no trained/queried basin |
| Hybrid Livnium attack features improve mate ranking | Falsified on tested data: decoded and hybrid complete rankings are identical |
| The mate ranker demonstrates advantage over elementary chess logic | Falsified: check plus fewest legal replies scores 100/100 versus the archived ranker's 84/100 |
| Basin tic-tac-toe nearly solves the heuristic opponent | Online-stream result only: the same games train the field and the source single run adapts to 491 draws with nine losses |
| Tic-tac-toe basins form a robust frozen policy | Falsified: across 2,500 frozen heuristic games X wins 810 and loses 729; against minimax it wins zero and loses 1,484 |
| Basin tic-tac-toe beats a simple symbolic policy | Falsified on this protocol: the standard heuristic draws all 2,500 games against both heuristic and minimax |
| A 25/40-step puzzle shuffle measures puzzle depth | Falsified: immediate reversals are allowed; 150 nominal 25-step starts have exact depths 1–17 and mean 8.493 |
| Persistent basin memory improves sliding-puzzle solution rate | Falsified: persistence worsens all three modes; greedy Manhattan solves 74% versus 1–9% for memory modes |
| Sliding state features uniquely identify a board | Falsified: 181,440 reachable boards map to 63,591 tuples and 63,383 cosine rays |
| A solved-state anchor attracts unsolved puzzle states under cosine distance | Falsified exactly: solved is the zero vector and has distance 1.0 from every nonzero feature |
| Sliding receipts are internally hash-chained | Verified engineering: zero adjacent breaks across 620,327 archived plus 3,000 live entries |
| The saved basin JSON's top-level state hash verifies its current canonical content | Falsified: the save routine hashes before replacing its own prior `state_hash` field |
| Livnium sorting learns an efficient sorting procedure | Retired: exact target Hamming is supplied and annealing takes 1,345 steps where direct placement is optimally nine swaps |

## NLI-Language lineage decisions

Detailed evidence: `NLI_LANGUAGE_LINEAGE_AUDIT.md`.

| Claim | Current verdict |
|---|---|
| The organized NLI-Language folder is disposable duplicate clutter | Falsified: its 39 top-level files have exact root copies, but the coherent verdict ladder, trimmed GloVe, neural checkpoint, bytecode provenance, and adjacent 729 MB basin state form a uniquely useful evidence bundle |
| Adaptive Livnium compression reaches about 1.782 bits/char | Measured narrowly: the normalized online order-4 context model freshly reproduces 1.781675 ideal bits/char on the fixed 389,335-byte corpus |
| The adaptive source is a fully self-contained compressor | Not implemented: it sums ideal `-log2 p` length but does not emit or decode a finite-precision coded stream |
| Rare contexts carry most useful predictive information | Negative on this corpus: pruning order-3 entries from 13,224 to 6,631 changes held-out prediction only from 1.800547 to 1.805276 bits/char |
| Surprising characters are where meaning lives | Unsupported: code cost measures local predictability; semantic importance needs an independent intervention or downstream test |
| Livnium character exposure classes capture predictive language structure | Negative: final-source replay is 3.942856 bits/char versus random 3.905738 and learned partition 3.752239 |
| The saved decisive CSV is output from the final source | Falsified by chronology and content: the CSV predates the source revision and its failed 4.1418 positive control replays as 3.7522 under the surviving source |
| Word-level Livnium geometry reaches about 60% SNLI | Narrowed to standard hashing: MD5 bucket occupancy reaches 59.8738%, but no cube neighborhood/distance is used, plain BoW reaches 60.1690%, and geometry summaries add nothing |
| The word-level comparison is size-matched | Falsified: cube occupancy has 39,366 columns versus 23,069 for plain BoW |
| Saved non-cheat basins use their 20 inputs competitively | Negative: saved basin modes score 41.60–42.27%, while logistic regression on exactly the same features reaches 53.13% |
| Cheat-mode 100% proves basin routing is sound | Falsified inference: the true label is appended as a one-hot input and all partial saved states decode it at 100% |
| The hidden SNLI receipt archives are internally chained | Verified engineering: zero adjacent breaks across 1,973,569 archived receipts; top-level JSON self-hashes separately fail because of save order |
| Mean-pooled GloVe clears the strongest SNLI artifact baseline | Negative: saved GloVe is 60.69% versus hypothesis-only 61.48%; fresh replay preserves the direction |
| The small neural n-gram beats matched count modeling | Negative: neural is 1.617459 bits/char versus train-size-matched order-6 Witten-Bell at 1.557914 |
| Nova-DTW/fracture establishes a Livnium NLI advantage | Negative overall: full compact features reach 51.47% on selected dev versus Count 55.73%, TF-IDF 56.65%, and hypothesis-only 58.03%; there is no untouched-test result |
| DTW summaries add information to the compact lexical feature set | Partial: nested dev ablation adds roughly 1.6 points, but design and evaluation reuse one development subset |
| Fracture contributes a distinct useful signal | Falsified on this protocol: it fires on about 96% of examples, D≈C, and two of four outputs are exact transforms of warp features |

## Demos lineage decisions

Detailed evidence: `DEMOS_LINEAGE_AUDIT.md`.

| Claim | Current verdict |
|---|---|
| The six organized demos are six unique projects | Falsified: all six are exact root mirrors and form a teaching layer over Core, Games, Karmic, and Nova-store mechanisms |
| Base-27 word conversion works | Verified engineering for canonical strings: ordinary positional numeral arithmetic and a 27-glyph digit alphabet |
| Integer/binary conversion is a reversible full-lattice codec | Falsified: leading zero/core glyphs are discarded, negative input returns an empty string, and no state decoder exists |
| The chosen random-start BasinField run learns to 100% | Falsified for that run: it is already 100% at epoch zero |
| BasinField improves the toy clusters across initialization | Measured: mean accuracy rises from 43.32% to 95.91% over 100 initializations, but 17 remain below 90% and a direct sign rule is 100% |
| The inside-engine demo solves a deep 25-move puzzle | Falsified: the start has exact depth 3; annealing fails after 5,001 proposals while greedy Manhattan solves in 3 |
| Pull feedback improves online wins over random | Measured on the training stream: seed 42 rises 1.6% to 47.2%; five-seed pull averages 37.32% wins |
| Pull+push has one stable benefit | Negative: beta changes win-versus-draw behavior and outcomes vary sharply by seed |
| Karmic law control is the best game policy | Falsified: it lies between naive pull and naive both depending objective, and a symbolic heuristic loses zero games |
| Karmic freshness is active | Falsified: the demo never calls `tick()` and all global freshness steps remain zero |
| Earned `O_Win` bad reputation governs push | Falsified: `O_Win` bad karma remains zero, so push stays at its fixed minimum scale |
| The source energy-jump metric is consecutive change | Falsified: it compares each score with the first score in the game |
| Anchors persist across bridge sessions | Verified engineering: ten anchors reload and affect later stream behavior |
| Warm persistence improves a matched cold continuation | Partial positive: five-pair warm wins are 31.6% versus 18.6%, but one seed regresses and both conditions train on reported games |
| The saved bridge is a strong frozen policy | Negative: 850/838/812 W/L/D across 2,500 games; direct symbolic heuristic is 0/0/2,500 |
| The saved ledger contains 1,557 total receipts | Falsified: 1,557 are archived and 1,000 are live; operation counts sum to 2,557 |
| Every spawn has a receipt | Falsified: new anchors begin with a reinforce receipt; no spawn operation appears |
| Receipt hashes protect every mutation | Falsified: the adjacent chain is intact but hashes centers only; authority, bad karma, counts, status, step, and ledger are excluded |
| Harm is attributed only to the responsible nearest anchor | Falsified in `NovaBasinStore`: one nearest decay increments harm and records a receipt for every anchor under the label |
| Promotion gates pull authority | Falsified: provisional and promoted scoring is identical |
| Promoted anchors can later be quarantined | Falsified: an early return makes promoted status permanent even after 30 harms and bad karma 1.0 |
| Saved bridge state was produced by the final sources | Falsified by chronology: the state predates bridge, store, and Karmic source revisions |

## Complete arch-archive root decisions

Detailed evidence: `ARCH_ARCHIVE_ROOT_AUDIT.md`.

| Claim | Current verdict |
|---|---|
| `arch-archive` is the oldest recovered Livnium work | Falsified: its surviving directory births begin 2025-12-12; the control-group transcript records 2025-02-14 and the conversation export begins 2025-03-03 |
| The first `arch-archive` path found is complete | Falsified: `clean-nova-livnium` lacks the 136 meaningful base-Core files |
| `clean=noba=back/arch-archive` is the oldest self-contained mirror | Verified from surviving filesystem birth times and subtree comparison |
| The workspace `arch-archive` is the artifact-complete mirror | Verified: it alone adds the previously audited K17/K4 Ramsey checkpoint; other meaningful shared branches match |
| The base Core is fully verified | Falsified: fresh full pytest gives 252 passes, 25 failures, and six collection errors |
| Base Core contains reusable mathematics and simulation code | Verified narrowly: odd-cube construction, exposure identities, quarter-turn actions, standard local gates/Born sampling, small exact state-vector simulation, and modular switches pass |
| The 24 cube group is fully enumerated and closure-checked | Narrowed: the code implements generators and compositions, not a complete enumerated closure proof |
| Core-C is a working structural prototype | Verified: 11/11 tests pass through a legal package alias for center-plus-cycle state, rotations/inverse, and structural work |
| Core-C is a semantic encoder | Falsified: it has no semantic input/task, encode/decode path, or empirically discovered encoding base |
| Archived market tension predicts next volatility | Negative: corrected mean correlation is -0.0109 across 321 finite symbols, versus 0.2495 for current absolute return and 0.3536 for rolling volatility |
| The saved market NaN sweep is a valid negative result | Falsified as written: the shifted target’s final NaN is not masked before `corrcoef` |
| The market “euphoria” state can occur under archived constants | Falsified: maximum possible tension is 0.62 while the threshold is 0.8 |
| O-A8 implements donor-backed transactional promotion and proves convergence | Falsified: its example deletes a node, no integrated promotion engine exists, and scalar monotonicity is insufficient for correctness or convergence |
| A monotone declared objective is a useful update guard | Preserved as an engineering principle, subject to explicit rollback, feasibility, and task-correctness checks |
| O-A9 establishes circular-frame physical equivalence | Unsupported: no contraction, topology, limit, or physical validation is defined |
| O-A10 proves information capacity from structural work/exposure | Unsupported: exposure is not an information measure and the fresh reinforcement test fails |
| `brain` contains a trained Livnium model | Falsified: it is a stock `sentence-transformers/all-mpnet-base-v2` Hugging Face cache with no custom state, trainer, or result log |
| The orphan SNLI and efficiency figures are reproducible evidence | Falsified: SNLI figures have no source/protocol binding; the efficiency chart uses hard-coded numbers without hardware, command, artifact, or accuracy context |
| The archived layer language survives as an implementation | Verified structurally: later exact parser copies preserve layer/depth/alternation/output notation, while explicitly making no word-meaning claim |

## Nova-and-Misc decisions

Detailed evidence: `NOVA_MISC_AUDIT.md`.

| Claim | Current verdict |
|---|---|
| The organized Nova-and-Misc scripts are thirteen independent experiments | Falsified: all thirteen are exact same-named root copies and bridge existing lineages |
| The static Sacred evaluator is label-blind | Verified by source path: no target label enters representation collapse |
| The dynamic Sacred comparison is a valid NLI evaluation | Falsified: gold labels are passed into dynamic collapse before classification |
| The hidden-state ablation is replayable as preserved | Falsified: stale `quantum_retrain` path resolution fails before evaluation despite surviving artifacts elsewhere |
| Angular-gradient sweeps prove an improvement | Unsupported: development-only hyperparameter sweeps, hard-coded baseline, no selected saved result, no untouched test |
| The swapped-head experiment produced a recoverable result | Unsupported: trainer exists, but no checkpoint or result artifact was found |
| Nova Memory v2 is a real maintenance improvement | Narrowly verified: arbitrary text is accepted, alphabet identity is global, supplied states can be checked, and live ledgers are bounded |
| Nova v2 encodes arbitrary text reversibly or semantically | Falsified: it folds text into an MD5-seeded permutation of the same 27 symbols |
| Nova v2's evaluator reports a weighted median and strict trend | Falsified: the median is unweighted, mean gain determines validity, and equality is accepted |
| Nova v2's semantic cache is mutation-safe | Falsified: its key omits weights although embeddings depend on them |
| Nova v2's ledger total count is total history | Falsified: fifty entries with live capacity ten report forty, the archived count |
| The basin dynamics script is a test | Narrowed to smoke diagnostic: no assertions or seed; one of five tensions rose in a fresh run |
| The saved energy landscape is proof of training or an external record | Falsified: it is a chosen potential around three saved anchors projected to two dimensions |
| Observer features add information over `[mean,max,std]` | Falsified algebraically: `R.mean=0`, `E.max=R.max+Om`, and `E.std=R.std` |

## Realcore legacy, snapshots, and Crux decisions

Detailed evidence: `REALCORE_LEGACY_SNAPSHOTS_AUDIT.md`.

| Claim | Current verdict |
|---|---|
| The February Python Core closes the archived Core failures | Falsified: bounded replay retains the same 25 failures |
| Its `n_qubits=27` fixture is small and CI-safe | Falsified: exact state-vector execution is killed with exit 137 |
| The recursive inheritance revision is useful | Verified narrowly: child configs now retain quantum feature switches |
| The February project assessment is a current verification report | Historical only: visually sound and strategically cautious, but completeness/all-tests/quantum wording is stale |
| Dual cubes implement discovered semantic physics | Falsified: contradiction, drift, cancellation, trapping, decay, and capacity are inserted rules |
| The pre-Core tests verify dual/trapped-cube claims | Falsified: 29 `test_*` functions contain zero assertions |
| The C Ramsey validator is useful | Verified on 200 seeded random K6/k=3 colorings against the independent Python checker |
| The Ramsey geometry distinguishes complete colorings | Falsified: 20 distinct K8 graph hashes all map to `(0.0,0.0,0.0)` |
| Ramsey speedup forecasts are measured | Unsupported: documents forecast 2–1000x without benchmark artifacts |
| GitNexus and Nova ZIPs contain stranded unique files | Falsified: 360/360 and 60/60 regular files respectively match extracted roots |
| Livnium Crux is a unique working release ancestor | Verified engineering: fresh 32/32 tests pass; Dart analysis has no errors but 41 warnings/info items |
| Livnium Crux proves semantics, compression, or quantum behavior | Unsupported: it is classical base-27/cube/Potts/tree engineering |
| Every deep Git root below `lab/infected` is a Livnium project | Falsified: three are exact third-party WikiExtractor checkouts |

## Open questions

1. Does collapse outperform matched MLP and linear heads when every model is
   trained end-to-end with the same embeddings, seeds, budget, and splits?
2. Does noun-collapse retain its result on a frozen matched corpus against SGNS
   and PPMI-SVD?
3. Does exact-gradient v2 provide useful task behavior beyond guaranteed energy
   descent?
4. Does the angular potential improve a locked, label-blind NLI protocol against
   direct embedding, linear, and matched nonlinear heads over multiple seeds?
5. Is Dart Realcore's A7 cross-lattice coupling worth implementing after its
   numerical conventions are reconciled with the Python core?
6. Can the compression result survive complete accounting and independent
   baselines?
7. After correcting its update directions and fusion schedule, does ECW-BT
   improve a fixed SBERT seed on standard intrinsic and downstream evaluations?
8. Can a predeclared dynamic-alpha policy improve its target observable without
   regressing matched fidelity, structure, and compute controls?
9. Does semantic memory triage help on a held-out access trace against LRU, LFU,
   TF-IDF/BM25, embedding, random-score, and learned policies?
10. Does the 76.34% EGAN checkpoint retain an advantage when collapse, MLP,
    residual, and standard NLI heads are independently trained with matched
    parameters, embeddings, seeds, and data?
11. Does Nova's retention policy improve an outcome independent of alpha when
    evaluated with a real gold set and matched memory-policy baselines?
12. Does the label-blind 76.42% Sacred-v2 torque-memory checkpoint retain an
    advantage when the iterative residual, MLP, fixed-force, and no-collapse
    variants are independently trained with matched seeds and budgets?
13. Can a repaired Nova Eye pipeline outperform character-CNN/transformer
    baselines after integrating the intended retina/glyph features and fixing
    its evaluation and conservation bugs?
14. Can dynamic basins provide an advantage when training may use labels but the
    inference API is structurally unable to receive gold labels?
15. Can the physics-embedding objective beat matched skip-gram/SGNS when collapse
    parameters are either honestly optimized or cleanly ablated under fixed
    seeds, corpus, vocabulary, and intrinsic/downstream benchmarks?
16. Can the useful NLI-ALL components—supervised lexical memory and explicit
    sequence alignment—beat matched lexical/hypothesis-only baselines when
    detached from gold-label debug routes and physics metaphors?
17. Can the emergent semantic learner avoid rank collapse and beat matched
    SGNS, PPMI, and fastText controls on predeclared intrinsic/downstream tasks?
18. Does a corrected mind-map using honest cosine/community metrics improve
    retrieval or human navigation over BM25, embedding search, and standard
    graph clustering on the same conversation corpus?
19. Can a pairwise-valid spherical variant produce a useful packing/search
    method against known spherical-code baselines when force is derived exactly
    from one continuous potential?
20. Does a corrected faithful A4 action provide any task advantage beyond a
    conventional tetrahedral graph representation?
21. Can basin-owned immutable candidate states beat random, greedy/local
    search, and simulated annealing on tiny exhaustive SAT/CSP/Max-Cut cases
    before scaling to the preserved UF20 corpus?
22. Can learned Sudoku value ordering beat LCV, randomized ordering, and
    standard search consistently across seeds on a fixed hashed corpus of
    unique puzzles with recognized difficulty?
23. Does the pure Sudoku learner exceed legal-candidate, location-only, and
    generator-bias baselines on the identical held-out puzzle/mask identities,
    while increasing independently valid completion rather than only
    source-solution equality?
24. On underlying-world-disjoint tasks with a real rotation-invariant property,
    does cube canonicalization or a fully equivariant model beat orbit lookup,
    augmentation, standard 3-D CNN/GNN, and group-pooling baselines?
25. Does directional locality remain above matched random/optimized partitions,
    wavelets, pyramids, convolution, and PCA across dataset splits after the
    complete autoencoder is made equivariant and total parameters are matched?
26. Do independently informed unelectable monitors reduce truth error and
    attacker profit without increasing false punishment under a fixed adversary
    budget, compared with the identical median-and-stake system?
27. Can a persistent integrated ledger with identity, stake, slashing, exit,
    Sybil return, and dumpable/shared exposure produce an incentive advantage
    once detection and truth are not supplied as oracle constants?
28. Does optimized distributed sensor placement remain robust under model
    mismatch, sensor corruption, and coherent attacks inside the assumed
    low-rank/equivariant subspace?
29. On a group-compatible underlying-world-disjoint task, does explicit
    representation decomposition improve accuracy, sample efficiency, or
    robustness beyond augmentation, group averaging, and matched generic
    priors?
30. Can a mathematically defined orientation-sensitive connection and
    path-ordered transport preserve semantic information and beat an identity
    transform, random probes/layouts, and direct embedding baselines on an
    untouched test set?
31. Can state-action or transition memory improve 8-puzzle search over BFS/A*,
    greedy Manhattan, and matched replay buffers when reward is attached to the
    resulting transition and the goal has a nondegenerate representation?
32. Can a Livnium chess state become a complete, independently ruled game
    representation after clocks/history are added and legality no longer
    depends on an external Python-chess board?

## Promotion rule

No claim moves upward because an AI, README, or exciting single run called it a
breakthrough. Promotion requires the command, code, data identity, seed set,
baseline, result artifact, and failure conditions.
