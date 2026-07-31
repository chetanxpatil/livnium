# Livnium Concept Map

This map prevents an old idea from looking new merely because its name or
programming language changed.

## 1. Foundation

| Idea | Names used | Meaning |
|---|---|---|
| Spatial alphabet | Base-27, `0+a..z`, cube language | Reversible mapping between symbols/numbers and a 3×3×3 spatial arrangement |
| Global anchor | Om, Origin Matrix, Observer | The fixed reference at the center |
| Local frame | LO, Local Observer | A temporary observer used for relative direction and context |
| Exposure | freedom, faces, boundary class | Number of coordinates touching the outer boundary |
| Symbolic weight | SE, SW, interaction potential | Canonical Python form is `SW = 9f`; old implementations contain other conventions |
| Spherical cap occupancy | Livnium-O exposure, blocked cap, continuous f | Implemented O-formula is a normalized occluded-cap quantity tending to 0.5, not a `[0,1]` free-exposure law |
| Cap-budget prefilter | generalized kissing weight, solid-angle budget | Sum of core cap weights can reject some impossible radius sets but cannot enforce neighbor-to-neighbor separation or prove a packing |

## 2. Lawful transformation

| Idea | Names used | Meaning |
|---|---|---|
| Rotation grammar | A4, quarter turns, cube group | The 24 orientation-preserving rotations and related face turns |
| Faithful finite-group action | tetrahedral A4, even permutation action, closure | A rotation table must act on canonical objects and satisfy closure; orthogonal determinant-1 matrices alone do not identify the intended group |
| Orbit canonicalization | canonical cube view, rotation quotient, orbit hash | Map every member of a known transformation orbit to one exact representative; guarantees invariance but can make transformed test inputs identical to training |
| Whole-map equivariance | tied encoder/latent/decoder, `f(gx)=g f(x)` | Weight tying in one layer is insufficient; biases, latent action, decoder, and output must obey the same group contract |
| Conservation | ledger, D2/D3, no cheating | Explicit quantities must remain invariant under allowed operations |
| Structural/task-state separation | immutable SW, basin score, confidence field | Do not mutate a declared conserved law to represent learning; task scores need separate state and contracts |
| Hierarchy | Livnium-H, recursive cube, wreath product | Cubes nested inside cubes with additive bookkeeping |
| Reversibility | hard machine, zero-sum permutation | Transformations preserve enough information to undo them |
| Unique center-cell anchor | observer anchor, odd-cube center, omcube center | Odd cube sizes have one natural central lattice cell; even cubes do not, although both retain proper cube rotations and exposure invariants |
| Cyclic substring flow | de Bruijn flow, pattern-frequency constraints, Rule-30 invariants | N-gram frequencies on a cyclic row obey generic incoming/outgoing substring balance plus normalization |
| Coarse-state non-closure | no 3-bit Markov closure, frequency insufficiency | Two full states can share the same aggregate N-gram statistic but evolve to different next aggregates |

## 3. Meaning and selection

| Idea | Names used | Meaning |
|---|---|---|
| Directional meaning | polarity, Φ, intent | Observer-relative cosine or directional relation |
| Difference from equilibrium | divergence, `0.38-alignment`, tension | A semantic/dynamic overlay; not derived from the discrete cube core |
| Attractor selection | basin, well, valley, gravity | State moves toward a stable region |
| Basin-owned candidate | assignment state, partition witness, reversible delta | A candidate solution must own an independently decodable/scorable state; coordinate lists in one mutable global lattice are not competing solutions |
| Strict solver outcome | witness valid, decision complete, objective gap | Distinguish candidate production, verified SAT witness, verified UNSAT proof, feasible optimization witness, timeout, and optimality gap |
| Objective identity | weighted GSET, metric contract, witness recomputation | Solver, baseline, saved witness, and literature target must use exactly the same signed/weighted objective |
| Puzzle identity contract | unique Sudoku, canonical grid, difficulty provenance | Generated clue count is not difficulty; preserve puzzle hash, uniqueness, source, givens, validity, target equality, and search metrics separately |
| Learned value ordering | Sudoku pull, digit ranker, search heuristic | A learned model may reorder candidates inside a standard complete solver; benefit requires paired comparison against LCV/random controls on identical puzzles |
| Rejected-action loop | unchanged-state argmax, policy stall | If a wrong action leaves state unchanged, deterministic greedy selection can repeat forever; mask, remember, perturb, or terminate the rejected action |
| Elimination | collapse, refusal, pruning, lawful forgetting | Remove unstable or inadmissible possibilities instead of enumerating all paths |
| Admissibility | law territory, border, watchtower | Test whether a proposed operation may survive the system's invariants |
| Geometric importance | alpha, semantic mass, rotation magnitude | A task-layer score derived from a word or rotation and used for pruning; not part of the proven discrete cube core |
| Resource-conditioned preservation | governor, entropy ceiling, dynamic alpha | Change tensor/memory compression policy under a fixed budget; usefulness depends on a predeclared task observable |
| Static versus label-routed collapse | static basin, dynamic basin, BasinField | Static inference settles without a gold label; historical dynamic routing selects a label-specific basin and must be prevented from receiving evaluation labels |
| Iterative learned refinement | collapse layer, learned residual, torque model | Repeated trainable state update; Sacred-v2 diagnostics show this learned residual, not its fixed torque/physics overlay, carries the saved prediction |
| Label-free basin lifecycle | emergent basin, birth, strengthen, decay, replace | Discover and maintain attractors without class routing; conservation must be implemented and tested rather than inferred from the name |
| Target-label shortcut | gold-routed basin, dynamic label basin, answer injection | Selecting a class-specific state transformation with the held-out target can make the answer directly decodable; lab Nova-SNLI rises from 32.99% proper test to 90.58% this way |
| Deterministic lexical geometry | geometry encoder, base-27 token cube, signature coordinates | Historical archive mapping that accidentally retains only the first character after modulo-27 conversion; preserve the interpretable-feature idea, not this word representation |
| Phonetic character bag | Sanskrit encoder, phoneme encoder, semantic phoneme features | Simplified articulatory character features followed by mean pooling; order loss makes anagrams equivalent, so it is at most an auxiliary feature idea |
| Polarity lexicon | brain state, SimpleLexicon, word polarity memory | Supervised word-to-entailment/contradiction/neutral moving averages; the only NLI-ALL saved state with a measurable current inference effect |
| Letter-bag lexical vector | native chain, deterministic letter geometry, chain encoder | Normalized sum of process-hashed character vectors; sentence position survives but within-word order does not, so unseen anagrams collide |
| Semantic sequence alignment | semantic warp, collision warp, path alignment | Dynamic-time-warping-like alignment using cosine cost; potentially useful engineering, not evidence of a new physical law |
| Lexical fracture | collision fracture, semantic fracture, negation fracture | Maximum aligned cosine mismatch; archive version fires on 86.6% of a focused SNLI sample and does not distinguish contradiction or negation |
| Opposition axis | inward/outward, signed divergence, resonance × sign(divergence) | v6/v7 scalar direction experiment; preserved as a formulation without demonstrated NLI advantage |
| Self-generated target | geometry teacher, geometry-first label, natural geometry label | Training/evaluating against labels produced by the same geometry; agreement with those labels is not task accuracy unless they independently match ground truth |
| Ephemeral reinforcement | geometry shaping, per-example law learning | Updating a fresh classifier instance after prediction and then discarding it; no learning carries to later examples unless state is shared or serialized |
| Reconstruction versus rollout | decoder accuracy, shadow density, predictive horizon | Same-state decoding or marginal-distribution matching does not establish autonomous future-trajectory accuracy |
| Exact sufficient-statistic baseline | analytic lookup, local-rule baseline, target already in features | Derive whether a target is an exact known function of current features before crediting a learned model; Rule-30 next density is `f_t` dotted with a fixed rule lookup |
| Exact transformed-input overlap | canonical train/test identity, orbit lookup | After deterministic invariance preprocessing, check whether held-out views become byte-identical to training before calling the result learned generalization |
| Algebraic feature redundancy | Om/LO identity, derived feature block | Norms, dot products, distances, and cosines may determine one another exactly; derive feature independence before crediting an observer/mechanism |
| Global density versus center bit | `c_t`, center column, density threshold | In the archived Rule-30 tracker, summing middle-position pattern frequencies equals whole-row one density, not a fixed spatial cell |
| Emergent distributional embedding | tabula rasa, online SGNS, global mind | Learn token vectors from local co-occurrence without a pretrained semantic seed; saved Livnium instance is real SGNS but geometrically collapsed |
| Per-word metabolic state | manifold, mass, radius, noise, age, crystallization | Persisted centroid plus lifecycle/policy scalars; metaphorical state management rather than established semantic physics |
| Effective semantic dimension | effective rank, PCA collapse, cone collapse | The number of independently used directions in a saved representation; nominal 64D/32D labels do not prevent rank-one collapse |
| Semantic graph basin | thought basin, anchor neighborhood, idea-field | In the mind-map, a greedy disjoint set of cosine-neighbors around a central node, not a demonstrated dynamical attractor |
| Pivot-distance inversion | mind-map tension, `|0.38-alignment|` | Once edges require alignment>0.4, larger “tension” means greater similarity; narrative conflict language is therefore inverted |
| Information-bearing governance | election economy, favor market, representative selection | A governance mechanism can clean information only if evidence quality affects what agents observe, value, or choose; a report-blind election is only power allocation |
| Independent monitor | judge community, unelectable arbiter, anomaly vote | Separation from eligibility is useful, but multiple copies of one global rule are one monitor unless members contribute distinct evidence/failure modes |
| Trust relocation | oracle, stake, structural anchor, same-layer tie | Every robustness mechanism depends on an information or incentive assumption; map the assumption and attack instead of calling the system trust-free |
| Incentive boundary | `1/q` law, slash, stake cap | Conditional expected-payoff equation linking gain, detection, penalty, collateral, horizon, and limited liability; not proof that detection or identity is reliable |
| Non-dumpable exposure | shared fate, locked commons, retained share | Harm coupling deters only while the attacker cannot sell, hedge, exit, or transfer the exposed asset before loss settlement |
| Progressive discipline | ten strikes, purge, public consequence | Persistent accusation history and escalating sanctions; requires correct adjudication, durable identity, appeal, replacement, and false-positive accounting |
| Temporal abstention | staggered silence, rest, quorum cap | Reliability-aware participation across rounds; a fixed oracle-known mask is filtering, not strategic staggered rest |
| Distributed observability | same-layer reality, local truth ties, sensor placement | Recover a known low-dimensional state from trusted measurements; rank/conditioning and sensor corruption—not philosophical layer names—determine identifiability |
| Matched-subspace projection | natural selector, Reynolds clean, equivariant decode | Exact group averaging removes off-subspace noise when truth lies in the invariant/equivariant space; valid in-subspace alternatives remain indistinguishable |
| Product-spectrum degeneracy | pull speeds, veins, P7-cube spectrum | Equal-axis Cartesian-product eigenvalues repeat through coordinate permutations and exact 1-D sum identities; shared levels need not be individual group irreps |
| Group-forced versus arithmetic collision | multiplet, irrep, accidental degeneracy | A commuting group preserves eigenspaces, while full multiplicity may combine several irreps or distinct separable mode families at one eigenvalue |
| Isotropy-breaking control | anisotropic axes, unequal pull | Give identical axis operators generic unequal weights to test which degeneracies require cube-axis equality rather than separability alone |
| Definition-derived baseline | checkmate zero replies, exact rule oracle | Before learning a target, derive whether its label is already an elementary consequence of legal-rule queries |
| Search-demo boundary | target energy, annealing demo, supplied objective | Reaching a known target with a generic optimizer verifies integration, not learning or discovery of the task algorithm |
| Random-walk depth fallacy | shuffle moves, scramble length | A reversible random walk can immediately undo moves; measure exact shortest-path depth or a certified lower bound instead of using walk length as difficulty |
| Transition credit | state-action memory, successor reward | Planning memory must distinguish which action produced which successor; a state-only label attached to a pre-action state cannot reliably rank candidate next states |
| Goal-distance degeneracy | zero-vector attractor, cosine basin | Verify that the goal has a unique, graded relationship to other states; cosine distance to a zero goal is constant and cannot guide attraction |
| Same-representation classifier control | basin versus linear head, decision-rule ablation | Train a direct classifier on exactly the same inputs before attributing failure or success to the representation |
| Answer-injection kill test | cheat mode, gold one-hot routing | Supplying the true answer can test plumbing and decodability, but its score is not task performance or proof that non-cheat routing works |
| Geometric address versus operation | word cube, MD5 cells, hashed BoW | Assigning features to cube coordinates is only hashing unless neighborhood, distance, rotation, or another geometric relation affects the computation |
| Nested feature ablation | A/B/C/D/E, incremental block test | Add feature families cumulatively against a fixed base to measure incremental information and expose saturation |
| Feature redundancy and saturation | fracture firing, derived energy | High firing rate is not selectivity, and algebraic transforms of existing features do not create new information |
| Predictive versus realized code length | ideal bits, arithmetic cost, compressed stream | Summed `-log2 p` is a model score; a compressor additionally needs a finite stream format, overhead accounting, and exact decoder round-trip |
| Surprise versus meaning | dark matter, hard bytes, residual bits | High code cost marks local unpredictability, not semantic importance; meaning needs an independent causal or downstream test |
| Model-payload accounting | float16 amortization, checkpoint cost | Predictive parameters, stripped deployment weights, and optimizer-bearing resumable checkpoints are distinct payloads and must be reported separately |
| Artifact/source binding | CSV chronology, source hash, replay identity | A result belongs to the exact source/data/environment that produced it; later code cannot be retroactively assigned to an older artifact |
| Canonical numeral versus fixed-width codec | base-27 word, core-zero digit | Positional integers discard leading zero width; preserve digits or encode length when cell position is part of the state |
| Epoch-zero control | pre-training row, initial accuracy | A final score is not learned improvement unless it is compared with the same model before its first update |
| Multi-initialization control | random anchors, seed sweep | Initialization-sensitive learners need a distribution of starts, not one selected seed |
| Outcome-vector feedback | win/loss/draw, aggression/defense | A coupling can trade one outcome for another; declare the objective across all outcome axes before tuning |
| State-variable reachability | written-but-unread karma, inactive freshness | For each claimed mechanism, verify the protocol writes the variable, it varies, and the decision path reads it |
| Paired prequential persistence | warm session, matched cold future | Compare warm and cold learners on the same future stream, label continued adaptation, and separate it from frozen generalization |
| Hash-coverage contract | center hash, metadata mutation | A valid chain protects only the fields committed by its hash; enumerate coverage rather than saying the whole mutation is verified |
| Court transition completeness | promotion, quarantine, demotion | Status machines need reachable forward and reverse transitions plus observable behavioral effects |
| Responsible-event attribution | nearest harm, wrapper fan-out | Metadata and receipts must identify the object actually mutated by the lower-level operation |

## 4. Cognition and memory

| Idea | Names used | Meaning |
|---|---|---|
| Association | BellPair, concept bond, entanglement | In most Livnium code this is classical coupled state change or a detached pair record, not physical entanglement |
| Joint-state boundary | multi-qubit register, tensor state, correlated collapse | Multi-node quantum claims require one consistent joint state that owns gates and measurement; separate local nodes plus a pair sidecar are insufficient |
| Propagation | Stabilizer, domino, wavefront | A local change activates linked changes until the system settles |
| Adaptive links | Hebbian learner, cells that fire together | Repeated co-activation creates associations |
| Persistent state | MemoryLattice, Nova memory, GrowthMind | Store, recall, blend, decay, branch, or merge previous states |
| Self-management | memory borders, metabolism, bureaucracy | Control admission, decay, and pruning of memory |
| Auditable memory mutation | receipt, state hash, archive-only maintenance | Make memory changes deterministic and inspectable; auditability does not by itself prove semantic usefulness |
| Receipt chain versus self-hash | state-hash-before/after, top-level JSON hash | An adjacent receipt chain can be valid even when a saved document's self-referential hash does not verify its current serialization; test both contracts separately |
| Online versus frozen policy | prequential adaptation, held-out games | A model that updates during reported games measures stream adaptation; frozen evaluation is a different claim and must use no outcome updates |
| Regression court | gold queries, self-check, P1–P9 | Re-run declared memory invariants and retrieval expectations; requires a real task-derived gold set |
| Source-version memory | Eyes, before/after session snapshot | Preserve code evolution even when Git history is missing; repeated snapshots are provenance, not automatically active source |
| Failure observer | failure memory, confusion memory, difficulty ledger | Record recurring errors for later analysis or curriculum design; Sacred-v2's saved implementation observes only and does not affect learning |
| Detached trajectory critic | watcher, second head, collapse witness | A separate head reads detached intermediate dynamics and raw signal; preserved as an unfinished Eye experiment, not a verified improvement |
| Conversation graph memory | thought graph, externalized mind, mindmap JSON | Paragraphs and code blocks embedded into a searchable cosine graph with viewer export; useful navigation layer, not native physics |
| Read-only narration | basin narrator, observer summary, optional polish | Convert graph regions into tentative text without changing the measured graph; summary faithfulness must be checked separately |

## 5. Implementations and aliases

| Name | Intended role |
|---|---|
| Livnium Core | The conserved cube mathematics |
| Realcore | Clean Dart implementation of core and cognition experiments |
| Nova | A learning/cognition system built above the core |
| LUGK | Livnium Unified Geometric Kernel: proposed immutable laws |
| LUGE | Livnium Unified Geometry Engine: proposed runtime dynamics |
| Livnium-O / T / C / H | Spherical, triangular/tetrahedral, cubic, and hierarchical variants; archived O/T are historical prototypes with failed packing/A4/quantum boundaries |
| ECW-BT | Embedding/collapse training branch using geometric forces |
| Cortex / cube_embed / holonomy | Representation and retrieval branches |
| Nova Eye | Unfinished raw-character/retina/glyph/watcher branch; actual saved trainer uses ordinal character signals and produced no checkpoint |
| Clean semantics packages | Refactored online SGNS, per-word metabolic repository, and incompatible SNLI-head generations under `archives-public/semantics` |
| Livnium mind-map | MiniLM/fallback paragraph graph, greedy anchor neighborhoods, deterministic narration, and browser JSON export |
| Geometry-direct | robust graph-signal IRLS, Laplacian denoiser | Standard smoothness-prior reconstruction on a grid; not a low-dimensional code or intrinsic truth oracle |
| Cube economy | energy/favor election and reciprocal-affinity feedback; preserved as a rich-capture counterexample because reports do not affect the election |
| Livnium Judges | unelectable median-deviation filter with replicated noisy votes; not an independently informed judge community |
| Same-layer reality | known-basis sparse sensing from trusted samples; strongest surviving interpretation is distributed observability |
| Livnium Selector / Vector Decode | exact rank-21 scalar-invariant and rank-42 vector-equivariant projectors used as matched structural priors |
| Symmetry spectrum | exact 70-level Laplacian spectrum of P7³; full cubic symmetry plus separable/arithmetic degeneracy, with task advantage still open |
| Cube semantic field | historical graph-response and 27-position angle transform; PPMI-SVD carries meaning, while sign-losing edge/loop roughness is preserved as a negative representation lesson rather than validated holonomy |
| Livnium chess transport | conserved piece/metadata token state verified against Python chess; useful reversible representation, not yet a complete independent rules engine |
| Sliding basin memory | historical state-only attractor/repulsor experiment whose persistence harms greedy Manhattan; preserve as the transition-credit and representation-collision negative result |
| NLI-Language honesty ladder | six-generation SNLI/ANLI baseline, partition, compression, and neural sequence whose main value is explicit kill tests; standard context prediction survives while Livnium-specific language advantages do not |
| Livnium Demos ladder | six base-27, prototype, puzzle, feedback, karma, and persistence teaching scripts; partial warm-stream memory survives while selected-seed, policy, court, and full-receipt claims do not |
| Nova Memory v2 | maintenance revision with deterministic arbitrary-text identity, supplied-state conservation, bounded live ledger, and archive sidecar; encoder is hash-to-permutation and evaluator/cache/count contracts remain incomplete |
| Dual/negative/trapped cube | pre-Core three-state conflict/history machine with inserted drift, cancellation, trapping, random decay, and capacity rules; a reusable application metaphor, not semantic physics |
| Livnium Crux | archive-only Dart/JS/CLI/docs release for base-27 arithmetic/codecs, cube moves, couplers, Potts recall, and hierarchy; fresh 32-test classical ancestor distinct from current Realcore |
| Transactional monotone update | evaluate a candidate mutation before commit and roll it back when a declared invariant/objective worsens; useful O-A8 engineering pattern, not by itself proof of correctness or convergence |
| Fixed-resource information density | O-A10 research hypothesis requiring defined bits, decoder, budget, held-out task, and direct baseline; structural exposure alone is not information |
| Circular reference frame | O-A9 multiscale boundary metaphor retained for formalization, not established physical equivalence |
| Market reachability preflight | derive feature ranges, prove every regime is reachable, mask aligned finite targets, split chronologically, and compare direct baselines before interpreting a market state |
| Provenance-bound result artifact | bind figure/table/model to source hash, command, data split, model hash, environment/hardware, metric, and timestamp; a cache or hard-coded chart is not result evidence |
| Livnium layer language | hollow/filled, depth, alternation, and output notation preserved by later exact structural parsers; no word-meaning claim |

## 6. Experiment families

- Language: SNLI, ANLI, noun embeddings, sentence paths, generation, chat, and
  the incorporated NLI-Language honesty ladder; its reusable pieces are direct
  baselines, same-feature controls, nested ablations, and adaptive context
  prediction rather than cube/basin/fracture advantage.
- Geometry: cube rotations, Sokoban, compression, symmetry, hierarchy.
- Dynamics: basins, collapse, funnels, shells, cascades, slow structure.
- Quantum-related: small exact statevectors, MPS, quantum-inspired collapse,
  teleportation simulations, geometric locality.
- Discrete systems: Rule 30, Sudoku, SAT/CSP, pathfinding, games.
- Games: chess state transport, mate ranking, online tic-tac-toe, sliding
  memory, and sorting/sliding SearchEngine demos are incorporated in
  `GAMES_LINEAGE_AUDIT.md`; exact-rule, frozen-policy, shortest-path, and direct
  algorithm baselines define their current boundaries.
- Demos: base-27 presentation, toy prototype learning, transparent annealing,
  feedback/karma, and the persistent bridge are incorporated in
  `DEMOS_LINEAGE_AUDIT.md`; epoch-zero, multi-start, paired warm/cold,
  frozen-policy, state-machine, and hash-coverage checks define their boundaries.
- Complete arch-archive root: base Core/Core-C, market-killer, layer language,
  O-A8/O-A9/O-A10, stock model cache, and orphan figures are incorporated in
  `ARCH_ARCHIVE_ROOT_AUDIT.md`; exact-mirror, chronology, full historical tests,
  reachability, direct-baseline, and provenance checks define their boundaries.
- Nova-and-Misc: Sacred evaluators/ablations, angular gradients, Nova v1/v2,
  basin diagnostics, observer features, and the landscape plot are incorporated
  in `NOVA_MISC_AUDIT.md`; target-blind dataflow, artifact, cache-dependency,
  count, and feature-independence checks define their boundaries.
- Realcore legacy and snapshots: February Core revisions, pre-Core/dual-cube/
  Ramsey history, embedded archives, full release snapshots, Crux, and deep Git
  roots are incorporated in `REALCORE_LEGACY_SNAPSHOTS_AUDIT.md`.
- Social/governance: economy, judges, deterrence, majority capture, shared fate,
  purge/rest, same-layer sensing, and scalar/vector structural selectors; the
  full 40-file historical lineage is incorporated in
  `GOVERNANCE_ECONOMY_LINEAGE_AUDIT.md`.
- Symmetry: exact cube rotations/projectors and the P7³ spectrum are preserved;
  spectral multiplicity, irrep decomposition, and task utility must remain
  separate claims.
- Semantic cube fields: corpus vectors, field transforms, loop statistics,
  counter-fitting, and sequential sentence state are incorporated in
  `CUBE_EMBED_LINEAGE_AUDIT.md`; direct semantic features beat the sign-losing
  transform and lexical constraints must be firewalled from evaluation.
- External domains: markets, banking, vision, document reconciliation.

## 7. The recurring restart loop

The same progression appears repeatedly:

1. define a cube or geometric container;
2. define weights and conservation;
3. add rotation or motion;
4. add observer-relative meaning;
5. add coupling and memory;
6. add collapse or basin selection;
7. apply it to NLI, quantum, Ramsey, Rule 30, or another domain;
8. see an exciting number;
9. discover leakage, a weak baseline, or a category mismatch;
10. clean the repository and restart under a new name.

Future work should enter at the unresolved scientific question, not at step 1.
