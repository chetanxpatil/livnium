# Recovered Idea Ledger

Updated: 2026-07-27

This ledger records the mechanism hiding inside each recovered experiment, the
evidence actually present, and the boundary that must survive future retellings.
It complements `CLAIM_LEDGER.md`: the claim ledger tracks scientific confidence;
this file prevents a useful mechanism from being lost merely because its original
claim was overstated.

## Attractor Dynamics family

Source audited:
`/Users/chetanpatil/Desktop/test/_ORGANIZED/02_Experiments/Attractor-Dynamics`

Coverage: five scripts, five verdict documents, four JSON files, one CSV, and five
figures. The saved artifacts were inspected, but the experiments were not
regenerated during this audit because their scripts are not portable in their
current locations.

### Family-level conclusion

The strongest recovered idea is not “everything flows toward Om.” It is:

> A geometric hierarchy helps only when its grouping is aligned with structure
> already present in the data or operator.

For images and smooth fields, locality-aligned pooling retained useful signal. A
radial shell interpretation did not generalize. For text, semantic placement was
required before geometric grouping could retain semantic variance. For iterative
solvers, depth improved precision through standard multigrid rather than a
cube-specific effect. For ordering dynamics, irreversible selection created order
while reversible rotations could only permute it.

### AD-01 — Slow structure

- **Actual mechanism:** a weighted ferromagnetic XY model on the 3×3×3 lattice,
  cooled by Metropolis Monte Carlo. Face-neighbour coupling is chosen as
  `sqrt(f_i f_j)`, where exposure `f` also defines `SW = 9f`. Om is decoupled
  because `f = 0`.
- **Artifact evidence:** coherence rises from about 0.395 at `T=3` to 0.995 at
  `T=0.05`; a slow anneal ends near 0.994 versus 0.972 for a fast quench. All 24
  rotations change the coherence by at most machine precision and preserve the
  ledger total 486.
- **What survives:** a reversible permutation cannot change a
  permutation-invariant order parameter. Order enters through the irreversible
  accept/reject dynamics. This is a clean conceptual bridge between Livnium's
  reversible core and a selective/collapse layer.
- **Boundary:** thermal ordering, annealing, and the XY transition are standard.
  The exposure-weighted coupling was selected by the experiment; the resulting
  exposure-class behavior does not independently validate the symbolic weight as
  a law of nature.
- **Decision:** **keep as measured demonstration and conceptual component**. Do
  not promote it as new physics.

### AD-02 — “Shell funnel” on synthetic fields

- **Actual mechanism:** one-step pooling of a 7×7×7 field into 27 directional
  blocks formed by negative/zero/positive bins on each coordinate. Despite the
  name and prose, the encoder is not a sequence of concentric shell means and does
  not execute a literal 7→5→3→1 cascade.
- **Artifact evidence:** on generated smooth fields, held-out variance explained
  is 0.746 for directional pooling versus 0.589 for size-matched random pooling at
  `sigma=1`; the gap is zero for white noise and reverses at `sigma=4`.
- **What survives:** a fixed local partition is useful when its region scale
  matches the data's correlation scale. The negative endpoints are as important
  as the positive middle.
- **Boundary:** the data generator and partition share the same coordinate axes,
  so the result demonstrates a matched locality prior rather than unique cube
  optimality. PCA remains better. Calling this a shell result obscures the actual
  mechanism.
- **Decision:** **keep as a partial synthetic ablation** and reusable baseline.
  Rename the mechanism “directional block pooling” in future work.

### AD-03 — Cascade on real digit images

- **Actual mechanism:** reconstruct 7×7 crops of `sklearn` digits using
  coarse-to-fine features. The useful variant is global mean → nine local
  rectangular region means → raw cells. The radial variant uses four Chebyshev
  shell means.
- **Artifact evidence:** at code size `K=10`, local grid pooling explains 0.596
  held-out variance, random pooling 0.474, radial shell features 0.353, and PCA
  0.747.
- **What survives:** local coarse-graining is a meaningful inductive bias for
  spatial images; radial pooling mixes distant pixels and is a poor general image
  compressor. This is the clearest real-data result in the family.
- **Boundary:** it is one small bundled dataset, one shuffled split, and five
  random partitions. The radial feature schedule is not stage-matched to the
  nine-region grid: after its global plus four shell summaries, raw pixels enter
  the first `K` features. The reported “sigma” is dispersion across five random
  partitions, not broad statistical significance. PCA still wins.
- **Decision:** **keep as a measured pilot, not a novelty claim**. If revived,
  compare against standard pyramid, wavelet, convolutional, and learned pooling
  baselines over repeated splits.

### AD-04 — Directional funnel for text

- **Actual mechanism:** learn 30-dimensional PPMI-SVD word vectors from SNLI; bin
  the first three axes to place words; reduce the coordinates to 27 sign regions;
  reconstruct the same word vectors from region means. Hash placement is the
  negative control.
- **Artifact evidence:** the semantic placement reports `R² ≈ 0.079` versus
  `-0.022` for random partitions; hash placement is near `-0.021` for both the
  directional and random groupings.
- **What survives:** placement must carry the relevant structure before pooling
  can preserve it. Hash placement is inert. This explains a recurring Livnium
  failure mode: geometry cannot recover semantics that were never encoded into
  position.
- **Boundary:** the target vector is also the source of the placement, so this is
  self-reconstruction of a quantized embedding—not independent evidence that the
  cube understands or filters meaning. The “60 sigma” statement uses the standard
  deviation of only five random partitions and must not be repeated as an
  inferential result. Standard sign quantization, product quantization, k-means,
  and learned bottlenecks were not compared.
- **Decision:** **keep the placement-first principle; classify the semantic claim
  as partial**. Any revival needs an independent held-out target or downstream
  task and non-cube quantization baselines.

### AD-05 — Attractor depth

- **Actual mechanism:** solve `A x = b` for `A = I + 8L` using damped Jacobi and a
  343→27→1 multigrid hierarchy. “Depth” is solver iteration and “precision” is
  distance to the linear-system solution.
- **Artifact evidence:** saved convergence factors are 0.935 for flat Jacobi,
  0.538 for two-grid, 0.576 for three-grid, and 0.465 for nested solving on a
  random-graph operator.
- **What survives:** iterative contraction yields geometric error decay, and
  coarse correction can accelerate diffusion-like solves. The random operator
  accelerating at least as much rules out a cube-specific win in this run.
- **Boundary:** every method draws a different random right-hand side inside
  `run`, so methods were not compared on the same problem. The work counter treats
  fine relaxations and dense coarse solves as if they had comparable cost; the
  flat method's `halt_work=160` is merely the loop limit because it never reaches
  the threshold. The script's printed “random nesting gives no gain” contradicts
  its own result. These issues prevent a clean performance claim.
- **Decision:** **historical/instructive, not promotable evidence**. Rebuild only
  with a shared problem, equal compute or wall time, repeated matrices and
  right-hand sides, and conventional multigrid baselines.

## Deduplicated principles recovered

1. **Placement before dynamics.** Geometry is useful only if data placement
   preserves the relationships the task needs.
2. **Locality before radial symbolism.** Local neighbourhood pooling worked on
   spatial data; general inward collapse toward Om did not.
3. **Reversible core, irreversible selection.** Permutations preserve global
   invariants; selection, dissipation, or learning is needed to create a new
   ordered state.
4. **Depth is an algorithmic resource.** More contraction steps can buy
   precision, but this is standard numerical behavior unless a Livnium-specific
   advantage survives matched controls.
5. **Conditional wins are useful.** White-noise parity, over-smooth reversal,
   hash-placement failure, and random-operator acceleration define where the
   mechanisms stop working.

## Reproducibility repairs required

- Four scripts hard-code
  `/sessions/beautiful-sharp-shannon/mnt/test` for data or outputs.
- `livnium_slow_structure.py` imports `livnium-public` relative to its experiment
  folder, but that dependency is absent there.
- Dependencies and environment versions are not locked with the artifacts.
- Several “sigma” descriptions summarize a handful of random partitions rather
  than replicated datasets, model fits, and splits.

Until repaired and rerun, the JSON/CSV files are historical result artifacts, not
a freshly verified benchmark.

## Incorporation decision

Do not copy these five experiments wholesale into the July repository. Preserve
the family here, carry the five principles into future designs, and port code only
when a specific controlled experiment needs it. The best candidate for a future
revival is a placement/locality ablation with independent targets and matched
non-cube baselines.

## Quantum family

Sources audited:

- `/Users/chetanpatil/Desktop/test/lab/infected/quantum`
- quantum submaps under `/Users/chetanpatil/Desktop/test/lab/index`
- organized successors under `/Users/chetanpatil/Desktop/core/learn`
- MPS narrative under the July repository's `archive/cortex-v2`

### Q-01 — Exact small state-vector simulation

- **Actual mechanism:** conventional complex state vectors, tensor-product gates,
  Born-rule measurement, Bell/GHZ states, and a fixed three-qubit teleportation
  circuit.
- **Fresh evidence:** 200 seeded random complex input states teleported with
  minimum fidelity `0.9999999999999996`; GHZ support was exactly 0.5 on `|000>`
  and 0.5 on `|111>`.
- **Boundary:** this is correct classical simulation with exponential state size,
  not hardware or a quantum speedup.
- **Decision:** **verified engineering; keep**.

### Q-02 — Quantum islands and geometric qubits

- **Actual mechanism:** many independent one-qubit states, exact two-qubit pairs,
  or a graph of local classical correlations.
- **Evidence:** the project's own GHZ comparison shows the geometric graph can
  emit outcomes forbidden to a global GHZ state.
- **Boundary:** linear scaling is obtained by not storing global entanglement.
- **Decision:** **keep as a classical probabilistic/complex-valued design
  pattern; retire physical-qubit wording**.

### Q-03 — Hierarchical omcubes and high site counts

- **Actual mechanism:** recursive address spaces, local state holders, projections,
  or compressed structures.
- **Boundary:** millions of addressable cells are not millions of amplitudes in
  one arbitrary global wavefunction; “entangled” often means recorded pair or
  graph correlation.
- **Decision:** **keep the hierarchy and capacity accounting; retire scalable
  global-quantum interpretations**.

### Q-04 — MPS/DMRG

- **Actual mechanism:** standard tensor-network compression. GHZ reaches long
  chains cheaply because its bond dimension is 2; generic highly entangled states
  do not.
- **Decision:** **verified engineering and useful simulation component**, already
  represented honestly in the July archive.

### Q-05 — Quantum applications

- **Actual mechanism:** toy Grover/SAT simulation, conflict resolvers, policy
  classifiers, cryptography mappers, and quantum-inspired word embeddings.
- **Boundary:** the Grover implementation classically scans basis states to mark
  solutions, so it does not demonstrate end-to-end asymptotic speedup. The
  semantic and policy uses lack matched classical ablations.
- **Decision:** **historical/open components**, not promoted claims.

### Q-06 — Cortex dynamic-alpha compression

- **Source:** `/Users/chetanpatil/Desktop/uantum`, with later corrected code and
  artifacts under `/Users/chetanpatil/Desktop/livnium`.
- **Actual mechanism:** use a geometry- or semantics-derived alpha scalar to
  alter an MPS entropy/bond ceiling under a fixed resource budget.
- **Fresh implementation evidence:** the `uantum` copy passes its forward/GHZ
  paths but fails reverse CNOT directionality. The later Desktop copy fixes the
  adjacent and non-adjacent reverse cases.
- **Saved benchmark evidence:** on 30 random-circuit seeds, dynamic alpha lowers
  internal truncation by about 13.1% but leaves L1 output fidelity effectively
  unchanged. On structured noise it improves some observables and worsens
  others.
- **What survives:** conditioning a compression policy on task information is a
  valid research direction. The task objective must be declared before tuning.
- **Boundary:** this is standard classical tensor-network simulation plus a
  policy heuristic. The current governor shifts a tradeoff surface; it does not
  dominate the baseline or demonstrate quantum advantage.
- **Decision:** **partial/mixed; preserve and redesign only as a matched policy
  experiment**.

## Semantic memory triage family

Detailed source and command evidence:
`UANTUM_AUDIT.md`.

### SMT-01 — GloVe/PCA geometric sensor

- **Actual mechanism:** map GloVe-50 word vectors through PCA-3 and an
  axis/angle transform to an alpha importance score.
- **Fresh diagnostic evidence:** on the script's 64-word seed fit, cosine alpha
  gives mean fact/filler scores 0.8318/0.2642. The seed-frequency IDF mode gives
  1.0000/0.9998 and is therefore nearly non-discriminative.
- **What survives:** inspect the score distribution before attaching a semantic
  interpretation to a governor.
- **Boundary:** a small hand-selected vocabulary and a single hand-authored
  document are not a held-out retrieval benchmark.
- **Decision:** **keep the X-ray diagnostic; replace the frequency and benchmark
  design before reuse**.

### SMT-02 — Capacity-constrained fact retention

- **Actual mechanism:** retain the highest-scored tokens in a bounded memory.
- **Toy evidence:** the hand-coded mock gets 10/10 facts versus 4/10 FIFO/LRU;
  the live cosine mapping gets 6/10 versus 4/10 on one 50-token document.
- **Stronger evidence:** across 150 saved documents, Alpha-Only and LIVNIUM-B are
  identical at P@10=0.0120 and both trail TF-IDF (0.0147) and YAKE (0.0160).
  LIVNIUM-B loses significantly to YAKE at P@5 (`p=0.0286`).
- **Boundary:** the mock table contains the answer key, and the toy's LRU is
  exactly FIFO because the stream never records reuse. The MPS stage adds no
  ranking value in the stronger run.
- **Decision:** **retain the bounded-memory question and negative result; retire
  the toy win as evidence of general retrieval superiority**.

## Ramsey search family

### RAM-01 — Incremental stochastic R(5,5) hunter

- **Source:**
  `/Users/chetanpatil/Desktop/uantum/ramsey/livnium_ramsey_v2_stochastic.py`.
- **Actual mechanism:** incremental K5 violation deltas, simulated annealing,
  heavy-tailed multi-edge kicks, and recent-state similarity repulsion.
- **Fresh evidence:** on random `N=8, K=5` graphs, 100/100 flip deltas matched a
  full recount.
- **Boundary:** no `best_graph.json` exists. The configured 43-vertex run would
  require approximately 9.6 million stored subset tuples and 533 billion inner
  adjacency checks. Its stated `[43,48]` bound is outdated; the published bound
  is now `[43,46]`.
- **Decision:** **keep the delta counter and novelty-pressure idea; classify the
  full script as an uncompleted historical search, not a lower-bound result**.

## ECW-BT family

See `LINEAGE_MAP.md` for paths and copy relationships.

### ECW-01 — Random-seeded Level-0

- **Actual mechanism:** unit-sphere word vectors updated from Wikipedia context
  windows with mass weighting, a cosine pivot, negative samples, and
  renormalization.
- **Decision:** **historical origin**.

### ECW-02 — SBERT-seeded pairwise distillation

- **Actual mechanism:** start from SBERT word vectors, apply Wikipedia
  co-occurrence updates, and repeatedly fuse back toward the teacher.
- **Artifact evidence:** the `(50000, 256)` output has mean cosine `0.99999908`
  with its seed and mean displacement `0.00027669`. The saved validation has sane
  inherited neighbours and one correct result among two analogy probes.
- **Boundary:** the main trainer's pull/repel signs contradict its documented
  force directions, and default per-shard fusion replaces about 95% of the student
  with the seed. The acceptance test does not compare student performance against
  the seed and accepts any nonempty neighbour lists.
- **Decision:** **pipeline engineering preserved; semantic improvement
  unproven**. Fix, rerun, and compare against the identical seed before revival.

## Desktop Livnium family

Detailed evidence and paths:
`DESKTOP_LIVNIUM_AUDIT.md`.

### DL-01 — Energy-Guided Attractor Network for SNLI

- **Actual mechanism:** pretrained bag-of-words embeddings pass through a
  learned collapse MLP for six repeated steps, then into a supervised NLI head.
- **Fresh artifact evidence:** the original `triple_crown_slow` checkpoint and
  tracked evaluator reproduce 7,513/9,842 = 0.7634 dev accuracy. A post-hoc
  co-adapted no-collapse diagnostic scores 0.6559. All audited later saved
  checkpoints score lower, from 0.5053 to 0.7115.
- **What survives:** a real, reproducible supervised NLI pilot in which the
  learned collapse path is load-bearing.
- **Boundary:** the diagnostic is not an independently trained control; no
  matched parameter/budget baseline, multiple seeds, or external NLI split was
  evaluated. Historical class, parameter-count, and speed claims need
  correction.
- **Decision:** **preserve the original checkpoint as the best measured March
  artifact; do not replace it with the later alpha/memory experiments**.

### DL-02 — Deterministic Nova memory

- **Actual mechanism:** a GrowthMind tree with deterministic receipts and state
  hashes, archive-only maintenance, bundle indexing, deterministic retrieval,
  and a regression-court structure.
- **Fresh engineering evidence:** nine collected pytest tests and all fifteen
  direct test scripts pass.
- **What survives:** auditable mutation and replayable invariants are useful
  memory infrastructure independent of the alpha theory.
- **Boundary:** the gold set is three placeholder queries, embeddings are toy,
  and the system has no competitive retrieval or performance benchmark.
- **Decision:** **keep the audit infrastructure as verified small-system
  engineering; replace the evaluation before claiming semantic value**.

### DL-03 — Alpha-ordered archival

- **Actual mechanism:** sort candidates by `(alpha, age, id)` and archive the
  lowest-alpha nodes until under capacity.
- **Artifact evidence:** the arXiv benchmark reports 199/374 high-alpha survivors,
  0/342 low-alpha survivors, and a `+0.5321` survival gap.
- **Boundary:** alpha is both the independent variable and the explicit sort key.
  P6 validates implementation of the chosen policy, not that alpha causes
  usefulness. The benchmark's TRL thresholds are self-defined.
- **Decision:** **retain as a deterministic policy mechanic; retire its use as
  semantic-causality or TRL4 evidence**.

### DL-04 — Alpha/MPS generalization test

- **Actual mechanism:** use GloVe/PCA-derived alpha to change retention or
  tensor-compression policy.
- **Artifact evidence:** a synthetic 60-word setup shows a large gap, but ten
  arXiv documents produce only +0.0068 mean Mode A gap and -0.0006 Mode B
  aggregate gap. The 150-document retrieval result also trails TF-IDF/YAKE.
- **What survives:** the saved failure is a useful kill-test showing that direct
  alpha separation does not automatically survive the MPS governor or improve
  retrieval.
- **Decision:** **preserve the negative generalization evidence; redesign around
  an independent downstream observable if revived**.

## Sacred Livnium family

Detailed evidence, hashes, paths, and replay boundaries:
`SACRED_VAULT_AUDIT.md`.

### SL-01 — Static pretrained collapse plus supervised NLI head

- **Actual mechanism:** initialize a quantum-named bag-of-words encoder from a
  pretrained collapse embedding table, apply one static collapse step, and train
  an SNLI classification head.
- **Fresh artifact evidence:** the recovered `collapse1-static` checkpoint scores
  76.01% dev and 76.12% test in deterministic noise-free replay. The historical
  workbook records 76.01% test and 2,357 errors in an older stochastic run.
- **What survives:** this is the strongest self-contained sacred replay. It uses
  label-blind inference, the exact embedding backbone is preserved by hash, and
  the error workbook exposes 1,554 cases shared across three models.
- **Boundary:** the model has no saved matched end-to-end MLP/no-collapse
  retraining, multiple seeds, or current standard NLI architecture comparison.
- **Decision:** **preserve as a measured artifact and the sacred replay bundle's
  current best model**.

### SL-02 — Dynamic label-specific BasinField

- **Actual mechanism:** maintain multiple sub-basins per class and route each
  sample through a label-specific field during dynamic collapse.
- **Code evidence:** the saved training and test evaluators can pass gold labels
  into dynamic routing, contrary to the README. For the surviving
  collapse4-dynamic checkpoint, correct-label, shuffled-label, and static routing
  all score about 75.3%.
- **What survives:** class-conditioned training geometry remains a testable idea,
  provided the inference interface cannot receive labels.
- **Boundary:** there is no artifact behind the remembered 95.76%/96.07% result,
  and the surviving dynamic model is worse than the static collapse1 model.
- **Decision:** **keep the mechanism as open research; retire the high-accuracy
  claim and the old evaluator design**.

### SL-03 — Corrected discrete-time attractor equation

- **Actual mechanism:** combine a learned residual field with anchor-directed
  Euclidean radial updates whose magnitudes use cosine divergence, plus a
  heuristic neutral-boundary term.
- **Saved evidence:** the project's own corrected equation withdraws the
  exact-gradient and path-dependent descriptions. A Jacobian experiment reports
  mean spectral norm 41.11, maximum 198.07, and no sampled norm below 1.
- **What survives:** an iterative geometry-shaped readout and an honest open
  question about sufficient local descent conditions.
- **Boundary:** it is not exact gradient flow, the saved map is not generally
  contractive, and the proposed cosine-gradient formula is not established by
  the saved experiment.
- **Decision:** **preserve the corrected formulation, not the original physics
  language**.

### SL-04 — Repaired label-blind retraining branch

- **Actual mechanism:** explicitly optimize collapse parameters and
  label-supervised anchors during training, then evaluate through a static,
  label-blind path.
- **Fresh artifact evidence:** `collapse_retrain/model_nli_v1/nli_epoch23.pt`
  scores 69.76% dev and 68.87% test.
- **What survives:** a cleaner separation between supervised training and
  inference, plus a reusable held-out evaluator.
- **Boundary:** only epoch 23 remains; seed, training log, and earlier model
  selection history are absent. Its accuracy is below the older static sacred
  checkpoint.
- **Decision:** **preserve as a distinct measured repair branch, not as a
  replacement for the best artifact**.

### SL-05 — Eyes source-version museum

- **Actual mechanism:** record before/after file content around AI-assisted
  sessions.
- **Artifact evidence:** 5,278 source/document rows in the sacred copy collapse
  to only 72 distinct hashes, including several historical versions not present
  in the artifact vault.
- **What survives:** recovery of code evolution when `.git` history is missing.
- **Boundary:** it is not a clean active code tree, and repeated snapshots should
  not be counted as independent implementations.
- **Decision:** **preserve as provenance; use hashes to reconstruct lineage
  before porting source**.

### SL-06 — Repeated learned residual with torque overlay

- **Actual mechanism:** fine-tune the full 50,000×256 encoder, repeatedly apply a
  learned residual update, add named fixed axial/anchor and torque forces, then
  classify from the final state plus a phase-shift feature.
- **Fresh artifact evidence:** torque-v1 scores 76.00% test,
  torque-replace-256 scores 75.82%, and torque-256-memory scores 76.42% under
  deterministic label-blind replay.
- **Kill-test:** zeroing all fixed forces leaves predictions essentially
  unchanged. Bypassing the collapse layers or zeroing the learned update drops
  test accuracy to roughly 59–66%.
- **What survives:** iterative learned residual refinement is the load-bearing
  architecture idea. The pretrained embeddings also move substantially during
  supervised end-to-end training.
- **Boundary:** these are post-hoc co-adapted diagnostics, not independently
  trained matched ablations. “Torque” and “physics” are not established causes.
- **Decision:** **preserve the learned iterative residual; retire causal credit
  to the fixed-force overlay pending matched retraining**.

### SL-07 — Basin replacement and failure-memory observer

- **Actual mechanism:** expand/replace basin representatives and separately
  record repeated class-confusion failures with nominal difficulty weights.
- **Artifact evidence:** replacement scores 75.82% test; memory scores 76.42%.
  The memory state contains 589,046 recorded and 1,610,960 resolved observations.
- **Kill-test:** the training loop never applies the defined difficulty boost.
  Failure memory cannot affect the loss, gradients, routing, or inference.
- **What survives:** basin replacement is useful negative history; the failure
  ledger is a potentially reusable observability/error-curriculum input.
- **Boundary:** neither mechanism has a matched-seed causal result, and the saved
  improvement cannot be attributed to memory.
- **Decision:** **preserve replacement as historical evidence and memory as a
  silent observer, not a demonstrated learning mechanism**.

### SL-08 — Nova Eye raw-character and watcher branch

- **Actual mechanism:** explore raw-character signals, optional
  center-surround/retina and glyph features, a label-free emergent basin field,
  and a second watcher head over detached trajectories.
- **Artifact evidence:** source, server, Flutter UI, and an incomplete 100k
  encoding log survive; `model/eye-v1` contains no checkpoint.
- **Failure evidence:** the actual trainer uses clipped ordinal characters, not
  the retina/glyph pipeline; its post-epoch stats keys do not match the basin
  API; signal evaluation removes the trained basin field; the renderer has
  low-six-bit character collisions.
- **Conservation kill-test:** one-basin strengthening changes energy 1.0→1.1,
  and decay later reduces it to 0.84 without redistribution.
- **What survives:** raw-character experimentation, local visual preprocessing,
  label-free basin birth/replacement, a trajectory watcher, and a diagnostic UI
  are distinct ideas worth retaining.
- **Boundary:** there is no completed model, benchmark, or strict conservation.
- **Decision:** **preserve as an unfinished design archive; repair and compare
  components separately before calling it Nova vision**.

## Lab Nova-SNLI predecessor family

Detailed evidence:
`LAB_NOVA_SNLI_AUDIT.md`.

### NS-01 — Physics-wrapped skip-gram embeddings

- **Actual mechanism:** train a 50,000×256 word table on WikiText context pairs
  with a cosine equilibrium/margin objective while passing vectors through a
  fixed randomly initialized collapse update and procedurally updated dynamic
  basins.
- **Artifact evidence:** the full final embedding tensor is exactly equal to the
  epoch-1 tensor. It gives 54.97% context-over-random discrimination on a focused
  held-out diagnostic versus 51.03% for a row-shuffled control.
- **Failure evidence:** all six runnable repository analogy prompts miss their
  conventional answer; no successful training log, seed, or matched SGNS
  baseline survives.
- **Critical boundary:** the optimizer contains only the embedding table, not
  collapse-engine parameters. “Physics” is a fixed training transform and basin
  lifecycle, not a jointly learned law.
- **Decision:** **preserve as a one-epoch representation artifact and training
  idea; do not promote semantic or causal physics claims**.

### NS-02 — Ground-truth-routed SNLI basins

- **Actual mechanism:** use the target class to select an entailment,
  contradiction, or neutral micro-basin immediately before the classifier
  during training.
- **Fresh evidence:** the unique checkpoint scores 32.99% test through proper
  static inference, 90.58% through gold-label routing, and 48.85% through
  globally shuffled routing.
- **What survives:** a powerful regression/kill-test for any future
  class-conditioned geometry. The later static evaluator correctly recognizes
  that held-out inference must not receive labels.
- **Failure mode:** the classifier decodes label identity injected by the router
  and fails near chance when the shortcut is removed.
- **Decision:** **preserve as canonical leakage evidence; never reuse this
  training interface for classification**.

## Infected Archive Nova source family

Detailed evidence:
`INFECTED_ARCHIVE_NOVA_AUDIT.md`.

### IA-01 — Orphan calibrated-error artifact

- **Actual artifact:** 2,243 unique, internally consistent SNLI test
  misclassifications with probabilities and class indices.
- **Conditional evidence:** if it is the complete error-only output over all
  9,824 valid test examples, it implies 7,581 correct and 77.17% accuracy.
- **Failure evidence:** it does not match the surviving collapse1-static error
  set, predates that checkpoint, and has no producing model, command, seed,
  calibration protocol, or complete predictions.
- **What survives:** an error-level historical artifact that could help identify
  a missing checkpoint or compare recurring hard cases.
- **Decision:** **preserve as partial evidence; never rank it above replayable
  checkpoints unless its missing provenance is recovered**.

### IA-02 — First-letter cube geometry encoder

- **Actual mechanism:** convert a token to a base-27 signature, reduce it to
  three cube coordinates and six derived geometric features, then optionally
  contextualize it with a Transformer.
- **Failure evidence:** the coordinate conversion uses only signature modulo 27,
  retaining the first character and discarding the rest. `cat`, `car`, and `c`
  have identical base features, as do `abc` and `acb`.
- **What survives:** deterministic, interpretable token features can remain a
  side channel when collision behavior is explicitly designed and measured.
- **Decision:** **retire this implementation as lexical geometry; preserve the
  broader deterministic-feature idea only**.

### IA-03 — Mean-pooled phoneme-character encoder

- **Actual mechanism:** map characters to simplified place/manner phoneme
  features, apply a learned projection, and average over the token.
- **Failure evidence:** order is discarded, so anagrams such as `abc/acb` and
  `cat/tac` produce the same output up to summation noise. Many characters also
  share the simplified feature defaults.
- **What survives:** phonetic or articulatory features may be useful as an
  auxiliary channel if sequence order and matched baselines are restored.
- **Decision:** **preserve as an abandoned component idea; retire semantic
  representation claims for the saved bag-of-characters form**.

### IA-04 — Source-only quantum-embed bridge

- **Actual mechanism:** the same physics-wrapped word-table training source
  already recorded as `NS-01`.
- **Copy evidence:** current and backup core sources are duplicates, meaningful
  files match Sacred-v2 `code/quantum-pretrain`, and the corpus files match the
  Lab Nova-SNLI physics-embedding corpus.
- **Failure evidence:** this root has no weights or results; its evaluator
  rebuilds vocabulary IDs from test text, and its analogy script points to a
  nonexistent model path.
- **Decision:** **preserve for lineage only; use `NS-01`, not this empty source
  copy, as the experiment record**.

## NLI-ALL pure-vector/geometric family

Detailed evidence:
`NLI_ALL_AUDIT.md`.

### NA-01 — Process-hashed letter-bag chain

- **Actual mechanism:** assign each character a pseudorandom 27-vector using
  Python's built-in hash, sum and normalize the character vectors for each
  word, then add sentence-position information.
- **Fresh evidence:** the simple classifier reaches 40.71% on all 9,824 valid
  SNLI test examples with its saved lexical memory.
- **Failure evidence:** within-word order is lost: `cat/act`, `not/ton`, and
  `dog/god` collide. Later results change when `PYTHONHASHSEED` changes, and no
  seed survives.
- **What survives:** a cheap deterministic-at-fixed-seed lexical baseline and a
  reproducibility/collision kill-test for future encoders.
- **Decision:** **preserve as historical baseline; replace process hash and
  restore order before reuse**.

### NA-02 — Supervised word-polarity memory

- **Actual mechanism:** maintain a dictionary from full word tokens to
  entailment/contradiction/neutral moving averages.
- **Artifact evidence:** simple/v3 hold the same 7,900-word state; legacy/v4
  share a 5,324-word state; v5/v8 share another 7,900-word state.
- **Fresh causal diagnostic:** on the first 1,000 test examples, the simple
  model falls from 40.9% to 34.4% when the saved lexicon is cleared. Clearing
  v3–v7 state has essentially no effect.
- **What survives:** lexical supervision is the only archived NLI-ALL learned
  state with a measurable current benefit.
- **Boundary:** it is a class-labeled lexical model, not emergent unsupervised
  geometry, and the simple decision never predicts neutral.
- **Decision:** **keep as a transparent baseline/component**.

### NA-03 — Basin, collapse, clarity, and prototype heads

- **Actual mechanism:** transform pair vectors through attraction/collapse
  features and classify from native class prototypes plus clarity statistics.
- **Artifact evidence:** v3 preserves `peak_clarity.pkl` and a small native
  `decision_head.pkl`.
- **Failure evidence:** the default classifier does not load the saved head; it
  is not the README-described MLP; static accuracy is 36.06% and never predicts
  contradiction.
- **Decision:** **preserve the feature/prototype design and orphan state as
  history; do not claim an NLI gain**.

### NA-04 — Layered planet trace and supervised rule reader

- **Actual mechanism:** expose intermediate encoding, attraction, stability,
  collapse, opposition, and decision features, then optionally fit a shallow
  supervised decision tree.
- **Artifact evidence:** the tree gives 48.37% on the same 13,988 rows it was
  fit on and 46.0% on a separate 100-row named test file.
- **What survives:** explicit intermediate state traces can support diagnostics,
  and small interpretable rule readers are testable components.
- **Boundary:** labels are used to fit the tree, the default classifier does not
  load it, the separate file has uncertain provenance, and 85.23% is example
  prose.
- **Decision:** **keep diagnostic layering and the partial tree artifact;
  retire unsupervised-law language**.

### NA-05 — Semantic warp sequence alignment

- **Actual mechanism:** dynamic-programming alignment of premise and hypothesis
  token paths using cosine mismatch, equivalent in spirit to dynamic time
  warping.
- **What survives:** a separable sequence-alignment feature that repairs part of
  the letter-bag pipeline's coarse word-position matching.
- **Boundary:** standard alignment machinery, no matched ablation, and no
  surviving NLI improvement.
- **Decision:** **preserve as the strongest reusable engineering component in
  v5/v8; evaluate under its ordinary algorithmic name**.

### NA-06 — Maximum-mismatch fracture

- **Actual mechanism:** identify the strongest aligned cosine mismatch above a
  threshold and interpret it as an opposition/fracture signal.
- **Fresh failure evidence:** 866 of the first 1,000 valid SNLI test examples
  fracture; only 15 contain an explicit diagnostic negation token; contradiction
  precision given fracture is 32.45%, essentially base rate.
- **What survives:** maximum local alignment mismatch can be a generic error or
  novelty feature.
- **Decision:** **retire negation/contradiction semantics; preserve only as an
  uncalibrated diagnostic**.

### NA-07 — Signed opposition axis

- **Actual mechanism:** define opposition as resonance multiplied by the sign
  of divergence, distinguishing inward and outward state movement.
- **Artifact evidence:** v6 and v7 both reach 35.84% full static test and never
  predict neutral.
- **What survives:** an explicit signed scalar is easier to inspect than
  metaphoric force descriptions.
- **Boundary:** no accuracy gain or physical-law evidence.
- **Decision:** **preserve as a mathematical feature experiment, not a result**.

### NA-08 — Per-example geometry shaping

- **Actual mechanism:** adjust a v7 classifier's local geometry after seeing the
  example's gold label.
- **Failure evidence:** each example receives a fresh classifier; reinforcement
  occurs after prediction; the instance is discarded; no geometry parameters
  are serialized. v7's complete predictions are exactly v6's.
- **What survives:** a definitive implementation rule—learning must update
  shared/persistent state and must be evaluated after the update can affect
  later examples.
- **Decision:** **keep as failed-method memory; do not revive without a
  persistent, label-blind evaluation protocol**.

### NA-09 — Geometry teacher and artificial force labels

- **Actual mechanism:** derive a “natural” class from the current geometry, or
  directly overwrite force channels from the gold class, then train/measure
  agreement with that constructed target.
- **Artifact evidence:** prediction of geometry-generated labels is 82.32%, but
  those labels agree with SNLI only 34.17%. Debug forces use fixed class-coded
  triples and gold-label decision paths reach 100%.
- **What survives:** self-labeling can be useful for clustering or
  representation discovery only when evaluated against an independent
  downstream target.
- **Decision:** **retire all associated numbers as NLI accuracy; preserve the
  files as target-definition and leakage evidence**.

## Archived application siblings

Detailed evidence:
`ARCHIVED_EXPERIMENTS_RULE30_AUDIT.md`.

### AE-01 — Violation-directed Ramsey search

- **Actual mechanism:** count monochromatic cliques exactly, score a coloring by
  violations, and flip implicated edges with restart/escape/checkpoint variants.
- **Verified boundary:** exhaustive K6/K5 checks confirm the small counter; the
  only saved K17/K4 state has 21 independently recounted violations.
- **What survives:** local constraint deltas, violation-directed proposals, and
  checkpoint/resume are reusable search components.
- **Decision:** **preserve as standard heuristic engineering and negative search
  history; no witness or new Ramsey bound**.

### AE-02 — Unique center-cell anchor

- **Actual mechanism:** require a single lattice cell at the geometric center as
  an observer anchor, which naturally selects odd cube sizes.
- **Focused correction:** all 24 proper rotations on N=2,4,6 even cubes preserve
  bijection, exposure class, and `SW=9f`.
- **What survives:** odd/even parity is a valid anchor distinction and an
  architectural definition.
- **Decision:** **preserve the center-anchor axiom; retire claims that even-cube
  rotations or exposure invariants fail**.

### AE-03 — Coordinate/pair search on a toy cipher

- **Actual mechanism:** sweep individual and paired key bytes by one-pair Hamming
  distance, then brute-force a local neighborhood.
- **Fresh evidence:** the disclosed 32-bit key is recovered after 555,863 local
  candidates, but the custom cipher fails its own decrypt round trip.
- **What survives:** a small discrete-landscape search exercise.
- **Decision:** **preserve as toy optimization; retire AES, geometric
  cryptanalysis, and quantum-search interpretation**.

### AE-04 — Small quantum-protocol regression

- **Actual mechanism:** run teleportation and Bell sampling in a three-qubit
  dense classical state-vector simulator.
- **Fresh evidence:** six hand states have fidelity 1.0; the sampled CHSH result
  fluctuates around the quantum expectation.
- **What survives:** compact simulator regression/demonstration scripts.
- **Decision:** **preserve as duplicate verified classical simulation; no
  physical or scalable quantum claim**.

## Rule-30 lineage

Detailed evidence:
`ARCHIVED_EXPERIMENTS_RULE30_AUDIT.md`.

### R30-01 — Cyclic de Bruijn flow coordinates

- **Actual mechanism:** represent a cyclic binary row by N-gram frequencies and
  remove linear redundancy using substring flow and normalization constraints.
- **Exact correction:** all four Phase-1 formulas lie in the generic
  flow-plus-normalization span; they are not Rule-30-specific.
- **What survives:** a clean feasible-polytope/null-space representation.
- **Decision:** **preserve under standard cyclic-flow terminology**.

### R30-02 — Explicit coarse-state non-closure witness

- **Actual mechanism:** find two rows with the same current coarse statistic but
  different next coarse statistics.
- **Proven evidence:** cyclic rows `001011` and `001101` have identical current
  3-gram counts and different next Rule-30 3-gram counts.
- **What survives:** a rigorous method and concrete proof that 3-gram frequency
  state is not dynamically closed.
- **Decision:** **promote as the strongest mathematical Rule-30 result in the
  archive**.

### R30-03 — Pattern-frequency manifold diagnostics

- **Actual mechanism:** project current/next N-gram summaries into null-space/PCA
  coordinates, fit one-step dynamics, and decode aggregate labels.
- **Boundary:** it is descriptive state compression; same-state reconstruction
  and explained variance are not autonomous prediction.
- **Decision:** **preserve for analysis and visualization, with target and
  information boundary declared explicitly**.

### R30-04 — Exact sufficient-statistic kill test

- **Actual mechanism:** before training, derive whether the target is already a
  known function of the input statistics.
- **Fresh evidence:** causal next density obeys `c_{t+1}=f_t·r` for a fixed
  64-entry Rule-30 lookup vector; all 5,000 saved rows match to
  `3.33e-16`.
- **What survives:** a mandatory baseline rule for every future Livnium model.
- **Decision:** **promote the diagnostic; narrow 99.6% model performance to
  learning an analytically available aggregate**.

### R30-05 — Reconstruction-versus-rollout boundary

- **Actual mechanism:** separate same-state decoding, one-step fit, marginal
  density, field reconstruction, and autonomous trajectory rollout.
- **Saved evidence:** later shadows fall near 50% with horizons two or three;
  perfect-grid rates are zero or nearly zero; one generator emits all zeros.
- **What survives:** first-divergence and multi-step rollout as mandatory
  evaluation outputs.
- **Decision:** **preserve the failed rollouts as high-value negative evidence;
  retire density matching as proof**.

## Semantics packages and conversation mind-map

Detailed evidence:
`SEMANTICS_MINDMAP_AUDIT.md`.

### SM-01 — Online tabula-rasa distributional learner

- **Actual mechanism:** learn a dynamically resized 64D vocabulary through
  skip-gram with negative sampling, frequency weighting, and persisted optimizer
  state.
- **Fresh boundary:** the mechanism is real, but both saved banks place about
  99.64% of centered variance in one principal component.
- **What survives:** compact online SGNS and explicit emergent-versus-pretrained
  source selection.
- **Decision:** **preserve as an engineering component; require matched SGNS,
  PPMI, and fastText controls plus effective-rank monitoring before semantic
  claims**.

### SM-02 — Sharded per-word metabolic state

- **Actual mechanism:** persist a word centroid with mass, radius, noise,
  velocity, age, and context frequency; update it through centroid attraction,
  an autoencoder, and scalar schedules.
- **Fresh boundary:** all 16,381 active centroids are effectively rank one and
  zero meet the declared mass/radius graduation rule.
- **What survives:** scalable sharded state, lifecycle instrumentation, and an
  explicit success predicate.
- **Decision:** **preserve repository/lifecycle design; call the scalars policy
  state rather than semantic physics and tie graduation to an external task**.

### SM-03 — Effective-rank collapse guard

- **Actual mechanism:** measure PCA variance, effective rank, pair cosine, and
  alignment to the mean direction before accepting a nominally high-dimensional
  representation.
- **Fresh evidence:** nominal 64D embeddings have effective rank about 1.04;
  nominal 64D active centroids have effective rank about 1.006.
- **What survives:** a general preflight/stop condition for every embedding and
  manifold experiment.
- **Decision:** **promote as mandatory diagnostics alongside retrieval/task
  metrics**.

### SM-04 — Artifact compatibility ledger

- **Actual mechanism:** record model state keys, tensor shapes, feature
  dimensions, code paths, and hashes before replaying a saved result.
- **Fresh evidence:** the saved semantic classifier is a 6→3 projection, the
  cache is 256D, and current source expects either 18D or a different 256D MLP.
- **What survives:** a cheap rule that prevents accidental cross-generation
  result attribution.
- **Decision:** **promote for all recovered checkpoints and caches**.

### SM-05 — Hypothesis-only semantic control

- **Actual mechanism:** split sentence-pair features into premise, hypothesis,
  absolute-difference, and product blocks and evaluate each alone.
- **Fresh evidence:** full cache accuracy is 49.5%, while hypothesis-only is
  48.8%; almost all linear signal is not relational.
- **What survives:** mandatory task-artifact diagnosis for every NLI generation.
- **Decision:** **promote; no SNLI claim survives without it and official split
  identity**.

### MM-01 — Conversation-to-graph external memory

- **Actual mechanism:** split files into paragraph/function blocks, embed them,
  form a cosine graph, compute connectivity mass, and export browser JSON.
- **Saved evidence:** the recovered conversation export yields 499 nodes and
  9,579 unique valid edges with a matched 1.98 MB graph artifact.
- **What survives:** a practical project archaeology and idea-navigation tool.
- **Decision:** **preserve and consider productizing after retrieval/navigation
  evaluation against standard search and graph baselines**.

### MM-02 — Greedy anchor-neighborhood basins

- **Actual mechanism:** rank nodes by alignment-weighted connectivity, take an
  anchor plus neighbors above a threshold, and prevent later overlap.
- **Boundary:** this is a greedy disjoint graph grouping heuristic despite
  documentation saying it is not clustering; only 8.82% of saved nodes are
  assigned.
- **Decision:** **preserve as a simple grouping option; compare with connected
  components, Leiden/Louvain, HDBSCAN, and nearest-neighbor retrieval**.

### MM-03 — Pivot-distance interpretation test

- **Actual mechanism:** derive how a named metric behaves after every threshold
  and selection rule, then compare that monotonic behavior to its narrative
  meaning.
- **Fresh evidence:** retained edges have alignment>0.4, so
  `|0.38-alignment|` increases exactly with similarity even though narration
  calls larger values conflict.
- **What survives:** an interpretation audit for every metaphorical metric.
- **Decision:** **promote the audit; rename or redesign mind-map tension and
  stability before use**.

### MM-04 — Read-only tentative narration

- **Actual mechanism:** narrate a graph region from its central text and nearby
  nodes without mutating graph geometry; optional LLM polish stays downstream.
- **What survives:** good observer/action separation and provenance-friendly
  summaries.
- **Decision:** **preserve, but fix the metric language and evaluate summary
  faithfulness on sampled basins**.

## Archived spherical/simplex cores and solver harness

Detailed evidence:
`ARCHIVED_CORE_VARIANTS_BENCHMARK_AUDIT.md`.

### OT-01 — Pairwise-valid spherical alphabet

- **Original mechanism:** admit tangent neighbor spheres when the sum of their
  core cap weights is at most two.
- **Fresh boundary:** for unit spheres the scalar rule admits 14, while actual
  three-dimensional kissing maximum is 12; generated placements overlap
  because neighbor-to-neighbor separation is never checked.
- **What survives:** a radius-aware continuous alphabet and cap budget as a
  cheap necessary prefilter.
- **Decision:** **preserve the exploration; require all pairwise angular and
  distance constraints before calling a configuration valid**.

### OT-02 — Energy/force single source of truth

- **Original mechanism:** soft repulsion plus a density-target potential,
  integrated through a Hamiltonian-style engine.
- **Fresh boundary:** away from overlap, force matches the numerical gradient;
  inside overlap, clamped kernel values and unclamped derivatives disagree by
  33 in the focused probe.
- **What survives:** explicit potential logging and finite-difference testing.
- **Decision:** **derive force exactly from one potential—preferably through
  automatic differentiation—and test every piecewise boundary**.

### OT-03 — Faithful finite-group action

- **Original mechanism:** build 12 tetrahedral rotations from named vertex
  permutations and SVD-derived SO(3) matrices.
- **Fresh boundary:** two permutations are not the stated 3-cycles, two
  matrices fail tetrahedron action, and 50/144 products fail closure.
- **What survives:** canonical tetrahedron coordinates and a small group that
  can be exhaustively tested.
- **Decision:** **generate from all even permutations, assert vertex action,
  identity, inverses, uniqueness, and closure, then store the resulting system
  permutation rather than returning unchanged nodes**.

### OT-04 — Immutable law versus mutable task state

- **Original mechanism:** use canonical `SW=9f` both as structural law and as
  the scalar modified by basin reinforcement.
- **Fresh boundary:** one update changes total SW from 108.0 to 108.2 and
  invalidates the ledger.
- **What survives:** the conservation ledger as a useful tripwire.
- **Decision:** **keep structural exposure/SW immutable; add separate mutable
  confidence, score, basin mass, and learning fields with their own contracts**.

### OT-05 — Joint-state quantum boundary

- **Original mechanism:** local two-amplitude nodes plus detached four-amplitude
  Bell-pair records.
- **Fresh boundary:** declaring a Bell pair does not change or correlate node
  measurements; CNOT cannot pass through the single-node API; separable `|++>`
  receives concurrence 1.0.
- **What survives:** normalized local state utilities and an association graph.
- **Decision:** **call the graph classical association until one joint
  state/tensor object owns gates, collapse, and exact entanglement measures**.

### BSH-01 — Basin-owned candidate state

- **Original mechanism:** represent a SAT/CSP candidate as a list of coordinates
  in one mutable Livnium lattice.
- **Fresh boundary:** every SAT/CSP candidate has the same coordinate list, the
  encoder overwrites one shared assignment, decoder ignores its basin argument,
  and constraint tension is global.
- **What survives:** explicit candidate IDs, constraint fields, and
  winner/loser lifecycle.
- **Decision:** **a candidate must own an immutable assignment or reversible
  delta plus its own independently recomputed objective**.

### BSH-02 — Strict solver-success semantics

- **Original mechanism:** set `solved=true` whenever any winner basin survives.
- **Fresh boundary:** SAT reports 10/10 although only one witness satisfies;
  CSP reports 6/6 although no saved assignment is valid.
- **What survives:** saved witnesses and external verification functions.
- **Decision:** **separate candidate production, witness validity, SAT witness,
  UNSAT proof, decision completion, objective, gap, timeout, and failure**.

### BSH-03 — Objective-identity preflight

- **Original mechanism:** parse only GSET endpoints, count unweighted crossing
  edges, and compare against stored literature values.
- **Fresh boundary:** signed weights are discarded, yielding ratios above 200%
  for the greedy baseline on G11/G12.
- **What survives:** GSET corpus, independent objective recomputation, greedy
  baseline, and result schema.
- **Decision:** **hash the input, preserve weights/signs, recompute every saved
  witness outside the solver, and refuse comparisons when objectives differ**.

### BSH-04 — Tiny exhaustive oracle before scale

- **Actual mechanism:** use exhaustive enumeration for tiny SAT, CSP, and
  Max-Cut instances to check state identity, witness validity, optimum, and
  gap before running large preserved corpora.
- **Historical lesson:** a four-cycle can return 4/4 at step one by lucky
  initialization while the same method reaches only 26.37% of G1's stored
  target.
- **Decision:** **make the exhaustive oracle, random search, greedy/local
  search, and simulated annealing mandatory baselines before any basin claim or
  UF20 run**.

## Sudoku learning and search

Detailed evidence: `SUDOKU_LINEAGE_AUDIT.md`.

### SUD-01 — Local candidate-context learner

- **Original mechanism:** predict an empty cell's digit from row, column, box,
  and optionally location presence features.
- **Fresh boundary:** the original training resubstitution is 58.61% versus
  53.22% on fresh masks; the strongest pure artifact is 43.36% unseen-cell
  accuracy, while legal-candidate baselines are already substantial.
- **What survives:** a compact same-task learned heuristic and explicit
  train/validation/test solution-board split.
- **Decision:** **preserve, but compare on identical masks against legal-random,
  smallest-legal, frequency, location-only, and generator-only controls**.

### SUD-02 — First-error rollout boundary

- **Original mechanism:** fill the globally most-confident pure prediction with
  no legality mask, backtracking, or rescue.
- **Fresh boundary:** the strongest saved run's average first error falls from
  37.34 moves on easy to 5.01 on expert, after which corrupted state compounds.
- **What survives:** first-mistake, calibration, and recovery as more
  informative measurements than average final cell equality alone.
- **Decision:** **keep pure rollout as a focused error-propagation experiment,
  with independent final-grid validity and alternate-solution handling**.

### SUD-03 — Paired value-ordering scaffold

- **Original mechanism:** inject MLP digit probabilities into conventional
  propagation/MRV/backtracking.
- **Fresh boundary:** exact saved means reproduce, but no bucket has a
  significant paired win; expert p=0.755 and LCV has the lower mean.
- **What survives:** shared puzzle identities, independent witness validation,
  and a clean ordering injection point.
- **Decision:** **benchmark learned, LCV, ascending, descending, and random
  orderings across multiple training seeds on hashed unique puzzles**.

### SUD-04 — Rejected-action loop guard

- **Original mechanism:** wrong policy actions leave the grid unchanged, after
  which deterministic greedy argmax selects the same action again.
- **Fresh boundary:** the saved 41-hole example takes 123 steps and changes zero
  cells.
- **What survives:** a reusable environment regression test.
- **Decision:** **mask rejected actions, change state, add explicit penalty
  memory/exploration, or terminate; never allow silent identical-state loops**.

### SUD-05 — Puzzle identity contract

- **Original mechanism:** generate a full board and randomly remove clues,
  equating clue count with difficulty and source-solution equality with success.
- **Fresh boundary:** 50-puzzle checks find multiple-solution rates from 20–24%
  at 51 givens to 100% at 23/26.
- **What survives:** the generators as data-production utilities, not as
  benchmark authorities.
- **Decision:** **record puzzle text/hash, provenance, uniqueness, recognized or
  solver-derived difficulty, independent validity, givens preservation, exact
  target equality, and unambiguous search counters separately**.

## Cube and geometry lineage

Detailed evidence: `CUBE_GEOMETRY_LINEAGE_AUDIT.md`.

### CG-01 — Exact group-orbit normalization

- **Original mechanism:** canonicalize all 24 rotations of a 5x5x5 binary world
  to one lexicographic representative.
- **Fresh boundary:** all 720 canonical test rows exactly match training; hash
  lookup and one-template nearest neighbor also score 100%.
- **What survives:** 24 valid cube permutations, orbit hashing, and exact
  nuisance removal.
- **Decision:** **call this deterministic invariance, not learned
  generalization; test task properties on underlying-world-disjoint splits**.

### CG-02 — Odd-cube directional partition

- **Original mechanism:** divide each axis into negative, zero, and positive
  groups, producing core/face/edge/corner sign blocks.
- **Fresh boundary:** the count is exactly the binomial identity
  `(2m+1)^3=1+6m+12m²+8m³`; counting alone does not prove optimal pooling.
- **What survives:** a simple position-preserving locality scheme.
- **Decision:** **benchmark against grids, pyramids, wavelets, convolution, and
  random/optimized layouts**.

### CG-03 — Learned local-filter boundary

- **Original mechanism:** one learned encoder filter per directional block plus
  a dense decoder.
- **Fresh evidence:** 0.6357 beats fixed means 0.5956 but trails PCA 0.7181;
  crossed control wins 30/40 random layouts while some random layouts reach
  0.6458.
- **Decision:** **preserve the masked-autoencoder scaffold and report full
  layout/split/seed distributions before crediting cube geometry**.

### CG-04 — Whole-map equivariance contract

- **Original mechanism:** tie only the 13 core/face/corner encoder weights under
  C4 and call the autoencoder equivariant.
- **Fresh boundary:** independent encoder biases and dense decoder/bias break
  `f(gx)=g f(x)`; whole totals are 512 versus 548 parameters.
- **Decision:** **tie encoder, biases, latent action, decoder, and output bias;
  numerically test every group element after training**.

### CG-05 — Robust graph denoising, not truth

- **Original mechanism:** IRLS L1 report fit plus grid-Laplacian smoothness.
- **Fresh boundary:** the generator is full-dimensional; clean smooth error is
  0.016 but clean checkerboard error is 0.637; the “social” control is exactly
  naive reports.
- **What survives:** standard robust graph-signal smoothing and a useful
  prior-mismatch test.
- **Decision:** **rename it, add standard denoisers, sweep corruption and signal
  bandwidth, and never identify prior preference with truth**.

### CG-06 — Algebraic feature-independence preflight

- **Original mechanism:** add LO distance/cosines beside Om
  norms/dot/cosine.
- **Fresh boundary:** all LO values reconstruct from Om to `3.75e-15`.
- **Decision:** **symbolically and numerically check new feature blocks for
  deterministic redundancy before attaching a conceptual mechanism**.

## Governance and economy lineage

Detailed evidence: `GOVERNANCE_ECONOMY_LINEAGE_AUDIT.md`.

### GOV-01 — Information-bearing governance

- **Original mechanism:** elect one representative through energy, favor,
  backing, repayment, and reciprocal coalition growth.
- **Fresh boundary:** observations have zero reads inside the election; reversing
  every observation leaves winner and trajectories identical.
- **What survives:** a compact model of wealth/favor lock-in and a clean
  oligarchic-capture counterexample.
- **Decision:** **require candidate evidence quality to affect agent information,
  utility, voting, and outcomes before calling a process an information
  economy**.

### GOV-02 — Independent-monitor contract

- **Original mechanism:** an unelectable community averages judge votes to prune
  anomalous candidates.
- **Fresh boundary:** judge identity and local evidence are unused; every judge
  repeats the same median-anomaly vector plus noise.
- **What survives:** separation of monitoring from eligibility and a useful
  honest-majority median-outlier filter.
- **Decision:** **give monitors heterogeneous evidence, reliability, cost,
  history, and corruptibility; cloned computations are one rule, not a
  community**.

### GOV-03 — Trust-relocation matrix

- **Original mechanism:** compare social median, oracle weight, earned stake,
  and structural tolerance under targeted attacks.
- **Fresh boundary:** oracle/reference receives truth directly, stake reputation
  is assigned, and the three components are never integrated.
- **What survives:** a strong taxonomy of distinct fatal assumptions—majority,
  oracle corruption, patient reputation, and valid in-tolerance lies.
- **Decision:** **retain this as a threat-model matrix and test joint failure
  with explicit, independently sourced trust channels**.

### GOV-04 — Exact incentive boundary

- **Original mechanism:** expected gain until detection minus capped slashing.
- **Fresh boundary:** one-shot threshold is exactly `1/q`; finite horizon 40 at
  `q=.15` needs 1.00911; the saved large-gain one-shot breaks even at stake 200.
- **What survives:** an exact, reusable risk-neutral design equation and the
  important limited-liability boundary.
- **Decision:** **calculate expectations exactly, state detection/identity/risk
  assumptions, and distinguish sufficient full-fine collateral from necessary
  expected deterrence collateral**.

### GOV-05 — Non-dumpable shared exposure

- **Original mechanism:** subtract global truth-quality loss from every cartel
  member's payoff.
- **Fresh boundary:** the result is a direct payoff identity with no strategic
  behavior; reducing retained share immediately restores profit.
- **What survives:** attackers must be unable to dump, hedge, or externalize the
  harmed exposure before settlement.
- **Decision:** **preserve the mechanism hypothesis; test endogenous holdings,
  timing, hedging, side payments, and exit in an integrated ledger**.

### GOV-06 — Progressive discipline with identity

- **Original mechanism:** ten detected lies trigger permanent public deletion
  and visible punishment reforms others.
- **Fresh boundary:** majority detection is hardcoded to zero without an anchor,
  while deletion directly flips remaining liar identities with a stipulated
  probability.
- **What survives:** strike histories, progressive sanctions, and detection
  boundaries as an institutional scaffold.
- **Decision:** **model false positives, appeal, durable identity, Sybil return,
  replacement, and endogenous behavioral response before claiming
  deterrence**.

### GOV-07 — Temporal abstention and quorum

- **Original mechanism:** low-confidence voters may rest for up to three
  staggered rounds while quorum is protected.
- **Fresh boundary:** the script has no time and permanently removes the
  oracle-known noisiest voters; 95% silence receives hardcoded error 9.99 below
  quorum.
- **What survives:** confidence-aware abstention can help only with honest
  reliability estimates and explicit participation constraints.
- **Decision:** **implement cooldown state, scheduling, strategic confidence,
  simultaneous-rest attacks, and measured quorum/welfare tradeoffs**.

### GOV-08 — Distributed observability, not metaphysical layer

- **Original mechanism:** sparse read-only local reality ties select truth
  against a coordinated globally valid lie.
- **Fresh boundary:** truth is restricted to ten known modes, reports are
  ignored, random 40-sensor layouts are full rank, and the comparison block is
  rank 6; corruption also has a cell-index bug.
- **What survives:** distributed sensor placement and condition-number checks
  are powerful tools for rejecting untrusted reports under a trusted channel.
- **Decision:** **rename as low-rank sensing; compare random, concentrated,
  optimized, and adversarial placement under basis mismatch and sensor
  corruption**.

### GOV-09 — Matched-subspace selector

- **Original mechanism:** average 24 cube rotations to retain only symmetric
  information.
- **Fresh boundary:** the rank-21 orthogonal projector receives truth generated
  in its own image; saved 0.2485 error matches analytic projected-noise 0.2474.
- **What survives:** exact group projection and an honest matched-versus-
  mismatched prior demonstration.
- **Decision:** **call it a linear known-prior selector, compare with group
  pooling/learned equivariance, and never infer truth from membership alone**.

### GOV-10 — Equivariant vector prior

- **Original mechanism:** decode vector fields on the rotation-equivariant
  manifold with robust whole-vector weighting.
- **Fresh boundary:** the projector is exact rank 42, but cell orbits have sizes
  1/6/8/12/24 and coherent equivariant lies remain valid.
- **What survives:** a strong known-subspace denoiser for incoherent
  off-subspace arrow corruption.
- **Decision:** **state the corruption model, test in-subspace adversaries, and
  avoid coding-distance language for an unrestricted real-valued subspace**.

## Symmetry spectrum

Detailed evidence: `SYMMETRY_SPECTRUM_LINEAGE_AUDIT.md`.

### SYM-01 — Exact product-spectrum preflight

- **Original mechanism:** interpret grid-Laplacian eigenvalue degeneracy as a
  pattern of cube pull.
- **Fresh boundary:** the operator is exactly `P7 □ P7 □ P7`; analytic sums
  reproduce all eigenvalues to `3.55e-14` and explain the 70 levels.
- **What survives:** a compact exact spectrum oracle and multiplicity regression
  test.
- **Decision:** **derive Cartesian-product spectra analytically before attaching
  physical or learned interpretations to numerical eigensolver output**.

### SYM-02 — Symmetry versus arithmetic degeneracy

- **Original mechanism:** identify every shared-speed multiplicity with proper
  cube-group irreps.
- **Fresh boundary:** full 48-element cubic symmetry commutes, but 15/18-fold
  levels merge several 3/6-permutation families because complementary path
  eigenvalues sum to four.
- **What survives:** eigenspaces are invariant representations and can be
  decomposed explicitly.
- **Decision:** **separate group-forced degeneracy, product permutation orbits,
  and accidental/arithmetic eigenvalue collisions with character projectors**.

### SYM-03 — Connected and anisotropic controls

- **Original mechanism:** compare one isotropic cube grid with one same-edge
  random graph.
- **Fresh boundary:** archived random max multiplicity five is disconnected
  zero modes; ten connected controls and a generic anisotropic separable control
  each have 343 distinct levels.
- **What survives:** random/simple-spectrum and symmetry-breaking controls.
- **Decision:** **match connectivity and relevant structural statistics, then
  break one symmetry at a time rather than relying on edge count alone**.

### SYM-04 — Structure-before-utility boundary

- **Original mechanism:** establish the pattern first and leave usefulness as a
  separate question.
- **Fresh boundary:** spectral existence is exact; no task or model is present.
- **Decision:** **test a predeclared group-compatible task against augmentation,
  group pooling, full equivariance, anisotropy, and locality-destroying
  isospectral controls before claiming computational value**.

## Cube embeddings and path geometry

### CEM-01 — Information-preservation preflight

- **Original mechanism:** map a semantic vector to a 27-position field and
  compare its within-field angles.
- **Fresh boundary:** all measured edge cosines depend only on squared
  coefficients in the orthonormal probe basis; independent sign flips leave
  the 94D signature unchanged.
- **Decision:** **derive invariances and collisions before training; reject any
  semantic transform that discards distinctions the downstream task needs**.

### CEM-02 — True transport versus loop roughness

- **Original mechanism:** sum `arccos` edge angles around 13 fixed loops and
  call the result holonomy/winding.
- **Fresh boundary:** the sum is unsigned, reversal-invariant, and has no
  connection, transported state, path ordering, or group product.
- **Decision:** **reserve holonomy for an explicitly oriented transport law;
  otherwise name the feature loop perimeter, variation, or roughness**.

### CEM-03 — Independent-channel accounting

- **Original mechanism:** concatenate 54 edge phases, 27 neighbor cosines, and
  13 loop sums as a 94D signature.
- **Fresh boundary:** all 40 later entries reconstruct from the first 54 to
  `6.06e-8`.
- **Decision:** **separate measurement dimension, deterministic feature
  expansion, and effective rank; include edge-only ablations**.

### CEM-04 — Constraint/evaluation firewall

- **Original mechanism:** derive synonym/antonym constraints from SimLex human
  scores, counter-fit, select a blend, and report on the same pairs.
- **Fresh boundary:** same-pair rho 0.8459 becomes pair-held-out 0.0337 and the
  cube channel becomes -0.0620.
- **Decision:** **hash and isolate lexical constraints from evaluation pairs;
  select every weight on train/dev only and report untouched test once**.

### CEM-05 — Identity and random-layout baselines

- **Original mechanism:** interpret a transformed embedding's score spread as
  geometric semantic signal.
- **Fresh boundary:** direct SVD retrieval and held-out SNLI beat the cube
  transform; the source layout ranks last of 21 probe layouts.
- **Decision:** **every learned or handcrafted geometric transform must beat
  the identity representation, random probes/layouts, and a matched direct
  classifier**.

### CEM-06 — Reproducible OOV identity

- **Original mechanism:** fixed random table plus Python trigram hashes.
- **Fresh boundary:** built-in `hash()` changes across processes and is
  load-bearing at 84–89% OOV.
- **Decision:** **use a specified stable hash with versioned salt and corpus
  manifest; never rely on interpreter hash randomization for model identity**.

## Games and discrete planning

Detailed evidence: `GAMES_LINEAGE_AUDIT.md`.

### GAM-01 — Conserved-symbol game-state transport

- **Original mechanism:** represent pieces and chess metadata as named tokens
  moved through bijective swaps.
- **Fresh boundary:** continuous 1,000-move and adversarial controls pass without
  symbol drift, but Python chess supplies legality/semantics and clocks/history
  are incomplete.
- **What survives:** a strong representation/transport scaffold and continuous
  multiset-conservation regression test.
- **Decision:** **name this verified state transport; call it a chess engine
  only after the Livnium state independently owns complete rules and history**.

### GAM-02 — Definition-derived tactical baseline

- **Original mechanism:** manually weight check, king mobility, attack, support,
  and mobility-drop features as a “basin” mate ranker.
- **Fresh boundary:** no basin is trained or queried; check plus zero legal
  replies finds all 100 generated mates while the scorecard finds 84.
- **What survives:** candidate-analysis records and an explicit reminder to
  derive the target definition before modeling.
- **Decision:** **run rule/definition baselines before crediting geometric or
  learned tactical intelligence**.

### GAM-03 — Online adaptation versus frozen policy

- **Original mechanism:** reinforce X-winning state anchors and repel states
  from O-loss anchors during tic-tac-toe play.
- **Fresh boundary:** the source result trains on its evaluation stream; frozen
  five-seed performance regresses and a symbolic heuristic is perfect for the
  tested draw objective.
- **What survives:** compact online negative-memory behavior.
- **Decision:** **report prequential/online adaptation separately from frozen
  generalization and compare both with deterministic policies**.

### GAM-04 — Transition-aware credit assignment

- **Original mechanism:** reward or penalize a sliding-puzzle state according
  to whether the following move improves Manhattan distance.
- **Fresh boundary:** history records the pre-action state without the action,
  while future selection scores post-action candidates; persistent memory makes
  every tested mode worse.
- **What survives:** a clear failure case demonstrating why state-only outcome
  anchors do not identify useful actions.
- **Decision:** **key planning memory by state-action/transition and attach
  credit to the resulting transition; freeze evaluation across held-out starts**.

### GAM-05 — Goal-distance representation preflight

- **Original mechanism:** use a per-position Manhattan-distance vector and
  cosine basin distance for 8-puzzle states.
- **Fresh boundary:** 181,440 boards collapse to 63,591 tuples/63,383 rays, and
  the solved zero vector is equally distant from every nonzero vector.
- **What survives:** exhaustive finite-state collision and goal-distance tests.
- **Decision:** **enumerate representation collisions and verify a graded,
  unique goal distance before training memory on a finite puzzle**.

### GAM-06 — Search-demo versus algorithm boundary

- **Original mechanism:** supply exact target energy to generic annealing over
  legal puzzle moves or arbitrary sorting swaps.
- **Fresh boundary:** the puzzle's exact depth is eight versus 3,714 annealing
  steps; sorting needs nine direct swaps versus 1,345 annealing steps.
- **What survives:** integration tests for `SwapSymbol`, `EnergyModel`, and
  `SearchEngine`.
- **Decision:** **label objective-supplied annealing a search-engine demo, not
  learned planning or algorithm discovery, and always report direct/exact
  baselines**.

## NLI, language modeling, and compression controls

Detailed evidence: `NLI_LANGUAGE_LINEAGE_AUDIT.md`.

### NLI-01 — Same-representation classifier control

- **Original mechanism:** route compact lexical features through persistent
  attractor/repulsor basins.
- **Fresh boundary:** saved non-cheat basins score 41.60–42.27% while logistic
  regression on the exact same 20 values reaches 53.13%.
- **What survives:** persistent prototypes, receipts, and a clean way to
  separate representation quality from decision-rule quality.
- **Decision:** **before blaming features, train a matched direct classifier on
  the identical inputs; credit the basin only for lift beyond that control**.

### NLI-02 — Answer-injection kill test

- **Original mechanism:** append a true-label one-hot vector in a deliberately
  named cheat condition and test whether the basin can route it.
- **Fresh boundary:** all partial cheat states score 100%; the answer itself
  dominates distance, so this cannot validate non-cheat geometry.
- **What survives:** an excellent plumbing/leakage positive control.
- **Decision:** **use answer injection only to prove label decodability and
  evaluator sensitivity; never treat it as task accuracy or mechanism proof**.

### NLI-03 — Hash buckets versus geometry

- **Original mechanism:** MD5-map words into 27³ cube cells and classify sparse
  premise/hypothesis occupancy.
- **Fresh boundary:** this is a 19,683-bucket hashing vectorizer per sentence;
  no cube adjacency or distance is used, plain BoW is smaller and slightly
  better, and 14 geometry summaries do not help.
- **What survives:** collision accounting and a compact sparse hashing baseline.
- **Decision:** **a geometric address is not a geometric computation; compare
  hashed occupancy with direct BoW before assigning meaning to the layout**.

### NLI-04 — Predictive code length versus emitted compression

- **Original mechanism:** an online normalized byte-context model with
  Witten-Bell-style interpolation reaches 1.781675 ideal bits/char at order 4.
- **Fresh boundary:** the implementation sums `-log2 p`; it contains no
  finite-precision coder, stream format, length field, or round-trip decoder.
- **What survives:** standard adaptive context prediction and a valid
  cross-entropy comparison that needs no transmitted learned count table.
- **Decision:** **report ideal/predictive bits separately from actual compressed
  bytes; claim a compressor only after an encoded stream round-trips**.

### NLI-05 — Surprise concentration is not semantic importance

- **Original mechanism:** call high-cost characters “dark matter” and infer that
  meaning lives in the surprising residue.
- **Fresh boundary:** 44.32% of K3 bytes cost under one bit and contribute only
  6.79% of code length, but the easy set is dominated by spaces, common letters,
  and newlines.
- **What survives:** a useful quantitative map of where prediction error is
  concentrated.
- **Decision:** **use surprise to allocate modeling capacity; require deletion,
  intervention, or downstream evidence before calling it meaning**.

### NLI-06 — Nested feature ablation and saturation

- **Original mechanism:** add Nova mean-cosine, DTW warp, and fracture summaries
  to a compact 13D lexical NLI feature set.
- **Fresh boundary:** DTW adds about 1.6 dev points to that compact set, but
  fracture adds none, fires on roughly 96%, and includes two exact transforms of
  existing warp values; all variants trail direct lexical baselines.
- **What survives:** nested ablations that distinguish incremental information
  from a larger concatenated feature set.
- **Decision:** **add blocks cumulatively, test algebraic redundancy and firing
  rate, and retain the strongest ordinary baseline in the same protocol**.

### NLI-07 — Checkpoint payload accounting

- **Original mechanism:** amortize float16 neural parameters over held-out text
  and compare predictive plus model cost.
- **Fresh boundary:** the neural predictor is 1.617459 bits/char, ideal float16
  amortization is 6.344796, and the actual optimizer-bearing 2.84 MB checkpoint
  yields 77.407781; the artifacts answer different deployment questions.
- **What survives:** explicit separation of predictive score, deployable model
  payload, and resumable training state.
- **Decision:** **state which payload is counted and never present an optimizer
  checkpoint as though it were a stripped compressed model**.

### NLI-08 — Artifact/source chronology binding

- **Original mechanism:** save decisive CSV/PNG output and later revise the
  learned-partition positive control in source.
- **Fresh boundary:** the CSV predates the source and reports 4.1418 for a
  control that the surviving code produces at 3.7522.
- **What survives:** both the historical failed apparatus and the corrected
  source-level negative result.
- **Decision:** **bind every result to source hash or timestamp; preserve an
  older result as history rather than silently attributing it to newer code**.

## Demos, feedback, and stream persistence

Detailed evidence: `DEMOS_LINEAGE_AUDIT.md`.

### DEM-01 — Canonical numeral versus fixed-width codec

- **Original mechanism:** serialize a 27-cell lattice string by interpreting its
  `0abcdefghijklmnopqrstuvwxyz` alphabet as one base-27 integer.
- **Fresh boundary:** positional integers erase leading zero glyphs, including
  the fixed core marker; the demo round-trips only a word without leading zero.
- **What survives:** reversible digit-list/string mapping and standard canonical
  integer arithmetic.
- **Decision:** **store width/length or operate on digits when position matters;
  never call a bare canonical integer a lossless fixed-width state codec**.

### DEM-02 — Epoch-zero and multi-initialization controls

- **Original mechanism:** show a randomly initialized labeled prototype field
  reaching 100% on two antipodal Gaussian clusters.
- **Fresh boundary:** the nominated seed is already 100% before training; across
  100 initializations mean accuracy rises 43.32%→95.91%, while a sign rule is
  perfect and 17 trained fields remain below 90%.
- **What survives:** a compact prototype-update demo and a strong adversarial
  swapped-anchor failure case.
- **Decision:** **always print epoch zero, repeat initialization, and compare the
  simplest rule implied by the data geometry**.

### DEM-03 — Outcome-vector feedback

- **Original mechanism:** combine X-win attraction and O-win repulsion with one
  beta and report win rate, draw rate, drift, jump, and diversity.
- **Fresh boundary:** increasing beta trades aggression for draws rather than
  producing one monotonic improvement; seeds select very different regimes.
- **What survives:** separate pull/push budgets and separate win/loss/draw axes.
- **Decision:** **declare the objective vector before choosing a coupling weight;
  do not rank policies by one outcome after inspecting all three**.

### DEM-04 — State-variable reachability

- **Original mechanism:** let positive authority, bad karma, and freshness govern
  pull/push.
- **Fresh boundary:** freshness never advances; X bad karma is written but not
  read by the scorer; O bad karma is read but never written and remains zero.
- **What survives:** separation of positive, negative, and age evidence.
- **Decision:** **for every claimed control variable, test that an update reaches
  it, it changes under the protocol, and the decision path actually reads it**.

### DEM-05 — Paired prequential persistence

- **Original mechanism:** compare a second warm session with the first cold
  session and infer memory benefit.
- **Fresh boundary:** seeds differ and both runs train during measurement; a
  matched five-pair control still gives warm 31.6% versus cold 18.6% wins, with
  one regression.
- **What survives:** task-relevant saved state and a prequential continuation
  advantage.
- **Decision:** **pair the same future stream, label prequential results, report
  per-seed regressions, and add frozen evaluation before claiming generalization**.

### DEM-06 — Explicit hash-coverage contract

- **Original mechanism:** call every receipt hash-verifiable.
- **Fresh boundary:** the receipt chain has zero breaks but hashes only ordered
  anchor centers; all decay and court events mutate excluded metadata with
  unchanged hashes.
- **What survives:** a bounded live log, archive sidecar, and center-history
  continuity check.
- **Decision:** **state exactly which fields a receipt commits to; include all
  decision-relevant mutation state when claiming a complete audit ledger**.

### DEM-07 — Reversible court-state transitions

- **Original mechanism:** provisional anchors earn promotion and harmful anchors
  enter quarantine.
- **Fresh boundary:** provisional/promoted score identically and an early return
  prevents any promoted anchor from later being quarantined.
- **What survives:** explicit evidence counts and status labels.
- **Decision:** **test every state-machine transition, including demotion after
  promotion, and require each status to have a declared behavioral effect**.

### DEM-08 — Responsible-event attribution

- **Original mechanism:** penalize only the nearest anchor responsible for harm.
- **Fresh boundary:** `KarmicBasinField` mutates the nearest anchor, but the Nova
  wrapper increments harm and emits decay receipts for every label anchor.
- **What survives:** the intention to localize negative credit.
- **Decision:** **return the mutated anchor from the lower layer and attribute
  metadata/receipts only to that object; test multi-anchor labels explicitly**.

## Complete arch-archive root

Detailed evidence: `ARCH_ARCHIVE_ROOT_AUDIT.md`.

### AR-01 — Chronology is evidence, not a folder name

- **Original mechanism:** use `arch-archive` as the presumed earliest source of
  the project.
- **Fresh boundary:** surviving arch-archive directory births begin
  2025-12-12, after the 2025-02-14 control-group transcript and the conversation
  export beginning 2025-03-03.
- **What survives:** `clean=noba=back` is the oldest self-contained
  arch-archive copy; workspace is the artifact-complete copy.
- **Decision:** **record both chronological and completeness canon; never infer
  age from a folder name or select a canonical mirror from age alone**.

### AR-02 — Transactional monotonicity needs a correctness contract

- **Original mechanism:** O-A8 proposes accepting promotions only when a global
  scalar does not increase.
- **Fresh boundary:** the markdown example deletes a node rather than
  implementing donor-backed promotion, and monotone descent alone does not
  establish feasibility, correctness, convergence, or freedom from stagnation.
- **What survives:** evaluate a candidate update before commit, preserve the
  prior state, and reject changes that violate a declared objective or invariant.
- **Decision:** **keep transactional rollback as an engineering pattern, while
  separately testing task correctness, reachability, and convergence**.

### AR-03 — Fixed-resource capacity is a hypothesis, not exposure

- **Original mechanism:** O-A10 interprets structural work/exposure at fixed
  resource as compressed information density.
- **Fresh boundary:** exposure counts are not information measures, the example
  code is incomplete, and the fresh reinforcement test fails.
- **What survives:** a testable question: under a fixed resource budget, does a
  defined representation improve held-out task information or compression?
- **Decision:** **define bits, decoder, resource budget, task, and direct
  baseline before using information-density language**.

### AR-04 — Reachability before market interpretation

- **Original mechanism:** classify market regimes with alignment, tension, and
  a fixed C=0.38 multiplier, then correlate tension with future volatility.
- **Fresh boundary:** the euphoria branch is mathematically unreachable; saved
  correlations are NaN from an unmasked shifted target; corrected tension
  correlation is near zero and weaker than direct volatility baselines.
- **What survives:** a reusable preflight: derive feature ranges, test every
  state’s reachability, mask finite aligned targets, split chronologically, and
  compare direct baselines.
- **Decision:** **retain the branch as a negative research harness, not an alpha
  signal; require data provenance, walk-forward evaluation, and trading-cost
  accounting before reconsideration**.

### AR-05 — Artifacts must bind to provenance

- **Original mechanism:** preserve generated SNLI geometry images, an efficiency
  chart, and a `brain` model directory as evidence of completed work.
- **Fresh boundary:** the images have no source/command/model/hash binding, the
  chart numbers are hard-coded, and `brain` is only a stock model cache.
- **What survives:** visual history and a reusable pretrained dependency.
- **Decision:** **a result artifact needs source hash, command, data split, model
  hash, environment/hardware, metric definition, and timestamp; cache presence
  is not training evidence**.

### AR-06 — Circular frames and layer notation

- **Original mechanism:** O-A9 treats circular reference frames as scale-free
  physical equivalence, while the layer note records hollow/filled symbols,
  depth, alternation, and function output.
- **Fresh boundary:** no contraction/topology/limit makes O-A9 a physical
  theorem; the archived note even loses one superscript and mismatches displayed
  versus described depth.
- **What survives:** the multiscale boundary metaphor and a later exact
  structural parser copied across three maintained roots.
- **Decision:** **preserve the notation/parser as structure and the circular
  frame as a research metaphor; require formal maps and measurable predictions
  before restoring physics or semantic claims**.

### Incorporation decision

The three archive roots are one historical lineage, not independent projects.
Preserve the oldest self-contained and artifact-complete roles, verified
Core/Core-C mechanisms, negative market result, hypotheses, notation, cache
provenance, and orphan visual history. Do not port the full base Core, market
classifier, O-A8/O-A9/O-A10 claims, cached “brain,” or hard-coded figures into
the active implementation without a new scoped experiment.

## Nova-and-Misc closure

Detailed evidence: `NOVA_MISC_AUDIT.md`.

### NM-01 — Target-blind representation firewall

- **Original mechanism:** compare static and dynamic collapse during NLI
  evaluation.
- **Fresh boundary:** the dynamic route consumes gold labels before
  classification and rises through a known shortcut.
- **What survives:** the label-blind static evaluator and hidden-state
  ablation shape.
- **Decision:** **assert that no target, target-derived basin, or answer token
  enters encoding/collapse; make this a testable dataflow contract**.

### NM-02 — Hash-to-permutation is identity, not semantics

- **Original mechanism:** fold arbitrary text into a 27-symbol permutation to
  avoid pair-dependent alphabets.
- **Fresh boundary:** every output is a permutation of the same alphabet and
  the operation is lossy and non-reversible.
- **What survives:** deterministic identity bucketing for fixtures or storage.
- **Decision:** **label deterministic hashes as identity features; require a
  decoder or held-out semantic task before calling them language geometry**.

### NM-03 — Cache keys must cover all dependencies

- **Original mechanism:** memoize semantic embeddings by cell identity.
- **Fresh boundary:** weights affect the embedding but not the key, so weight
  mutation can return stale state.
- **What survives:** a useful cache layer after dependency repair.
- **Decision:** **hash every input that influences output or invalidate on
  mutation; test same cells/different weights explicitly**.

### NM-04 — Live, archived, and historical counts are different

- **Original mechanism:** maintain a bounded live receipt ledger with archive
  spillover.
- **Fresh boundary:** the reported “total” equals archived count, not archived
  plus live, and each append rewrites full history.
- **What survives:** bounded hot state plus recoverable cold history.
- **Decision:** **declare live/archive/total equations and use append-safe or
  chunked storage; test counts across truncation boundaries**.

### NM-05 — Observer feature invertibility preflight

- **Original mechanism:** add residual-observer statistics to mean/max/std.
- **Fresh boundary:** residual mean is zero and the original max/std are
  reconstructed exactly.
- **What survives:** an interpretable centered coordinate system.
- **Decision:** **before training, derive whether new features are independent,
  invertible reparameterizations, constants, or genuinely lossy summaries**.

### NM-06 — Angular potential as a controlled candidate

- **Original mechanism:** use smooth log-sum-exp angular attraction to class
  anchors and sweep its gradient parameters.
- **Fresh boundary:** development-only selection, fixed baseline prose, no
  final artifact, and no untouched test.
- **What survives:** a differentiable candidate collapse potential.
- **Decision:** **evaluate it under a locked label-blind protocol against
  direct-vector, linear, and matched nonlinear heads over multiple seeds**.

## Realcore legacy, snapshots, and Crux

Detailed evidence: `REALCORE_LEGACY_SNAPSHOTS_AUDIT.md`.

### RL-01 — Capacity fixtures must respect representation complexity

- **Original mechanism:** supply `n_qubits=27` so capacity scripts collect.
- **Fresh boundary:** exact state-vector memory is exponential and the run is
  killed rather than verified.
- **What survives:** explicit capacity boundaries and infeasibility skips.
- **Decision:** **derive resource cost before choosing a fixture; separate
  exact-register capacity from local/pairwise/site-count capacity**.

### RL-02 — Dual-cube state machine, not semantic physics

- **Original mechanism:** positive, negative, and trapped cubes model meaning,
  contradiction, drift, cancellation, fossilization, and decay.
- **Fresh boundary:** all thresholds and transitions are inserted; “move”
  copies state; 29 test-named scripts have no assertions.
- **What survives:** a three-state event/history model for application-level
  conflict handling.
- **Decision:** **reuse only as an explicit domain state machine with supplied
  labels, learned/calibrated transitions, assertions, and outcome metrics**.

### RL-03 — Representation-collision kill test

- **Original mechanism:** compress a complete Ramsey coloring into three
  coordinates for geometric search.
- **Fresh boundary:** modulo-one arithmetic maps all complete binary colorings
  to `(0,0,0)`.
- **What survives:** the need for a compact search index.
- **Decision:** **measure collision rate and task-label retention before any
  geometric search; reject encodings that collapse the complete state space**.

### RL-04 — Separate validator value from search narrative

- **Original mechanism:** combine geometric mutation with a compiled C Ramsey
  constraint checker.
- **Fresh boundary:** the geometric state collapses, but the C checker matches
  an independent Python checker on 200 seeded cases.
- **What survives:** the validator as a useful reusable component.
- **Decision:** **promote independently verified components even when the
  surrounding search hypothesis fails; benchmark them separately**.

### CRUX-01 — Release mining before rewriting

- **Original mechanism:** Crux packages base-27 arithmetic/codecs, cube moves,
  couplers, Potts recall, hierarchy, CLI, JS bridge, visualizer, and docs.
- **Fresh boundary:** it is absent from the current Dart Core and survives only
  as a ZIP; its saved test count is stale.
- **What survives:** a fresh 32-test classical release and strong educational
  surface.
- **Decision:** **compare APIs and port deliberately before another rewrite;
  keep source history, tests, CLI, and documentation as one release unit**.

### CRUX-02 — Runtime/API version contract

- **Original mechanism:** a Dart 3.3-compatible package with a modern web
  bridge.
- **Fresh boundary:** the bridge uses APIs introduced in Dart 3.6 while the
  package permits 3.3.
- **What survives:** working code on the current SDK.
- **Decision:** **set the minimum SDK to the oldest actually supported API
  level and test that boundary in CI**.

### SNAP-01 — Content delta, not filename, defines a snapshot

- **Original mechanism:** preserve several similarly named Core, Sacred, Nova,
  and GitNexus ZIP/tar releases.
- **Fresh boundary:** some are exact, some near-exact, and Crux is unique; names
  alone do not reveal that.
- **What survives:** hashes, normalized entry manifests, and explicit
  exact/different/only-in counts.
- **Decision:** **list before extraction, normalize paths, compare content,
  assign unique files to a lineage, and retain archive hashes**.
