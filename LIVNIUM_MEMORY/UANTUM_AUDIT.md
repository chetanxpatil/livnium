# `Desktop/uantum` Audit

Audited: 2026-07-26

Source:
`/Users/chetanpatil/Desktop/uantum`

This folder is a small June-era bridge between the legacy quantum bundle, the
MPS/Cortex work, semantic memory triage, and a Ramsey search. It contains useful
ideas, but it is not the best current implementation of the MPS engine and it
does not contain a new quantum or Ramsey result.

No source file was modified during this audit.

## Inventory and duplication

- Size: approximately 1.5 MB.
- Python files: 58.
- Distinct Python content hashes: 56.
- Of those 56 hashes, 49 have external copies, all under
  `/Users/chetanpatil/Desktop/livnium`.
- The top-level and nested copies of `livnium_cortex_v1.py` are byte-identical.
- The nested `livnium_extension/livnium/` tree is principally a copy of the
  already-audited legacy state-vector, quantum-islands, Grover/SAT, and geometric
  simulator family.

Eight Python paths, representing seven content hashes, were confined to this
root at inventory time:

1. `livnium_cortex_v1.py`
2. `livnium_extension/benchmark/benchmark_retrieval_triage.py`
3. `livnium_extension/livnium_cortex_v1.py`
4. `livnium_extension/mps/bridge_fidelity_test.py`
5. `livnium_extension/mps/mps_simulator.py`
6. `livnium_extension/mps/test_mps.py`
7. `livnium_extension/sensor_xray.py`
8. `ramsey/livnium_ramsey_v2_stochastic.py`

The Cortex hash occurs twice inside `uantum`, so these eight paths represent
seven unique content hashes.

## Cortex and MPS

### What the code is

The Cortex combines:

- the 24 proper cube rotations;
- SO(3) axis/angle extraction and an SU(2) single-qubit map;
- a standard matrix-product-state simulator;
- entropy/bond-budget pruning;
- an `alpha` scalar used to alter the compression ceiling;
- optional semantic words mapped through GloVe-50 and PCA-3.

The useful idea is a task- or content-conditioned tensor compression policy:
keep the quantum simulation and the policy objective conceptually separate, then
test which observable the policy preserves at a fixed resource budget.

### Fresh checks

Running the `uantum` Cortex self-test printed `12/12` passes. That output is not a
complete correctness certificate:

- Test 9, which is meant to show that `alpha` changes pruning, compares zero
  pruning events against zero pruning events. It passes vacuously.
- Test 12 samples a 15-qubit GHZ chain. GHZ has bond dimension 2, so this is a
  standard low-entanglement MPS case, not a general 15-qubit capacity result.

Focused package tests reported 35 passes. Pytest also warned that
`test_ghz_correctness` and `test_ghz_scale` return values instead of asserting
them, so those two collected functions can appear green without enforcing their
printed verdicts.

A direct directionality check found a real defect in this snapshot:

- adjacent `cnot(1, 0)` corrupts the tensor layout and makes dense contraction
  fail with a reshape error;
- non-adjacent `cnot(2, 0)` leaves `|001>` unchanged instead of producing
  `|101>`;
- forward `cnot(0, 2)` behaves correctly.

The later implementation at
`/Users/chetanpatil/Desktop/livnium/livnium_extension/mps/mps_simulator.py`
passes all three directionality checks with probability 1 on the expected state.
It also adds a non-vacuous alpha-pruning test and a thirteenth Cortex regression
test. The July repository's `archive/cortex-v2` remains the best honest narrative
because it records further swap-network/norm fixes and states the entanglement
limit explicitly.

Decision: **preserve `uantum` as lineage, but do not use its MPS/Cortex files as
the canonical engine**.

## Dynamic-alpha benchmark evidence

The unique `bridge_fidelity_test.py` idea was continued in the later
`Desktop/livnium` root, where saved artifacts exist.

Random-circuit run:

- 12 qubits, 24 steps, 30 seeds;
- static alpha truncation error: 32.8847;
- dynamic alpha truncation error: 28.5665, about 13.1% lower;
- static/dynamic L1 distance from the reference: 1.2573 / 1.2560;
- static/dynamic prune events: 17.0 / 19.8.

This shows that dynamic alpha changes internal compression behavior. It did not
produce a meaningful output-fidelity advantage.

The saved structured-noise run is mixed:

- dynamic alpha improves endpoint correlation and global L1;
- it worsens GHZ validity, even-parity retention, and truncation error.

Decision: **keep as a partial policy experiment**. The next scientific version
must declare the preservation objective before tuning and compare policies across
matched `chi`, entropy ceiling, noise, circuit, and seed sweeps. “Useful
structure” cannot be inferred after observing whichever metric improved.

Evidence:

- `/Users/chetanpatil/Desktop/livnium/docs/BENCHMARK_RESULTS.md`
- `/Users/chetanpatil/Desktop/livnium/livnium_extension/mps/benchmark_logs/bridge_fidelity_20260605T231405Z/summary.json`

## Semantic sensor and retrieval triage

### Mechanism

The semantic bridge maps:

`word → GloVe-50 → PCA-3 → axis/angle → SO(3)/SU(2) and alpha`.

The triage script retains high-alpha tokens under a fixed capacity and compares
them with FIFO and LRU.

### Fresh toy results

The mock benchmark reports 10/10 fact recall against 4/10 for FIFO/LRU at 40%
capacity. This is circular evidence: the hand-written high-alpha table contains
the same words used by the ground-truth fact set.

With the live GloVe/PCA cosine mapping, the same single-document toy reports 6/10
against 4/10. This is a small pilot, not a reliable retrieval benchmark:

- the document and fact set were hand-authored together;
- FIFO and LRU are identical in a single-write stream;
- there is one document, one ordering, and no uncertainty estimate;
- there is no TF-IDF, BM25, embedding-similarity, learned, or randomized semantic
  baseline;
- surviving tokens include several common filler words.

The sensor X-ray is still useful as a diagnostic. On its 64-word seed fit:

- cosine mode mean alpha: facts 0.8318, fillers 0.2642;
- IDF mode mean alpha: facts 1.0000, fillers 0.9998.

The IDF mode therefore has almost no discrimination because frequency is
estimated from a tiny seed list in which nearly every word appears once.

### Stronger saved evaluation

The 150-document Milestone 1 evaluation under `Desktop/livnium/results` is the
stronger evidence:

| Extractor | P@5 | P@10 |
|---|---:|---:|
| TF-IDF | 0.0200 | 0.0147 |
| YAKE | 0.0213 | 0.0160 |
| Alpha-Only | 0.0093 | 0.0120 |
| LIVNIUM-B | 0.0093 | 0.0120 |

LIVNIUM-B loses to TF-IDF at P@10 (`p = 0.3956`) and to YAKE at P@5
(`p = 0.0286`). LIVNIUM-B and Alpha-Only are numerically identical, so the
circuit/MPS stage added no ranking value in this run.

Decision: **preserve the sensor X-ray and capacity-constrained memory question;
retire the toy win as evidence of general retrieval superiority**. Any revival
should use a held-out corpus, repeated query sets, a real access/update trace, and
matched FIFO, LRU, LFU, TF-IDF/BM25, embedding, random-score, and learned
baselines.

Evidence:

- `/Users/chetanpatil/Desktop/livnium/results/eval_m1_report.txt`
- `/Users/chetanpatil/Desktop/livnium/results/eval_m1_results.json`

## Ramsey stochastic search

### Mechanism

`livnium_ramsey_v2_stochastic.py` is a conventional graph-search prototype:

- binary edge flips;
- incremental counting of monochromatic/clique and independent-set K5
  violations;
- simulated annealing;
- heavy-tailed multi-edge kicks;
- a rolling similarity penalty to repel the search from recently visited graphs.

The incremental counter is a valid useful component. A fresh test on random
8-vertex graphs compared 100 proposed flips with a full recount; all 100 deltas
matched.

### What it does not establish

- No `best_graph.json` witness exists in `Desktop/uantum`.
- The configured goal is a zero-violation graph on 43 vertices. Such a checked
  witness would improve the lower bound to `R(5,5) ≥ 44`; the script does not
  contain one.
- Its header says the known interval is `[43, 48]`. The current published bound
  is `43 ≤ R(5,5) ≤ 46`; the upper bound was improved by Angeltveit and McKay.
- Precomputation stores 903 × 10,660 = 9,625,980 five-vertex subset tuples.
- Five million ordinary flip attempts imply roughly 533,000,000,000 inner
  adjacency lookups before multi-edge kicks and similarity comparisons. The
  “millions is fine” runtime description is not credible for these Python loops.

Current-bound source:
`https://doi.org/10.1002/jgt.70029`

Decision: **keep the incremental-delta and novelty-penalty ideas; classify this
file as an uncompleted historical search prototype, not a Ramsey result**. A
revival should use compact bitsets or a SAT/local-search solver, save
checkpoints, and independently verify any zero-violation adjacency certificate.

## Final classification

| Part | Classification |
|---|---|
| Nested exact simulator and quantum applications | Duplicated historical lineage; already audited |
| `uantum` MPS/Cortex implementation | Superseded and directionality-buggy |
| Dynamic-alpha compression policy | Partial, mixed, worth controlled study |
| Semantic X-ray | Useful diagnostic |
| Mock retrieval win | Circular demonstration, not evidence |
| Live one-document retrieval win | Small pilot, contradicted by stronger evaluation |
| 150-document retrieval result | Negative result; preserve |
| Ramsey incremental counter | Correct reusable component |
| Ramsey 43-vertex claim/result | No witness; uncompleted search |

## Incorporation decision

Nothing needs to be copied from `uantum` into another source repository now.
The folder must remain preserved because it records the bridge between several
ideas. Future implementation should start from the corrected July Cortex/MPS
lineage and the negative retrieval evidence, not from this snapshot.
