# Realcore Legacy and Snapshot Audit

Date: 2026-07-27

Status: incorporated. This closes the remaining `Desktop/core/learn`,
`lab/index`, P2 archive, and nested-Git boundaries.

No source archive was modified or unpacked into a project root. Archive
contents were listed first; the few necessary replays were extracted only to
temporary directories.

## 1. February Python Core versus the archived ancestor

Compared:

- February copy:
  `/Users/chetanpatil/Desktop/core/learn/core`
- artifact-complete archived ancestor:
  `lab/infected/workspace/clean-nova-livnium/archives-local/arch-archive/core`

Ignoring caches and macOS metadata, the February tree adds exactly three paths:

- `Livnium_Project_Assessment.docx`
- `recursive/run_recursive_example.py`
- `tests/conftest.py`

Only two shared files differ:

- `recursive/inheritance.py` now copies all quantum feature switches into a
  child configuration; and
- `quantum/tests/test_true_quantum_layer.py` adds five-qubit, ten-qubit,
  infeasibility-skip, and repeated Bell-measurement checks.

The inheritance change is a real repair: a recursive child can now retain the
parent's quantum configuration. The example is a useful runnable hierarchy
walkthrough.

### 1.1 Fresh test boundary

The archived ancestor previously produced **252 passed, 25 failed, 6 collection
errors**.

The February `conftest.py` supplies `n_qubits=27` to the six capacity scripts.
That removes collection errors but is unsafe: `TrueQuantumRegister(27)` attempts
an exact `2^27` complex state and the full run was killed by the operating
system with exit 137.

A bounded replay that excluded the three capacity-script files and the explicit
ten-qubit dense-gate stress test produced:

- **254 passed**
- **25 failed**
- **2 skipped**
- **1 deselected**
- **4 return-not-assert warnings**

The same 25 old failures remain: corner gating, basin reinforcement,
entanglement configuration/API drift, Moksha/recursive API drift, semantic API
drift, and temporal API drift. The fixture converts an honest collection error
into an out-of-memory route; its “small for CI” comment is false.

### 1.2 Historical assessment document

`Livnium_Project_Assessment.docx` is a seven-page February 7, 2026 viability
memo. Every page rendered legibly with no clipping. Its conservative product
conclusion remains useful: research/education may be viable, commercial utility
needs validation, and the next step should be user/problem/alternative
validation.

Its technical wording is stale. “Complete implementation,” “all tests,” “it
does work,” and broad quantum-layer claims conflict with the fresh failure and
memory-scaling boundary above. Keep it as a historical assessment, not a
current verification report.

## 2. The older `archives-local/archive` layer

The generated `lab/index` maps a second archive tree distinct from
`arch-archive`. Its canonical live copy is:

`/Users/chetanpatil/Desktop/test/livnium.core-0.0.1/archive`

It contains 231 meaningful regular files covering:

- self-labelled broken simulators/tests and incomplete capacity scripts;
- legacy `quantum_2`/quantum-islands;
- an earlier hierarchical `quantum_computer`;
- MPS extracts;
- pre-Core hierarchical/islands/Livnium code;
- dual positive/negative cubes and trapped-phi “scar” experiments; and
- a Python/C Ramsey search.

Mirror reconciliation:

- `clean=noba=back/archive` matches the 231-file canonical content and adds four
  preserved binary/archive artifacts;
- `clean-nova-livnium/archives-local/archive` has the same boundary plus five
  later dual-cube files that introduce a shared `step()` evolution order; and
- the live `livnium.core-0.0.1/archive` omits only the embedded release tarball
  and compiled Ramsey build artifacts from those mirror extras.

The legacy `quantum_2` mechanisms are the already incorporated Q1
quantum-islands lineage. Most pre-Core and quantum-computer sources reappear in
the organized Realcore museum. They are lineage copies, not independent
quantum systems.

### 2.1 What survives

- exact three-qubit state-vector gates, GHZ, and teleportation are standard
  verified simulation;
- sparse/MPS structures are useful classical representations with their normal
  approximation and bond-dimension limits;
- the later dual-cube `step()` provides one explicit order for contradiction
  migration, drift, cancellation, and measurement; and
- a compiled C Ramsey validator matched the independent Python validator on
  **200 seeded random K6, k=3 colorings**.

### 2.2 What does not survive

The pre-Core tree contains 29 functions named `test_*` and **zero assert
statements**. These are print experiments, not a verification suite.

The “dual cube” is a hand-designed two-list model:

- contradiction is coordinate proximity plus an amplitude threshold;
- “move” creates a negative copy without removing the positive state;
- drift and cancellation are inserted rules; and
- no semantic task supplies or validates the amplitudes.

Trapped phi is likewise a configurable state machine: fixed contradiction and
decoherence thresholds, minimum age, random one-percent leakage after a
half-life, and a fixed 25% capacity ceiling. It is a reusable memory metaphor,
not a discovered physical law.

The Ramsey solver correctly calls itself classical, but its geometric encoding
collapses completely for full binary colorings. In
`RamseyGraph.to_coordinates`, integer-colored weighted terms are reduced modulo
one. Twenty distinct seeded K8 colorings therefore produced:

- **20 distinct graph hashes**
- **1 coordinate**
- the coordinate `(0.0, 0.0, 0.0)` in every case.

The C validator is useful engineering, but the geometry cannot guide completed
colorings and no new Ramsey witness is preserved. The documents' speculative
2–1000x speedups remain forecasts, not measurements.

## 3. Embedded archive-within-archive objects

Two nested releases were listed before temporary extraction:

| Artifact | Size | Entries | SHA-256 | Result |
|---|---:|---:|---|---|
| `livnium-quantum-7b27e33.tar.gz` | 369 KB | 284 | `d1d49b10b8bc0205f374a0659b0e3bff4129a2778f9cf0225510c694a4835e77` | Earlier organized quantum snapshot; later Realcore adds hierarchy/dual-cube work while the tar preserves old docs/broken simulators |
| `quantum_computer_code.zip` | 19 KB | 27 | `9ccee2849c374e74c5de82eefc4b33d6bc05b1fdabf15e2ca39185ad6df81fb4` | Early subset of the later legacy quantum-computer tree |

No unique validated quantum claim is stranded only in either nested archive.
They remain valuable chronology snapshots.

## 4. Top-level snapshot inventory

| Snapshot | Entries | SHA-256 | Reconciliation |
|---|---:|---|---|
| `Desktop/livnium-sacred.zip` | 131 | `05d3b7d3ecdfe39e2338454ca7f98c279bcc8132165cf6058d706508a5a95431` | Earlier `test/livnium-sacred` snapshot: 81 meaningful files exact and 3 meaningful Nova-v3 files revised later |
| `livnium.core-0.0.1-multi-basin.zip` | 1,860 | `5e93a01b6a242a6a2829c9d16f995935918897f1df716339b74de8c657aef832` | Historical full bundle; 1,721 non-cache files |
| `livnium.core-0.0.1.tar.gz` | 1,502 | `a30378421a8f8da0ebae840778419850c6e70f5ef56546f7f185d2d8099cdd6d` | Live tree differs meaningfully in two NLI files and adds `run_collapse.py` |
| `livnium.core-0.0.3.zip` | 1,845 | `f8ce4dea41b3b901f18a610451a7a45dac1d57d9c4f4c9c03e976197686be70d` | Follow-up full bundle; 1,713 non-cache files |
| `livnium-crux-main.zip` | 206 | `8ffdef14cfe2a9e3e5fa2dd77b4ea41e6d8b44510a010fc7c8f6fbcc345be6f4` | Unique polished Dart/JS/docs release; incorporated below |
| `GitNexus-main.zip` | 476 | `3038f4e7af5aef529486f6c6fc068bfe5cb2c367d5a625013b960b0189c0ca40` | All 360 regular files exactly match extracted `GitNexus-main` |
| `nova-memory-main.zip` | 71 | `61d055cbea32be0f9fd42bd9e6f405d0662b77e7b190a2e1b42f3799a969964c` | All 60 regular files exactly match extracted Nova Memory v1 |

The two full Core ZIPs share 1,701 meaningful paths:

- 1,691 are byte-exact;
- 10 changed;
- 20 occur only in multi-basin, all under the already audited Nova-v3
  preservation branch; and
- 12 occur only in 0.0.3, comprising already audited NLI, Rule-30, and Nova
  model/result artifacts.

Both store `.githash` `548bc6f`; their ZIP comments record different packaging
commits. Their unique material is now assigned to an audited lineage.

## 5. Livnium Crux: unique release recovered

`livnium-crux-main.zip` is not a copy of the current Dart Realcore. It is a
self-contained historical release with:

- base-27 alphabet, codecs, canonical/balanced/cyclic arithmetic;
- 3x3x3 coordinates, exposure classes, rotations, and face-move permutations;
- coupler fields and a standard Potts associative memory;
- a 27-child recursive tree and center/core-bit bias;
- a command-line interface;
- a Dart-to-JavaScript bridge and browser visualizer;
- a generated Docusaurus documentation site; and
- citation, contribution, benchmark, example, and test material.

Fresh replay on the temporary extraction:

- `dart test`: **32 passed, 0 failed**
- `dart analyze`: **0 errors**, but 41 warnings/info items

The saved `TEST_RESULTS.md` says 23 tests; the archive now collects 32, so that
document is stale. Analyzer warnings include an SDK contract mismatch: the web
bridge uses APIs introduced in Dart 3.6 while `pubspec.yaml` permits Dart 3.3.

Crux is good classical engineering and the strongest recovered packaged
base-27/cube teaching release. It does not establish semantic understanding,
compression advantage, physical quantum behavior, or task advantage. Preserve
it as a distinct release ancestor; do not collapse it into the broader current
Realcore without a deliberate port.

## 6. `_ORGANIZED` and `lab/index` closure

Fresh `tree`/content reconciliation found:

- `_ORGANIZED`: 203 meaningful regular files, of which 202 are exact
  same-named root copies; the only unique meaningful file is the generated
  `INDEX.md`;
- 28 project/data/archive symlinks: 27 resolve and the old
  `04_Projects/quantum_retrain` link is broken;
- `lab/index`: 54 generated maps/indexes totaling about 9.4 MB.

Every `_ORGANIZED` experiment family is now incorporated. Its Core-Theory,
reports, data links, project links, and archive links point to already
registered roots. `lab/index` is a source locator, not an evidence ledger; its
remaining legacy/pre-Core/Ramsey submaps are covered by this audit.

Do not “fix” the broken `quantum_retrain` link by inventing a new root. The
surviving artifacts are under Sacred and `collapse_retrain`; the stale link is
useful provenance.

## 7. Nested Git repositories

Nine Git worktrees exist under `test/lab/infected`:

### First-party Livnium/Nova worktrees

1. `python/clean-nova` — `b6d9c07e1d88`
2. `python/clean-nova-livnium` — `f589b90195aa`
3. `python/clean-nova-livnium 2` — `443298242bf1`
4. `python/nova-livnium` — `369ab2ae6bea`
5. `realcore` — `588dfc89f9a9`
6. `workspace/clean-nova-livnium` — `f58be369d997`

### Third-party corpus tooling

Three ECW-BT roots contain exact Git checkouts of WikiExtractor at commit
`8f1b434a8060`, remote `attardi/wikiextractor`.

The previous machine registry found the six first-party roots but missed the
three deeper WikiExtractor worktrees because its search stopped at depth six.
The inventory builder is updated to search deeply enough on its next run. The
third-party checkouts are dependencies/corpus tools, not Livnium projects.

## 8. Closure decision

The remaining Realcore, `lab/index`, snapshot, and nested-repository material is
now either:

- assigned to an existing audited lineage;
- identified as an exact or near-exact preservation snapshot;
- incorporated here as unique Crux/dual-cube/Ramsey history; or
- explicitly marked third-party.

No unindexed P0, P1, or named P2 source remains in the recovery queue.
