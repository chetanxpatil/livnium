# Desktop Livnium Audit

Audited: 2026-07-26

Source:
`/Users/chetanpatil/Desktop/livnium`

This is a read-only recovery audit. No source file, checkpoint, result, or
document in the project was changed.

## Executive conclusion

This root preserves two valuable but different engineering threads:

1. a classical SNLI system whose best saved checkpoint reproducibly reaches
   **76.34% accuracy on the complete 9,842-example SNLI development split**; and
2. a small Nova memory system with deterministic receipts, state hashes,
   archive-only maintenance, bundle indexing, and reproducible retrieval.

Neither thread establishes the broader claims that later proposal documents
attach to them. The NLI run still lacks matched end-to-end baselines and
multi-seed replication. Nova's reported `+0.5321` alpha survival gap is largely
guaranteed by a pruning policy that explicitly archives the lowest-alpha nodes;
its regression “gold set” contains three placeholder queries. The iDEX material
is therefore preserved as proposal history, not accepted as TRL3/TRL4 evidence.

The best thing in this folder is not a physical-quantum claim. It is the
combination of a real measured NLI pilot and careful memory-audit
infrastructure.

## Boundary and lineage

- Size at audit: approximately 956 MB and 368 non-Git files.
- Major payloads: `data/` 541 MB, `pretrained/` 252 MB, `runs/` 154 MB.
- File mix: 125 Python files, 114 bytecode files, 49 Markdown files, 25 JSON
  files, 18 JSONL files, 8 PyTorch checkpoints, and 2 Word documents.
- Git branch: `main`, commit `37faef0`, one commit ahead of `origin/main`.
- The recorded Git history spans 2026-03-15 to 2026-03-16 and tracks only 24
  files. It began under quantum/Nova terminology and was renamed to
  **Energy-Guided Attractor Network (EGAN)**.
- Cortex, Nova, results, experiments, scripts, documentation, and both iDEX
  documents are untracked in this Git root. Git history is therefore not a
  complete history of the folder.
- At audit time, six tracked SNLI files and `.gitignore` were modified,
  `.eval_results/snli.yaml` was deleted, and eleven top-level groups/files were
  untracked. These user changes were preserved.

The source/document manifest contains 180 rows and 172 distinct content hashes.
Of those hashes, 137 have copies elsewhere and 35 are confined to this root.
The locally confined material is concentrated in `system/`,
`livnium_extension/`, `docs/`, `results/`, `experiments/`, Nova, and the
top-level explanatory files.

Most embedded Nova files have copies under `test/lab/nova/public`,
`livnium-sacred copy`, or `test/nova-memory-main`. Four inspected Nova files were
confined to this root at inventory time:

- `nova-memory-main/benchmark/arxiv_nova_benchmark.py`
- `nova-memory-main/tests/test_alpha_retention.py`
- `nova-memory-main/livnium/growth.py`
- `nova-memory-main/ai/encoder.py`

For Nova Git history, use `/Users/chetanpatil/Desktop/test/lab/nova/public`
rather than assuming the embedded untracked directory is canonical. It is a Git
worktree at commit `38287d7` on `feature/interaction-audit-hashing`.

## SNLI / EGAN checkpoint audit

### Actual architecture

The saved system is a classical neural NLI model:

- pretrained word embeddings;
- bag-of-words sentence encoding;
- a learned iterative collapse MLP repeated for six steps; and
- a supervised classifier head.

There is no physical quantum execution in this model.

### Reproduced best result

The original tracked code at Git HEAD was evaluated in an isolated temporary
copy against the complete SNLI development split:

- checkpoint: `runs/triple_crown_slow/best_model.pt`;
- examples: 9,842;
- correct: 7,513;
- accuracy: **0.7634**.

Per-class results using the code's actual label mapping were:

| Class | Correct / total | Accuracy |
|---|---:|---:|
| Entailment | 2,674 / 3,329 | 0.8032 |
| Contradiction | 2,542 / 3,278 | 0.7755 |
| Neutral | 2,297 / 3,235 | 0.7100 |

The evaluation report header prints `E N C`, but the true mapping is
entailment, contradiction, neutral. Historical claims of 87.5% entailment and
81.2% contradiction were not reproduced.

A post-hoc diagnostic that bypassed the collapse module while retaining the
co-adapted trained classifier scored **0.6559**. This shows that collapse is
load-bearing in this checkpoint, a gain of about 10.75 percentage points over
that diagnostic. It is not a fair causal baseline because the no-collapse model
was not independently trained.

The checkpoint contains approximately 13.0 million parameter-state elements:
12.8 million in the embedding table, 132,608 in collapse, and 72,617 in the
head. The README's “about 2M parameters” description is not a correct total.

### Saved checkpoint comparison

Several newer checkpoints required temporary evaluator repairs: the evaluator
hard-codes a 256-dimensional collapse despite 259-dimensional saved models, and
some checkpoints store a non-portable relative embedding path. The source and
checkpoints were not modified.

| Saved run | Dev accuracy | Interpretation |
|---|---:|---|
| `triple_crown_slow` with original tracked code | **0.7634** | Best saved checkpoint |
| `alpha_gated_v1` | 0.7115 | Regresses from the original |
| `my_run` | 0.6898 | Regresses |
| `lyapunov_run` | 0.6877 | Regresses |
| `memory3_run` | 0.6831 | Regresses |
| `infersent_run` | 0.5053 | Large regression |

The current dirty alpha-gated evaluator silently applies a new alpha map even to
the old checkpoint and lowers the original checkpoint from 0.7634 to 0.6825.
The original tracked evaluator is required to reproduce the best result.

### SNLI decision

Preserve `triple_crown_slow` as a **measured pilot**. It proves that the trained
collapse pathway contributed to this saved system and achieved 76.34% on this
split. It does not yet prove superiority to a parameter-matched MLP, residual
network, InferSent-style head, or another standard baseline. The next scientific
promotion test requires matched training, multiple seeds, frozen data identity,
and error analysis.

Do not repeat the unreproduced per-class numbers, the “2M total parameters”
claim, or the end-to-end speed claim without a new timed protocol.

## Nova memory audit

### What is genuinely implemented

The embedded Nova package contains useful small-system engineering:

- a `GrowthMind` tree;
- deterministic UUID receipts and state hashes;
- archive-only maintenance;
- deterministic retrieval;
- bundle indexing;
- branch and counter integrity checks; and
- a regression-court/self-check structure.

All nine collected pytest tests passed. Running every `tests/test_*.py` script
directly also passed all fifteen scripts, covering P1–P9 retention checks,
receipts, deterministic retrieval, haircut, indexing, conservation health,
branch/counter integrity, and reasoning self-checks.

The README honestly identifies important limits: toy embeddings, roughly 100
nodes, a three-query gold set, no competitive benchmark, no authentication, and
no performance hardening.

### Why `+0.5321` is not semantic evidence

Nova reads alpha as the mean of precomputed values from `data/alphamap.json`;
out-of-vocabulary tokens receive 0.5. Its `haircut` policy sorts candidates by
`(alpha, age, id)` and archives the lowest-alpha nodes first.

The arXiv benchmark inserts 716 tokens as flat root children, limits the system
to 200 nodes, and then verifies that low-alpha nodes were removed. The saved
artifact reports:

- 374 high-alpha and 342 low-alpha tokens;
- 517 archived nodes;
- 199 surviving high-alpha and 0 surviving low-alpha nodes;
- survival gap `+0.5321`;
- `P6 = true`; and
- `trl4_pass = true`.

Because the tested policy directly sorts by alpha, the alpha-retention
relationship is a policy invariant, not independent evidence that retained
nodes are more useful or meaningful. The benchmark also defines its own TRL
thresholds and contains a hard-coded session path.

`eval/gold/v0.jsonl` contains three records whose query text is literally
`<fill query>`, all pointing to the same expected ID. `gold_queries.jsonl` is
also a three-query toy. The regression court is good infrastructure but is not
yet backed by a real gold set.

### Nova decision

Classify deterministic receipts, state hashes, archive-only mutation, and
regression checks as **verified engineering at small scale**. Classify
alpha-pruning and its `+0.5321` result as a **verified policy mechanic**, not a
semantic or causal result. TRL4 is not established by the local evidence.

Keep the architecture. Replace the placeholder gold set, use task outcomes
independent of the pruning score, compare against LRU/LFU/BM25/embedding/random
and learned policies, and report usefulness, latency, storage, and failure modes
before promoting it.

## Cortex, alpha, and retrieval evidence

The later Cortex code fixes reverse-direction CNOT failures found in
`Desktop/uantum`, but its empirical story is mixed:

- the 150-document M1 evaluation gives P@10 of 0.0147 for TF-IDF, 0.0160 for
  YAKE, and 0.0120 for both Alpha-Only and LIVNIUM-B;
- LIVNIUM-B is identical to Alpha-Only and loses significantly to YAKE at P@5
  (`p=0.0286`);
- a synthetic 60-word retention test reaches a `+0.3667` survival gap at one
  threshold but mixes label and position effects;
- across ten arXiv documents, Mode A's mean MPS survival gap is only `+0.0068`;
  Mode B's aggregate gap is `-0.0006`; and
- dynamic-alpha MPS policies change internal truncation/structure tradeoffs but
  have not produced a matched fidelity or downstream-task win.

The alpha signal itself separates the hand-defined “technical” and “common”
token groups. That shows the score and labels are correlated; it does not show
that the selected words improve retrieval, reasoning, or memory usefulness.

Preserve these artifacts because the negative generalization result is
scientifically valuable. Retire the claim that the synthetic gap establishes a
general semantic filter.

## iDEX documents

Both Word documents were rendered and visually inspected page by page.

### `LIVNIUM_TechFeasibility_iDEX_TRL3.docx`

Useful qualities:

- correctly describes the MPS implementation as classical;
- does not claim quantum hardware;
- states the synthetic-corpus, benchmark, and baseline limitations; and
- describes TRL4 as future work.

Evidence problems:

- TRL3 relies on the older `12/12` Cortex suite, whose reverse CNOT tests were
  incomplete and whose alpha-pruning test was vacuous;
- the semantic bridge correlation and 27-to-149-word comparison use synthetic
  configurations without standard NLP baselines or a sufficient statistical
  protocol; and
- long GHZ chains are a standard low-bond-dimension MPS case, not generic
  high-entanglement capacity.

Verdict: preserve as an honest-leaning technical proposal, but update its
evidence package before external reuse.

### `idex_spark_executive_summary.docx`

This document overstates the local evidence by describing alpha as
non-statistical/physically grounded, asserting unique rotations for every word,
using the policy-created `+0.5321` gap as TRL evidence, and presenting TRL4 and
funding language without independent validation. Alpha is derived from
pretrained statistical GloVe vectors plus PCA, and vector databases/RAG systems
do not universally treat all information equally.

Verdict: preserve as proposal history. Do not use it as scientific evidence or
repeat “TRL4 complete” until independent task-based validation exists.

## Documentation classification

- `docs/LIVNIUM_CORE_SPEC.md`: useful, clean core specification; overlaps with
  the July repository and records the corrected `N=7` total of 2646.
- `docs/QUANTUM_PROOF.md`: a careful description of complex-amplitude
  interference as operationally quantum-like, not physical quantum evidence.
- `MEDIUM_POST.md`: thoughtful EGAN narrative, but its class numbers, speed,
  parameter-count, and novelty language require correction.
- `docs/livnium/FINAL_REPORT.md` and `REAL_WORLD_SOLUTION.md`: promotional
  historical documents built on hand-authored tests. Their “100%,” “problem
  solved,” production, and quantum language are not accepted evidence.
- `STRUCTURE.md`: useful navigation for the March reorganization.

## What to preserve

1. `runs/triple_crown_slow/best_model.pt` and the exact tracked-code/data
   environment needed to reproduce 0.7634.
2. The later checkpoints as negative/ablation history, not as replacements.
3. Nova's receipt, hash, archive-only, indexing, and regression-court design.
4. All saved Cortex/arXiv/M1 results, including the negative results.
5. Both iDEX documents as dated proposal history.
6. The 35 content hashes currently confined to this root until the sacred roots
   are compared.

## Next recovery action

Audit `/Users/chetanpatil/Desktop/livnium-sacred` and
`/Users/chetanpatil/Desktop/livnium-sacred copy` together, with
`/Users/chetanpatil/Desktop/test/checkpoints-sacred` as the artifact comparison
target. Compare source hashes, checkpoint identities, configs, result logs, and
Git history before selecting a canonical artifact vault.

