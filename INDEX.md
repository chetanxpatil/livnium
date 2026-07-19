# Livnium project index

Updated 2026-07-15. The repository is arranged by evidence and responsibility,
not by chronology.

## Importance grades

| grade | component | role | evidence |
|---|---|---|---|
| **A+** | `packages/livnium-core/` | proven mathematical core | dependency-free; 67 focused tests |
| **A+** | `packages/vector-collapse/` | reusable dynamics engine | installable package; ledger and routing regressions tested |
| **A+** | `models/noun-collapse/` | strongest trained ML result | SimLex noun ρ = 0.3616; random control ≈ 0.022 |
| **A** | `models/premise-generator/` | contextual generator | real 5.98M checkpoint; ~6 ms CPU short-reply latency |
| **A** | `research/ramsey/` | verified mathematical research | exhaustive clique checks; independent witness verifier |
| **A−** | `models/collapse-nli/` | supervised NLI | 68.87% SNLI; matched end-to-end ablation pending |
| **A−** | `benchmarks/` | comparison evidence | frozen/matched protocols and baseline harnesses |
| **B+** | `research/chat-brain/` | active generation research | working ladder; reasoning remains weak |
| **B** | `research/exact-gradient/` | conservative collapse branch | exact potential, matched model result pending |
| **B** | `research/language-probes/` | representation probes | synthetic and SNLI-aware tests |
| **B−** | `research/vision/` | image-collapse prototypes | smoke/cross checks; full-scale result pending |
| **C+** | `research/discrete-chat/` | discrete generation probe | mechanism experiment |
| **C** | `research/qwen-hook/` | external-model hook | narrow integration probe |
| **D** | `archive/` | historical record | preserved, not maintained |

Grades mean:

- **A** — keep visible; independently useful and evidence-backed.
- **B** — active research with a clear question, but incomplete validation.
- **C** — exploratory probe; useful context, not a headline result.
- **D** — historical or superseded; retained for auditability.

## Canonical layout

```text
packages/     stable installable code
models/       trained systems with checkpoints and evaluation
benchmarks/   controlled comparisons and shared external datasets
research/     active questions and probes
archive/      superseded work
apps/         visual/demo applications
docs/         maintained explanation and results record
artifacts/    checkpoint manifests
```

No component is allowed to hide inside another component's `model/` or `data/`
folder. Noun-collapse, premise generation, and personal chat now have separate
code, weights, data paths, and READMEs.

## Reading order

1. `README.md`
2. `docs/core/STATE_OF_THE_CORE.md`
3. `packages/livnium-core/README.md`
4. `packages/vector-collapse/README.md`
5. `models/noun-collapse/README.md`
6. `models/premise-generator/README.md`
7. The README for the research question you care about.

## Verification

```bash
python3 -m pytest -q
python3 -m ruff check .
python3 -m black --check .
python3 research/ramsey/independent_check.py
```

Checkpoints and datasets remain gitignored. See `artifacts/checkpoints.md` for
canonical paths and hashes. Repository-root `conversations.json` is private input
for chat preprocessing and must never be committed.
