# Trained models

This directory contains model-sized experiments that have a defined task,
checkpoint schema, evaluation record, and runnable entry point. It is separate
from exploratory `research/` code.

| grade | component | why it matters |
|---|---|---|
| **A+** | `noun-collapse/` | Strongest trained evidence: 23,758 noun targets, SimLex-999 Spearman ρ = 0.3616 on 662/666 covered noun pairs. |
| **A** | `premise-generator/` | Working 5.98M-parameter contextual generator; a hypothesis and NLI label unfold into a premise. |
| **A−** | `collapse-nli/` | 68.87% SNLI test accuracy; useful result, with a matched end-to-end ablation still pending. |

Grades describe current evidential strength, not personal preference. Checkpoints
are gitignored; their canonical locations and hashes are in
`artifacts/checkpoints.md`.
