# Chat-brain checkpoints

| checkpoint | run | measured | status |
|---|---|---|---|
| `model/chat_typer.pt` | word typer, 20k vocab, 6k steps MPS | held-out per-word 98.0%, exact 86.4%, clean OOV-free 100.0% in the saved run log | full-vocab retrain pending |
| `model/chat_reply.pt` | reasoning v1, 8 epochs | dev reply-NLL/word 7.98 (uniform ≈ 9.9) | under-trained, superseded by later local runs |
| `model/char_typer_all.pt` | character rung | CE ≈ 0.002 by step 1500 in the recorded run | canonical-data rerun status should be rechecked |

Mechanism probes on the saved runs found order sensitivity (reorder: mean-pool
cosine 1.000 vs collapse cosine 0.072), typed-word memory retrieval at rank 1,
and minted-word isolation (maximum cosine ≈ 0.54 from trained wells). These are
mechanism measurements, not reasoning benchmarks.
