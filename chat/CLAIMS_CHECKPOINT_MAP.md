# Claim → checkpoint map (verification notes)

## Chat-brain checkpoints (added 2026-07-02)

| checkpoint | run | measured | status |
|---|---|---|---|
| `model/chat_typer.pt` | word typer, 20k vocab, 6k steps MPS | held-out per-word 98.0%, exact 86.4%, **clean OOV-free 100.0%** (from the run log) | ✅ verified; **superseded soon** — full-vocab (`--max-vocab 0`, ~67k wells) retrain pending |
| `model/chat_reply.pt` | reasoning v1, 8 epochs @ lr 3e-4 | dev reply-nll/word 7.98 (uniform = 9.9); unigram-stage generations | ⚠️ under-trained by design of the run, superseded by the 60-epoch run |
| `model/char_typer_all.pt` | char rung, raw canonical lines, batch 1024 | CE ~0.002 by step 1500 (in training at time of writing) | ⏳ first converged run was flatten-sourced; canonical rerun in progress |

Mechanism verifications (numpy replicas on real data, no torch): reader
order-sensitivity (reorder → meanpool cos 1.000000 vs collapse cos 0.072),
self-attend memory retrievability (typed-word key rank 1), minting isolation
(new-word wells max-cos 0.54 from all trained wells).

---

## SNLI premise-generator claims (original)

Which checkpoint backs each public claim, what's verified, and what needs fixing.
All checkpoints inspected directly (config + weight tensors read from the `.pt`
files; param counts from storage sizes). PyTorch was not run — speed numbers are
cross-checked against the repo's own benchmark docs, not re-measured.

## Measured this session (NumPy reimplementation of the real weights)

PyTorch wouldn't install in the sandbox, so the model's `generate()` /
`premise_nll()` were reimplemented in pure NumPy using the actual trained weights
read from `premise_from_hyp_align_53.pt`. It reproduces the post's exact sample
output ("is the girl standing" → neutral → "a girl in a pink shirt standing in a
doorway."), confirming the reimplementation is faithful.

| claim | measured | verdict |
|---|---|---|
| 5.98M params | 5.975M | ✅ |
| ~53% generative-classifier accuracy | **52.9%** on 1,500 SNLI dev pairs | ✅ |
| ~5 ms / reply on CPU | **4.46 ms median** (8.8 tok/reply, ~1,937 tok/s, 1-thread NumPy) | ✅ sub-10ms holds |

Caveat: NumPy on Linux sandbox CPU, not PyTorch on M-series. Confirms magnitude +
the launch-bound-decode story, not the exact MPS-vs-CPU split (no Apple GPU here).

## The three generator checkpoints

| file | `align` | attention weights present? | size | role |
|---|---|---|---|---|
| `chat/model/premise_from_hyp_align_53.pt` | **True** | yes (`att_key`, `att_query`) | 24.31 MB | shipped chat demo, ~53% gen |
| `…/pure-cleaned/model/premise_from_hyp_align_52.pt` | **True** | yes | 24.31 MB | earlier/weaker run, ~52% gen |
| `…/pure/model/premise_from_hyp.pt` | **False** | **none** | 22.99 MB | the only attention-free generator |

Key point: the 52% and 53% models are the **same architecture** (both
`align=True`, both carry attention weights). The 52→53 gap is training, not
mechanism. The genuinely attention-free model is `premise_from_hyp.pt`, which has
**no published generative-accuracy number** in the repo.

## Claim-by-claim

| public claim | correct checkpoint / source | status |
|---|---|---|
| 5.98M params | `align_53` (counted 5.975M float32) | ✅ verified |
| ~53% generative-classifier acc | `align_53` (`align=True`) | ✅ consistent with name + verdict log (0.534) |
| "attention-free / no attention" | only true of `premise_from_hyp.pt` (`align=False`) | ⚠️ misleading — the shipped `align_53` runs a `torch.softmax` step (`align_context`, line 124) |
| classifier 66.1% mean-pool | CollapseNLI baseline (5.25M) | ✅ matches `SNLI_BASELINES.md` |
| classifier "72.7% with alignment" | should be **74.7% dev / 74.4% test** (5.52M) | ❌ stale — 72.7% only survives in superseded `pure-cleaned/`; current docs all say 74.7/74.4 |
| CPU 5.3 ms vs MPS 13.1 ms | `chat_bench.py` on `align_53` | ✅ directional (CPU wins at batch 1); ⚠️ simplified — `BENCHMARKS.md` shows GPU overtakes at batch ≥4 |
| 78.2% same-footing baseline | Bowman '15 lexical features | ✅ sourced |
| SNLI only, no pretrained embeddings | code: `word_anchors` not from GloVe/word2vec/BERT | ✅ true (note: chat ckpt is warm-started from the internal char-collapse "wells", not pure random init as the README phrases it) |

## Recommended fixes before re-posting

1. **72.7% → 74.7% dev / 74.4% test.** The real number is better than what was
   posted; just update it.
2. **Reword "attention-free."** Accurate version: *"the collapse engine is
   attention-free; the shipped checkpoint adds one lightweight label-conditioned
   alignment (single-head attention) step on top."*
   - Alternative: ship/benchmark `premise_from_hyp.pt` (`align=False`) instead and
     quote *its* generative accuracy — then "no attention" is literally true.
3. **Speed claim:** optionally note that CPU wins only at batch 1; the GPU wins on
   throughput at batch ≥4 (already documented honestly in `BENCHMARKS.md`).
