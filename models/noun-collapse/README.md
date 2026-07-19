# Noun Collapse — grade A+

The strongest trained ML result in this repository. Every vocabulary item has a
distinct learned well. A context is read as an ordered trajectory through those
wells, and the endpoint is trained to identify the missing noun.

The model does not use Word2Vec initialization, an MLP, attention, a transformer,
or a separate learned output matrix. Meaning is learned from prediction pressure
on raw Wikipedia contexts.

## Measured checkpoint

| property | value |
|---|---:|
| wells | 100,001 × 256 |
| noun targets | 23,758 |
| training signal | 94.75M noun-eligible occurrences |
| parameters | ~25.6M |
| SimLex-999 noun ρ | **0.3616** |
| coverage | 662 / 666 noun pairs |
| random same-shape control | mean ρ ≈ 0.022 |

The checkpoint lives at `model/noun_collapse_pure.pt`; its SHA-256 is recorded
in `artifacts/checkpoints.md` and the published model is on Hugging Face.

## Run

```bash
python3 models/noun-collapse/noun_collapse_pure.py --probe cat physics war india
python3 models/noun-collapse/embed_eval.py \
  --model models/noun-collapse/model/noun_collapse_pure.pt
python3 models/noun-collapse/noun_bench.py
```

Training and evaluation share `text.py`; this component no longer imports the
personal chat-data pipeline. `noun_embed.py` is the PPMI+SVD comparison pipeline,
while `benchmarks/embeddings/matched-corpus/` is the controlled same-corpus test.

## Honest boundary

This demonstrates learned lexical semantics and order-sensitive context encoding.
It does not by itself demonstrate multi-step reasoning or open-domain generation.
