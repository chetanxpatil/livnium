# Premise Generator — grade A

This model reverses natural-language inference:

```text
(hypothesis, label) → generated premise
```

Each trained word is already a distinct semantic attractor. The input sentence
creates an ordered trajectory through that learned geometry; a small controller
then unfolds the trajectory into output words. The shipped aligned checkpoint
uses lightweight label-conditioned cross-attention over hypothesis word wells;
it does not use transformer self-attention.

## Measured checkpoint

| property | value |
|---|---:|
| parameters | 5,975,042 |
| vocabulary | 20,000 words |
| best recorded token accuracy | 52.71% |
| best recorded NLL | 2.6355 |
| CPU median latency | ~6 ms per short reply |

Single-word inputs are contextual rather than simple echoes: trained nouns such
as `microscope` and `dog` unfold into scene-like premises. Label control is the
weaker part—the output is contextual, but entail/neutral/contradiction adherence
is not yet consistently strong.

## Run

```bash
python3 models/premise-generator/chat_premise.py
python3 models/premise-generator/chat_bench.py --device cpu
python3 models/premise-generator/verify_lyapunov.py
```

The canonical checkpoint path is
`models/premise-generator/model/premise_from_hyp_align_53.pt`. Training stages,
interactive inference, benchmarks, claim map, and Lyapunov measurement now live
together in this folder—there is no compatibility shim back into chat-brain.

See `CLAIMS_CHECKPOINT_MAP.md` for the exact attention caveat and
`SNLI_BASELINES.md` for classifier comparisons.
