# Discrete chat probes — grade C+

These scripts ask whether sequences can be represented by compositions of the
24 cube rotations or a small higher-dimensional signed-permutation group.

- `discrete_cube_collapse.py` — synthetic word-to-rotation classification.
- `test_discrete_chat.py` — 5D discrete classifier on local chat contexts.
- `train_group_lookup.py` — replaces matrix composition with a precomputed group table.
- `findings.md` — continuous-vs-discrete speed measurements and the negative
  result for treating the v1 noun force as an exact cosine potential.

The exact continuous gradient generator is a different experiment and lives in
`research/exact-gradient/`. Both experiments read chat data directly from
`research/chat-brain/data/`; no copied dataset is maintained here.

```bash
python3 research/discrete-chat/discrete_cube_collapse.py
python3 research/discrete-chat/test_discrete_chat.py
python3 research/discrete-chat/train_group_lookup.py
```

The discrete speed result is a mechanism benchmark, not evidence of semantic
reasoning or a universal accelerator.
