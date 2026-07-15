# Exact-gradient collapse — grade B

`pure_reply.py` replaces the chord-directed v1 update with the exact gradient of
`V(h) = -cos(h, target)`. It is the conservative dynamics branch and deliberately
lives outside the promoted chat model until it earns a matched result.

It reuses the active chat-brain data/model helpers and uses the canonical noun
checkpoint from `models/noun-collapse/`; those paths are resolved from the file
location rather than from the current working directory.

Local checkpoints live in `research/exact-gradient/model/`.

```bash
python3 research/exact-gradient/pure_reply.py --epochs 50 --device mps
python3 research/exact-gradient/pure_reply.py --chat \
  --ckpt research/exact-gradient/model/chat_reply_pure.pt
```
