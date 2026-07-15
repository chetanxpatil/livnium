# STATUS: archived (loose scripts, individually concluded)

**Why it was attempted.** A holding pen for standalone experiment scripts
that never grew into full projects (moved from the repo root 2026-07-02,
archived in the 2026-07 layout refactor).

**What happened / what superseded each.**

- `livnium.py` — the 43 KB Ramsey pattern-replication stress test. Concluded;
  the Ramsey line of work continues outside this folder.
- `geometry_discriminator_test.py` — backs
  `docs/results/GEOMETRY_DISCRIMINATOR_VERDICT.md`; verdict written, case
  closed.
- `char_typer_symbols.py` — pure-geometry whole-symbol typer; superseded by
  the chat-brain ladder in `research/chat-brain/`.
- `ledger.py` + `container_stream.py` — ledger-geometry stream experiments;
  the observability idea was productized as `DynamicsLedger` in
  `packages/vector-collapse/`.

**Note.** `qwen_probe.py` was NOT archived — it moved to
`research/qwen-hook/` because the Qwen hook is on the active
revival list.

**What remains reusable.** `container_stream.py` imports `ledger.py`; run
from inside this folder.
