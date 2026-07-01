# experiments/ — standalone experiment scripts

Loose research scripts, moved from the repo root (2026-07-02). Each is
self-contained unless noted.

- `livnium.py` — the 43 KB pattern-replication / Ramsey stress-test monolith.
- `geometry_discriminator_test.py` — backs `../GEOMETRY_DISCRIMINATOR_VERDICT.md`.
- `char_typer_symbols.py` (+ `.pt`) — pure-geometry whole-symbol typer.
- `ledger.py` + `container_stream.py` — ledger geometry + container stream.
  `container_stream` imports `ledger`, so run from inside this folder:
  `cd experiments && python3 container_stream.py`.
- `qwen_probe.py` — Qwen probe experiment.
