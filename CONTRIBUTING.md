# Contributing to Livnium

Thanks for your interest. Livnium is a small, honest project, and the bar for
anything that lands in `packages/livnium-core/src/livnium_core/` is the same as for the rest of it:
**proven, tested, and reversible.**

## Ground rules

- The core (`packages/livnium-core/src/livnium_core/`) is pure Python with **zero runtime dependencies**.
  Keep it that way — anything needing `numpy`, `torch`, etc. belongs in the
  experimental folders (`archive/cortex-v2/`, `models/collapse-nli/`) or in `benchmarks/nli/`.
- Every change to the core must come with tests, and every existing test must
  stay green.
- Honesty over hype. If something doesn't work, we write down that it doesn't
  work (see `docs/history/FINDINGS.md`). Negative results are welcome here.

## Development setup

```bash
git clone https://github.com/chetanxpatil/livnium.git
cd livnium
python -m pip install -e "packages/livnium-core[dev]" \
    -e "packages/vector-collapse[test]" scipy
```

## Before you open a pull request

Run the same checks CI runs:

```bash
python -m pytest -q          # all tests must pass
python -m ruff check .       # lint must be clean
python -m black --check .    # formatting must match
```

To auto-fix formatting and the safe lint issues:

```bash
python -m black .
python -m ruff check . --fix
```

Style is enforced by `ruff` and `black` with a 100-character line length
(configured in `pyproject.toml`), so you don't have to argue about it.

## Reporting issues

Open an issue describing what you expected and what happened. For math/claims,
point to the relevant file in `docs/` so we can check it against the record.

## License

By contributing, you agree that your contributions are licensed under the
project's [PolyForm Noncommercial License](LICENSE). Commercial use requires a
separate paid license — see [`COMMERCIAL.md`](COMMERCIAL.md).
