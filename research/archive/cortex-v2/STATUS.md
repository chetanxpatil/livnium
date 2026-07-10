# STATUS: archived (superseded)

**Why it was attempted.** cortex_v2 was the "500-site computer" era: an MPS
(matrix-product-state) simulator plus the first collapse-training prototype,
built to test how far entangled amplitude-like sites could scale and whether
collapse dynamics could train NLI on top of them.

**What happened.** The site-count claim conflated faithful entangled sites
(limited by entanglement) with addressable structured sites (limited only by
memory) — see `SESSION_FINDINGS.md` for the code-verified numbers. The NLI
training run had bugs that were forensically documented in
`../../../docs/collapse/COLLAPSE_ENGINE_VERDICT.md`.

**What superseded it.** The standalone engine in
`packages/vector-collapse/` (clean config, basin field, DynamicsLedger) and
the supervised model in `research/nli/supervised-collapse/`.

**What remains reusable.** The `validate_*.py` scripts are honest,
self-contained checks of core lattice/hierarchy properties, and
`test_regressions.py` still guards them. `SESSION_FINDINGS.md` is the
canonical record of the site-count resolution. Run from this folder;
scripts add `packages/livnium-core/src` to `sys.path` themselves.
