# Session Findings — "How many sites?" investigation

A code-verified investigation into the retired "500-site / exotic-physics" claim,
and where it actually holds. Every number below was produced by running a script
in this folder, not asserted. Run any of them from the repo root with
`python archive/cortex-v2/<name>.py`.

---

## The headline

The "500-site computer" claim conflated two different things:

- **Faithful entangled amplitude-like sites** — limited by *entanglement*, not hardware.
- **Addressable / structured sites** — limited only by memory.

Separating them resolves everything:

| Regime | Deepest stable reached | What limits it |
|---|---|---|
| Faithful entangled (random/worst case) | **13 sites** | entanglement (hard wall) |
| Exact statevector (uncapped) | n≈20 | exponential memory/time |
| Faithful **structured** (limited interaction) | **50,000 sites** | RAM (linear cost) |
| Faithful **repeating** pattern | **1,000,000 sites** | RAM (linear cost) |
| Nested-cube address capacity | **10^143 cells** (depth 100) | nothing (closed form) |
| Nesting depth, conserved | **unlimited** (≈247 layers in float64) | numeric underflow only |

**Bottom line:** 500 faithful sites is trivially reachable for *structured* states
(we hit a million). It is impossible only for *fully random* states (wall at 13) —
and that limit is fundamental, not an engineering gap. Random data is incompressible
for everyone, classical or amplitude-like.

---

## Verified results, by script

- `validate_sites.py` — exact statevector match to 1e-16 at small n; random circuit
  faithful only to ~13 sites at chi=64.
- `validate_nocap.py` — removing the bond cap restores exactness but reveals the
  exponential blowup (n=20 → bond 1024, 42.7 MB, 11 s; n=22+ intractable).
- `validate_cube.py` — driving the MPS with the cube's SU(2) lift gives the *same*
  13-site wall; the cube stays a valid bijection with ΣSW=486 throughout.
- `validate_combined.py` — conserved capacity scales freely while the faithful
  entangled slice stops at ~12–13 sites. Both truths in one run.
- `validate_conservation_vs_result.py` — **key result:** conservation (ΣSW=486) stays
  perfect while the computed result is 61% wrong (fidelity 0.39). Conservation is a
  checksum on geometry, NOT proof the amplitudes are correct.
- `validate_pattern_reuse.py` — **the area law.** Limited-interaction and repeating
  states stay faithful at n=hundreds/thousands with tiny bond dimension. The wall is
  only for noise. (Pushed to 1,000,000 sites in 61 MB.)
- `validate_compression.py` — structured data compresses 100–585× losslessly; random
  data does not compress at all (it's incompressible).
- `validate_representation.py` — you can change basis freely, but entanglement
  (Schmidt rank) is basis-invariant: no representation shrinks a truly entangled state.
- `validate_sites_via_nesting.py` — address sites via nesting are effectively unlimited
  (10^143); faithful amplitude-like sites stay ~13. Nesting does not raise the entangled count.
- `validate_energy_sites.py` — feeding shrinking/dividing energy into the governor does
  NOT raise the 13 ceiling; tighter energy only prunes more (fewer faithful sites).
- `validate_negative_energy.py` — negativity (signed amplitudes) is REQUIRED for
  interference, but the MPS already has it; it doesn't remove the exponential cost.
- `validate_hierarchy.py` / `validate_nesting_values.py` — the repo's *additive*
  nesting: global ledger = N³·SW(M) + SW(N); total grows with depth.
- `validate_split_nesting.py` — the *split-and-decrease* model (your design): a parent's
  value is partitioned among children, so per-cell value shrinks geometrically but the
  GLOBAL total stays exactly SW(top) at every depth. Cleaner, depth-independent
  conservation.
- `validate_nesting_depth.py` — conservation holds to depth 10,000 (exact); float64
  per-cell values underflow around depth ~247; capacity is closed-form unlimited.
- `validate_forward_cone.py` — forward best-first growth from a seed (open randomness
  in bands, follow the closeness gradient) reaches the answer touching 0.26% of the
  space. Constructive search, never materializing the full space.
- `validate_triangle_search.py` — narrowing search (bisection) collapses 1024
  possibilities in 10 steps WITH a closeness signal; without structure it's brute force.

## New engine

- `mps_qudit.py` — MPS generalized to d-level sites (qudits).
- `validate_qutrit_base27.py` — with d=3, **base-27 fits perfectly in 3 qutrits**
  (3³ = 27, zero waste). All 27 symbols roundtrip exactly. Qutrit entanglement
  (Fourier + SUM gates) works. Contrast: 5 qubits hold 32 states (5 wasted).

---

## The one idea under all of it

Computation — classical or amplitude-like — is the **search for and exploitation of
structure**, reshaping a conserved budget it can never inflate.

- Compression, distribution, and amplitude-like speedup all work *exactly when structure
  exists*, and all die on pure randomness.
- The bond dimension is simultaneously the memory cost, the cross-machine
  communication cost, and the entanglement — they are the same number.
- Conservation bounds the books but does not pin the result; the result lives in the
  amplitudes (reconstructed by walking the bonds, never stored as one list).

## Livnium as a "digital atom"

The honest, overclaim-free identity: a **conserved, composable digital atom** —
27 cells, 24 rotational symmetries, base-27 native (= 3 qutrits), with a boundary-
exposure weight law (SW = 9 × exposure) that mirrors valence/outer-shell reach.

Unlike a real atom, it has no opposing force, so no stability ceiling: it never
decays at any size or nesting depth. Real atoms stop at ~118 (Oganesson) because
long-range proton repulsion (∝ N²) overtakes short-range binding (∝ N). The digital
atom has only one rule — conservation — and nothing fighting it.
