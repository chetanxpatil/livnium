# Livnium — Complete Findings Inventory (Mar–Dec 2025 + audits to Jun 2026)

Every finding from the year, with honest status. Nothing real is dropped here —
including the wins that didn't survive. Status legend:

- ✅ **PROVEN** — pure math, re-derived from scratch and unit-tested in this repo.
- 🔧 **VERIFIED ENGINEERING** — code is correct and tested, but standard/expected (not a discovery).
- 🟡 **PARTIAL** — a real result on limited evidence (one checkpoint / one seed); not yet confirmed.
- ❌ **FALSIFIED** — claimed as a win during the year, tested, did not hold. Recorded on purpose.

---

## ✅ Proven mathematics (the durable core — all tested in this repo)

| # | Finding | Status | Where |
|---|---|---|---|
| 1 | **Base-27 codec** — positional numeral system (`0,a..z`), lossless, exact carry (`z+a=a0`), binary-convertible | ✅ | `base27.py`, `test_base27.py` |
| 2 | **Exposure classes** — every cell f∈{0,1,2,3}; closed-form counts (N−2)³, 6(N−2)², 12(N−2), 8 | ✅ | `lattice.py` |
| 3 | **Symbolic Weight law** SW = 9f and total **ΣSW(N)=54(N−2)²+216(N−2)+216** (486/1350/2646/4374) | ✅ | `lattice.py`, `test_lattice.py` |
| 4 | **Cube rotation group** — exactly 24 elements (≅S₄), generated from X/Y/Z 90° turns, R⁴=I | ✅ | `rotations.py`, `test_rotations.py` |
| 5 | **Conservation** — every rotation preserves class, counts, ΣSW, bijection (reversible permutation) | ✅ | `test_rotations.py` |
| 6 | **T13 orientation-independence** — a rotation's class-response is identical from all 24 start orientations → simulation reduces to 24 constants (≈497× speedup) | ✅ | verified; see note below |
| 7 | **Hierarchy / wreath product** — macro-N hosting micro-M; additive global ledger; G_M≀G_N | ✅ | `hierarchy.py`, `test_hierarchy.py` |

*Note on #6:* the orientation-independence property is verified here (one distinct
response pattern across all 24 orientations). The specific 497× engineering
speedup lived in the original simulation code; the math that licenses it is proven.

## 🔧 Verified engineering (correct, but standard — not discoveries)

| Finding | Status | Note |
|---|---|---|
| MPS state simulator — GHZ to ~1000 sites at χ=2 | 🔧 | Correct, but area-law GHZ at χ=2 is textbook MPS behaviour |
| SU(2) lift + SO(3)→SU(2) homomorphism | 🔧 | Implemented and checked (selftest) |
| Self-test suite (13/13 invariants), multi-seed harness, forensic logging | 🔧 | Genuine research hygiene |

## 🟡 Partial results (real, on limited evidence — confirm before claiming)

| Finding | Status | What's needed to promote it |
|---|---|---|
| **Grad-V reduction** — a trained ~1.2M-param MLP update ≈ analytic gradient of V(h)=−logsumexp(β·cos(h,Aₖ)), no accuracy loss (~82% vs 82%, one checkpoint) | 🟡 | multi-seed + MNLI; relate to modern-Hopfield/DEQ literature |
| **Collapse-engine ablation** — contributes +4 to +9 pts over a dummy on SNLI (one ablation, 3 checkpoints) | 🟡 | transformer-vs-carrier test (train head on h₀ only) |

## ❌ Falsified (claimed during the year; tested; did not hold)

| Claim | Verdict |
|---|---|
| "Geometry beats standard methods at NLI" | ❌ At chance (~29–34%) on ANLI; below the ~37–41% bag-of-words bar. SNLI's ~76% was hypothesis-artifact leakage, not reasoning. |
| "R(5,5) ≥ 44 via Paley(43)" | ❌ False — 316+1064 monochromatic K₅s; 43≡3 (mod 4) so no Paley graph exists. Re-proven. |
| ΣSW(7) = 3024 | ❌ Typo; correct value 2646. |
| Cortex v1 "100% vs 30% retrieval" | ❌ Rigged mock (answer key hard-coded); real geometry 60%, loses to IDF (70–80%). |
| "Operationally exotic" / "500-site computer (new hardware)" | ❌ Overstated; a correct classical simulator of small-n state vectors. |
| ×3 numerology → R(5,5)=54 | ❌ Falsified (proven R(5,5) ≤ 48). |
| Ramsey power-law "field geometry" | ❌ Cannot hold; diagonal Ramsey growth is exponential. |

---

## One-line summary

Seven proven mathematical results (codec, classes, weight law, the 24-group,
conservation, orientation-independence, hierarchy), solid-but-standard
engineering, two partial NLI results worth chasing, and a set of overclaims that
were tested and retired. The math is the keeper. The "beats AI" story is not.
Both are kept here on purpose — that is what makes the repo trustworthy.
