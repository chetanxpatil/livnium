# Geometry Discriminator Verdict

The Livnium core is a conserved recursive geometry with rotation, phase-like
structure, projection, and nested addressability. Under its native ledger
measurement, it remains classical: values are additive and non-negative, so
destructive cancellation does not appear, and the CHSH scan stays at `S = 2.0`.

The missing ingredient is not the geometry. It is the measurement algebra.
Behavior based on complex amplitudes, cancellation, and Born-style readout
belongs to the separate state-vector/MPS simulator, not the native
conserved-ledger core.

**One-line summary:** Livnium core is a classical conserved geometry with
phase-shaped structure; amplitude behavior requires the separate complex-state
layer.

---

## Measured results

*Reproduce:* `python3 research/archive/experiments/geometry_discriminator_test.py`

A complex-amplitude reference is run alongside the core as a control. It lands on
the textbook values (null = 0, CHSH = 2√2 ≈ 2.8284), which confirms the core's
results are a real property of the core and not an artifact of a weak test.

| Discriminator | Livnium core (native ledger) | Core + complex-amplitude / Born readout | Complex-state reference (control) |
|---|---|---|---|
| **A. Destructive cancellation** — two paths each > 0, together = 0 | **No cancellation** — combined intensity ≥ 1 | **Cancels to 0** | **Cancels to 0** |
| **B. CHSH correlation** — classical bound = 2 | **S = 2.000** — does not exceed | — | **S = 2.8284** — exceeds 2 |

## Why the core stays classical (structural, not a bug)

1. **No cancellation.** The native measurement is the additive, non-negative
   symbolic-weight / occupancy ledger. Two paths deposit non-negative amounts
   that only sum; non-negative quantities never cancel to zero.

2. **CHSH capped at 2.** The most natural shared resource the geometry offers is
   a shared cube rotation read out locally on each side — a local
   hidden-variable model by construction, which is bounded by `S ≤ 2`. The scan
   confirms it sits exactly at 2.000 and cannot be pushed higher.

The decisive observation: keeping the **same cube rotations** but swapping the
**readout** from the additive ledger to a complex-amplitude / Born readout makes
the cancellation appear (column 3). The missing ingredient is the measurement
algebra, not the geometry.

## The phase-shaped structure is real

- The cube rotations `ROT_X / ROT_Y / ROT_Z` have eigenvalues `{1, i, −i}` —
  genuine 4th roots of unity. The geometry carries **latent phase**.
- The complex-amplitude layer already exists in the repo:
  `research/archive/cortex-v2/mps_qudit.py` (Fourier gate, SUM gate, complex tensors). Amplitude
  behavior is the job of that layer, not the conserved-ledger core.
