# Empirical evidence that the collapse decoder is a Lyapunov-compatible driven system

This is **not** a claim that the collapse decoder is a strict global contraction,
and **not** a formal proof of global Lyapunov stability. It is a strong empirical
result on sampled, task-driven trajectories.

Reproduce: `python3 verify_lyapunov.py` (NumPy only; loads
`model/premise_from_hyp_align_53.pt` without torch). Numbers below are from that
run on the shipped 5.98M-param chat checkpoint (`strength=0.438`).

## The energy used

```
V(h) = 1 - cos(h, target)
```

This is a **directional** Lyapunov energy: it measures angular distance to the
target word-well. Its zero set is the **target ray** (every positive multiple of
the target — `target`, `2·target`, `0.5·target` all have cosine 1), or the single
target point only if states are explicitly normalized. So V tracks *direction*,
not magnitude.

## 1. Lyapunov candidate — empirically validated ✅

Across **300 sampled driven trajectories / 12,000 measured steps** (fixed target =
the task), the energy was **monotone non-increasing on 100.00% of steps**:

```
max  dV observed : -7.275e-05     (any positive value = a Lyapunov violation)
mean dV observed : -2.482e-02
```

The maximum single-step change is strictly negative, so on every sampled step the
forced collapse reduced (or held) angular distance to the target. This is an
empirical validation of a Lyapunov **candidate** on sampled trajectories — not an
analytic proof that `ΔV ≤ 0` for all possible `h`.

Convergence note: 0% of trajectories reach `V<1e-3` within 40 steps. Expected —
the step scales with `(1 - cos)`, so the pull weakens near the well: it **eases in
asymptotically** rather than snapping to a fixed point.

## 2. Local spectral behaviour — predominantly contracting, a few mild stretches

Singular values of one collapse step's Jacobian `dF/dh ∈ ℝ²⁵⁶ˣ²⁵⁶`:

| point | dims with S<1 | S_max | S_mean |
|---|---|---|---|
| random | 99.6% (255/256) | ~1.00 | 0.91 |
| near attractor | 100% (256/256) | 0.94 | 0.89 |
| mid (h = 2·target) | 86.3% (221/256) | ~1.00 | 1.00 |

Precision sweep over 150 random driven points (38,400 singular values):

```
worst sampled S_max         : 1.006756
singular values > 1 + 1e-6  : 84 / 38400   (~0.22%)
```

So the honest spectral statement is: the step is **predominantly contracting**
(mean S ≈ 0.89, ~99.8% of sampled directions ≤ 1), but it is **not strictly
non-expansive** — a small fraction (~0.2%) of directions mildly stretch, with the
worst observed spectral norm ≈ **1.007**, not ≤ 1. These few stretching directions
match the "class-separating" stretch dimensions noted for the classifier engine in
`docs/collapse/COLLAPSE_STRUCTURE_REPORT.md`.

## 3. Driven, not static ✅

During real generation of *"two men are playing football"* `[neutral]`, each task
token pulls the state toward its well in one collapse step:

```
word        cos(h,target) before -> after   |h|
two         -0.016 -> +0.685   0.54
men         +0.141 -> +0.759   0.53
are         +0.153 -> +0.765   0.53
playing     +0.208 -> +0.768   0.53
football.   -0.252 -> +0.784   0.47
<eos>       -0.332 -> +0.860   0.45
```

Each step jumps from near-zero/negative alignment to ≈ +0.7–0.86 after the
collapse, and the state norm shifts across steps (std 0.034). The state **reacts
to the task at each step** and the trajectory **moves** — a forced dynamical
process, not a frozen lookup vector.

## Precise public claim

> For a fixed target word-well, the forced collapse step admits a simple
> directional Lyapunov candidate, `V(h) = 1 − cos(h, target)` (zero on the target
> ray). Across 300 sampled driven trajectories and 12,000 measured steps this
> energy was monotone non-increasing on 100% of steps (max ΔV = −7.3e-5). Sampled
> per-step Jacobians were predominantly contracting (mean singular value ≈ 0.89;
> ~99.8% of sampled directions ≤ 1), with a small fraction (~0.2%) of mildly
> expanding directions (worst sampled spectral norm ≈ 1.007).
>
> So the decoder behaves like a **Lyapunov-compatible driven dynamical system**:
> each task token pulls the state toward a target well, the state moves, and no
> measured chaotic expansion occurs. It is **not** a static lookup table, and it is
> **not** a claimed strict global contraction — the precise, defensible property is
> *inspectable, predominantly contracting, task-driven dynamics with an empirically
> validated directional Lyapunov candidate.*

The right headline is not "the model understands." It is: **the decoder has
inspectable, stable dynamics under task forcing** — a specific, defensible claim.
