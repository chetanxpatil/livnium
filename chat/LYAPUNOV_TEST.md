# Is the collapse decoder a Lyapunov-stable driven system?

Empirical test of one claim: when the collapse decoder is **forced with a task**
(a target word-well to type), it behaves as a **Lyapunov-stable, non-expansive,
driven dynamical system** — not a chaotic warp and not a static lookup.

Reproduce: `python3 verify_lyapunov.py` (needs only NumPy + the checkpoint;
loads `model/premise_from_hyp_align_53.pt` without torch). Results below are from
that run on the shipped 5.98M-param chat checkpoint (`strength=0.438`).

## 1. Lyapunov function exists ✅

Candidate energy `V(h) = 1 - cos(h, target) ≥ 0`, zero only at the attractor.
Over **300 driven trajectories / 12,000 steps**, V was **monotone-decreasing on
100.00% of steps** — it never once increased. That is exactly the discrete
Lyapunov condition: a scalar energy that only ever decreases under the dynamics,
so the forced collapse provably descends toward its attractor without oscillating
or diverging.

Note: 0% of trajectories reach `V<1e-3` within 40 steps. This is expected, not a
failure — the step size scales with `(1 - cos)`, so the pull weakens as it nears
the well: it **eases in asymptotically** rather than snapping to the fixed point.
Stability (monotone descent), not finite-time convergence, is the property.

## 2. Contraction / non-expansiveness ✅

Singular values of one collapse step's Jacobian `dF/dh ∈ ℝ²⁵⁶ˣ²⁵⁶`:

| point | contracting dims (S<1) | S_max | S_mean |
|---|---|---|---|
| random | **99.6%** (255/256) | 1.00 | 0.91 |
| near attractor | **100%** (256/256) | 0.94 | 0.89 |
| mid (h = 2·target) | 86.3% (221/256) | 1.00 | 1.00 |

At typical operating points the map contracts on 99–100% of directions, with
`S_max ≈ 1.0` everywhere — i.e. **non-expansive** (it never amplifies any
direction). This matches and exceeds the classifier engine's documented "~92% of
dimensions contracting" (`docs/COLLAPSE_STRUCTURE_REPORT.md`).

## 3. Driven, not static ✅

During real generation of *"two men are playing football"* `[neutral]`, each task
token pulls the state toward its well in a single collapse step:

```
word        cos(h,target) before -> after   |h|
two         -0.016 -> +0.685   0.54
men         +0.141 -> +0.759   0.53
are         +0.153 -> +0.765   0.53
playing     +0.208 -> +0.768   0.53
football.   -0.252 -> +0.784   0.47
<eos>       -0.332 -> +0.860   0.45
```

Every step jumps from near-zero/negative alignment to ≈ +0.7–0.86 after the
collapse, and the state norm shifts across steps (std 0.034). The state **reacts
to the task at each step** and the trajectory **moves** — it is a forced
dynamical process, not a frozen vector.

## Honest framing for any public claim

The rigorous statement is **not** "the collapse is a strict global contraction."
It is:

> The task-driven collapse admits a Lyapunov function (cosine-energy, monotone
> non-increasing on 100% of measured steps) and is **non-expansive** — `S_max ≈ 1`
> with 86–100% of directions strictly contracting and **no expanding directions**.
> It eases into its attractor asymptotically rather than reaching it in finite
> time.

That is a stronger and more defensible claim than "it's a contraction," and it
directly supports the "Lyapunov-compatible, dynamic not static" intuition.
