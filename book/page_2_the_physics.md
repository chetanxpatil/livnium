# Page 2 — The Physics

## Chapter 1: The File That Runs Everything

```
system/snli/model/core/physics_laws.py
```

This file is tiny — under 50 lines. But every single force in the system flows through it. Three functions:

```python
BARRIER = 0.38

def divergence_from_alignment(alignment):
    return BARRIER - alignment

def tension(divergence):
    return divergence.abs()

def boundary_proximity(a_e, a_c):
    return torch.exp(-4.0 * (a_e - a_c).pow(2))
```

That's the entire physics layer. Let's unpack each one.

---

## Chapter 2: Alignment

Alignment is the cosine similarity between the current state vector `h` and an anchor direction `A`:

```
alignment = cos(h, A) = dot(normalize(h), normalize(A))
```

It ranges from **-1** (pointing exactly opposite) to **+1** (pointing exactly the same direction).

The three anchors are `anchor_entail`, `anchor_contra`, `anchor_neutral` — learned vectors in the same space as `h`. During training they shift to positions that best separate the three classes.

When a sample is *entailment*, its h₀ should be close to `anchor_entail`, meaning `cos(h, anchor_entail)` should be high. When it's *contradiction*, close to `anchor_contra`, and so on.

---

## Chapter 3: Divergence — The Core Force

```python
divergence = BARRIER - alignment
```

This is the heart of Livnium.

`BARRIER` is set to **0.38** by default. What does that number mean?

It means: the equilibrium point for each basin is at **cos(h, A) = 0.38**, which is approximately 67.6 degrees off-axis from the anchor. The state vector doesn't have to point *exactly* at the anchor to be in equilibrium. It rests at a specific angular distance from it.

The force this divergence produces:

| Situation | Alignment | Divergence | Force direction |
|---|---|---|---|
| Far from anchor | cos = 0.0 | div = +0.38 | Strong pull toward anchor |
| At equilibrium | cos = 0.38 | div = 0 | No force |
| Too close | cos = 0.6 | div = -0.22 | Push away from anchor |
| Pointing at anchor | cos = 1.0 | div = -0.62 | Strong push away |

This is crucial: **the attractor doesn't try to perfectly align h with its anchor**. It creates a shell at radius 0.38. States inside the shell (too close) get pushed out. States outside (too far) get pulled in. The stable orbit is on the shell.

Three such shells in 256-dimensional space, one per label. The collapse dynamics drive h onto the shell of the correct basin.

---

## Chapter 4: Why 0.38?

You can change this with `--barrier` in train.py:

```bash
python train.py --barrier 0.38   # default
python train.py --barrier 0.0    # equilibrium = orthogonality (cos = 0)
python train.py --barrier 1.0    # equilibrium = perfect alignment
```

**barrier = 0** means the system tries to make `h` orthogonal to each anchor it doesn't belong to. Hard to achieve, often unstable.

**barrier = 1** means the system tries to make `h` perfectly aligned with its anchor. Forces are strongest far away, weakest when nearly aligned — can cause basins to collapse on top of each other.

**barrier = 0.38** is empirically chosen. It's large enough that the three anchor shells have room to be distinct (they sit far enough off-axis to not overlap), but small enough that the forces remain strong throughout training. It was found through grid search and kept because it consistently produces clean basin separation.

The key property: at `cos = 0.38`, the three anchor directions (which are distributed roughly tetrahedral in high-D space) can each maintain a separate shell without the shells intersecting.

---

## Chapter 5: Tension

```python
tension = |divergence| = |BARRIER - alignment|
```

Tension is simply how far the state is from its equilibrium. It's always non-negative.

Tension is used in two places:

**1. Basin spawning decisions**

When training with dynamic basins, the system checks:
```python
if tension > basin_tension_threshold AND alignment < basin_align_threshold:
    spawn a new basin
```

High tension + low alignment means: the state is being pulled hard but isn't getting close. Something is wrong with the geometry — the current basin doesn't fit this state well. Spawn a new one.

**2. Diagnostics / trace**

The tension at each collapse step is logged in the trace. You can watch it over training: early on, tension is high everywhere (the system is far from equilibrium). As training progresses, tension drops for correctly classified samples (they settle into their basins) and stays high only for ambiguous or hard cases.

---

## Chapter 6: Boundary Proximity

```python
boundary_proximity(a_e, a_c) = exp(-4 * (a_e - a_c)²)
```

This function returns a value near **1** when the state is equidistant between entailment and contradiction (i.e., `a_e ≈ a_c`), and near **0** when it's clearly closer to one of them.

It's used to give neutral a boost at the E-C boundary:

```python
force_on_h += strength_neutral_boost * boundary_proximity(a_e, a_c) * force_toward_neutral
```

The intuition: if a sentence pair is ambiguous between entailment and contradiction, it's geometrically sitting on the boundary between those two basins. That position is exactly where neutral cases live. So the system applies an extra pull toward the neutral anchor specifically in that region.

Without this, the system would occasionally let E-C boundary cases fall randomly into entailment or contradiction instead of resolving to neutral. With it, the neutral basin "claims" that territory.

---

## Chapter 7: The Force Equation in Full

Putting it all together, the update at each collapse step is:

```
h_{t+1} = h_t
         + delta(h_t)                                    ← learned residual update
         - s_E * divergence(h_t, anchor_E) * dir_E       ← entailment force
         - s_C * divergence(h_t, anchor_C) * dir_C       ← contradiction force
         - s_N * divergence(h_t, anchor_N) * dir_N       ← neutral force
         - s_boost * boundary_proximity * dir_N          ← extra neutral at E-C boundary
```

Where `dir_X = normalize(h - anchor_X)` is the unit vector pointing from anchor X toward h (so subtracting it moves h toward X).

`delta(h_t)` is a small learned network (Linear → Tanh → Linear) that adds a learned residual — it can steer trajectories in ways the pure physics forces can't, essentially giving the model a "correction term" it learns from data.

---

## Chapter 8: The Barrier as Implicit Quantisation

This is the theoretical insight from the architecture.

Because the system has discrete stable shells (not a continuous gradient pointing everywhere), the dynamics produce **emergent categorical behaviour from continuous equations**. The state doesn't smoothly interpolate between E, N, and C. It moves and then *snaps* — rapidly converges onto one shell.

This is what makes the trace readable. Once h is inside a basin's attraction zone, it accelerates toward that shell and stabilises. The transition between "being attracted to E" and "being attracted to C" is sharp, not gradual.

Formally: the system is a **piecewise dynamical system with discrete fixed points**. The fixed points are the three shells. Training shapes the anchors so that the correct sentences collapse to the correct shells.

This is different from a classifier learning a boundary. The boundary here is not learned directly — it *emerges* from the geometry of three competing force fields.

---

*Next: Page 3 — The Collapse Engine (VectorCollapseEngine, locking zones, rotation, and the null endpoint)*
