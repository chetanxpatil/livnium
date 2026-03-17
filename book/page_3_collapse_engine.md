# Page 3 — The Collapse Engine

## Chapter 1: The File

```
system/snli/model/core/vector_collapse_engine.py
```

This is the largest and most important file in the system. The `VectorCollapseEngine` class is the machine that turns `h0` into `h_final`. Everything else — the encoder, the head, the basin field — feeds into or out of this engine. Understanding this file is understanding Livnium.

---

## Chapter 2: Initialisation — What the Engine Holds

```python
class VectorCollapseEngine(nn.Module):
    def __init__(self, dim=256, num_layers=6, ...):
```

The engine has six categories of parameters and state:

**1. The state update network**
```python
self.update = nn.Sequential(
    nn.Linear(dim, dim),
    nn.Tanh(),
    nn.Linear(dim, dim)
)
```
A two-layer MLP that takes the current `h` and produces a learned correction `delta`. This is not the physics — it's the model's learnable residual on top of the physics. It learns to make small adjustments that the pure force equations can't handle alone.

**2. The three static anchors**
```python
self.anchor_entail  = nn.Parameter(torch.randn(dim))
self.anchor_contra  = nn.Parameter(torch.randn(dim))
self.anchor_neutral = nn.Parameter(torch.randn(dim))
```
These are learned vectors — the positions of the three gravitational wells in the embedding space. They start random and shift during training to positions that best separate E, N, and C.

**3. The virtual null endpoint (v8)**
```python
if strength_null > 0.0:
    self.anchor_null = nn.Parameter(null_init)
```
A fourth attractor with no label attached. Enabled with `--strength-null 0.03`. Gives genuinely ambiguous or confusing samples somewhere to go that is *not* Neutral. Prevents Neutral from becoming a catch-all dumping ground for hard cases.

**4. The adaptive metric (v9)**
```python
if adaptive_metric:
    self.metric_log_scale = nn.Parameter(torch.zeros(dim))
```
A learned vector of 256 scale factors. When enabled, alignment is computed in a warped space where the model has stretched dimensions that help separate classes. Some dimensions of the embedding space matter more for NLI than others; the metric learns which ones.

**5. The rotation matrix (v6)**
```python
if rot_rank > 0:
    self.W_rot_U = nn.Parameter(torch.randn(dim, rot_rank) * 0.01)
    self.W_rot_V = nn.Parameter(torch.randn(dim, rot_rank) * 0.01)
```
A low-rank skew-symmetric matrix that adds a *circulating component* to the trajectory. Instead of falling straight toward an anchor, `h` spirals in. This prevents different samples' trajectories from colliding and merging into undifferentiated blobs. Enabled with `--rot-rank 8`.

**6. Locking zone parameters (v7)**
```python
self.lock_threshold = lock_threshold
self.lock_gain = lock_gain
self.lock_temp = lock_temp
```
Once `h` is clearly inside a basin (alignment > `lock_threshold`), the attractive force is amplified. The basin "grabs" the state more firmly once it's inside, creating a capture zone. Prevents borderline cases from oscillating at the boundary.

---

## Chapter 3: The Three Collapse Modes

The engine has three distinct operating modes, each for a different situation.

### Mode 1: `collapse(h0)` — Static inference

Used during evaluation and baseline training. Three fixed anchors, no dynamic basins.

```python
def collapse(self, h0):
    return self._collapse_static(h0)
```

The simplest mode. All samples feel the same three forces from the same three anchors. Fast, stable, deterministic.

### Mode 2: `collapse_dynamic(h0, labels, basin_field)` — Training with dynamic basins

Used during training when dynamic basins are enabled (the default).

```python
def collapse_dynamic(self, h0, labels, basin_field, global_step, ...):
```

Each sample is routed to the basin of its *true label* (entailment to an E basin, contradiction to a C basin, etc.). The basin centers adapt after each batch. New basins can spawn when the current ones don't fit well. This is where the geometry actively grows during training.

Note a critical design decision: `collapse_dynamic` uses the *true label* to route. This means it "knows" the answer during training. So the actual gradient loss uses a different path:

### Mode 3: `collapse_inference(h0, basin_field)` — Label-free routing

Used for the forward pass that produces gradients during training (with dynamic basins).

```python
def collapse_inference(self, h0, basin_field):
```

No labels. Routes each sample to the best-matching basin using the current state vector alone. Also uses **bidirectional routing**: it considers both `h0` and `-h0` (the reversed direction), because a contradiction pair looks like entailment if you flip premise/hypothesis. The forward direction attracts; the reverse direction repels.

The training loop is:
```
Step 1 (no grad): collapse_dynamic  → update basin geometry using true labels
Step 2 (with grad): collapse_inference → forward pass for loss, no labels
```

This closes the train/eval gap. The model learns to work without knowing the answer.

---

## Chapter 4: Inside `_collapse_static` — Step by Step

This is the core loop. At each of `num_layers` (default 6) steps:

```python
for step in range(self.num_layers):

    # 1. Normalize h into the metric space
    h_n = self._metric_normalize(h)    # applies learned scale if adaptive_metric=True

    # 2. Compute alignment with each anchor
    a_e = (h_n * e_dir).sum(dim=-1)   # cos(h, anchor_entail)
    a_c = (h_n * c_dir).sum(dim=-1)   # cos(h, anchor_contra)
    a_n = (h_n * n_dir).sum(dim=-1)   # cos(h, anchor_neutral)

    # 3. Compute divergence and tension (from physics_laws.py)
    d_e = 0.38 - a_e
    d_c = 0.38 - a_c
    d_n = 0.38 - a_n

    # 4. Compute neutral boundary proximity
    ec_boundary = exp(-4 * (a_e - a_c)²)   # high when equidistant from E and C

    # 5. Learned residual
    delta = self.update(h)

    # 6. Direction vectors (from anchor TOWARD h, so subtracting = moving toward anchor)
    e_vec = normalize(h - anchor_entail)
    c_vec = normalize(h - anchor_contra)
    n_vec = normalize(h - anchor_neutral)

    # 7. Apply all forces
    h = h + delta
        - s_E * d_e * e_vec
        - s_C * d_c * c_vec
        - s_N * d_n * n_vec
        - s_boost * ec_boundary * n_vec   ← extra neutral pull at E-C boundary

    # 8. Optional: locking zone (amplify force once h is inside basin)
    # Optional: rotational dynamics (spiral trajectories)
    # Optional: null endpoint force

    # 9. Soft norm control — prevent h from growing too large
    if |h| > 10.0:
        h = h * (10.0 / |h|)
```

Six iterations of this loop. The state moves from its starting position toward whichever basin has the strongest pull given the actual sentence pair geometry.

---

## Chapter 5: Locking Zones

```python
if self.lock_threshold > 0.0:
    gate_e = sigmoid((a_e - lock_threshold) / lock_temp)
    s_e = strength_entail * (1 + lock_gain * gate_e)
```

When `cos(h, anchor_entail)` exceeds `lock_threshold`, the sigmoid gate activates and multiplies the entailment force by `(1 + lock_gain * gate)`. With `lock_gain=2.0`, the force can triple once the state is clearly inside the basin.

This creates two zones:
- **Outside the basin** (alignment < threshold): normal fluid dynamics, state can flow freely
- **Inside the basin** (alignment > threshold): high-gain capture, state is held firmly

Why does this help? Without locking, borderline cases (where E and C forces are nearly equal) can oscillate indefinitely near the boundary without settling. The lock zone ensures that once a state tips even slightly toward one basin, that basin wins decisively.

---

## Chapter 6: Rotational Dynamics

```python
S = W_rot_U @ W_rot_V.T - W_rot_V @ W_rot_U.T   # antisymmetric, rank-2k
h += rot_strength * (h @ S)
```

`S` is a skew-symmetric matrix (S = -Sᵀ). Multiplying by a skew-symmetric matrix rotates a vector — it adds a component orthogonal to `h` that circles around rather than pointing toward or away from any anchor.

Why does this help? Without rotation, all trajectories for E samples converge straight toward the entailment anchor. All those straight-line paths tend to collapse into a single region, making the basin dense and undifferentiated. Rotation spreads trajectories out — they spiral in, arriving from slightly different directions and populating a wider region of the basin.

The analogy from physics is angular momentum: planets don't fall straight into the sun, they orbit. The rotation parameter gives h a bit of orbital momentum so it doesn't just tunnel into the nearest anchor.

---

## Chapter 7: The Virtual Null Endpoint

```python
if strength_null > 0.0:
    a_null = (h_n * null_dir).sum(dim=-1)
    d_null = divergence_from_alignment(a_null)
    h = h - strength_null * d_null * normalize(h - null_dir)
```

This is the fourth attractor — position is learned, but it has *no label*. Nothing trains it to correspond to E, N, or C. It just occupies space in the embedding geometry.

The problem it solves: without it, every sample must fall into E, N, or C. But some sentence pairs are genuinely nonsensical or ambiguous — they don't cleanly belong anywhere. These samples used to drift into Neutral because that was the least-bad option. Neutral then became a catch-all "I don't know" bucket.

With the null endpoint, ambiguous samples have a fourth option — a region of space that corresponds to "unclassifiable." This keeps Neutral clean (it only attracts true neutral cases) and improves the overall geometry.

---

## Chapter 8: The SNLIHead

```
system/snli/model/tasks/snli/head_snli.py
```

After the engine produces `h_final`, classification happens in the head:

```python
class SNLIHead(nn.Module):
    def forward(self, h_final, v_p, v_h):
        # Directional signals from the original premise/hypothesis vectors
        diff   = v_h - v_p          # same as h0, before noise and collapse
        prod   = v_h * v_p          # element-wise product (captures agreement)

        # Concatenate state + geometry signals
        x = cat([h_final, diff, prod])

        # Classify
        logits = self.classifier(x)  # Linear → ReLU → Linear → 3
        return logits
```

The head doesn't just use the collapsed state. It also feeds in the original difference and product of the premise/hypothesis vectors. The difference captures directionality, the product captures where the two sentences agree. Together with `h_final` (which has been shaped by the physics to reflect the relationship type), the head has rich geometric information to work with.

---

*Next: Page 4 — The Basin Field (dynamic basins, how they spawn, update, merge, and die)*
