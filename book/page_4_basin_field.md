# Page 4 — The Basin Field

## Chapter 1: The Problem with Three Fixed Anchors

The simplest version of Livnium has exactly three attractors — one for E, one for N, one for C. Every entailment sample feels the same E anchor. Every contradiction sample feels the same C anchor.

This works. But it hits a ceiling.

The reason: *entailment is not one thing*. There are many geometrically distinct ways a hypothesis can follow from a premise:

- "A dog is running" → "An animal is moving" (categorical generalisation)
- "The man is sleeping" → "The man is not awake" (negation)
- "The woman bought three apples" → "The woman bought some fruit" (quantity abstraction)

In embedding space, these three sentence pair types produce very different `h0` vectors. But the static collapse engine routes all of them to the same single `anchor_entail` point. The single anchor has to simultaneously attract all these different geometric regions. It ends up as a compromise — not perfectly tuned to any of them.

The Dynamic Basin Field solves this by allowing **multiple attractors per label**.

---

## Chapter 2: The File

```
system/snli/model/core/basin_field.py
```

And the simpler vectorised version used during embedding pretraining:

```
system/snli/embed/basin_field.py
```

---

## Chapter 3: Structure of the BasinField

```python
class BasinField:
    def __init__(self, max_basins_per_label=64):
        # Centers: (3 labels, 64 max basins, dim)
        self.centers = torch.zeros(3, max_basins_per_label, dim)

        # Which basins are currently active
        self.active = torch.zeros(3, max_basins_per_label, dtype=torch.bool)

        # How many samples have been routed to each basin
        self.counts = torch.zeros(3, max_basins_per_label, dtype=torch.int32)

        # Last training step each basin was used
        self.last_used = torch.zeros(3, max_basins_per_label, dtype=torch.int32)
```

Three labels (0=E, 1=C, 2=N), up to 64 basins each = **192 possible attractors** total. At the start of training, all slots are inactive. Basins are added as training proceeds and the geometry needs them.

---

## Chapter 4: Routing — How a Sample Finds Its Basin

When a sample arrives with label E, the system looks at all currently active E basins and finds the one whose center is most aligned with the current state:

```python
# h: current state vector (dim,)
# Find best E basin
sims = dot(normalize(h), centers[E_label])   # cosine sim with all active E centers
best_basin = argmax(sims)
alignment  = sims[best_basin]
```

The best basin becomes the *target* for that sample's collapse. All force from the E anchor during that sample's collapse step points toward this specific basin center, not the global `anchor_entail`.

This is routing. Each sample finds *its* micro-region of the E space, not just "E" in general.

---

## Chapter 5: Spawning — When a New Basin is Born

After routing, the system checks whether the assigned basin is a good fit:

```python
tension   = |BARRIER - alignment|
spawn_ok  = (tension > tension_threshold) AND (alignment < align_threshold)
```

In words: **tension is high** (the sample is far from equilibrium in the assigned basin) **AND alignment is low** (the sample isn't even pointing toward that basin very well).

This combination means: "I have been assigned to a basin but that basin doesn't represent me well. I need my own region."

When the condition triggers, a new basin is created at the sample's current position:

```python
def add_basin(self, label_idx, vector, step):
    # Find a free slot
    idx = first_inactive_slot(label_idx)
    # Store the normalized vector as the new basin center
    self.centers[label_idx, idx] = normalize(vector)
    self.active[label_idx, idx] = True
    self.counts[label_idx, idx] = 0
    self.last_used[label_idx, idx] = step
```

The new basin is placed exactly where the misfit sample is. Future samples of the same label that are geometrically similar will route to this new basin.

Over training, the E basin field builds up a collection of micro-attractors that each cover a different geometric sub-type of entailment. Same for N and C. The field grows from 0 basins to potentially 64 per label.

---

## Chapter 6: Updating — How Basin Centers Move

After collapse, the basin centers adapt to the samples that routed to them:

```python
# EMA update: move the center slightly toward where the samples ended up
new_center = (1 - lr) * old_center + lr * mean(h_final for samples in this basin)
basin.center = normalize(new_center)
```

`lr` is `basin_anchor_lr` (default 0.05) — a slow exponential moving average. The basin center doesn't jump to each new sample; it drifts gradually toward the average position of its assigned samples. This prevents a single outlier from moving the center dramatically.

The normalisation is critical. All basin centers live on the unit hypersphere. The geometry is purely angular — distance in the basin field is measured in terms of direction, not magnitude.

---

## Chapter 7: Pruning and Merging

Over time, the basin field can accumulate:
- **Dead basins**: spawned early, rarely used since
- **Redundant basins**: two basins very close together covering the same region

Pruning and merging cleans this up (triggered every N steps with `--basin-prune-every`):

**Pruning:**
```python
# Remove basins that haven't accumulated enough samples
if basin.count < min_count:
    deactivate(basin)
```

A basin that was spawned but never attracted many samples probably represents noise, not a real sub-type. It gets removed.

**Merging:**
```python
# Merge pairs of basins that are too similar
for each pair (i, j) of active basins in the same label:
    if cosine_sim(center_i, center_j) > merge_threshold:
        # Weighted average by usage count
        new_center = (count_i * center_i + count_j * center_j) / (count_i + count_j)
        replace center_i with normalize(new_center)
        deactivate basin_j
```

Two basins that ended up pointing almost the same direction are redundant. Merging them frees a slot for future spawning and keeps the geometry clean.

---

## Chapter 8: Neutral Seeding at the E-C Boundary

There is one special spawning rule that only applies to the neutral basin:

```python
# After each batch, check where the final states landed
boundary_score = boundary_proximity(alignment_to_E, alignment_to_C)

# If a state is sitting on the E-C boundary after collapse...
if boundary_score > threshold:
    # Seed a neutral basin at this position
    basin_field.add_basin(N_label, h_final, step)
```

The geometric intuition: the E-C boundary is the region equidistant from the entailment and contradiction attractors. In the geometry of NLI, this is precisely where neutral cases live — pairs where the premise neither confirms nor denies the hypothesis are pulled equally toward E and C, ending up in between.

By planting neutral basin seeds at positions discovered to be on the E-C boundary during training, the neutral field learns to populate exactly that geometric region. The neutral basins cluster around the E-C midplane, which is where they belong.

---

## Chapter 9: Static vs Dynamic — When to Use Which

The system supports both modes. The choice is `--disable-dynamic-basins` to turn off dynamic basins and use only the three fixed anchors.

| Mode | Basins | Speed | Accuracy | Use when |
|---|---|---|---|---|
| Static | 3 (fixed) | Faster | Lower ceiling | Debugging, baseline |
| Dynamic | up to 192 (adaptive) | Slightly slower | Higher ceiling | Full training runs |

The training loop with dynamic basins runs two collapse passes per batch (see Page 3, Chapter 3) — one to update basin geometry, one for the loss gradient. This makes each batch slightly more expensive but produces significantly better geometry.

The current 76.32% baseline used dynamic basins with the pretrained BoW encoder. The BERT-based runs will also use dynamic basins by default.

---

## Chapter 10: The Embed/ Version vs The Model/ Version

There are two implementations of the basin field:

**`system/snli/embed/basin_field.py`** — vectorised, used during embedding pretraining. Simpler, slightly faster. Does routing in bulk for whole batches per label group.

**`system/snli/model/core/basin_field.py`** — fuller implementation, used during SNLI training. Supports per-sample routing, the neutral boundary seeding, the full anchor object model, richer tracking.

Both implement the same conceptual thing — a collection of per-label attractors that spawn, adapt, and prune — but at different levels of sophistication. The embed/ version shaped the word embeddings. The model/ version shapes the sentence-pair representations during the actual NLI task.

---

*Next: Page 5 — The Encoder Story (56% → 76% → BERT, what changed and why)*
