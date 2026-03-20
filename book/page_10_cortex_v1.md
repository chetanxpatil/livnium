# Page 10 — Cortex v1

## Chapter 1: A Different Kind of Machine

Everything in Pages 1–9 was about classification. Sentence pairs go in, labels come out. The dynamics are elegant, the physics is real, but the task is narrow: three categories, one dataset, one kind of reasoning.

Cortex v1 asks a different question: *what if Livnium's geometric principles could govern something larger than a classifier?*

The answer is a machine that reads text and processes it through a quantum-inspired simulator — not to classify it, but to *filter* it. To decide, word by word, what matters and what does not. To build a structured representation of a document that preserves the important parts and discards the noise, guided by the same geometric signals that Livnium uses to separate entailment from contradiction.

The tagline that emerged during development captures it precisely:

> **A Deterministic Geometric Controller driving a Stochastic Information Filter.**

The geometric controller is a 3×3×3 cubic lattice with 24 rotational symmetries. The information filter is a Matrix Product State quantum simulator. And the bridge between them — the thing that makes this genuinely new — is a signal called α.

---

## Chapter 2: The Cubic Lattice

The heart of the controller is a lattice: 27 nodes arranged in a 3×3×3 cube. Each node has an integer position `(x, y, z)` where x, y, z ∈ {0, 1, 2}. Each node has a scalar weight, initially set to its distance from the center of the cube.

```
Node (0,0,0): corner   → weight = √3 ≈ 1.73
Node (1,1,1): center   → weight = 0
Node (1,0,0): face     → weight = √2 ≈ 1.41
Node (1,1,0): edge     → weight = 1
```

The lattice is a geometric object — it has spatial structure, symmetry, and an invariant. The invariant is the sum of all squared weights:

```
ΣSW = Σᵢ wᵢ² = 486
```

This number does not change under any of the 24 rotations of the cube. It is the lattice's "energy" — a global property that is conserved by every symmetry operation.

Why a cube? Because the rotation group of the cube is the largest discrete rotation group in three dimensions that is also the symmetry group of a regular polyhedron. It has exactly 24 elements, cleanly classified into four conjugacy classes:

| Class | Angle | Count | Example |
|---|---|---|---|
| Identity | 0° | 1 | Do nothing |
| Face rotations | 90° | 6 | Quarter-turn around a face axis |
| Vertex rotations | 120° | 8 | Third-turn around a body diagonal |
| Edge rotations | 180° | 9 | Half-turn around an edge midpoint axis |

Every rotation permutes the 27 nodes. The positions change. The weights stay attached to the positions. When you apply a 90° face rotation, the corner nodes move to different corners, the face-center nodes move to different faces, and the center stays put.

The α signal is computed from this permutation.

---

## Chapter 3: The α Signal

After a rotation is applied to the lattice, every node has moved from its original position to a new one. For each node, compute the cosine of the angle between its original position vector and its new position vector (both relative to the cube center):

```python
cos_theta = dot(original_pos - center, new_pos - center) / (|original_pos - center| * |new_pos - center|)
```

Then α is the mean of the absolute values:

```python
α = mean(|cos θᵢ|)   for all 27 nodes
```

α ranges from 0 (the identity rotation, where nothing moves so all cos θ = 1 except the center which is undefined — but the absolute mean is 0 by convention for identity) to approximately 0.84 (for 180° rotations, where half the nodes point in the opposite direction).

The α values fall into clean bands by rotation class:

```
Identity (0°):    α = 0.000
Face (90°):       α ≈ 0.595
Vertex (120°):    α ≈ 0.722
Edge (180°):      α ≈ 0.836–0.841
```

This is not a coincidence. The rotation angle determines how much the lattice is disturbed. Small rotations disturb it a little (low α). Large rotations disturb it a lot (high α). α is a measure of geometric disruption — how far the lattice has moved from its identity state.

The key insight: **α is a geometric quantity with a direct physical interpretation, and it is different for different rotations.** If you map words to rotations, α becomes a per-word signal that measures how much geometric disruption that word causes. High-disruption words (mapped to 180° rotations) get high α. Low-disruption words (mapped to identity or 90°) get low α.

---

## Chapter 4: The Semantic Bridge

The original cortex prototype mapped words to rotations using MD5 hashing — deterministic but semantically meaningless. The semantic bridge replaces this with a geometrically grounded mapping:

```
word → GloVe-50 vector v ∈ R⁵⁰
     → PCA projection v' ∈ R³
     → normalise: n̂ = v'/‖v'‖     (rotation axis on the unit sphere)
     → angle θ = f(word)           (see below)
     → SO(3) matrix M = axis_angle(n̂, θ)
```

Each word gets an axis (from its GloVe embedding projected into 3D) and an angle (from its information content). The axis determines the *direction* of the rotation. The angle determines the *magnitude*.

Two angle assignment strategies:

**IDF mode:** `θ = π × (1 − exp(−IDF(word)))`. Rare, high-information words get θ → π (half-turn, high α). Common function words ("the", "is", "a") get θ → 0 (near-identity, low α).

**Cosine mode:** `θ = arccos(cos(v, corpus_mean))`. Words far from the average concept get large θ. Words near the corpus centroid get small θ.

Both modes produce the same qualitative behavior: semantically important words cause large geometric disruptions (high α), while filler words cause small disruptions (low α).

The critical property that makes this a bridge rather than a mapping: **semantically similar words produce nearly commuting rotations.** If two GloVe vectors are close, their PCA projections yield nearby axes. Rotations around nearby axes nearly commute — applying them in either order gives nearly the same result. This means that semantically coherent passages produce low entropy growth in the MPS, while semantically diverse passages produce high entropy growth.

This is constructive versus destructive interference, driven by actual word meaning.

---

## Chapter 5: The MPS Simulator

The stochastic information filter is a Matrix Product State (MPS) quantum simulator. This is the single most technically ambitious component of cortex v1.

An MPS represents a quantum state of n qubits as a chain of tensors:

```
ψ = A[0] · A[1] · A[2] · ... · A[n-1]
```

Each tensor A[i] has shape `(χ_left, 2, χ_right)` where χ is the bond dimension — the amount of entanglement the representation can hold.

The key numbers:

- **Memory:** O(n × χ²) — linear in the number of qubits, quadratic in the bond dimension. Compare to the full quantum state vector which requires O(2ⁿ) — exponential.
- **Bond dimension χ = 1:** Product state, no entanglement. Trivial.
- **Bond dimension χ = 2:** Can represent GHZ states (maximally entangled in a structured way). This is where cortex v1 lives.
- **Bond dimension χ = 2^(n/2):** Exact representation of any quantum state. Equivalent to the dense vector.

The simulator supports:

- Single-qubit gates: Hadamard, Pauli X/Y/Z, Rx, Ry, Rz, arbitrary unitary
- Two-qubit gates: CNOT (adjacent and non-adjacent via SWAP chains)
- Measurement: Born-rule sampling from the MPS
- Environment computation: correct left and right environment contraction for measurement probabilities

The CNOT implementation is where the MPS machinery earns its keep. When a CNOT is applied between adjacent qubits, the two tensors are contracted into a single 4-index tensor, the CNOT gate is applied, and then the tensor is split back into two using SVD (singular value decomposition). The SVD is where entanglement grows — the number of non-zero singular values is the new bond dimension. If it exceeds `max_chi`, the smallest singular values are truncated.

Every truncation discards information. The truncation error is logged. This is where the information filter acts: the MPS can only hold χ² parameters per bond. When the state requires more entanglement than that, something has to be thrown away.

---

## Chapter 6: The Polarity Governor

The polarity governor is the module that connects α to the MPS. It controls *how aggressively* the MPS truncates its bonds based on the geometric signal from the lattice.

The core formula:

```
effective_S_max(bond i) = S_max_base × (1 + α × polarity(i))
```

Where polarity at bond i is:

```
polarity(i) = 1 − S(i) / S_max_theoretical(i)

S_max_theoretical(i) = min(i+1, n−i−1) × log(2)
```

Polarity measures how far the current entanglement entropy at a bond is below the theoretical maximum. A GHZ state (highly structured, low entropy relative to the theoretical limit) has polarity ≈ 1. A random deep-circuit state (maximal entropy) has polarity ≈ 0.

The governor's logic:

- **High α (semantically important word) + high polarity (structured bond):** The entropy ceiling is raised. The bond survives. Structured information is preserved.
- **Low α (filler word) + low polarity (noisy bond):** The entropy ceiling stays low. The bond is pruned more aggressively. Noise is discarded.
- **High α + low polarity:** The ceiling is raised, but the bond is already near the theoretical limit. The governor relaxes the ceiling but cannot create more physical capacity. α is an "entanglement counsellor, not a dictator."

This last case is the capacity limit — the point where geometry meets physics. The governor can recommend preservation, but if the bond dimension has hit the hard χ cap, there is no room to keep more information. The governor shapes the pruning priorities within the available capacity, not the capacity itself.

---

## Chapter 7: The SO(3) → SU(2) Bridge

One of the most elegant pieces of cortex v1 is the mapping from classical rotations to quantum gates.

The lattice operates in SO(3) — the group of 3D rotations. The MPS operates in SU(2) — the group of 2×2 unitary matrices (single-qubit gates). These two groups are related by the covering map:

```
U(n̂, θ) = cos(θ/2)·I − i·sin(θ/2)·(n̂·σ)
```

Where σ = (σₓ, σᵧ, σᵤ) are the Pauli matrices. Every SO(3) rotation around axis n̂ by angle θ maps to a unique SU(2) matrix (up to a global phase). This mapping is:

- **Homomorphic:** U(A) @ U(B) ≈ U(A @ B) — composing rotations in SO(3) and then mapping is the same as mapping each rotation and composing in SU(2).
- **Verified:** tested on all 20 random rotation pairs, confirmed to machine precision.

This means that when a word is mapped to an SO(3) rotation via the semantic bridge, that rotation can be faithfully converted to a quantum gate and applied to a qubit in the MPS. The geometric content of the rotation (its axis, its angle, its class) is preserved in the quantum domain.

The 24 cube rotations produce 24 distinct SU(2) gates. These gates, applied sequentially to qubits in the MPS as words are processed, build up the quantum state that encodes the document's structure. Semantically similar words apply nearly-commuting gates (constructive interference). Semantically diverse words apply non-commuting gates (destructive interference, entropy growth, potential truncation).

---

## Chapter 8: The Full Stack

Putting it all together, the cortex v1 pipeline for processing a text stream is:

```
Word arrives
  → Semantic bridge: word → SO(3) rotation → SU(2) gate, α signal
  → Lattice: apply rotation, compute new node positions, verify ΣSW = 486
  → Governor: use α and current bond polarities to set entropy ceilings
  → MPS: apply SU(2) gate to next qubit, SVD truncation under governor ceilings
  → Log: which bonds survived, which were pruned, the α value, the polarity

After all words processed:
  → Survivors: the bonds that survived the full stream represent the "important" structure
  → Measurement: sample the MPS to get a binary string (a unique fingerprint of the sequence)
  → Output decoder: report which words caused the most pruning, which bonds are most structured
```

The machine reads text one word at a time. Each word causes a geometric event in the lattice and a quantum event in the MPS. The two events are linked by α. High-α words expand the MPS's capacity to remember. Low-α words contract it. At the end, what survives in the MPS is a compressed, structured representation of the semantically important content.

This is not a classifier. It is a *filter*. It does not output a label. It outputs a quantum state — a compressed encoding of the entire text, shaped by the geometry of Livnium's lattice.

---

## Chapter 9: What the Self-Test Verifies

Cortex v1 ships with a comprehensive self-test suite that verifies 12 invariants:

```
✓ R⁴ = I  for all three generators           — generators are order-4
✓ 24 distinct orientations generated by BFS   — all symmetries found
✓ Class counts: {core:1, face:6, edge:12, vertex:8}  — correct symmetry classes
✓ ΣSW = 486 invariant under all 24 rotations  — lattice energy conserved
✓ Lattice bijection preserved after every rotation  — no nodes lost or duplicated
✓ cos θ signals in [−1, +1], identity → α = 0  — α is well-defined
✓ Generator SO(3) matrices orthogonal, det=+1  — proper rotations
✓ Angle classes: {0°:1, 90°:6, 120°:8, 180°:9}  — correct count per class
✓ All 24 SU(2) gates unitary, all distinct     — quantum gates are valid
✓ Homomorphism: U(A)@U(B) ≈ U(A@B) for 20 random pairs  — mapping is faithful
✓ High α → fewer governor prune events         — α controls pruning as designed
✓ GHZ-15 circuit: 200/200 valid outcomes       — MPS simulator produces correct physics
```

Every invariant is independently testable. If any single one fails, the system is broken in a diagnosable way. This is the engineering standard that Livnium's physics-first approach enables: when every component has a mathematical specification, every component has a testable invariant.

---

## Chapter 10: What Cortex v1 Proves

Cortex v1 is a proof of concept for a specific claim:

> Livnium's geometric principles — attractors, polarity, energy landscapes — can govern systems beyond classification.

The cubic lattice is not an NLI classifier. The MPS is not a sentence encoder. The polarity governor is not a basin field. But the same conceptual framework underpins all of them: a geometric signal (α / cosine alignment) controls a dynamical process (entropy pruning / attractor collapse) that separates meaningful structure from noise.

In the original Livnium, "meaningful structure" means "which of three NLI labels does this sentence pair belong to." In cortex v1, "meaningful structure" means "which words in this text stream carry the most semantic weight."

The vocabulary is different. The mathematics is analogous. The principle is the same: **geometry decides what matters.**

---

*Next: Page 11 — The Benchmark That Beats FIFO and LRU (α-triage retrieval and what it means for memory)*
