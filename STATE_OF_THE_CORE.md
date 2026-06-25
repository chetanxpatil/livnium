# State of the Core

A measured status of Livnium. Every claim below has a number behind it and a
script that reproduces it. The method throughout: *prune until truth survives* —
take each claim, test it against the dumb baseline, and keep only the version the
data defends. Inflated claims were not deleted; they were **scoped**.

---

## 1. Collapse engine (NLI) — `chat/`

A 5.98M-parameter sequence model trained only on SNLI, no pretrained embeddings.

| claim | measured | status |
|---|---|---|
| parameters | 5.975M (read from checkpoint) | ✓ |
| generative-classifier accuracy | 52.9% on 1,500 SNLI dev pairs (chance 33%) | ✓ |
| CPU latency | ~4.5 ms / reply (NumPy reimpl of trained weights) | ✓ |
| classifier sibling w/ alignment | 74.7% dev / 74.4% test | ✓ |
| collapse dynamics | Lyapunov-stable, non-expansive: monotone energy descent on 100% of 12,000 sampled steps; ~99.8% of directions contracting; worst S_max ≈ 1.007 | ✓ |

**Scope (pruned):** not a chatbot, no understanding. "Attention-free" → *no
transformer self-attention; one lightweight cross-attention alignment step*. Not a
strict global contraction → *Lyapunov-stable and non-expansive*. See
`chat/LYAPUNOV_TEST.md`, `chat/CLAIMS_CHECKPOINT_MAP.md`.

---

## 2. Cube geometry → Ramsey — `ramsey/`

### 2a. Cayley graphs on group symmetry reconstruct lower bounds

Verified Ramsey witnesses built as Cayley graphs, exhaustively checked:

| Ramsey | n | group | verified |
|---|---|---|---|
| R(3,3)=6 | 5 | Z₅ (pentagon) | red/blue K₃ = 0/0 |
| R(4,4)=18 | 17 | Z₁₇ (Paley) | red/blue K₄ = 0/0 |
| **R(4,5)=25** | **24** | **cube rotation group (S₄)** | red K₄ = 0, blue K₅ = 0 |

The cube's rotation group has order 24 = R(4,5)−1, so its 24 rotations *are* the
vertices; symmetry collapsed the search to 16 bits. **Scope:** reconstructs known
exact values, not new bounds; the cube group is *a* working order-24 group.
(`cayley_cube_ramsey.py`)

### 2b. Compass solver — locality + compass + branching

On R(4,4), n=17 (the witness size), raced wall-clock, solutions verified:

| solver | solved | median time |
|---|---|---|
| tuned SA | 3–4 / 8 | ~1.2 s |
| canonical WalkSAT | 8/8 | ~0.25 s |
| **COMPASS** (net-delta move) | **8/8** | **~0.07 s** |

Reach: solves K₄ to n=17, K₅ to n≈35, then hits the glassy wall. **Scope:**
COMPASS is a WalkSAT/min-conflicts variant; its edge is the *net-delta* move rule
(fix-most, not break-least). R(5,5) is **not reachable** by generic search — its
records need algebraic construction + symmetry, not a faster walker.
(`compass_solver.py`)

---

## 3. Recursive conservation = exact conserved sum-tree — `ramsey/`

Forcing each parent node to equal the sum of its 27 children turns the nested
geometry into a self-maintaining cache: aligned sub-region aggregates are read
straight off a macro-node, never by traversing leaves. Benchmarked vs four
standard structures (n = 27⁴ = 531,441):

| structure | global | aligned region | arbitrary region | point update | mixed (update+aligned ×5000) |
|---|---|---|---|---|---|
| flat+cached total | 0.02µs | 0.71µs | 93µs | 0.15µs | (no cheap region) |
| prefix-sum | 0.11µs | 0.11µs | **0.11µs** | 435µs | 218 ms |
| Fenwick/BIT | 0.82µs | 1.11µs | 1.02µs | 2.4µs | 18.7 ms |
| **recursive 27-tree** | 0.07µs | **0.07µs** | 55µs | **1.2µs** | **9.9 ms** |

Memory overhead: 0.0385× (≈ 27/26). **Verdict:** the recursive layer is **not a
universal accelerator** — it is an exact conserved sum-tree matched to a 27-ary
nested geometry. Its niche is update-heavy, hierarchically aligned, conserved
multiscale aggregation: O(1) global/aligned-region queries, O(depth) updates,
~3.8% overhead. It loses to prefix-sum on arbitrary regions and ties a cached
scalar on bare totals, but wins where the cube geometry naturally asks aligned
questions. *The recursion didn't create magic — it created native addressability
for its own geometry.* (`recursive_sumtree_bench.py`, `FINDINGS.md`)

---

## 4. What was pruned (overclaims dropped, with the reason)

- **"amplitude-like computer / 500 qubits"** → the simulator is correct amplitude-like math
  (unitary gates, real entanglement, Born rule) but full state-vector, capped at
  ~25–30 qubits by memory; 500 qubits needs ~5×10¹⁵¹ bytes (≈10⁷¹× the atoms in
  the universe). Kept: a correct small-scale amplitude-like simulator. Dropped: the scale
  and the "advantage" claim.
- **"universal geometry engine"** → kept the measured sum-tree niche; dropped
  "universal."
- **"attention-free"** → kept "no self-attention"; dropped the absolute.
- **"72.7%"** → corrected to the better, real 74.4%.
- **"R(5,5) is reachable"** → it isn't, by generic search; records need structure.

The pattern: each prune traded an unprovable assertion for a smaller,
unassailable, *measured* result.

---

## Reproduce

```bash
# Ramsey
python3 ramsey/cayley_cube_ramsey.py        # Cayley ladder incl. R(4,5)>=25 on cube group
python3 ramsey/compass_solver.py 17 12 4    # COMPASS vs SA vs WalkSAT on R(4,4) witness
python3 ramsey/recursive_sumtree_bench.py   # 5-structure aggregation benchmark
# Collapse engine
python3 chat/verify_lyapunov.py             # Lyapunov / contraction verification
python3 chat/chat_bench.py                  # latency
```
