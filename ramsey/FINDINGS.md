# Livnium → Ramsey: verified findings

Two results connecting the cube core to Ramsey theory. Both are **exhaustively
verified** (every clique checked, no sampling) and reproducible from the scripts
in this folder. Honest scope is stated for each.

## 1. Cayley graphs on cube symmetry reconstruct Ramsey lower bounds

A 2-coloring with no forbidden monochromatic clique is a Ramsey lower-bound
witness. Building such colorings as **Cayley graphs on a group** collapses the
search from `2^(n choose 2)` edges to a few connection-set bits. Three verified
rungs (`cayley_cube_ramsey.py`):

| Ramsey | n = R−1 | group | construction | verified |
|---|---|---|---|---|
| R(3,3)=6 | 5 | Z₅ | pentagon C₅ | red/blue K₃ = 0/0 |
| R(4,4)=18 | 17 | Z₁₇ | Paley(17) | red/blue K₄ = 0/0 |
| **R(4,5)=25** | **24** | **cube rotation group (S₄)** | Cayley, \|S\|=10, 10-regular | red K₄ = 0, blue K₅ = 0 |

The R(4,5) rung is the structural one: the cube's rotation group has order 24,
and R(4,5)−1 = 24, so the 24 rotations *are* the vertex set. The group action
reduced the search to 16 connection-set bits; a verified (no red K₄, no blue K₅)
coloring was found and checked over all 10,626 K₄ and 42,504 K₅ subsets.

**Honest scope:** these *reconstruct known exact values*; not new bounds. The cube
group is *a* working order-24 group, not provably the only one. The win is that
the cube's own symmetry does real, verified Ramsey work — a structural link, not a
value coincidence.

## 2. Recursive conservation = an exact conserved sum-tree (measured niche)

The recursive/nested geometry implements an exact **conserved sum-tree** over
`27^L` leaves: each node = sum of its 27 children. Benchmarked against four
standard structures (`recursive_sumtree_bench.py`, n = 27⁴ = 531,441):

| structure | global | aligned region | arbitrary region | update |
|---|---|---|---|---|
| naive flat | 127µs | 0.71µs | 93µs | total O(n) |
| flat+cached total | **0.02µs** | 0.71µs | 93µs | O(1) |
| prefix-sum | 0.11µs | 0.11µs | **0.11µs** | O(n) rebuild |
| Fenwick/BIT | 0.82µs | 1.11µs | 1.02µs | O(log n) |
| **recursive 27-tree** | 0.07µs | **0.07µs** | 55µs | **O(depth)=4** |

**Verified capability:** O(1) global and *aligned-regional* conserved queries,
O(depth) updates, ~27/26 ≈ 1.038× memory overhead (measured 0.0385×).

Update performance is *measured*, not just complexity-labelled:

| | point update (µs) | mixed 5000×(update + aligned query) |
|---|---|---|
| prefix-sum | 435 (O(n)) | 218 ms |
| Fenwick/BIT | 2.4 (O(log n)) | 18.7 ms |
| **recursive 27-tree** | **1.2 (O(depth))** | **9.9 ms** |

On the update-heavy + aligned-region workload the 27-tree wins: ~2× over Fenwick
(no log factor — a depth-4 path, aligned region = one stored node) and ~22× over
prefix-sum (which can't update cheaply). `flat+cached` updates faster (0.15µs) but
cannot answer aligned-region queries cheaply, so it is not a competitor here.

**Honest scope:** this is the segment-tree / quadtree / Fenwick capability,
enabled by conservation (`node = Σ children`). Its genuine niche is
**update-heavy, hierarchically-aligned, conserved multiscale aggregation** — there
it beats prefix-sum (can't update) and Fenwick (pays a log on aligned blocks).
Tied by a cached scalar on the bare global total; beaten by prefix-sum on
arbitrary non-aligned regions. A real data-structural niche matched to the cube's
aligned geometry — **not a universal engine**, but the right engine for its own
query pattern.

## Verdict

The recursive layer is **not a universal accelerator**. It is an exact conserved
sum-tree matched to a 27-ary nested geometry. Its real niche is update-heavy,
hierarchically aligned, conserved multiscale aggregation: O(1) global/aligned-region
queries, O(depth) updates, and ~3.8% memory overhead. It loses to prefix-sum on
arbitrary regions and ties a cached scalar on bare totals, but it wins where the
cube geometry naturally asks aligned questions.

The recursion did not create magic. It created **native addressability for its own
geometry**. That is the claim worth keeping.

## Reproduce

```bash
python3 cayley_cube_ramsey.py        # ladder + R(4,5)>=25 on the cube group, exhaustively verified
python3 recursive_sumtree_bench.py   # 5-structure aggregation benchmark
```
