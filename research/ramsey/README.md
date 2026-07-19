# Ramsey and conserved trees — grade A

This component contains two independently useful results:

1. Cayley graphs on the 24-element cube rotation group reconstruct the known
   `R(4,5) ≥ 25` witness, exhaustively checking every red K4 and blue K5.
2. A 27-ary conserved sum-tree provides O(1) aligned-region queries and O(depth)
   updates in its natural workload.

The COMPASS race adds a solver comparison on the R(4,5) frontier, including an
independent witness checker and an n=25 unsatisfiable control. These are known
Ramsey values, not new mathematical bounds; the contribution is the structural
construction and measured search behavior.

Start with `FINDINGS.md`, then `R45_RACE_FINDINGS.md`.

```bash
cd research/ramsey
python3 cayley_cube_ramsey.py
python3 independent_check.py
python3 recursive_sumtree_bench.py
```
