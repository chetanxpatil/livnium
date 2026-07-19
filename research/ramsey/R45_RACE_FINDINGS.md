# COMPASS vs WalkSAT vs SA — the real fight: R(4,5) witness search at n=24

This is the experiment that decides whether COMPASS is a real heuristic or just a
faster demo on easy instances. It moves off the trivial R(4,4)/n=17 problem and onto
the **R(4,5) frontier SAT side**: 2-color K₂₄ with `red K4 = 0` and `blue K5 = 0`.
n=24 is satisfiable (R(4,5)=25); n=25 is not, and is used as a negative control.

## Protocol (designed so it cannot become a "nice demo")

1. **Asymmetric target.** red K4 (size 4) and blue K5 (size 5) are different cliques,
   handled separately — not the symmetric R(4,4) shortcut.
2. **100 seeds per solver, equal per-seed wall-clock budget (8s).** Compare
   **solve-rate**, not just median time.
3. **Independent exhaustive verifier.** Every claimed witness is re-checked by
   recomputing red-K4 / blue-K5 counts straight from the coloring (`verify()` in
   `r45_race.py`), not from the incremental sums. Column `fake` counts any case where
   the solver's internal "0 violations" disagrees with the verifier. It is 0 everywhere.
4. **n=25 negative-control trap.** R(4,5) at n=25 is UNSAT, so no solver may produce a
   valid witness. If any did, it would expose a verifier/bookkeeping bug.
5. **Saved witness + fully independent checker.** One COMPASS witness is dumped to
   `witness_n24.json` and re-verified by `independent_check.py`, which shares no code
   with the solver (plain `itertools`, no numpy, no incremental state).

## Result

| solver  |  n | target              | seeds | solved | solve-rate | median | p90 | best viol (unsolved) | fake |
|---------|---:|---------------------|------:|-------:|-----------:|-------:|----:|---------------------:|-----:|
| COMPASS | 24 | red K4=0/blue K5=0  |   100 |     26 |       26%  |  4.61s | 6.78s |                1-2 |    0 |
| WalkSAT | 24 | red K4=0/blue K5=0  |   100 |      1 |        1%  |  6.17s | 6.17s |                1-4 |    0 |
| SA      | 24 | red K4=0/blue K5=0  |   100 |      3 |        3%  |  3.60s | 5.51s |                1-3 |    0 |
| COMPASS | 25 | (UNSAT control)     |    30 |      0 |        0%  |    -   |   -   |                3-8 |    0 |
| WalkSAT | 25 | (UNSAT control)     |    30 |      0 |        0%  |    -   |   -   |               10-18 |    0 |
| SA      | 25 | (UNSAT control)     |    30 |      0 |        0%  |    -   |   -   |                6-10 |    0 |

Independent witness check (`python3 independent_check.py`):
`n=24 seed=0: red K4=0  blue K5=0  -> VALID R(4,5)>=25 WITNESS`

## What this shows

- **The separation is in solve-rate, not just speed.** COMPASS solves **26/100**;
  WalkSAT **1/100**; SA **3/100** under identical budget. That is the real signal the
  earlier R(4,4) test could not give (there WalkSAT also hit 100%, so only time differed).
  Here COMPASS is ~9–26× more likely to find a witness.
- **It is even sharper on near-misses.** COMPASS's unsolved seeds plateau at just
  **1–2** violations; WalkSAT/SA spread wider. COMPASS lands closer even when it fails.
- **No fakes, control clean.** `fake=0` across all 390 runs, and every solver scores
  **0/30** at the UNSAT n=25 control — the verifier and the negative control both hold.

## Honest scope

- This is a **lower-bound witness search**, not a new bound: R(4,5)=25 is known. The
  contribution is heuristic quality, not a new Ramsey number.
- 8s/100-seed is a meaningful but not exhaustive budget; absolute solve-rates would rise
  for all three with more time. The **ratio** between solvers is the claim, and it is large
  and consistent.
- COMPASS's edge is its net-delta "compass" move (flip the edge that fixes the most
  violations) plus locality and branch restarts — a real algorithmic difference, now
  demonstrated where structure actually matters.

**Verdict:** COMPASS separates from canonical WalkSAT and tuned SA on the R(4,5)
frontier by solve-rate, with a clean UNSAT control and an independently verified witness.
This is the result the earlier tiny R(4,4) demo was missing.

## Reproduce

```bash
python3 r45_race.py 24 100 8            # full single-process race (slow; ~tens of min)
# or chunked + parallel (4 workers), resumable, appends to master.csv:
python3 chunk.py 24 COMPASS 0 100 8 master.csv
python3 chunk.py 24 WalkSAT 0 100 8 master.csv
python3 chunk.py 24 SA      0 100 8 master.csv
python3 chunk.py 25 COMPASS 0 30  6 master.csv   # + WalkSAT, SA  (UNSAT control)
python3 independent_check.py            # re-verify the saved witness from scratch
```
