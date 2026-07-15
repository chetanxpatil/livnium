"""
r45_race.py — the real fight: R(4,5) witness search at n=24.

Goal: 2-color K_n so that  red K4 = 0  AND  blue K5 = 0.
  n=24 is SAT (R(4,5)=25), the frontier SAT side.
  n=25 is UNSAT  -> negative control / verifier trap.

Three local-search solvers raced under EQUAL per-seed wall-clock budget:
  COMPASS  = locality (edges in violated cliques) + net-delta flip + noise + branch restart
  WalkSAT  = canonical: violated clique, zero-break else min-break, with noise
  SA       = simulated annealing on net-delta

Every claimed witness is verified by an INDEPENDENT exhaustive checker that
recomputes red-K4 / blue-K5 counts straight from the coloring (not the
incremental sums) — so an incremental-bookkeeping bug cannot fake a solve.

c[e] in {0,1}: 1 = red, 0 = blue.
  red  K4 violation: all 6 edges of a 4-set are red  (redsum4 == 6)
  blue K5 violation: all 10 edges of a 5-set are blue (redsum5 == 0)

No torch; numpy + itertools only.
Run:  python3 r45_race.py <n> <seeds> <budget_s> [out_prefix]
"""
import itertools, time, sys, json, numpy as np

# ---------------- problem build ----------------
def build(n):
    edges = list(itertools.combinations(range(n), 2))
    eidx = {e: i for i, e in enumerate(edges)}
    K4 = np.array([[eidx[(a, b)] for a, b in itertools.combinations(q, 2)]
                   for q in itertools.combinations(range(n), 4)], dtype=np.int32)   # (.,6)
    K5 = np.array([[eidx[(a, b)] for a, b in itertools.combinations(q, 2)]
                   for q in itertools.combinations(range(n), 5)], dtype=np.int32)   # (.,10)
    E = len(edges)
    inc4 = [[] for _ in range(E)]
    inc5 = [[] for _ in range(E)]
    for ci, ce in enumerate(K4):
        for e in ce: inc4[e].append(ci)
    for cj, ce in enumerate(K5):
        for e in ce: inc5[e].append(cj)
    inc4 = [np.array(x, dtype=np.int32) for x in inc4]
    inc5 = [np.array(x, dtype=np.int32) for x in inc5]
    return E, K4, K5, inc4, inc5

# ---------------- independent exhaustive verifier ----------------
def verify(c, n, K4, K5):
    """Recompute from coloring. Returns (redK4, blueK5). 0,0 == valid witness."""
    redK4  = int((c[K4].sum(1) == 6).sum())
    blueK5 = int((c[K5].sum(1) == 0).sum())
    return redK4, blueK5

# ---------------- incremental state ----------------
def fresh(c, K4, K5):
    rs4 = c[K4].sum(1)          # red-count per K4
    rs5 = c[K5].sum(1)          # red-count per K5
    v = int((rs4 == 6).sum()) + int((rs5 == 0).sum())
    return rs4, rs5, v

def delta_break(e, c, inc4, inc5, rs4, rs5):
    """net change in violations, and break-count (newly-violated), for flipping e."""
    d = -1 if c[e] == 1 else 1           # red->blue is -1
    a4 = rs4[inc4[e]]; n4 = a4 + d
    old4 = (a4 == 6); new4 = (n4 == 6)
    a5 = rs5[inc5[e]]; n5 = a5 + d
    old5 = (a5 == 0); new5 = (n5 == 0)
    net = int(new4.sum() - old4.sum()) + int(new5.sum() - old5.sum())
    brk = int((new4 & ~old4).sum()) + int((new5 & ~old5).sum())
    return net, brk

def apply_flip(e, c, inc4, inc5, rs4, rs5):
    d = -1 if c[e] == 1 else 1
    rs4[inc4[e]] += d
    rs5[inc5[e]] += d
    c[e] ^= 1

def viol_cliques(rs4, rs5, K4, K5):
    """return list of edge-index arrays for currently-violated cliques."""
    out = []
    for ci in np.where(rs4 == 6)[0]: out.append(K4[ci])
    for cj in np.where(rs5 == 0)[0]: out.append(K5[cj])
    return out

# ---------------- solvers ----------------
def compass(E, K4, K5, inc4, inc5, seed, budget, noise=0.25, branch=2000):
    rng = np.random.default_rng(seed); t0 = time.time()
    c = rng.integers(0, 2, E); rs4, rs5, v = fresh(c, K4, K5); best = v; it = 0
    while time.time() - t0 < budget:
        if v == 0: break
        viols = viol_cliques(rs4, rs5, K4, K5)
        es = viols[int(rng.integers(0, len(viols)))]
        if rng.random() < noise:
            e = int(es[int(rng.integers(0, len(es)))])
        else:
            nets = [delta_break(int(x), c, inc4, inc5, rs4, rs5)[0] for x in es]
            e = int(es[int(np.argmin(nets))])
        nd = delta_break(e, c, inc4, inc5, rs4, rs5)[0]
        apply_flip(e, c, inc4, inc5, rs4, rs5); v += nd; best = min(best, v); it += 1
        if it % branch == 0:
            c = rng.integers(0, 2, E); rs4, rs5, v = fresh(c, K4, K5)
    return c, best, it, time.time() - t0

def walksat(E, K4, K5, inc4, inc5, seed, budget, p=0.3, maxflips=20000):
    rng = np.random.default_rng(seed); t0 = time.time()
    c = rng.integers(0, 2, E); rs4, rs5, v = fresh(c, K4, K5); best = v; it = 0; f = 0
    while time.time() - t0 < budget:
        if v == 0: break
        viols = viol_cliques(rs4, rs5, K4, K5)
        es = viols[int(rng.integers(0, len(viols)))]
        brks = [delta_break(int(x), c, inc4, inc5, rs4, rs5)[1] for x in es]
        zero = [int(es[i]) for i, b in enumerate(brks) if b == 0]
        if zero:
            e = int(rng.choice(zero))
        elif rng.random() < p:
            e = int(es[int(rng.integers(0, len(es)))])
        else:
            e = int(es[int(np.argmin(brks))])
        nd = delta_break(e, c, inc4, inc5, rs4, rs5)[0]
        apply_flip(e, c, inc4, inc5, rs4, rs5); v += nd; best = min(best, v); it += 1; f += 1
        if f >= maxflips:
            c = rng.integers(0, 2, E); rs4, rs5, v = fresh(c, K4, K5); f = 0
    return c, best, it, time.time() - t0

def sa(E, K4, K5, inc4, inc5, seed, budget, T0=2.0, Tend=0.03, period=30000):
    rng = np.random.default_rng(seed); t0 = time.time()
    c = rng.integers(0, 2, E); rs4, rs5, v = fresh(c, K4, K5); best = v; it = 0
    while time.time() - t0 < budget:
        if v == 0: break
        e = int(rng.integers(0, E))
        nd = delta_break(e, c, inc4, inc5, rs4, rs5)[0]
        T = T0 * (Tend / T0) ** ((it % period) / period)
        if nd <= 0 or rng.random() < np.exp(-nd / T):
            apply_flip(e, c, inc4, inc5, rs4, rs5); v += nd
        best = min(best, v); it += 1
    return c, best, it, time.time() - t0

SOLVERS = {"COMPASS": compass, "WalkSAT": walksat, "SA": sa}

# ---------------- race ----------------
def race(n, seeds, budget, out_prefix=None):
    E, K4, K5, inc4, inc5 = build(n)
    print(f"n={n}  edges={E}  K4s={len(K4)}  K5s={len(K5)}  "
          f"target: red K4=0, blue K5=0  seeds={seeds}  budget={budget}s/seed", flush=True)
    results = {}
    for name, fn in SOLVERS.items():
        solved = 0; times = []; bests = []; fake = 0
        for sd in range(seeds):
            c, best, it, tt = fn(E, K4, K5, inc4, inc5, sd, budget)
            rK4, bK5 = verify(c, n, K4, K5)        # INDEPENDENT check
            valid = (rK4 == 0 and bK5 == 0)
            if best == 0 and not valid: fake += 1   # incremental claimed 0 but verifier disagrees
            if valid:
                solved += 1; times.append(tt)
            else:
                bests.append(best)
            print(f"  [{name:7}] seed {sd:3d}  best={best:4d}  verify(redK4={rK4},blueK5={bK5})"
                  f"  {'SOLVED' if valid else ''}  {tt:.2f}s", flush=True)
        rate = solved / seeds
        row = {
            "n": n, "seeds": seeds, "solved": solved, "solve_rate": rate,
            "median_time": float(np.median(times)) if times else None,
            "p90_time": float(np.percentile(times, 90)) if times else None,
            "best_violations_min": (min(bests) if bests else 0),
            "best_violations_max": (max(bests) if bests else 0),
            "verifier_disagreements": fake,
        }
        results[name] = row
        print(f"==> {name}: solved {solved}/{seeds} ({rate:.0%})  "
              f"median={row['median_time']}  p90={row['p90_time']}  "
              f"best_viol={row['best_violations_min']}-{row['best_violations_max']}  "
              f"fake={fake}", flush=True)
    if out_prefix:
        with open(f"{out_prefix}_n{n}.json", "w") as f:
            json.dump(results, f, indent=2)
    return results

if __name__ == "__main__":
    n      = int(sys.argv[1]) if len(sys.argv) > 1 else 24
    seeds  = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    budget = float(sys.argv[3]) if len(sys.argv) > 3 else 8.0
    prefix = sys.argv[4] if len(sys.argv) > 4 else None
    race(n, seeds, budget, prefix)
