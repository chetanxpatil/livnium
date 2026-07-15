"""
chunk.py — run seeds for ONE solver in parallel (4 workers), append VERIFIED
results to a master CSV, one row flushed as each finishes. Resumable: skips
(solver,seed) already in the CSV, so re-running a range is idempotent and a
timeout never loses completed work.

usage: python3 chunk.py <n> <solver> <seed_start> <seed_count> <budget> <csv>
"""
import sys, csv, os, time
from multiprocessing import Pool
import r45_race as R

FIELDS = ["solver","n","seed","best","redK4","blueK5","solved","fake","time","iters"]

def worker(args):
    n, solver, seed, budget = args
    E, K4, K5, inc4, inc5 = worker.cache[n]
    fn = R.SOLVERS[solver]
    c, best, it, tt = fn(E, K4, K5, inc4, inc5, seed, budget)
    rK4, bK5 = R.verify(c, n, K4, K5)
    valid = (rK4 == 0 and bK5 == 0)
    return dict(solver=solver, n=n, seed=seed, best=int(best),
                redK4=rK4, blueK5=bK5, solved=int(valid),
                fake=int(best == 0 and not valid), time=round(tt, 3), iters=it)

def init(n):
    worker.cache = {n: R.build(n)}

def done_set(csvf):
    s = set()
    if os.path.exists(csvf):
        with open(csvf) as f:
            for r in csv.DictReader(f):
                s.add((r["solver"], int(r["n"]), int(r["seed"])))
    return s

if __name__ == "__main__":
    n = int(sys.argv[1]); solver = sys.argv[2]
    s0 = int(sys.argv[3]); cnt = int(sys.argv[4]); budget = float(sys.argv[5])
    csvf = sys.argv[6]
    have = done_set(csvf)
    jobs = [(n, solver, s, budget) for s in range(s0, s0 + cnt)
            if (solver, n, s) not in have]
    if not jobs:
        print(f"{solver} n={n} {s0}..{s0+cnt-1}: all done already"); sys.exit(0)
    header = not os.path.exists(csvf)
    f = open(csvf, "a", newline=""); w = csv.DictWriter(f, fieldnames=FIELDS)
    if header: w.writeheader(); f.flush()
    t0 = time.time(); solved = fake = n_done = 0
    with Pool(4, initializer=init, initargs=(n,)) as p:
        for r in p.imap_unordered(worker, jobs):
            w.writerow(r); f.flush()
            solved += r["solved"]; fake += r["fake"]; n_done += 1
    f.close()
    print(f"{solver} n={n} ran {n_done} new seeds: solved {solved}  fake={fake}  "
          f"wall={time.time()-t0:.1f}s", flush=True)
