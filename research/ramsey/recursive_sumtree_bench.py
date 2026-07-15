"""
recursive_sumtree_bench.py — the recursive/nested Livnium geometry is an exact
conserved sum-tree (node = sum of its 27 children). This benchmarks its real
data-structural niche against four standard structures, on three query types.

Verdict (see FINDINGS.md): the 27-tree wins on ALIGNED regional aggregates +
point UPDATES (its native geometry); it ties a cached scalar on the global total
and loses to prefix-sum on arbitrary non-aligned regions. Not universal — the
right engine for its own aligned query pattern.

No torch; numpy only.  Run:  python3 recursive_sumtree_bench.py
"""
import numpy as np, time

def bench(fn, reps):
    t = time.perf_counter()
    for _ in range(reps): fn()
    return (time.perf_counter() - t) / reps * 1e6  # microseconds/op

def main(B=27, L=4):
    n = B ** L
    rng = np.random.default_rng(0); arr = rng.integers(0, 28, n).astype(np.int64)
    total = int(arr.sum())                              # flat + cached total
    P = np.concatenate([[0], np.cumsum(arr)])           # prefix-sum
    bit = np.zeros(n + 1, dtype=np.int64)               # Fenwick / BIT
    for i, v in enumerate(arr, 1):
        j = i
        while j <= n: bit[j] += v; j += j & -j
    def bit_prefix(i):
        s = 0
        while i > 0: s += bit[i]; i -= i & -i
        return s
    levels = [arr.copy()]; cur = arr                    # recursive 27-tree (conserved sums)
    for _ in range(L): cur = cur.reshape(-1, B).sum(1); levels.append(cur)

    l, r = 12345, 400000                                # arbitrary non-aligned range
    lvl, blk = 2, 5; al, ar = blk * B ** lvl, (blk + 1) * B ** lvl   # aligned region
    def rec_arb():
        s = 0; i = l
        while i < r:
            k = 0
            while k < L and i % (B ** (k + 1)) == 0 and i + B ** (k + 1) <= r: k += 1
            s += int(levels[k][i // (B ** k)]); i += B ** k
        return s
    assert rec_arb() == int(P[r] - P[l])                # tree result == ground truth

    print(f"n = {n} leaves (B={B}, L={L})\n")
    print(f"{'structure':22} {'global':>9} {'aligned reg':>13} {'arbitrary reg':>15} {'update':>16}")
    def row(name, g, a, arb, upd): print(f"{name:22} {g:>7.2f}us {a:>11.2f}us {arb:>13.2f}us {upd:>16}")
    row("1 naive flat",
        bench(lambda: int(arr.sum()), 50), bench(lambda: int(arr[al:ar].sum()), 2000),
        bench(lambda: int(arr[l:r].sum()), 200), "total O(n)")
    row("2 flat+cached total",
        bench(lambda: total, 200000), bench(lambda: int(arr[al:ar].sum()), 2000),
        bench(lambda: int(arr[l:r].sum()), 200), "O(1)")
    row("3 prefix-sum",
        bench(lambda: int(P[n] - P[0]), 200000), bench(lambda: int(P[ar] - P[al]), 200000),
        bench(lambda: int(P[r] - P[l]), 200000), "O(n) rebuild")
    row("4 Fenwick/BIT",
        bench(lambda: bit_prefix(n), 20000), bench(lambda: bit_prefix(ar) - bit_prefix(al), 20000),
        bench(lambda: bit_prefix(r) - bit_prefix(l), 20000), "O(log n)")
    row("5 recursive 27-tree",
        bench(lambda: int(levels[L][0]), 200000), bench(lambda: int(levels[lvl][blk]), 200000),
        bench(rec_arb, 20000), f"O(depth)={L}")
    print(f"\ntree memory overhead vs leaves: {sum(len(x) for x in levels[1:]) / n:.4f}x  (~27/26 = 1.038)")

if __name__ == "__main__":
    main()
