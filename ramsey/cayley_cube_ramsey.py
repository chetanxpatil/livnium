"""
cayley_cube_ramsey.py — Ramsey lower-bound witnesses as Cayley graphs on groups,
including the cube's rotation group. Exhaustively verified (every clique checked).

Ladder:
  R(3,3)=6   -> Z_5  pentagon C5            (no mono K3 either color)
  R(4,4)=18  -> Z_17 Paley(17)              (no mono K4 either color)
  R(4,5)=25  -> cube rotation group (S_4)   (no red K4, no blue K5)  <-- structural:
               |rotation group| = 24 = R(4,5)-1, so the 24 rotations are the vertices.

No torch; numpy + itertools only.  Run:  python3 cayley_cube_ramsey.py
"""
import itertools, numpy as np

def mono_counts(A, n, reds, blues):
    """count monochromatic red-cliques of size `reds` and blue-cliques of size `blues`."""
    r = sum(1 for s in itertools.combinations(range(n), reds)
            if all(A[a][b] for a, b in itertools.combinations(s, 2)))
    b = sum(1 for s in itertools.combinations(range(n), blues)
            if all(not A[a][b] for a, b in itertools.combinations(s, 2)))
    return r, b

# ---------- R(3,3): pentagon on Z_5 ----------
def pentagon():
    n, S = 5, {1, 4}
    A = [[(j - i) % n in S for j in range(n)] for i in range(n)]
    r, b = mono_counts(A, n, 3, 3)
    print(f"R(3,3): Z_5 pentagon C5      red K3={r}  blue K3={b}  -> R(3,3) >= 6" + ("  OK" if r==b==0 else "  FAIL"))

# ---------- R(4,4): Paley(17) on Z_17 ----------
def paley17():
    n = 17; qr = {(x * x) % n for x in range(1, n)}
    A = [[(i != j and (i - j) % n in qr) for j in range(n)] for i in range(n)]
    r, b = mono_counts(A, n, 4, 4)
    print(f"R(4,4): Z_17 Paley(17)       red K4={r}  blue K4={b}  -> R(4,4) >= 18" + ("  OK" if r==b==0 else "  FAIL"))

# ---------- R(4,5): Cayley graph on the cube ROTATION group (order 24) ----------
def cube_rotation_group():
    mats = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product([1, -1], repeat=3):
            M = np.zeros((3, 3), int)
            for i, p in enumerate(perm): M[i, p] = signs[i]
            if round(np.linalg.det(M)) == 1: mats.append(M)
    assert len(mats) == 24
    return mats

def cube_ramsey_45(seed=0, restarts=400):
    G = cube_rotation_group(); k = lambda M: M.tobytes()
    idx = {k(M): i for i, M in enumerate(G)}
    I0 = idx[k(np.eye(3, dtype=int))]
    inv = [idx[k(G[i].T)] for i in range(24)]
    diff = [[idx[k(G[g] @ G[h].T)] for h in range(24)] for g in range(24)]
    seen, orbits = set(), []
    for e in range(24):
        if e == I0 or e in seen: continue
        o = {e, inv[e]}; seen |= o; orbits.append(sorted(o))   # symmetric-set free bits
    rng = np.random.default_rng(seed)
    others = [v for v in range(24) if v != 0]
    K4 = [(0,) + c for c in itertools.combinations(others, 3)]  # through v0 (vertex-transitive)
    K5 = [(0,) + c for c in itertools.combinations(others, 4)]
    def adj(bits):
        S = [False] * 24
        for ob, b in zip(orbits, bits):
            if b:
                for e in ob: S[e] = True
        A = np.zeros((24, 24), bool)
        for g in range(24):
            for h in range(24):
                if g != h and S[diff[g][h]]: A[g][h] = True
        return A, S
    def cost(A):
        r = sum(1 for s in K4 if all(A[a][b] for a, b in itertools.combinations(s, 2)))
        b = sum(1 for s in K5 if all(not A[a][b] for a, b in itertools.combinations(s, 2)))
        return r + b
    best = None
    for _ in range(restarts):
        bits = rng.integers(0, 2, len(orbits)).tolist(); A, _ = adj(bits); c = cost(A)
        for _ in range(60):
            if c == 0: break
            i = int(rng.integers(0, len(orbits))); bits[i] ^= 1
            A, _ = adj(bits); c2 = cost(A)
            if c2 <= c: c = c2
            else: bits[i] ^= 1
        if best is None or c < best[0]: best = (c, bits[:])
        if c == 0: break
    _, bits = best; A, S = adj(bits)
    # FULL exhaustive verify over ALL cliques (no transitivity shortcut)
    r = sum(1 for s in itertools.combinations(range(24), 4) if all(A[a][b] for a, b in itertools.combinations(s, 2)))
    b = sum(1 for s in itertools.combinations(range(24), 5) if all(not A[a][b] for a, b in itertools.combinations(s, 2)))
    deg = sorted(int(A[v].sum()) for v in range(24))
    print(f"R(4,5): cube group (S_4)     red K4={r}  blue K5={b}  |S|={sum(S)} {deg[0]}-regular"
          + (f"  -> R(4,5) >= 25  OK" if r == 0 and b == 0 else "  FAIL"))

if __name__ == "__main__":
    pentagon()
    paley17()
    cube_ramsey_45()
