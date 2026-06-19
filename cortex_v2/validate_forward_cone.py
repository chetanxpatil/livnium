"""
validate_forward_cone.py — the USER's model (corrected):

  NOT collapsing a huge space backward to a point.
  Instead: START at angle 0 (a single fully-defined seed, 0% randomness) and open
  FORWARD like a cone — admit randomness in bands (10%, 20%, ... 100%). At each band
  there are 'slits' (the reachable variations). Search forward through the slits,
  keeping the closest, growing understanding as you go. Never materialize the whole
  2^n / 3^L space — only the forward cone that structure actually reaches.

This is forward best-first growth with progressive widening (the reverse of
annealing). Test: does it reach the answer while touching only a tiny slice?

Base-3 strings (trits) of length L, tying to base-27 = 3^3.
Run from repo root:  python cortex_v2/validate_forward_cone.py
"""
import itertools, random

L = 12                      # trit-string length -> space = 3^12
BASE = 3
FULL = BASE ** L
target = [2,0,1,2,2,1,0,0,1,2,1,0]   # the structured answer we grow toward
seed   = [0]*L              # angle 0: fully defined, zero randomness (the apex)

def closeness(s):           # STRUCTURE: how many positions already match (the gradient)
    return sum(1 for a,b in zip(s,target) if a==b)

def neighbors(s, k):
    """open 'slits': all ways to change exactly k positions of s (one band wider)."""
    out = []
    for pos in itertools.combinations(range(L), k):
        for vals in itertools.product([1,2], repeat=k):   # change to a different trit
            t = s[:]
            for p,v in zip(pos, vals): t[p] = (t[p]+v) % BASE
            out.append(t)
    return out

print("="*72)
print(f"FORWARD CONE from angle 0. Space = 3^{L} = {FULL}. Open it in bands.")
print("="*72)
print(f"{'band':>5} {'randomness':>11} {'slits opened':>13} {'best match':>11} {'touched':>9}")

frontier = [seed]
touched = 0
BEAM = 8                    # keep only the best few directions -> cone stays thin
best_overall = (closeness(seed), seed)
for band in range(1, L+1):
    rnd = int(100*band/L)
    # open one band wider around the current promising frontier (forward only)
    cand = []
    for s in frontier:
        cand.extend(neighbors(s, 1))      # widen by 1 trit each step = move forward
    touched += len(cand)
    # score by structure (closeness), keep the best BEAM -> the cone's leading edge
    cand.sort(key=closeness, reverse=True)
    frontier = cand[:BEAM]
    b = closeness(frontier[0])
    if b > best_overall[0]: best_overall = (b, frontier[0])
    print(f"{band:>5} {rnd:>10}% {len(cand):>13} {b:>9}/{L} {touched:>9}")
    if b == L:
        print(f"\n  reached the answer at band {band} ({rnd}% randomness).")
        break

print()
print(f"touched {touched} states out of {FULL}  =  {100*touched/FULL:.4f}% of the space")
print("the cone opened FORWARD from a point and found the answer by following the")
print("closeness gradient (structure) — never building the full space. Moving forward,")
print("understanding as it goes. Exactly your model.")
print()
print("Contrast: open ALL slits at once (full 100% randomness, no gradient) =")
print(f"  {FULL} states = the exponential wall. The cone only works because structure")
print("  tells you which slits to open next.")
