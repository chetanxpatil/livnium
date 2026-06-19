"""
validate_triangle_search.py — testing the user's "triangle" idea:

  "A structure starts wide (all possibilities), then slowly shrinks the possibility
   down to a point — narrow a range, keep the part closest to the answer, repeat."

That IS structured search by narrowing (bisection / annealing). The key test:
narrowing only works when there is a CLOSENESS signal (structure pointing the way).
With structure -> the triangle collapses in ~log2(N) steps. Without it (random,
flat landscape) -> no signal to narrow by -> you are back to scanning one-by-one.

Run from repo root:  python cortex_v2/validate_triangle_search.py
"""
import math

N = 1024              # size of the space (all possibilities)
target = 731          # the one answer we want

def bar(width, full=N):
    return "#" * max(1, int(40 * width / full))

print("=" * 70)
print(f"YOUR TRIANGLE: space of {N}, narrow the range toward target #{target}")
print("=" * 70)

# --- WITH STRUCTURE: a closeness signal (we can tell which half is warmer) ---
print("\nWITH structure (closeness tells us which way to go):")
lo, hi = 0, N
step = 0
while hi - lo > 1:
    mid = (lo + hi) // 2
    width = hi - lo
    certainty = 100 * (1 - (width - 1) / (N - 1))     # 0% wide -> 100% at a point
    print(f"  step {step:2d}: range width {width:4d}  certainty {certainty:5.1f}%  {bar(width)}")
    if target < mid:        # closeness signal: target is in the lower half
        hi = mid
    else:
        lo = mid
    step += 1
print(f"  step {step:2d}: range width    1  certainty 100.0%  -> found #{lo}")
print(f"  => collapsed in {step} steps  (log2({N}) = {int(math.log2(N))}). The triangle.")

# --- WITHOUT STRUCTURE: flat landscape, no closeness, only hit/miss ----------
print("\nWITHOUT structure (random/flat: you only learn hit or miss, no 'closer'):")
print("  narrowing gives NO traction — each test rules out just 1, not half.")
print(f"  expected tests to find it = N/2 = {N//2}  (vs {int(math.log2(N))} with structure)")
import random
rng = random.Random(0)
tries = 0
seen = set()
while True:
    g = rng.randrange(N); tries += 1
    if g == target: break
    seen.add(g)
print(f"  random scan actually took {tries} tries this run.")

print()
print("VERDICT: your triangle is REAL — it's bisection/annealing. Narrowing a range")
print("toward the closest result collapses possibility geometrically (wide -> point)")
print("in log2(N) steps... but ONLY if a closeness signal (structure) exists.")
print("No structure = no 'closer' = the triangle can't form = brute force returns.")
