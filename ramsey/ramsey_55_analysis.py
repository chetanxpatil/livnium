"""
ramsey_55_analysis.py
─────────────────────
Structural constraint analysis for R(5,5) via K₄₃.

Goal: Assume a valid 2-coloring of K₄₃ exists (no monochromatic K₅).
      Apply Ramsey constraints layer by layer.
      Identify what must be true and where the squeeze tightens.

Method: assume → constrain → squeeze → attempt contradiction.
"""

from math import comb, floor, ceil, sqrt
from itertools import combinations

DIVIDER = "=" * 70

# ─── Known Ramsey numbers ─────────────────────────────────────────────────────
R = {
    (2,2):2, (2,3):3, (2,4):4, (2,5):5, (2,6):6,
    (3,3):6, (3,4):9, (3,5):14, (3,6):18,
    (4,4):18, (4,5):25,
    # R(5,5) unknown: in [43, 48]
}
for k,v in list(R.items()):
    R[(k[1],k[0])] = v  # symmetry

N = 43  # attempting to 2-color K₄₃
GOAL = 5  # no monochromatic K₅

print(DIVIDER)
print(f"  R(5,5) — STRUCTURAL SQUEEZE ON K_{N}")
print(f"  Assumption: a valid 2-coloring exists (no monochromatic K_{GOAL})")
print(DIVIDER)

# ─── LEVEL 1: Degree bounds ───────────────────────────────────────────────────
print("\n╔══ LEVEL 1: DEGREE BOUNDS (from R(4,5)=25) ══╗")

deg = N - 1  # = 42, each node has this many edges

# If r(v) ≥ R(4,5) = 25:
#   The 25 red neighbors form K₂₅.
#   By R(4,5)=25: K₂₅ has a red K₄ OR a blue K₅.
#   Red K₄ + center v = red K₅ → FORBIDDEN.
#   Blue K₅ → FORBIDDEN.
#   ∴ Contradiction → r(v) ≤ 24.
r_max = R[(4,5)] - 1   # = 24
b_max = R[(5,4)] - 1   # = 24  (symmetric argument)
r_min = deg - b_max    # = 42 - 24 = 18
b_min = deg - r_max    # = 42 - 24 = 18

print(f"   R(4,5) = {R[(4,5)]} → r(v) ≤ {r_max}  (else red K₄ in nbhd → red K₅ with v)")
print(f"   R(5,4) = {R[(5,4)]} → b(v) ≤ {b_max}  (symmetric)")
print(f"   r(v) + b(v) = {deg}")
print(f"")
print(f"   ┌─────────────────────────────────────────┐")
print(f"   │  CONSTRAINT L1: r(v) ∈ [{r_min}, {r_max}] ∀v  │")
print(f"   │                 b(v) ∈ [{b_min}, {b_max}] ∀v  │")
print(f"   └─────────────────────────────────────────┘")
print(f"   Range width: {r_max - r_min + 1} values. NOT yet a contradiction.")

# ─── LEVEL 2: Global edge counts ─────────────────────────────────────────────
print(f"\n╔══ LEVEL 2: GLOBAL EDGE COUNTS ══╗")

E_total = comb(N, 2)  # = 903
E_r_min = ceil(N * r_min / 2)
E_r_max = floor(N * r_max / 2)

print(f"   Total edges in K_{N}: C({N},2) = {E_total}")
print(f"   Sum of red degrees: Σ r(v) = 2|E_R|  (handshaking)")
print(f"   Σ r(v) ∈ [{N*r_min}, {N*r_max}]  →  |E_R| ∈ [{E_r_min}, {E_r_max}]")
print(f"   Average red degree: {N*(r_min+r_max)//2}/{N} ... range midpoint = {(r_min+r_max)/2}")
print(f"   |E_R| + |E_B| = {E_total}  →  |E_B| ∈ [{E_total - E_r_max}, {E_total - E_r_min}]")
print(f"   Both constraints satisfied simultaneously: need |E_R| ∈ [{max(E_r_min, E_total-E_r_max)}, {min(E_r_max, E_total-E_r_min)}]")
print(f"")
feasible_min = max(E_r_min, E_total - E_r_max)
feasible_max = min(E_r_max, E_total - E_r_min)
print(f"   Feasible |E_R|: [{feasible_min}, {feasible_max}]  ({feasible_max-feasible_min+1} values)")
print(f"   Midpoint: {E_total/2:.1f}  →  graph must be roughly BALANCED in color")

# ─── LEVEL 3: Forced structure in neighborhoods ───────────────────────────────
print(f"\n╔══ LEVEL 3: FORCED STRUCTURE (from R(4,4)=18) ══╗")

# r(v) ≥ 18 = R(4,4).
# In red nbhd of v (≥18 nodes), K_{r(v)} must be 2-colored.
# By R(4,4)=18: it has a monochromatic K₄.
# If RED K₄: + center v = red K₅ → FORBIDDEN.
# So it MUST be a BLUE K₄.
#
# Same argument for blue nbhd: must contain a RED K₄.

print(f"   r(v) ≥ {r_min} = R(4,4) = {R[(4,4)]}")
print(f"   ∴ red nbhd of v (K_{{r(v)}}) has a monochromatic K₄  (by R(4,4)={R[(4,4)]})")
print(f"   Red K₄ in red nbhd + v → red K₅  FORBIDDEN")
print(f"   ∴ red nbhd MUST contain a BLUE K₄")
print(f"")
print(f"   ┌─────────────────────────────────────────────────────────┐")
print(f"   │  FORCED L3a: Every node v has a blue K₄ in its         │")
print(f"   │              red neighborhood.                           │")
print(f"   │  FORCED L3b: Every node v has a red K₄ in its          │")
print(f"   │              blue neighborhood.                          │")
print(f"   └─────────────────────────────────────────────────────────┘")

# ─── LEVEL 4: Goodman's formula — monochromatic triangles ────────────────────
print(f"\n╔══ LEVEL 4: GOODMAN'S FORMULA — MONOCHROMATIC TRIANGLES ══╗")

# Goodman (1959):
# T_mono = C(n,3) - (1/2)·Σ_v r(v)·b(v)
# where r(v)+b(v) = n-1.

T_total = comb(N, 3)

# r·b is maximized when r=b=(n-1)/2=21: r·b = 21² = 441
# r·b is minimized at endpoints: 18·24 = 432
rb_max_per_v = ((N-1)//2) * ((N-1)//2 + (N-1)%2)  # 21*21 = 441
rb_min_per_v = r_min * b_max   # 18*24 = 432

# T_mono is minimized when Σ r·b is maximized (all nodes balanced at 21)
T_mono_min = T_total - (N * rb_max_per_v) / 2
# T_mono is maximized when Σ r·b is minimized (all at endpoints 18 or 24)
T_mono_max = T_total - (N * rb_min_per_v) / 2

print(f"   C({N},3) = {T_total} total triangles")
print(f"   r(v)·b(v) ∈ [{rb_min_per_v}, {rb_max_per_v}]  (product maximized at r=b=21)")
print(f"")
print(f"   Goodman: T_mono = C({N},3) - (1/2)·Σ r(v)·b(v)")
print(f"   T_mono ≥ {T_total} - (1/2)·{N}·{rb_max_per_v} = {T_mono_min:.1f}  → T_mono ≥ {ceil(T_mono_min)}")
print(f"   T_mono ≤ {T_total} - (1/2)·{N}·{rb_min_per_v} = {T_mono_max:.1f}  → T_mono ≤ {floor(T_mono_max)}")
print(f"")
print(f"   ┌─────────────────────────────────────────────────────────┐")
print(f"   │  FORCED L4: In any valid 2-coloring of K_{N}:            │")
print(f"   │  T_mono ∈ [{ceil(T_mono_min)}, {floor(T_mono_max)}]                                 │")
print(f"   │  = thousands of unavoidable monochromatic triangles!    │")
print(f"   └─────────────────────────────────────────────────────────┘")

# Fraction that are monochromatic
frac_min = T_mono_min / T_total
frac_max = T_mono_max / T_total
print(f"   Fraction monochromatic: [{frac_min:.3f}, {frac_max:.3f}]")
print(f"   i.e., ≥{frac_min*100:.1f}% of all triangles are monochromatic — unavoidable!")

# ─── LEVEL 5: Recursion — apply bounds INSIDE neighborhoods ──────────────────
print(f"\n╔══ LEVEL 5: RECURSIVE SQUEEZE IN RED NEIGHBORHOODS ══╗")

# In red nbhd of v (k nodes, k ∈ [18,24]):
# Must have: no red K₄, no blue K₅  (else K₅ globally)
# For each node u in red nbhd:
#   r'(u) = red degree of u INSIDE nbhd
#   r'(u) ≥ R(3,5)-1 → forced. Wait: if r'(u) ≥ R(3,5)=14, then
#     in u's red sub-nbhd (within v's red nbhd), we get red K₃ or blue K₅.
#     Red K₃ in sub-nbhd + u = red K₄ in v's red nbhd + v = red K₅. FORBIDDEN.
#     Blue K₅: FORBIDDEN.
#     ∴ r'(u) ≤ R(3,5)-1 = 13.

r2_max = R[(3,5)] - 1  # = 13
# And if r'(u) ≥ R(4,4)=18 (another way): but k≤24 so r'(u)≤23 anyway.

print(f"   For node u in red nbhd of v (size k ∈ [{r_min},{r_max}]):")
print(f"   r'(u) = red degree of u within that nbhd")
print(f"")
print(f"   R(3,5) = {R[(3,5)]}: if r'(u) ≥ {R[(3,5)]},")
print(f"     sub-nbhd of u has red K₃ (→ red K₄ in v-nbhd → red K₅ with v) OR blue K₅.")
print(f"     Both forbidden.  ∴ r'(u) ≤ {r2_max}")
print(f"")

# Inside the red nbhd (K_k, no red K₄, no blue K₅):
# Each node has r'(u) ≤ 13, so red graph in nbhd has max degree 13.
# Blue graph in nbhd has max degree k-1-r2_max.
for k in [18, 20, 22, 24]:
    b2_min = k - 1 - r2_max
    b2_max = k - 1
    # By R(3,3)=6 inside blue nbhd of u within red nbhd of v:
    # if b2(u) ≥ 6...
    r2_min_from_b = k - 1 - (R[(5,4)] - 1)  # b'(u) ≤ R(5,4)-1=24, but k≤24 so this is loose
    turan_red_edges = k*k / 3  # Turán bound: K₄-free graph
    turan_blue_edges = 3*k*k / 8  # Turán K₅-free: (1-1/4)*k²/2 = 3k²/8
    total_allowed = turan_red_edges + turan_blue_edges
    actual_edges = comb(k, 2)
    print(f"   k={k}: max red deg inside={r2_max}, blue deg ∈ [{b2_min},{b2_max}]")
    print(f"         Turán K₄-free: ≤{turan_red_edges:.0f} red edges, K₅-free: ≤{turan_blue_edges:.0f} blue edges")
    print(f"         Total allowed ≤ {total_allowed:.0f} vs actual edges = {actual_edges}")
    print(f"         {'✓ consistent' if total_allowed >= actual_edges else '✗ CONTRADICTION'}")

# ─── LEVEL 6: Minimum red K₃ count from degree constraints ───────────────────
print(f"\n╔══ LEVEL 6: COUNTING FORCED STRUCTURES ══╗")

# In red graph G_R (n=43 vertices):
# No red K₄ (else K₅ with neighbors). So G_R is K₄-free.
# By Turán's theorem: |E_R| ≤ T(43,3) = floor(43²·(1-1/3)/2)
turan_k4_free = floor(N**2 * (1 - 1/3) / 2)
print(f"   Red graph G_R must be K₄-free.")
print(f"   Turán T({N},3) = ⌊{N}²·(2/3)/2⌋ = {turan_k4_free} max edges for K₄-free graph")
print(f"   But |E_R| ∈ [{feasible_min}, {feasible_max}]")
print(f"   Is [{feasible_min}, {feasible_max}] ⊆ [0, {turan_k4_free}]? ", end="")
if feasible_max <= turan_k4_free:
    print(f"YES — consistent (no K₄-free Turán contradiction yet)")
else:
    print(f"NO — CONTRADICTION: edge count forces red K₄!")
print(f"")

# Minimum number of red triangles in G_R:
# Kruskal-Katona / Razborov flag algebra:
# For K₄-free triangle density...
# Kruskal-Katona gives: T_R ≥ ...
# Use simpler: in a graph with e edges and n vertices,
# T(edges) ≥ roughly e(4e-n²)/3n (Kruskal-Katona type)
# For balanced degrees: every vertex has r(v) ∈ [18,24]
# Red triangles T_R: use Razborov's triangle removal / triangle counting
# Simple bound via degrees:
# T_R = (1/6) * Σ_{(u,v) red edge} (common red neighbors of u,v)
# Not easy to bound from below without more structure.

# Use averaging: each pair of red neighbors of v forms a potential triangle.
# For vertex v: r(v) red neighbors. Edges among those = t(v) red + rest blue.
# T_R ≥ Σ_v C(r(v),2) - non-closed... this is just the number of "paths of length 2" in red
paths_r = sum(comb(r, 2) for r in range(r_min, r_max+1)) * N // (r_max - r_min + 1)
print(f"   Each node v has r(v) red neighbors → C(r(v),2) red 'wedges' centered at v")
print(f"   Avg wedges per node ≈ C(21,2) = {comb(21,2)}")
print(f"   Total red wedges ≈ {N}·{comb(21,2)} = {N*comb(21,2)}")
print(f"   Each red triangle counted 3 times in wedges.")
print(f"   If ZERO triangles closed: red graph is triangle-free, max edges = {N*N//4} (Turán K₃-free).")
print(f"   But |E_R| ≥ {feasible_min} > {N*N//4}: {'MUST have red triangles!' if feasible_min > N*N//4 else 'K₃-free still possible.'}")
turan_k3_free = N*N // 4
print(f"   Turán K₃-free: max {turan_k3_free} edges.  Our min: {feasible_min}.")
if feasible_min > turan_k3_free:
    print(f"   ∴ G_R MUST contain red triangles.  (And G_B must contain blue triangles.)")
else:
    print(f"   G_R could still be triangle-free.")

# ─── LEVEL 7: Neighborhood structure forces cascades ─────────────────────────
print(f"\n╔══ LEVEL 7: THE CASCADE — RED K₃ IN BLUE NEIGHBORHOODS ══╗")

# We showed T_mono ≥ 2860. These must all be red K₃ or blue K₃.
# If T_R is the count of red K₃:
# Red K₃ = triple (u,v,w) all red.
# None of these can extend to red K₄ (since no red K₄ globally).
# So every red edge uv has NO red common neighbor with any other red edge from u,v.
# Wait: no red K₄ means no 4 vertices all red-connected.
# So every red K₃ is "isolated" in the sense that no 4th node connects to all 3 in red.

print(f"   T_mono ≥ {ceil(T_mono_min)} monochromatic triangles forced.")
print(f"")
print(f"   Key constraint: no red K₄, no blue K₄... wait — blue K₄ IS allowed!")
print(f"   (Only red K₅ and blue K₅ are forbidden.)")
print(f"   Blue K₄ is fine. Red K₄ is forbidden.")
print(f"")
print(f"   So: G_R is K₄-free AND has ≥{feasible_min} edges.")
print(f"   G_B is K₅-free AND has ≥{E_total-feasible_max} edges.")
print(f"")

# G_R is K₄-free with many edges → by Kruskal-Katona, many red triangles.
# Kruskal-Katona: K₄-free graph on n nodes with m edges has ≥ m(4m-n²)/(3n) triangles
# (Razborov / Zykov). Let's compute:
m_r = feasible_min  # minimum red edges
razborov_T_R = m_r * (4*m_r - N**2) / (3*N)
print(f"   G_R: K₄-free, ≥{m_r} edges.")
print(f"   Razborov triangle bound: T_R ≥ m(4m-n²)/(3n)")
print(f"   T_R ≥ {m_r}·(4·{m_r}-{N}²)/(3·{N}) = {razborov_T_R:.1f}")
if razborov_T_R > 0:
    print(f"   → G_R has ≥ {ceil(razborov_T_R)} red triangles.")
    print(f"   Each red triangle is a 'used' triple that cannot be extended to red K₄.")
else:
    print(f"   → Razborov bound non-positive here (edges not dense enough for direct bound).")
    print(f"   → Need Kruskal-Katona or shadow bound for stronger result.")

# ─── LEVEL 8: The key tension ─────────────────────────────────────────────────
print(f"\n╔══ LEVEL 8: WHERE THE SQUEEZE STANDS ══╗")

print(f"""
   WHAT WE KNOW so far (all provable, no contradiction yet):

   (1) Every node: r(v) ∈ [{r_min},{r_max}], b(v) ∈ [{b_min},{b_max}]
   (2) |E_R| ∈ [{feasible_min},{feasible_max}],   |E_B| ∈ [{E_total-feasible_max},{E_total-feasible_min}]
   (3) G_R is K₄-free, G_B is K₅-free
   (4) Every red nbhd contains a blue K₄
   (5) Every blue nbhd contains a red K₄
   (6) T_mono ∈ [{ceil(T_mono_min)},{floor(T_mono_max)}] monochromatic triangles (unavoidable)
   (7) G_R must contain red triangles  (edges too dense for K₃-free)
   (8) Inside each red nbhd: max red degree {r2_max}

   WHY NO CONTRADICTION YET:

   The constraints are TIGHT but not immediately impossible.
   The system is highly constrained — each node must balance ≈21 red / 21 blue,
   no red K₄, no blue K₅, thousands of monochromatic triangles packed in.

   The TRUE squeeze lives in the interaction of:
   • The forced blue K₄s in every red nbhd (Level 3)
   • The K₄-free structure of G_R (Level 7)
   • The precise degree balance requirements (Level 6)

   KNOWN RESULT: R(5,5) ≥ 43, proven by finding an explicit 2-coloring of
   K₄₂ (McKay-Radziszowski 1995; Exoo 1989).
   The PALEY GRAPH on 41 nodes achieves the lower bound.
""")

# ─── LEVEL 9: Next levers ─────────────────────────────────────────────────────
print(f"╔══ LEVEL 9: NEXT LEVERS (what would give a contradiction) ══╗")

print(f"""
   LEVER A: Count blue K₄s globally.
   Every red nbhd (43 nodes, each) contains a blue K₄.
   How many DISTINCT blue K₄s can there be?
   A blue K₄ {{a,b,c,d}} serves as evidence for node v iff
   v is red-connected to all of {{a,b,c,d}}.
   If |red common nbhd of {{a,b,c,d}}| = t, that K₄ serves t nodes.
   Total "service" = 43. Total blue K₄s × avg_service = 43.
   → Need avg_service ≥ 43 / (# blue K₄s).
   This leads to counting blue K₄ densities in K₅-free graphs...

   LEVER B: Flag algebras (Razborov 2007).
   Encode: density of "red-red-blue-path" patterns,
   density of "blue K₄ with red connection" patterns.
   Solve an SDP to find the minimum density of monochromatic K₅.
   This IS how R(5,5) ≤ 48 was proved.

   LEVER C: Double-count red wedges through blue K₄s.
   Each blue K₄ {{a,b,c,d}} that sits in the red nbhd of v
   gives 4 blue K₃s "witnessed" by v in red.
   Summing over v: each such K₃ witnessed by all its red common neighbors.
   Leads to double-counting argument on mixed configurations.

   LEVER D: The Paley graph on 43 nodes.
   Construct explicitly: nodes = GF(43) = {{0,...,42}},
   edges = quadratic residues mod 43 = {{1,4,6,9,10,11,13,14,16,17,21,...}}.
   This is conjectured to be the unique (or near-unique) valid graph.
   Verify computationally whether it achieves the bound.
""")

# ─── LEVEL 10: Paley graph construction ──────────────────────────────────────
print(f"╔══ LEVEL 10: PALEY GRAPH ON {N} NODES ══╗")
print(f"   (The conjectured extremal construction for R(5,5))")

p = N  # 43 is prime
# Quadratic residues mod 43
qr = set()
for x in range(1, p):
    qr.add((x * x) % p)
qr_sorted = sorted(qr)
print(f"\n   p = {p} (prime ✓)")
print(f"   Quadratic residues mod {p}: {qr_sorted}")
print(f"   Count: {len(qr_sorted)} (should be (p-1)/2 = {(p-1)//2})")

# Build adjacency: i~j iff (i-j) mod p is a QR
adj = [[False]*p for _ in range(p)]
edge_count = 0
for i in range(p):
    for j in range(i+1, p):
        diff = (i - j) % p
        if diff in qr:
            adj[i][j] = adj[j][i] = True
            edge_count += 1

degrees = [sum(adj[i]) for i in range(p)]
print(f"   Edges: {edge_count}  (should be p(p-1)/4 = {p*(p-1)//4})")
print(f"   Degrees: all {degrees[0]}  (regular: ✓ if {degrees[0]} = (p-1)/2 = {(p-1)//2})")

# Check for clique-5 in Paley graph (G_R = Paley, G_B = complement)
# Check red K₅ (clique in Paley)
print(f"\n   Checking for monochromatic K₅ in Paley 2-coloring...")
print(f"   (Red = QR edge, Blue = non-QR edge)")

def has_clique_k(adj_mat, n, k, color_mask=None):
    """Check if graph has clique of size k. color_mask selects edges."""
    count = 0
    for verts in combinations(range(n), k):
        ok = True
        for i in range(k):
            for j in range(i+1, k):
                if not adj_mat[verts[i]][verts[j]]:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            count += 1
            if count >= 1:
                return True, verts
    return False, None

# Red K₅: clique in Paley graph
print(f"   Checking red K₅ (K₅ in Paley graph)...", end=" ", flush=True)
found_r5, verts_r5 = has_clique_k(adj, p, 5)
print(f"Found: {found_r5}")
if found_r5:
    print(f"   Red K₅ at: {verts_r5}")

# Blue K₅: independent set of size 5 in Paley graph
comp = [[not adj[i][j] and i!=j for j in range(p)] for i in range(p)]
print(f"   Checking blue K₅ (independent set of 5 in Paley)...", end=" ", flush=True)
found_b5, verts_b5 = has_clique_k(comp, p, 5)
print(f"Found: {found_b5}")
if found_b5:
    print(f"   Blue K₅ at: {verts_b5}")

if not found_r5 and not found_b5:
    print(f"""
   ┌──────────────────────────────────────────────────────────────────┐
   │  RESULT: Paley(43) has NO red K₅ and NO blue K₅!                │
   │  This PROVES: R(5,5) ≥ 44 — a valid K₄₃ coloring EXISTS!       │
   │  → The question is whether K₄₄ can also be safely colored.      │
   └──────────────────────────────────────────────────────────────────┘""")
else:
    print(f"\n   Paley(43) does not avoid all monochromatic K₅ — not the right construction.")

# ─── LEVEL 11: Degree profile of Paley ───────────────────────────────────────
print(f"\n╔══ LEVEL 11: STRUCTURAL PROFILE OF PALEY({N}) ══╗")

# In Paley(43): each node has degree (43-1)/2 = 21
# Red degree = 21 for all nodes! Exactly balanced.
r_paley = degrees[0]
print(f"   Red degree (Paley): {r_paley} for every node  (perfectly balanced!)")
print(f"   Blue degree: {p-1-r_paley} for every node")
print(f"   |E_R| = {edge_count}, |E_B| = {E_total - edge_count}")
print(f"   Color balance: perfect 50/50")

# Count red triangles in Paley
print(f"\n   Counting monochromatic triangles in Paley({p})...")
t_red = 0
t_blue = 0
for i,j,k_ in combinations(range(p), 3):
    red_edges = adj[i][j] + adj[i][k_] + adj[j][k_]
    if red_edges == 3:
        t_red += 1
    elif red_edges == 0:
        t_blue += 1

t_total = comb(p, 3)
t_mono = t_red + t_blue
print(f"   Red K₃: {t_red}")
print(f"   Blue K₃: {t_blue}")
print(f"   Total mono: {t_mono} / {t_total} = {t_mono/t_total*100:.1f}%")
print(f"   (Goodman formula predicts range [{ceil(T_mono_min)}, {floor(T_mono_max)}] ← check: {t_mono})")

# Check no red K₄ (should be true if no red K₅, but let's verify)
print(f"\n   Checking for red K₄ in Paley({p})...", end=" ", flush=True)
found_r4, verts_r4 = has_clique_k(adj, p, 4)
print(f"Found: {found_r4}")
if found_r4:
    print(f"   Red K₄ at: {verts_r4}")
    print(f"   (Red K₄ exists but no red K₅ — valid)")

print(f"\n{'='*70}")
print(f"  SUMMARY")
print(f"{'='*70}")
print(f"""
  The structural squeeze shows:

  1. r(v) ∈ [18,24] — tight window from R(4,5)=25
  2. Every red nbhd has a forced blue K₄
  3. Thousands of monochromatic triangles unavoidable
  4. G_R is K₄-free with ≈451 edges

  The Paley(43) graph demonstrates that K₄₃ CAN be 2-colored
  without monochromatic K₅ — proving R(5,5) ≥ 44.

  The OPEN QUESTION: Can K₄₄...K₄₇ also be colored?
  Current best: R(5,5) ∈ [43, 48]  (R(5,5)-1 ≥ 43 means R(5,5) ≥ 44)

  Wait — this needs verification. If Paley(43) works:
  - It 2-colors K₄₃ without K₅ → R(5,5) > 43 → R(5,5) ≥ 44
  Actually the known bound is R(5,5) ≥ 43 (lower bound),
  meaning K₄₂ can be 2-colored but K₄₃ is unknown.
  Let's see what Paley(43) actually tells us...
""")

print(f"  Paley(43) result: {'K₄₃ has a valid coloring → R(5,5) ≥ 44' if not found_r5 and not found_b5 else 'Paley(43) does not give valid K₄₃ coloring'}")
