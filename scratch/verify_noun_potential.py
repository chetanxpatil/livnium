"""
verify_noun_potential.py -- numerical audit of the noun-model attraction force.

Checks whether F(h) = -s(1-cos(h,c)) (h-c)/||h-c|| is the gradient of any scalar
potential. Findings:
  * -grad( (s/6)||h-c||^3 )  matches F in direction everywhere, magnitude on ||h||=1.
  * -grad( (sqrt2/3)s(1-cos)^1.5 ) is ~44 deg off even on the unit sphere -> WRONG.
  * Jacobian of F is asymmetric (max|J-J^T| ~ 2e-2) -> F is NON-conservative in R^d;
    no exact global scalar potential exists. Same Euclidean-radial vs cosine-gradient
    mismatch the NLI paper flagged for Livnium v1.
Referenced by experiment/findings.md section 1.
"""

import numpy as np
rng = np.random.default_rng(0)
D, s = 256, 0.7

def cos(h, c): return float(h@c/(np.linalg.norm(h)*np.linalg.norm(c)))

# the code's actual noun force:  F = -s(1-cos(h,c)) * (h-c)/||h-c||
def F_code(h, c):
    r = h - c
    return -s*(1-cos(h,c)) * r/np.linalg.norm(r)

# recovered cos-form potential:  E = (sqrt2/3) s (1-cos)^1.5
def E_cos(h, c): return (np.sqrt(2)/3)*s*(1-cos(h,c))**1.5
# exact euclidean central potential: E = (s/6) ||h-c||^3
def E_euc(h, c): return (s/6)*np.linalg.norm(h-c)**3

def grad(fn, h, c, eps=1e-6):
    g = np.zeros_like(h)
    for i in range(len(h)):
        hp, hm = h.copy(), h.copy(); hp[i]+=eps; hm[i]-=eps
        g[i] = (fn(hp,c)-fn(hm,c))/(2*eps)
    return g

def report(name, h):
    c = rng.standard_normal(D); c/=np.linalg.norm(c)   # c is a unit well (as in code: A is normalized)
    Fc = F_code(h,c)
    gcos = -grad(E_cos,h,c); geuc = -grad(E_euc,h,c)
    def ang(a,b): return np.degrees(np.arccos(np.clip(a@b/(np.linalg.norm(a)*np.linalg.norm(b)),-1,1)))
    print(f"[{name}]  ||h||={np.linalg.norm(h):.3f}")
    print(f"   -grad(E_cos) vs F_code : angle {ang(gcos,Fc):6.2f} deg   mag-ratio {np.linalg.norm(gcos)/np.linalg.norm(Fc):.4f}")
    print(f"   -grad(E_euc) vs F_code : angle {ang(geuc,Fc):6.2f} deg   mag-ratio {np.linalg.norm(geuc)/np.linalg.norm(Fc):.4f}")

# on the unit sphere
h = rng.standard_normal(D); h/=np.linalg.norm(h)
report("h on unit sphere", h)
# off the sphere (code clamps ||h||<=10, normalizes only for cos)
report("h off sphere ||h||=3", h*3.0)
report("h off sphere ||h||=0.4", h*0.4)

# conservativity of the CODE force off-sphere: is Jacobian symmetric?
def jac(fn, h, c, eps=1e-6):
    J=np.zeros((len(h),len(h)))
    for j in range(len(h)):
        hp,hm=h.copy(),h.copy(); hp[j]+=eps; hm[j]-=eps
        J[:,j]=(fn(hp,c)-fn(hm,c))/(2*eps)
    return J
c = rng.standard_normal(D); c/=np.linalg.norm(c)
for label,hh in [("unit",h),("||h||=3",h*3.0)]:
    J=jac(F_code,hh,c)
    asym=np.abs(J-J.T).max()
    print(f"conservativity (code force, {label}): max|J-J^T| = {asym:.2e}  -> {'symmetric/conservative' if asym<1e-4 else 'NON-conservative'}")
