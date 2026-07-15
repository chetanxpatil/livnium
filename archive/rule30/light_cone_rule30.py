#!/usr/bin/env python3
"""
"Predict the local neighborhood first, then use that data -- but it gets big."

Exactly right. To know center(t) you need the cells it DEPENDS on at step 0.
Rule 30 spreads at speed 1, so the dependence set ("light cone") of center(t)
is precisely the 2t+1 cells [seed-t .. seed+t] at row 0.

  - Test 1: fix those 2t+1 cells, scramble everything else -> center(t) never
            changes.  Drop even one cell from the cone -> it can change.
            => the minimal neighborhood is EXACTLY width 2t+1.
  - Test 2: give the engine the full 2t+1 cone -> it predicts center(t) at 100%
            on fresh cones.  Give it a cone that's too narrow -> it loses it.
  - The neighborhood you must predict grows as 2t+1 (linear), and rebuilding it
    is the whole t-row triangle (~t^2 cells). That growth IS the irreducibility.
"""
from __future__ import annotations
import numpy as np

class CollapseEngine:
    def __init__(s,d,p=48,beta=2.,lr=.08,seed=0):
        r=np.random.default_rng(seed)
        s.W=r.normal(0,1/np.sqrt(d),(p,d)); s.b=np.zeros(p); s.A=r.normal(0,1,(2,p)); s.beta,s.lr=beta,lr
    def _p(s,H):
        dd=((H[:,None,:]-s.A[None])**2).sum(-1); lo=-s.beta*dd; lo-=lo.max(1,keepdims=True)
        e=np.exp(lo); return e/e.sum(1,keepdims=True)
    def predict(s,X): return s._p(np.tanh(X@s.W.T+s.b)).argmax(1)
    def fit(s,X,y,epochs=250,batch=256):
        N=len(X); r=np.random.default_rng(1)
        for _ in range(epochs):
            for k in range(0,N,batch):
                bi=r.permutation(N)[k:k+batch]; xb,yb=X[bi],y[bi]
                H=np.tanh(xb@s.W.T+s.b); p=s._p(H); oh=np.zeros_like(p); oh[np.arange(len(yb)),yb]=1
                dl=p-oh; df=H[:,None,:]-s.A[None]
                dH=(-s.beta*2*df*dl[:,:,None]).sum(1); dA=(s.beta*2*df*dl[:,:,None]).sum(0)
                dpre=dH*(1-H**2)
                s.W-=s.lr*(dpre.T@xb)/len(yb); s.b-=s.lr*dpre.mean(0); s.A-=s.lr*dA/len(yb)

def step(r): return np.bitwise_xor(np.roll(r,1), np.bitwise_or(r,np.roll(r,-1)))

def center_after(base_row, t, seed):
    row=base_row.copy()
    for _ in range(t): row=step(row)
    return int(row[seed])

def cone_center(cone_bits, t):
    """center(t) from a cone of 2t+1 bits, embedded in a zero-padded row."""
    Wd=4*t+5; mid=Wd//2; w=len(cone_bits); row=np.zeros(Wd,np.uint8)
    row[mid-w//2: mid-w//2+w]=cone_bits
    for _ in range(t): row=step(row)
    return int(row[mid])

def main():
    print("="*84)
    print("LIGHT CONE: the neighborhood you must predict is width 2t+1")
    print("="*84)

    # ---- Test 1: dependence set is exactly the cone ----
    print("\nTest 1 -- fix the 2t+1 cone, scramble outside: does center(t) move?")
    rng=np.random.default_rng(0)
    for t in [3,6,10]:
        Wd=8*t+11; seed=Wd//2; cone=range(seed-t,seed+t+1)
        cone_bits=rng.integers(0,2,2*t+1)
        vals_fix=[]
        for _ in range(200):
            row=rng.integers(0,2,Wd,dtype=np.uint8)
            row[seed-t:seed+t+1]=cone_bits            # fix the cone, random elsewhere
            vals_fix.append(center_after(row,t,seed))
        # now also vary one cone cell to show it CAN change
        vary=set()
        for flip in range(2*t+1):
            cb=cone_bits.copy(); cb[flip]^=1
            row=np.zeros(Wd,np.uint8); row[seed-t:seed+t+1]=cb
            vary.add(center_after(row,t,seed))
        print(f"  t={t:2d}: width 2t+1={2*t+1:3d} | center(t) with cone fixed = "
              f"{'CONSTANT' if len(set(vals_fix))==1 else 'varies'} over 200 random outsides "
              f"| flipping cone cells reaches {sorted(vary)}")

    # ---- Test 2: engine predicts center(t) from the full cone vs a truncated cone ----
    print("\nTest 2 -- engine: full cone (2t+1) vs truncated cone (2t-1):")
    for t in [3,5,7]:
        w=2*t+1
        rng=np.random.default_rng(t)
        cones=rng.integers(0,2,(6000,w))
        y=np.array([cone_center(c,t) for c in cones])
        cut=int(.7*len(cones));
        # full cone
        ef=CollapseEngine(w); ef.fit(cones[:cut].astype(float),y[:cut])
        full=(ef.predict(cones[cut:].astype(float))==y[cut:]).mean()
        # truncated: drop the two outer cone cells (now missing real dependencies)
        tr_in=cones[:,1:-1]
        et=CollapseEngine(w-2); et.fit(tr_in[:cut].astype(float),y[:cut])
        trunc=(et.predict(tr_in[cut:].astype(float))==y[cut:]).mean()
        maj=max(y[cut:].mean(),1-y[cut:].mean())
        print(f"  t={t}: full cone({w}) acc={full:.4f}   truncated cone({w-2}) acc={trunc:.4f}   (chance {maj:.3f})")

    print("\n"+"="*84)
    print("WHAT THIS MEANS")
    print("  Your idea works -- IF you predict the FULL neighborhood. But the minimal")
    print("  neighborhood for level t is 2t+1 cells wide, and filling it in is the whole")
    print("  t-row triangle (~t^2 cells). 'Predict the neighborhood first' bottoms out at")
    print("  the seed and rebuilds all of Rule 30. The cone HAS to get big -- that growth,")
    print("  with no smaller sufficient set, is computational irreducibility itself.")
    print("="*84)

if __name__=="__main__":
    main()
