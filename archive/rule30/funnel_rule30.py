#!/usr/bin/env python3
"""
"Round both ends, make it a funnel -- the symmetry repeats, and the center is in
the wrong place."  Two precise claims, both testable.

CLAIM A -- "round the ends" (wrap the row into a RING / funnel mouth):
   a finite ring has finitely many states, so it MUST cycle. The pattern DOES
   repeat -- you're right. Question: how long is the repeat?

CLAIM B -- "the symmetry repeats / center is wrong":
   is the Rule 30 triangle mirror-symmetric about the seed (so the center is a
   real symmetry axis)?  Compare to Rule 90, which IS symmetric.
"""
from __future__ import annotations
import numpy as np

def ring_step30(state):
    L=np.roll(state,1); C=state; R=np.roll(state,-1)
    return np.bitwise_xor(L, np.bitwise_or(C,R))

def to_int(a):
    v=0
    for b in a: v=(v<<1)|int(b)
    return v

def ring_period(N, max_iter=4_000_000):
    state=np.zeros(N,np.uint8); state[N//2]=1
    seen={}; t=0
    while t<max_iter:
        key=to_int(state)
        if key in seen: return seen[key], t-seen[key]   # (transient, period)
        seen[key]=t; state=ring_step30(state); t+=1
    return None,None

def main():
    print("="*74)
    print("CLAIM A -- round the ends into a ring: it repeats. how long?")
    print("="*74)
    print(f"  {'ring N':>7} {'transient':>10} {'PERIOD':>12}   period/2^N")
    for N in [5,7,9,11,13,15,17,19,21,23,25]:
        tr,per=ring_period(N)
        if per is None: print(f"  {N:>7} {'>cap':>10} {'>cap':>12}"); continue
        print(f"  {N:>7} {tr:>10} {per:>12}   {per/2**N:.4f}")
    print("\n  -> yes, the funnel/ring REPEATS (you're right). but the period")
    print("     grows roughly exponentially with size -- a real cycle you can")
    print("     never reach at any useful scale. 'it repeats' with an")
    print("     astronomically long repeat IS irreducibility, wrapped.")

    # CLAIM B -- symmetry about the center
    print("\n"+"="*74)
    print("CLAIM B -- is the center a symmetry axis? (mirror the row about seed)")
    print("="*74)
    def triangle(rule_fn,N):
        W=2*N+5; s=W//2; row=np.zeros(W,np.uint8); row[s]=1
        g=np.empty((N,W),np.uint8)
        for t in range(N): g[t]=row; row=rule_fn(row)
        return g,s
    r90=lambda r: np.bitwise_xor(np.roll(r,1),np.roll(r,-1))
    r30=lambda r: np.bitwise_xor(np.roll(r,1),np.bitwise_or(r,np.roll(r,-1)))
    for name,fn in [("Rule 90",r90),("Rule 30",r30)]:
        g,s=triangle(fn,400); mism=0; tot=0
        for t in range(1,400):
            for k in range(1,t+1):
                tot+=1; mism+= int(g[t,s-k]!=g[t,s+k])   # left vs right of center
        print(f"  {name}: left/right mismatch about center = {mism/tot:.4f}  "
              f"({'SYMMETRIC' if mism==0 else 'ASYMMETRIC -- center is NOT a mirror axis'})")

    print("\n"+"="*74)
    print("VERDICT")
    print("  A) Rounding the ends makes it cycle -- the symmetry genuinely repeats.")
    print("     But the period blows up exponentially; the repeat is unreachable.")
    print("  B) Rule 90's center IS a true mirror axis. Rule 30's is NOT -- the rule")
    print("     treats left and right differently (L alone vs R inside the OR), so")
    print("     there is no reflection center. The geometric middle isn't a symmetry")
    print("     point; it's just where we read. You were right that 'the center is in")
    print("     the wrong place' -- for Rule 30 a symmetry center doesn't exist.")
    print("="*74)

if __name__=="__main__":
    main()
