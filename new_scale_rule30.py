#!/usr/bin/env python3
"""
"It's not chaos, just a NEW SCALE added."  Let's test that precisely.

A "new scale that is just a copy of the old scales" = SELF-SIMILAR = has a
closed-form shortcut across scales. Some CAs are exactly like this:

  Rule 90 :  a' = L XOR R                (purely linear)
             -> the whole pattern is the Sierpinski triangle, perfectly self-
                similar, and cell(t,x) = a BINOMIAL mod 2 you can read off in
                O(log t) via Lucas' theorem.  No iteration. "Just a new scale."

  Rule 30 :  a' = L XOR C XOR R XOR (C AND R)
             = Rule 90's idea PLUS the nonlinear (C AND R) term.
             -> that ONE extra term destroys self-similarity. No scale is a copy
                of the others; each adds genuinely new information.

So "just a new scale" is the signature of LINEAR rules. Rule 30 is one AND-gate
away from being one -- and that gate is the whole difference.
"""
from __future__ import annotations
import numpy as np, zlib

def evolve(rule_fn, N):
    W=2*N+5; s=W//2; row=np.zeros(W,np.uint8); row[s]=1
    g=np.empty((N,W),np.uint8)
    for t in range(N): g[t]=row; row=rule_fn(row)
    return g, s

rule90=lambda r: np.bitwise_xor(np.roll(r,1), np.roll(r,-1))
rule30=lambda r: np.bitwise_xor(np.roll(r,1), np.bitwise_or(r, np.roll(r,-1)))

def lucas_cell(t,x):
    """Closed-form Rule 90 cell from single center seed: binomial(t,k) mod 2."""
    if (t+x)%2 or abs(x)>t: return 0
    k=(t-x)//2
    return 1 if (k & (t-k))==0 else 0      # binom(t,k) odd  <=>  k submask of t

def compress(bits):
    b=np.packbits(bits.astype(np.uint8)).tobytes(); return len(zlib.compress(b,9))/max(len(b),1)

def main():
    N=600
    g90,s=evolve(rule90,N); g30,_=evolve(rule30,N)

    print("="*80)
    print("1) CLOSED-FORM ACROSS SCALES  (predict cell(t,x) WITHOUT iterating)")
    print("="*80)
    # Rule 90: check Lucas formula vs the real grid
    ok=sum(lucas_cell(t,x)==g90[t,s+x] for t in range(N) for x in range(-t,t+1,1) if abs(x)<=t)
    tot=sum(1 for t in range(N) for x in range(-t,t+1,1) if abs(x)<=t)
    print(f"  Rule 90 : Lucas closed-form matches the grid on {ok}/{tot} cells = {ok/tot:.4f}")
    # predict an ENORMOUS t instantly, no iteration:
    T=2**40
    print(f"  Rule 90 : cell(t={T}, x=12) via closed form  = {lucas_cell(T,12)}   (instant, no iteration)")
    # Does the SAME formula predict Rule 30? (use it as a would-be shortcut)
    ok30=sum(lucas_cell(t,x)==g30[t,s+x] for t in range(N) for x in range(-t,t+1) if abs(x)<=t)
    print(f"  Rule 30 : same closed-form matches on {ok30/tot:.4f}  (i.e. it does NOT -- chance)")

    print("\n"+"="*80)
    print("2) SELF-SIMILARITY / SCALE STRUCTURE  (compressibility of the pattern)")
    print("="*80)
    def patt(g): return np.concatenate([g[t, s-t:s+t+1] for t in range(1,N)])
    print(f"  Rule 90 whole triangle compresses to : {compress(patt(g90)):.3f}  (tiny = self-similar fractal)")
    print(f"  Rule 30 whole triangle compresses to : {compress(patt(g30)):.3f}  (~1 = no scale is a copy)")

    print("\n"+"="*80)
    print("3) THE DIFFERENCE IS EXACTLY ONE TERM")
    print("="*80)
    print("   Rule 90 :  L XOR R                         -> self-similar, closed form")
    print("   Rule 30 :  L XOR C XOR R XOR (C AND R)     -> irreducible")
    print("   The (C AND R) gate is the entire difference between")
    print("   'just a new scale' and 'genuinely new information each scale'.")
    print("="*80)
    print("VERDICT: you're right that it's deterministic, not random. But 'just a new")
    print("scale' means self-similar = linear (Rule 90), where a closed form predicts")
    print("any depth instantly. Rule 30's one nonlinear gate makes each new scale")
    print("depend on ALL the detail beneath it -- new information, not a rescaled copy.")
    print("="*80)

if __name__=="__main__":
    main()
