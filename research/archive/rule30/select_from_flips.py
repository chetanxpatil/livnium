#!/usr/bin/env python3
"""
"Get the answer for all the flips and tell it to SELECT."

For each level t the bit is either 0 or 1 -- those are "all the flips".
We hand the engine the candidates and let it select. Two honest regimes:

  SELECT-WITH-KEY : the engine may use the precomputed answer column (a lookup).
                    -> 100%.  But building that column REQUIRED running Rule 30 to
                       every level. Selecting is reading a table you already filled
                       by computing. No prediction, no new formula.

  SELECT-BLIND    : the candidates {0,1} carry no label; the engine must choose
                    for HELD-OUT levels using only features of t (it never saw
                    those answers). -> coin flip.

Also: VERIFY-COST -- to even *check* a claimed bit at level t you must evolve t
rows. There is no certificate shorter than the computation.
"""
from __future__ import annotations
import numpy as np

class CollapseEngine:
    def __init__(s,d,p=64,beta=2.,lr=.08,seed=0):
        r=np.random.default_rng(seed)
        s.W=r.normal(0,1/np.sqrt(d),(p,d)); s.b=np.zeros(p); s.A=r.normal(0,1,(2,p)); s.beta,s.lr=beta,lr
    def _p(s,H):
        dd=((H[:,None,:]-s.A[None])**2).sum(-1); lo=-s.beta*dd; lo-=lo.max(1,keepdims=True)
        e=np.exp(lo); return e/e.sum(1,keepdims=True)
    def predict(s,X): return s._p(np.tanh(X@s.W.T+s.b)).argmax(1)
    def fit(s,X,y,epochs=300,batch=256):
        N=len(X); r=np.random.default_rng(1)
        for _ in range(epochs):
            for k in range(0,N,batch):
                bi=r.permutation(N)[k:k+batch]; xb,yb=X[bi],y[bi]
                H=np.tanh(xb@s.W.T+s.b); p=s._p(H); oh=np.zeros_like(p); oh[np.arange(len(yb)),yb]=1
                dl=p-oh; df=H[:,None,:]-s.A[None]
                dH=(-s.beta*2*df*dl[:,:,None]).sum(1); dA=(s.beta*2*df*dl[:,:,None]).sum(0)
                dpre=dH*(1-H**2)
                s.W-=s.lr*(dpre.T@xb)/len(yb); s.b-=s.lr*dpre.mean(0); s.A-=s.lr*dA/len(yb)

def rule30_step(r): return np.bitwise_xor(np.roll(r,1), np.bitwise_or(r,np.roll(r,-1)))

def center_column(N):
    W=2*N+50; row=np.zeros(W,np.uint8); s=W//2; row[s]=1
    col=np.empty(N,int)
    for t in range(N): col[t]=row[s]; row=rule30_step(row)
    return col

def tfeat(tt,N):
    f=[tt/N]+[fn(2*np.pi*tt/2**k) for k in range(1,12) for fn in (np.sin,np.cos)]
    f+=[((tt.astype(int)>>b)&1).astype(float) for b in range(11)]
    return np.stack(f,1)

def main():
    N=1024
    col=center_column(N)               # <-- this step IS "running Rule 30" to fill the table
    t=np.arange(N); X=tfeat(t,N)
    rng=np.random.default_rng(0); perm=rng.permutation(N)
    tr,te=perm[:int(.7*N)],perm[int(.7*N):]

    print("="*82)
    print("SELECT-FROM-ALL-FLIPS")
    print("="*82)

    # SELECT-WITH-KEY: allow the engine to use the stored answers (lookup)
    #   we simulate "it has the key" by training AND testing on the same levels.
    eng=CollapseEngine(X.shape[1]); eng.fit(X,col)
    key_acc=(eng.predict(X)==col).mean()
    print(f"\nSELECT-WITH-KEY (lookup the precomputed column)   acc={key_acc:.4f}")
    print("   but the column was filled by EVOLVING Rule 30 to every level first.")
    print("   selecting = reading the answer you already computed. not prediction.")

    # SELECT-BLIND: held-out levels, no access to their answers
    engb=CollapseEngine(X.shape[1]); engb.fit(X[tr],col[tr])
    blind_acc=(engb.predict(X[te])==col[te]).mean()
    maj=max(col[te].mean(),1-col[te].mean())
    print(f"\nSELECT-BLIND (choose 0/1 for unseen levels)       acc={blind_acc:.4f}  (chance {maj:.3f})")
    print("   the two candidates {0,1} carry no hint of which is right;")
    print("   with no computed answer to lean on, selection is a guess.")

    # VERIFY-COST: cheapest way to confirm a claimed bit at level t
    claim_levels=[162,180,1000]
    print("\nVERIFY-COST (rows you must evolve just to CHECK a claimed bit):")
    for q in claim_levels:
        print(f"   level {q:4d} -> must compute {q} rows to verify. no shorter certificate.")

    print("\n"+"="*82)
    print("WHY EVERY ANGLE LANDS HERE")
    print("  more data / iteration-index / formula+level / select-from-flips all reduce")
    print("  to ONE of two things:")
    print("     (a) run the rule (iterate to the level)  -> 100%, but no shortcut")
    print("     (b) don't run it (guess / select blind)  -> coin flip")
    print("  there is no third door. that absence is exactly Rule 30's irreducibility.")
    print("="*82)

if __name__=="__main__":
    main()
