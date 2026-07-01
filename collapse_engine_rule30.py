#!/usr/bin/env python3
"""
Vector Collapse Engine vs Rule 30 center bit.

Faithful, self-contained NumPy reimplementation of the livnium "collapse engine"
mechanism:
  - a learnable linear projection  h = W x + b
  - K basin anchors  a_0 (bit=0), a_1 (bit=1)
  - "collapse" = soft assignment to nearest basin by squared distance
        logits_k = -beta * ||h - a_k||^2 ,  p = softmax(logits)
  - trained supervised with cross-entropy (gradient descent on W, b, anchors).

We give it Rule 30 data under several regimes and ask: does it reach 100%?

Setup: random initial conditions on a width-W ring, two DISJOINT trajectories
(train seed / test seed). We track one fixed column = "the center column".
Next value of that column is, by definition, Rule30(left, self, right) of the
current row -- a deterministic local function.
"""
from __future__ import annotations
import numpy as np
from sklearn.decomposition import PCA

# ----------------------------- Rule 30 data -----------------------------
def rule30_step(row):
    left = np.roll(row, 1); right = np.roll(row, -1)
    return np.bitwise_xor(left, np.bitwise_or(row, right))

def evolve(width, steps, seed):
    rng = np.random.default_rng(seed)
    row = rng.integers(0, 2, width, dtype=np.uint8)
    grid = np.empty((steps, width), dtype=np.uint8)
    for t in range(steps):
        grid[t] = row; row = rule30_step(row)
    return grid

# ----------------------------- Collapse Engine -----------------------------
class CollapseEngine:
    """Learnable projection + 2 basin anchors + alignment-softmax collapse."""
    def __init__(self, in_dim, proj_dim=16, beta=2.0, lr=0.05, seed=0):
        rng = np.random.default_rng(seed)
        self.W = rng.normal(0, 1/np.sqrt(in_dim), (proj_dim, in_dim))
        self.b = np.zeros(proj_dim)
        self.A = rng.normal(0, 1.0, (2, proj_dim))   # two basins
        self.beta, self.lr = beta, lr

    def _project(self, X):
        return np.tanh(X @ self.W.T + self.b)        # (N, proj_dim)

    def _probs(self, H):
        # squared distance to each anchor -> softmax over -beta*d
        d = ((H[:, None, :] - self.A[None, :, :]) ** 2).sum(-1)   # (N,2)
        logits = -self.beta * d
        logits -= logits.max(1, keepdims=True)
        e = np.exp(logits); return e / e.sum(1, keepdims=True), d

    def predict(self, X):
        p, _ = self._probs(self._project(X)); return p.argmax(1)

    def fit(self, X, y, epochs=60, batch=512, verbose=False):
        N = len(X); rng = np.random.default_rng(1)
        for ep in range(epochs):
            idx = rng.permutation(N)
            for s in range(0, N, batch):
                bi = idx[s:s+batch]; xb = X[bi]; yb = y[bi]
                pre = xb @ self.W.T + self.b
                H = np.tanh(pre)
                p, _ = self._probs(H)
                onehot = np.zeros_like(p); onehot[np.arange(len(yb)), yb] = 1
                dlogit = (p - onehot)                         # (n,2)
                # logits_k = -beta * ||H-A_k||^2
                diff = H[:, None, :] - self.A[None, :, :]     # (n,2,proj)
                dH = (-self.beta * 2 * diff * dlogit[:, :, None]).sum(1)  # (n,proj)
                dA = ( self.beta * 2 * diff * dlogit[:, :, None]).sum(0)  # (2,proj)
                dpre = dH * (1 - H**2)
                dW = dpre.T @ xb / len(yb)
                db = dpre.mean(0)
                self.W -= self.lr * dW; self.b -= self.lr * db
                self.A -= self.lr * dA / len(yb)
            if verbose and (ep+1) % 20 == 0:
                acc = (self.predict(X) == y).mean()
                print(f"   epoch {ep+1:3d}  train acc {acc:.4f}")

def run(name, Xtr, ytr, Xte, yte, epochs=60, **kw):
    eng = CollapseEngine(Xtr.shape[1], **kw)
    eng.fit(Xtr, ytr, epochs=epochs)
    tr = (eng.predict(Xtr) == ytr).mean()
    te = (eng.predict(Xte) == yte).mean()
    maj = max(yte.mean(), 1 - yte.mean())
    flag = "<-- 100%" if te > 0.999 else ("(chance)" if te < maj + 0.02 else "")
    print(f"{name:<52} train={tr:.4f}  test={te:.4f}  (majority {maj:.3f}) {flag}")
    return te

# ----------------------------- Experiment -----------------------------
def main():
    W, T = 401, 6000
    c = W // 2
    g_tr = evolve(W, T, seed=1)
    g_te = evolve(W, T, seed=2)

    # label = NEXT value of the center column
    ytr = g_tr[1:, c].astype(int); yte = g_te[1:, c].astype(int)

    print("=" * 92)
    print("VECTOR COLLAPSE ENGINE  ->  Rule 30 center bit")
    print(f"width={W}  steps={T}  proj_dim=16   (train seed1 / test seed2, disjoint)")
    print(f"center-bit balance: train p(1)={ytr.mean():.3f}  test p(1)={yte.mean():.3f}")
    print("=" * 92)

    # --- Regime 0: SAME-TIME leakage (answer is literally in the input) ---
    X0tr = g_tr[1:].astype(float); X0te = g_te[1:].astype(float)   # row that CONTAINS the bit
    run("0. LEAK  full row -> SAME-time center bit", X0tr, ytr, X0te, yte)

    # --- Regime 1: local 3 cells of current row -> next center bit ---
    loc = [c-1, c, c+1]
    X1tr = g_tr[:-1][:, loc].astype(float); X1te = g_te[:-1][:, loc].astype(float)
    run("1. LOCAL3  [c-1,c,c+1] -> next center bit", X1tr, ytr, X1te, yte, lr=0.1, epochs=120)

    # --- Regime 2: FULL current row -> next center bit ("all the data") ---
    X2tr = g_tr[:-1].astype(float); X2te = g_te[:-1].astype(float)
    run("2. FULLROW  entire current row -> next bit", X2tr, ytr, X2te, yte)

    # --- Regime 3a: GEOMETRY only -- PCA-8 of full row (the paper's setup) ---
    pca = PCA(n_components=8, random_state=0).fit(g_tr[:-1].astype(float))
    X3tr = pca.transform(g_tr[:-1].astype(float)); X3te = pca.transform(g_te[:-1].astype(float))
    run("3a. GEOMETRY PCA-8(row) -> next center bit", X3tr, ytr, X3te, yte)

    # --- Regime 3b: NON-LOCAL -- full row with the 3 local cells MASKED out ---
    mask = np.ones(W, bool); mask[loc] = False
    X4tr = g_tr[:-1][:, mask].astype(float); X4te = g_te[:-1][:, mask].astype(float)
    run("3b. NON-LOCAL  row minus [c-1,c,c+1] -> next bit", X4tr, ytr, X4te, yte)

    print("=" * 92)
    print("INTERPRETATION")
    print("  100% is reachable ONLY when the 3 local cells are present (regimes 0,1,2):")
    print("  the engine just relearns Rule 30's lookup table from input/output pairs.")
    print("  Strip the local neighborhood (3a geometry / 3b non-local) and it falls to")
    print("  chance: the center column is computationally irreducible. No engine fixes that.")
    print("=" * 92)

if __name__ == "__main__":
    main()
