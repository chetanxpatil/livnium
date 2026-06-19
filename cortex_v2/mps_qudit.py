"""
mps_qudit.py — MPS generalized from 2-level (qubit) to d-level (qudit) sites.

Setting d=3 makes the natural Livnium site a QUTRIT, so base-27 = 3^3 sits in
exactly 3 sites with ZERO wasted states (vs 5 qubits with 5 wasted).

Gates generalized from the qubit version:
  - Fourier gate F  (replaces Hadamard): F[j,k] = w^{jk}/sqrt(d), w = e^{2pi i/d}
  - SUM gate        (replaces CNOT):     SUM|a,b> = |a,(a+b) mod d>
"""
from __future__ import annotations
import numpy as np

class QuditMPS:
    """MPS on n d-level sites. tensors[i]: (chi_l, d, chi_r). Starts in |0...0>."""
    def __init__(self, n: int, d: int = 3, max_chi: int = 64):
        self.n, self.d, self.max_chi = n, d, max_chi
        self.trunc_error = 0.0
        self.max_chi_used = 1
        self.tensors = [np.zeros((1, d, 1), dtype=complex) for _ in range(n)]
        for t in self.tensors:
            t[0, 0, 0] = 1.0

    # gate constants -----------------------------------------------------
    def fourier(self):
        d = self.d; w = np.exp(2j * np.pi / d)
        return np.array([[w ** (j * k) for k in range(d)] for j in range(d)]) / np.sqrt(d)

    def sum_gate(self):
        d = self.d; G = np.zeros((d, d, d, d), dtype=complex)
        for a in range(d):
            for b in range(d):
                G[a, (a + b) % d, a, b] = 1.0   # out_a,out_b, in_a,in_b
        return G

    # ops ----------------------------------------------------------------
    def set_basis(self, values):
        """Prepare the computational basis state |v0 v1 ... v_{n-1}>."""
        for i, v in enumerate(values):
            t = np.zeros((1, self.d, 1), dtype=complex); t[0, v % self.d, 0] = 1.0
            self.tensors[i] = t

    def apply_1q(self, site, U):
        self.tensors[site] = np.tensordot(U, self.tensors[site], axes=([1], [1])).transpose(1, 0, 2)

    def apply_2q_adjacent(self, i, G):
        th = np.tensordot(self.tensors[i], self.tensors[i + 1], axes=([2], [0]))
        th = np.tensordot(G, th, axes=([2, 3], [1, 2])).transpose(2, 0, 1, 3)
        l, _, _, r = th.shape
        U, s, Vh = np.linalg.svd(th.reshape(l * self.d, self.d * r), full_matrices=False)
        k = min(self.max_chi, int((s > 1e-12).sum()) or 1)
        total = float((s * s).sum()); kept = float((s[:k] ** 2).sum())
        if total > 0: self.trunc_error += 1.0 - kept / total
        sk = s[:k] * np.sqrt(total / kept) if kept > 0 else s[:k]
        self.tensors[i] = U[:, :k].reshape(l, self.d, k)
        self.tensors[i + 1] = (sk[:, None] * Vh[:k]).reshape(k, self.d, r)
        self.max_chi_used = max(self.max_chi_used, k)

    def memory_bytes(self):
        return sum(t.nbytes for t in self.tensors)

    def measure_all(self, rng=None):
        rng = rng or np.random.default_rng()
        R = [None] * (self.n + 1); R[self.n] = np.array([[1.0 + 0j]])
        for i in range(self.n - 1, -1, -1):
            t = self.tensors[i]
            R[i] = sum(t[:, m, :] @ R[i + 1] @ t[:, m, :].conj().T for m in range(self.d))
        rho = np.array([[1.0 + 0j]]); out = []
        for i in range(self.n):
            t = self.tensors[i]; probs = []; conds = []
            for m in range(self.d):
                Xm = t[:, m, :].conj().T @ rho @ t[:, m, :]
                conds.append(Xm); probs.append(max(float(np.trace(Xm @ R[i + 1]).real), 0.0))
            tot = sum(probs); probs = [p / tot for p in probs] if tot > 0 else [1/self.d]*self.d
            r = rng.random(); c = 0.0; m = self.d - 1
            for j in range(self.d):
                c += probs[j]
                if r < c: m = j; break
            out.append(m); rho = conds[m] / (probs[m] * tot if probs[m] * tot > 0 else 1.0)
        return out

def to_trits(x, k=3):
    """base-27 symbol -> k base-3 digits (trits)."""
    return [(x // (3 ** i)) % 3 for i in range(k)]

def from_trits(trits):
    return sum(t * (3 ** i) for i, t in enumerate(trits))
