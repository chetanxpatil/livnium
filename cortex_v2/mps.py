"""
cortex_v2.mps — Matrix Product State simulator with built-in governor.

What changed vs v1 (334 lines + two governor classes, 522 lines):
  - one class, ~180 lines; governor folded into the SVD split
    (the only place truncation can physically happen) instead of a
    post-hoc per-bond enforcement pass with a second SVD
  - gate constants built once at module load
  - measure_all uses right environments (same algorithm, no copies)
  - honest accounting: trunc_error is the discarded squared weight

Governor semantics (simplified, documented):
  effective entropy ceiling = s_max * (1 + alpha)
  Higher alpha -> higher ceiling -> fewer prunes (v1 invariant T9).
"""

from __future__ import annotations

import numpy as np

_SQ2 = 1.0 / np.sqrt(2.0)
H = np.array([[_SQ2, _SQ2], [_SQ2, -_SQ2]], dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
CNOT = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex).reshape(
    2, 2, 2, 2
)
SWAP = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=complex).reshape(
    2, 2, 2, 2
)


def rx(theta: float) -> np.ndarray:
    c, s = np.cos(theta / 2), -1j * np.sin(theta / 2)
    return np.array([[c, s], [s, c]], dtype=complex)


def rz(theta: float) -> np.ndarray:
    return np.array([[np.exp(-1j * theta / 2), 0], [0, np.exp(1j * theta / 2)]], dtype=complex)


def entropy(s: np.ndarray) -> float:
    """von Neumann entropy (nats) of a singular-value spectrum."""
    p = s * s
    t = p.sum()
    if t <= 0:
        return 0.0
    p = p / t
    p = p[p > 1e-15]
    return float(-(p * np.log(p)).sum())


class MPS:
    """MPS on n two-level sites. tensors[i]: (chi_l, 2, chi_r). Starts in |0...0>."""

    def __init__(self, n: int, max_chi: int = 64, s_max: float | None = None, alpha: float = 0.5):
        self.n = n
        self.max_chi = max_chi
        self.s_max = s_max  # None -> no governor, chi cap only
        self.alpha = alpha  # geometric signal, set per word
        self.prune_events = 0
        self.trunc_error = 0.0
        self.max_chi_used = 1
        self.tensors = [np.zeros((1, 2, 1), dtype=complex) for _ in range(n)]
        for t in self.tensors:
            t[0, 0, 0] = 1.0

    # -- diagnostics ------------------------------------------------------
    def bond_dims(self):
        return [self.tensors[i].shape[2] for i in range(self.n - 1)]

    def memory_bytes(self):
        return sum(t.nbytes for t in self.tensors)

    def bond_entropies(self):
        out = []
        for i in range(self.n - 1):
            th = np.tensordot(self.tensors[i], self.tensors[i + 1], axes=([2], [0]))
            l, _, _, r = th.shape
            s = np.linalg.svd(th.reshape(l * 2, 2 * r), compute_uv=False)
            out.append(entropy(s))
        return out

    # -- gates ------------------------------------------------------------
    def apply_1q(self, site: int, U: np.ndarray):
        t = self.tensors[site]
        self.tensors[site] = np.tensordot(U, t, axes=([1], [1])).transpose(1, 0, 2)

    def hadamard(self, s):
        self.apply_1q(s, H)

    def pauli_x(self, s):
        self.apply_1q(s, X)

    def pauli_z(self, s):
        self.apply_1q(s, Z)

    def rx_gate(self, s, th):
        self.apply_1q(s, rx(th))

    def rz_gate(self, s, th):
        self.apply_1q(s, rz(th))

    def _apply_2q_adjacent(self, i: int, G: np.ndarray):
        th = np.tensordot(self.tensors[i], self.tensors[i + 1], axes=([2], [0]))
        th = np.tensordot(G, th, axes=([2, 3], [1, 2])).transpose(2, 0, 1, 3)
        self._split(th, i)

    def _split(self, theta: np.ndarray, left: int):
        l, _, _, r = theta.shape
        U, s, Vh = np.linalg.svd(theta.reshape(l * 2, 2 * r), full_matrices=False)
        w = s * s
        total = float(w.sum())
        k = int((s > 1e-12).sum()) or 1
        if k > self.max_chi:
            k = self.max_chi
        # governor: shrink k until entropy fits under the effective ceiling
        if self.s_max is not None:
            ceiling = self.s_max * (1.0 + self.alpha)
            pruned = False
            while k > 1 and entropy(s[:k]) > ceiling:
                k -= 1
                pruned = True
            if pruned:
                self.prune_events += 1  # one event per governed split
        kept = float(w[:k].sum())
        if total > 0:
            self.trunc_error += 1.0 - kept / total
        # Rescale only to compensate for *discarded* weight, preserving the
        # local block's own norm. Dividing by sqrt(kept) alone silently assumes
        # ||theta||^2 == 1 (true only in canonical gauge); that assumption fails
        # on the leftward swap-back leg of a long-range CNOT once a bond carries
        # real entanglement, collapsing the global norm. sqrt(total/kept) is a
        # no-op when there is no truncation (kept == total) and otherwise only
        # corrects the truncation loss.
        sk = s[:k] * np.sqrt(total / kept) if kept > 0 else s[:k]
        self.tensors[left] = U[:, :k].reshape(l, 2, k)
        self.tensors[left + 1] = (sk[:, None] * Vh[:k]).reshape(k, 2, r)
        if k > self.max_chi_used:
            self.max_chi_used = k

    def _swap_adjacent(self, i: int):
        self._apply_2q_adjacent(i, SWAP)

    def cnot(self, control: int, target: int):
        if abs(control - target) == 1:
            if control < target:
                self._apply_2q_adjacent(control, CNOT)
            else:
                # reverse CNOT = (SWAP, CNOT, SWAP) collapsed to a flipped gate
                G = CNOT.transpose(1, 0, 3, 2)
                self._apply_2q_adjacent(target, G)
            return
        # bring control next to target with swaps, apply, swap back
        c, t = control, target
        path = []
        while abs(c - t) > 1:
            step = c + (1 if t > c else -1)
            self._swap_adjacent(min(c, step))
            path.append(min(c, step))
            c = step
        self.cnot(c, t)
        for i in reversed(path):
            self._swap_adjacent(i)

    # -- measurement ------------------------------------------------------
    def measure_all(self, rng: np.random.Generator | None = None):
        rng = rng or np.random.default_rng()
        # right environments: R[i] = sum_s A_s R[i+1] A_s^dagger
        R = [None] * (self.n + 1)
        R[self.n] = np.array([[1.0 + 0j]])
        for i in range(self.n - 1, -1, -1):
            t = self.tensors[i]
            R[i] = sum(t[:, m, :] @ R[i + 1] @ t[:, m, :].conj().T for m in range(2))
        rho = np.array([[1.0 + 0j]])
        out = []
        for i in range(self.n):
            t = self.tensors[i]
            probs, conds = [], []
            for m in range(2):
                Xm = t[:, m, :].conj().T @ rho @ t[:, m, :]
                conds.append(Xm)
                probs.append(max(float(np.trace(Xm @ R[i + 1]).real), 0.0))
            ptot = probs[0] + probs[1]
            p0 = probs[0] / ptot if ptot > 0 else 0.5
            m = 0 if rng.random() < p0 else 1
            out.append(m)
            rho = conds[m] / (probs[m] if probs[m] > 0 else 1.0)
        return out


def ghz(n: int, **kw) -> MPS:
    m = MPS(n, **kw)
    m.hadamard(0)
    for i in range(n - 1):
        m.cnot(i, i + 1)
    return m
