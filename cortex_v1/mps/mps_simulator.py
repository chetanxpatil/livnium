"""
MPS (Matrix Product State) Quantum Simulator
=============================================

Implements quantum states as a chain of tensors:

  ψ = A[0] · A[1] · A[2] · ... · A[n-1]

Each tensor A[i] has shape (χ_left, 2, χ_right) where:
  χ = bond dimension — controls how much entanglement is represented
  2 = qubit dimension (|0> or |1>)

Memory: O(n × χ²)  vs  O(2^n) for dense state vector

The bond dimension χ is the key parameter:
  χ = 1  → product state (no entanglement), trivial
  χ = 2  → can represent GHZ exactly
  χ = 2^(n/2)  → exact representation of any state (= dense vector)

The "crash test" lives here: start small χ, add entanglement, watch χ grow.
"""

import numpy as np
from typing import List, Tuple, Optional


class MPSSimulator:
    """
    Quantum simulator using Matrix Product States.

    Can represent thousands of qubits for LOW-entanglement states.
    Bond dimension χ grows when entanglement grows.
    At max χ = 2^(n/2) this is equivalent to the full dense vector.
    """

    def __init__(self, n_qubits: int, max_bond_dim: int = 64):
        """
        Args:
            n_qubits    : number of qubits
            max_bond_dim: maximum allowed bond dimension χ.
                          When the state needs χ > this, it gets truncated
                          (introduces approximation error).
                          Set to 2^(n//2) for exact simulation.
        """
        self.n = n_qubits
        self.max_chi = max_bond_dim

        # Initialise all qubits to |0>
        # Each tensor: shape (chi_left, 2, chi_right)
        # For |0>: A[0] = 1, A[1] = 0
        self.tensors: List[np.ndarray] = []
        for i in range(n_qubits):
            t = np.zeros((1, 2, 1), dtype=complex)
            t[0, 0, 0] = 1.0   # |0>
            self.tensors.append(t)

        self.truncation_errors: List[float] = []  # track approximation loss

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def bond_dims(self) -> List[int]:
        """Bond dimensions between each pair of adjacent sites."""
        return [self.tensors[i].shape[2] for i in range(self.n - 1)]

    @property
    def max_bond_dim_used(self) -> int:
        return max(self.bond_dims) if self.bond_dims else 1

    @property
    def memory_bytes(self) -> int:
        return sum(t.nbytes for t in self.tensors)

    # ------------------------------------------------------------------
    # Contract to dense vector (only feasible for small n)
    # ------------------------------------------------------------------

    def to_dense(self) -> np.ndarray:
        """Contract full MPS to a 2^n dense state vector."""
        if self.n > 25:
            raise MemoryError(f"Cannot contract {self.n}-qubit MPS to dense vector (2^{self.n} entries).")
        result = self.tensors[0][:, :, :]   # (1, 2, chi)
        for i in range(1, self.n):
            # result shape: (1, 2^i, chi_left)
            # tensors[i]:   (chi_left, 2, chi_right)
            chi_l = result.shape[2]
            d_left = result.shape[1]
            chi_r = self.tensors[i].shape[2]
            # contract over chi_left
            result = np.tensordot(result, self.tensors[i], axes=([2], [0]))
            # result: (1, d_left, d_right, chi_r)
            result = result.reshape(1, d_left * 2, chi_r)
        return result[0, :, 0]

    # ------------------------------------------------------------------
    # Single-qubit gates
    # ------------------------------------------------------------------

    def _apply_single_gate(self, site: int, gate: np.ndarray):
        """Apply a 2×2 gate to qubit at `site`."""
        t = self.tensors[site]                    # (chi_l, 2, chi_r)
        t = np.tensordot(gate, t, axes=([1], [1]))  # (2, chi_l, chi_r)
        self.tensors[site] = t.transpose(1, 0, 2)   # (chi_l, 2, chi_r)

    _H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
    _X = np.array([[0, 1], [1, 0]], dtype=complex)
    _Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    _Z = np.array([[1, 0], [0, -1]], dtype=complex)

    def hadamard(self, site: int):
        self._apply_single_gate(site, self._H)

    def pauli_x(self, site: int):
        self._apply_single_gate(site, self._X)

    def pauli_z(self, site: int):
        self._apply_single_gate(site, self._Z)

    def rz(self, site: int, theta: float):
        """Rz rotation — introduces phase."""
        gate = np.array([[np.exp(-1j * theta / 2), 0],
                         [0, np.exp(1j * theta / 2)]], dtype=complex)
        self._apply_single_gate(site, gate)

    def rx(self, site: int, theta: float):
        """Rx rotation."""
        c, s = np.cos(theta / 2), np.sin(theta / 2)
        gate = np.array([[c, -1j * s], [-1j * s, c]], dtype=complex)
        self._apply_single_gate(site, gate)

    # ------------------------------------------------------------------
    # Two-qubit gates (CNOT between ADJACENT sites only in base MPS)
    # ------------------------------------------------------------------

    def cnot(self, control: int, target: int):
        """
        CNOT between adjacent qubits (control, target = control±1).
        Increases bond dimension by up to 2× — this is where entanglement grows.
        SVD truncation is applied if χ exceeds max_bond_dim.
        """
        if abs(control - target) != 1:
            # Swap to make adjacent, apply, swap back
            self._cnot_non_adjacent(control, target)
            return

        # Merge the two tensors into a 4-index theta tensor
        tc = self.tensors[control]   # (chi_l, 2, chi_mid)
        tt = self.tensors[target]    # (chi_mid, 2, chi_r)

        # Contract: theta[chi_l, d_c, d_t, chi_r]
        theta = np.tensordot(tc, tt, axes=([2], [0]))   # (chi_l, 2, 2, chi_r) — wait, need reshape
        chi_l = tc.shape[0]
        chi_r = tt.shape[2]
        theta = theta.reshape(chi_l, 2, 2, chi_r)  # (chi_l, d_c, d_t, chi_r)

        # Apply CNOT gate to (d_c, d_t)
        cnot_gate = np.array([[1, 0, 0, 0],
                              [0, 1, 0, 0],
                              [0, 0, 0, 1],
                              [0, 0, 1, 0]], dtype=complex).reshape(2, 2, 2, 2)
        # theta_new[chi_l, d_c', d_t', chi_r] = sum_{d_c, d_t} CNOT[d_c', d_t', d_c, d_t] * theta[chi_l, d_c, d_t, chi_r]
        theta = np.tensordot(cnot_gate, theta, axes=([2, 3], [1, 2]))
        # shape: (2, 2, chi_l, chi_r) → (chi_l, 2, 2, chi_r)
        theta = theta.transpose(2, 0, 1, 3)

        # SVD to split back into two tensors
        self._split_svd(theta, control, target)

    def _split_svd(self, theta: np.ndarray, left_site: int, right_site: int):
        """
        Split theta (chi_l, 2, 2, chi_r) via SVD into two MPS tensors.
        Truncates to max_bond_dim if needed.
        """
        chi_l, d1, d2, chi_r = theta.shape
        mat = theta.reshape(chi_l * d1, d2 * chi_r)

        U, S, Vh = np.linalg.svd(mat, full_matrices=False)

        # Truncate
        chi_new = min(len(S), self.max_chi)
        trunc_err = float(np.sum(S[chi_new:] ** 2)) if chi_new < len(S) else 0.0
        self.truncation_errors.append(trunc_err)

        U  = U[:, :chi_new]
        S  = S[:chi_new]
        Vh = Vh[:chi_new, :]

        # Absorb S into left tensor
        US = U * S[np.newaxis, :]

        self.tensors[left_site]  = US.reshape(chi_l, d1, chi_new)
        self.tensors[right_site] = Vh.reshape(chi_new, d2, chi_r)

    def _cnot_non_adjacent(self, control: int, target: int):
        """
        Handle CNOT between non-adjacent sites using SWAP chain.
        Moves target next to control, applies CNOT, swaps back.
        """
        if control > target:
            # Swap control toward target
            for i in range(control, target + 1, -1):
                self._swap(i - 1, i)
            self.cnot(target, target + 1)
            for i in range(target + 1, control):
                self._swap(i, i + 1)
        else:
            for i in range(control, target - 1):
                self._swap(i, i + 1)
            self.cnot(target - 1, target)
            for i in range(target - 1, control, -1):
                self._swap(i - 1, i)

    def _swap(self, i: int, j: int):
        """Swap qubits i and j (must be adjacent)."""
        assert abs(i - j) == 1
        swap_gate = np.array([[1, 0, 0, 0],
                              [0, 0, 1, 0],
                              [0, 1, 0, 0],
                              [0, 0, 0, 1]], dtype=complex).reshape(2, 2, 2, 2)
        tc = self.tensors[i]
        tt = self.tensors[j]
        chi_l, chi_r = tc.shape[0], tt.shape[2]
        theta = np.tensordot(tc, tt, axes=([2], [0])).reshape(chi_l, 2, 2, chi_r)
        theta = np.tensordot(swap_gate, theta, axes=([2, 3], [1, 2])).transpose(2, 0, 1, 3)
        self._split_svd(theta, i, j)

    # ------------------------------------------------------------------
    # Measurement
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Measurement (correct left/right environment propagation)
    # ------------------------------------------------------------------

    def _right_environments(self) -> List[np.ndarray]:
        """
        Compute right environments R[i] for all sites, right to left.

        R[n] = [[1]]  (trivial right boundary)
        R[i][a, a'] = sum_{σ, b, b'} A[i][a, σ, b] * R[i+1][b, b'] * conj(A[i][a', σ, b'])
                    = sum_σ  M_σ @ R[i+1] @ M_σ†

        R[i] is a chi_l[i] × chi_l[i] matrix.
        """
        R = [None] * (self.n + 1)
        R[self.n] = np.array([[1.0 + 0j]])
        for i in range(self.n - 1, -1, -1):
            t   = self.tensors[i]          # (chi_l, 2, chi_r)
            Rnext = R[i + 1]               # (chi_r, chi_r)
            chi_l = t.shape[0]
            Rnew  = np.zeros((chi_l, chi_l), dtype=complex)
            for sigma in range(2):
                M = t[:, sigma, :]         # (chi_l, chi_r)
                Rnew += M @ Rnext @ M.conj().T
            R[i] = Rnew
        return R

    def measure_all(self) -> List[int]:
        """
        Measure all qubits left to right using correct L/R environment propagation.

        P(m_i | m_0,...,m_{i-1}) ∝ Tr( L[i] @ M_m @ R[i+1] @ M_m† )

        Left env update after measuring m_i:
            L[i+1] = M_m.T @ L[i] @ M_m.conj()
        """
        R = self._right_environments()
        results = []
        L = np.array([[1.0 + 0j]])         # (1, 1) — trivial left boundary

        for i in range(self.n):
            t     = self.tensors[i]        # (chi_l, 2, chi_r)
            Rnext = R[i + 1]               # (chi_r, chi_r)

            probs = np.zeros(2)
            for m in range(2):
                M = t[:, m, :]             # (chi_l, chi_r)
                probs[m] = np.real(np.trace(L @ M @ Rnext @ M.conj().T))

            total = probs.sum()
            probs = probs / total if total > 1e-12 else np.array([0.5, 0.5])

            result = 1 if np.random.rand() < probs[1] else 0
            results.append(result)

            # Propagate left environment
            M_m = t[:, result, :]          # (chi_l, chi_r)
            L   = M_m.T @ L @ M_m.conj()  # (chi_r, chi_r)
            # Normalise to prevent numerical drift
            nl = np.linalg.norm(L)
            if nl > 1e-12:
                L /= nl

        return results

    def measure(self, site: int) -> int:
        """Measure a single qubit (uses full L/R contraction for correctness)."""
        # For a single measurement we need both environments.
        R = self._right_environments()
        # Build left env up to this site
        L = np.array([[1.0 + 0j]])
        for i in range(site):
            # We don't have previous outcomes — just contract the full transfer matrix
            t = self.tensors[i]
            chi_l = t.shape[0]
            Lnew  = np.zeros((t.shape[2], t.shape[2]), dtype=complex)
            for sigma in range(2):
                M = t[:, sigma, :]
                Lnew += M.T @ L @ M.conj()
            L = Lnew

        t     = self.tensors[site]
        Rnext = R[site + 1]
        probs = np.zeros(2)
        for m in range(2):
            M = t[:, m, :]
            probs[m] = np.real(np.trace(L @ M @ Rnext @ M.conj().T))

        total = probs.sum()
        probs = probs / total if total > 1e-12 else np.array([0.5, 0.5])

        result = 1 if np.random.rand() < probs[1] else 0

        # Project the tensor at this site
        proj = np.zeros((2, 2), dtype=complex)
        proj[result, result] = 1.0
        self._apply_single_gate(site, proj)
        n = np.linalg.norm(self.tensors[site])
        if n > 1e-12:
            self.tensors[site] /= n

        return result
