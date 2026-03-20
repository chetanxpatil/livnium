"""
Entanglement Governor
=====================

Monitors Von Neumann entropy across all bonds in an MPS and enforces
a maximum entropy ceiling by adaptive bond truncation.

Von Neumann entropy at bond i:
    S_i = -sum_k  λ_k² log(λ_k²)
    where λ_k are the normalised singular values at that bond.

    S = 0   → product state, no entanglement
    S = log(χ) → maximally entangled at this bond (saturated)

The governor:
  1. After every gate, recomputes S across all bonds
  2. If any bond exceeds S_max, truncates that bond
  3. Logs every pruning event with its entropy drop
  4. Exposes a live dashboard of the system's entanglement profile

This enforces "Area Law" behaviour — local concepts stay distinct,
long-range information is explicitly bounded.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from mps_simulator import MPSSimulator


def von_neumann_entropy(singular_values: np.ndarray) -> float:
    """
    S = -sum_k  p_k * log(p_k)  where  p_k = λ_k² / sum(λ²)

    singular_values: raw (unnormalised) singular values from SVD.
    Returns entropy in nats (use log2 for bits).
    """
    lam2 = singular_values ** 2
    norm = lam2.sum()
    if norm < 1e-15:
        return 0.0
    p = lam2 / norm
    # Clip to avoid log(0)
    p = p[p > 1e-15]
    return float(-np.sum(p * np.log(p)))


def bond_entropies(sim: MPSSimulator) -> List[float]:
    """
    Compute Von Neumann entropy at every bond of an MPS.

    For bond between site i and i+1:
      Reshape the left tensor to (chi_l*2, chi_r) and SVD.
      The singular values give the entanglement spectrum.
    """
    entropies = []
    for i in range(sim.n - 1):
        t = sim.tensors[i]              # (chi_l, 2, chi_r)
        chi_l, _, chi_r = t.shape
        mat = t.reshape(chi_l * 2, chi_r)
        _, s, _ = np.linalg.svd(mat, full_matrices=False)
        entropies.append(von_neumann_entropy(s))
    return entropies


class EntanglementGovernor:
    """
    Wraps an MPSSimulator and enforces a maximum Von Neumann entropy
    ceiling at every bond after every gate application.

    Parameters
    ----------
    sim        : the MPS to govern
    S_max      : maximum allowed entropy per bond (nats)
                 log(2)  ≈ 0.693 → at most 1 ebit per bond
                 log(64) ≈ 4.16  → up to 64-dimensional bond
    verbose    : print every pruning event
    """

    def __init__(
        self,
        sim: MPSSimulator,
        S_max: float = np.log(64),
        verbose: bool = True,
    ):
        self.sim     = sim
        self.S_max   = S_max
        self.verbose = verbose
        self.pruning_log: List[Dict] = []   # history of all prune events
        self.entropy_history: List[List[float]] = []  # per-gate snapshot

    # ------------------------------------------------------------------
    # Core: enforce entropy ceiling on a single bond
    # ------------------------------------------------------------------

    def _enforce_bond(self, site: int) -> bool:
        """
        Check bond between site and site+1. Truncate if S > S_max.
        Returns True if a pruning happened.
        """
        t = self.sim.tensors[site]         # (chi_l, 2, chi_r)
        chi_l, _, chi_r = t.shape
        mat = t.reshape(chi_l * 2, chi_r)

        U, s, Vh = np.linalg.svd(mat, full_matrices=False)
        S_current = von_neumann_entropy(s)

        if S_current <= self.S_max:
            return False

        # Find minimum chi that keeps S ≤ S_max
        # Greedily include singular values from largest down
        lam2 = s ** 2
        norm = lam2.sum()
        if norm < 1e-15:
            return False
        p = lam2 / norm

        cumulative_S = 0.0
        chi_keep = 1
        for k in range(len(p)):
            pk = p[k]
            if pk > 1e-15:
                cumulative_S -= pk * np.log(pk)
            if cumulative_S >= self.S_max and k > 0:
                chi_keep = k
                break
        else:
            chi_keep = len(s)

        chi_keep = max(1, chi_keep)
        if chi_keep >= len(s):
            return False

        S_after = von_neumann_entropy(s[:chi_keep])
        trunc_err = float(np.sum(s[chi_keep:] ** 2))

        # Apply truncation
        U_t  = U[:, :chi_keep]
        S_t  = s[:chi_keep]
        Vh_t = Vh[:chi_keep, :]

        US = U_t * S_t[np.newaxis, :]
        self.sim.tensors[site]     = US.reshape(chi_l, 2, chi_keep)
        self.sim.tensors[site + 1] = (Vh_t @ self.sim.tensors[site + 1].reshape(chi_r, -1)).reshape(chi_keep, 2, self.sim.tensors[site + 1].shape[2])

        event = {
            "bond"     : site,
            "chi_before": len(s),
            "chi_after" : chi_keep,
            "S_before"  : S_current,
            "S_after"   : S_after,
            "trunc_err" : trunc_err,
        }
        self.pruning_log.append(event)

        if self.verbose:
            print(f"  [PRUNE] bond {site}: χ {len(s)}→{chi_keep}, "
                  f"S {S_current:.3f}→{S_after:.3f}, "
                  f"trunc_err={trunc_err:.2e}")
        return True

    def enforce_all(self):
        """Sweep all bonds and enforce the entropy ceiling."""
        for i in range(self.sim.n - 1):
            self._enforce_bond(i)

    # ------------------------------------------------------------------
    # Gate wrappers (apply gate then enforce)
    # ------------------------------------------------------------------

    def hadamard(self, site: int):
        self.sim.hadamard(site)
        self.enforce_all()
        self._snapshot()

    def cnot(self, control: int, target: int):
        self.sim.cnot(control, target)
        self.enforce_all()
        self._snapshot()

    def rz(self, site: int, theta: float):
        self.sim.rz(site, theta)
        # Single-qubit gates don't change entanglement — no need to sweep
        self._snapshot()

    def rx(self, site: int, theta: float):
        self.sim.rx(site, theta)
        self._snapshot()

    def measure_all(self) -> List[int]:
        return self.sim.measure_all()

    # ------------------------------------------------------------------
    # Monitoring
    # ------------------------------------------------------------------

    def _snapshot(self):
        self.entropy_history.append(bond_entropies(self.sim))

    def current_entropies(self) -> List[float]:
        return bond_entropies(self.sim)

    def max_entropy(self) -> float:
        e = self.current_entropies()
        return max(e) if e else 0.0

    def total_entanglement(self) -> float:
        return sum(self.current_entropies())

    def dashboard(self):
        """Print a live entropy dashboard."""
        entropies = self.current_entropies()
        max_S = np.log(self.sim.max_bond_dim_used) if self.sim.max_bond_dim_used > 1 else 1.0
        print(f"\n{'─'*60}")
        print(f"  Entanglement Governor Dashboard")
        print(f"  S_max ceiling : {self.S_max:.3f} nats  (≈ χ={int(np.exp(self.S_max))})")
        print(f"  Bonds         : {len(entropies)}")
        print(f"  Max S         : {max(entropies):.4f}")
        print(f"  Mean S        : {np.mean(entropies):.4f}")
        print(f"  Total S       : {sum(entropies):.4f}")
        print(f"  Prune events  : {len(self.pruning_log)}")
        print(f"  Bond dims     : min={min(self.sim.bond_dims)} max={self.sim.max_bond_dim_used}")
        print(f"  Memory        : {self.sim.memory_bytes:,} bytes")
        print()

        # ASCII bar chart of entropy per bond (sample up to 40 bonds)
        step = max(1, len(entropies) // 40)
        print("  Entropy profile (each bar = one bond):")
        print("  0.0" + " " * 18 + f"S_max={self.S_max:.2f}")
        for i in range(0, len(entropies), step):
            s = entropies[i]
            bar_len = int(20 * s / max(self.S_max, 0.01))
            bar_len = min(bar_len, 20)
            bar = "█" * bar_len + "░" * (20 - bar_len)
            ceiling_marker = "⚠" if s > self.S_max * 0.9 else " "
            print(f"  [{i:3d}] {bar} {s:.3f} {ceiling_marker}")
        print(f"{'─'*60}")
