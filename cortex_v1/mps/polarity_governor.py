"""
Semantic Polarity Governor
==========================

Extends EntanglementGovernor with polarity-aware pruning.

Polarity score at bond i (between sites i and i+1):

    S_max_theoretical(i) = min(i+1, n-i-1) × log(2)   [Page / volume-law limit]

    polarity(i) = 1 - S(i) / S_max_theoretical(i)

    polarity = 1.0  →  far below the theoretical ceiling (GHZ is here!)
                     →  bond carries STRUCTURED information, like cos θ = ±1 in Livnium
    polarity = 0.0  →  at the theoretical ceiling (random deep circuit)
                     →  bond is at volume-law noise, like cos θ ≈ 0 in Livnium

Why the theoretical maximum?
    GHZ bonds have χ=2 and S=log(2).
    That looks "saturated" if you compare to log(χ)=log(2).
    But relative to the full bipartition capacity (e.g. min(i+1, n-i-1) × log(2)):
        bond i=9 of GHZ-20: S_max_theo = 10 × log(2), S = log(2) → polarity = 0.9
    GHZ is an Area Law state — it uses O(1) entanglement where Volume Law uses O(n).
    That's why it's highly polarised.

Reward policy:

    effective_S_max(bond i) = S_max_base × (1 + α × polarity(i))

    High polarity → looser ceiling → structured bonds survive
    Low polarity  → standard ceiling → noisy bonds pruned more aggressively

Connection to Livnium EGAN:
    cos θ(h, A*) measures how decisively a state is pulled toward an attractor.
    High |cos θ| = the state is in a semantically meaningful direction.

    polarity(bond i) measures how far below the theoretical entanglement ceiling
    the bond sits. GHZ is structured → far below ceiling → high polarity.
    Random circuit is at ceiling → polarity ≈ 0.

    Both quantities distinguish "meaningful signal" from "maximal noise."
"""

import numpy as np
from typing import List, Dict, Optional
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from mps_simulator import MPSSimulator
from entanglement_governor import EntanglementGovernor, von_neumann_entropy, bond_entropies


# ─────────────────────────────────────────────────────────────────────────────
# Polarity score
# ─────────────────────────────────────────────────────────────────────────────

def polarity_score_theoretical(
    singular_values: np.ndarray,
    bond_site: int,
    n_qubits: int,
) -> float:
    """
    polarity = 1 - S / S_max_theoretical

    S_max_theoretical(i) = min(i+1, n-i-1) × log(2)
        This is the maximum possible Von Neumann entropy at bond i
        for an n-qubit system (Page limit / volume-law ceiling).

    GHZ bonds have S=log(2) << S_max_theoretical → polarity → 1.0
    Random deep circuits have S → S_max_theoretical  → polarity → 0.0

    Args:
        singular_values: raw singular values at this bond
        bond_site      : bond index i  (bond between site i and i+1)
        n_qubits       : total number of qubits

    Returns:
        float in [0, 1]
    """
    n_left  = bond_site + 1
    n_right = n_qubits - n_left
    S_max_theo = min(n_left, n_right) * np.log(2)

    if S_max_theo < 1e-15:
        return 1.0

    S = von_neumann_entropy(singular_values)
    return float(np.clip(1.0 - S / S_max_theo, 0.0, 1.0))


def bond_polarities(sim: MPSSimulator) -> List[float]:
    """Compute polarity at every bond of an MPS using the theoretical maximum."""
    polarities = []
    for i in range(sim.n - 1):
        t = sim.tensors[i]
        chi_l, _, chi_r = t.shape
        mat = t.reshape(chi_l * 2, chi_r)
        _, s, _ = np.linalg.svd(mat, full_matrices=False)
        polarities.append(polarity_score_theoretical(s, i, sim.n))
    return polarities


# ─────────────────────────────────────────────────────────────────────────────
# Semantic Polarity Governor
# ─────────────────────────────────────────────────────────────────────────────

class SemanticPolarityGovernor(EntanglementGovernor):
    """
    Polarity-aware entanglement governor.

    Extends EntanglementGovernor so that the effective entropy ceiling at each
    bond depends on that bond's polarity relative to the theoretical maximum:

        polarity(i) = 1 - S(i) / (min(i+1, n-i-1) × log(2))
        effective_S_max(i) = S_max_base × (1 + α × polarity(i))

    Parameters
    ----------
    sim     : the MPS to govern
    S_max   : base entropy ceiling (nats)
    alpha   : polarity reward strength
              0.0  →  standard governor (no polarity adjustment)
              0.5  →  Area-Law bonds get 50% looser ceiling
              1.0  →  Area-Law bonds get double ceiling
    verbose : print every pruning event
    """

    def __init__(
        self,
        sim: MPSSimulator,
        S_max: float = np.log(64),
        alpha: float = 0.5,
        verbose: bool = True,
    ):
        super().__init__(sim, S_max=S_max, verbose=verbose)
        self.alpha = alpha
        self.polarity_log: List[Dict] = []

    # ─────────────────────────────────────────────────────────────────────────
    # Override: polarity-adjusted enforcement
    # ─────────────────────────────────────────────────────────────────────────

    def _enforce_bond(self, site: int) -> bool:
        """
        Check bond between site and site+1.
        Computes polarity first, adjusts effective S_max, then enforces.
        """
        t = self.sim.tensors[site]
        chi_l, _, chi_r = t.shape
        mat = t.reshape(chi_l * 2, chi_r)

        U, s, Vh = np.linalg.svd(mat, full_matrices=False)
        S_current = von_neumann_entropy(s)

        # ── Polarity-adjusted ceiling ──────────────────────────────────────
        p_score = polarity_score_theoretical(s, site, self.sim.n)
        effective_S_max = self.S_max * (1.0 + self.alpha * p_score)

        self.polarity_log.append({
            "bond"           : site,
            "polarity"       : p_score,
            "S_current"      : S_current,
            "effective_S_max": effective_S_max,
        })

        if S_current <= effective_S_max:
            return False

        # ── Find minimum chi that keeps S ≤ effective_S_max ───────────────
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
            if cumulative_S >= effective_S_max and k > 0:
                chi_keep = k
                break
        else:
            chi_keep = len(s)

        chi_keep = max(1, chi_keep)
        if chi_keep >= len(s):
            return False

        S_after   = von_neumann_entropy(s[:chi_keep])
        trunc_err = float(np.sum(s[chi_keep:] ** 2))

        # ── Apply truncation ───────────────────────────────────────────────
        U_t  = U[:, :chi_keep]
        S_t  = s[:chi_keep]
        Vh_t = Vh[:chi_keep, :]

        US = U_t * S_t[np.newaxis, :]
        self.sim.tensors[site]     = US.reshape(chi_l, 2, chi_keep)
        self.sim.tensors[site + 1] = (
            Vh_t @ self.sim.tensors[site + 1].reshape(chi_r, -1)
        ).reshape(chi_keep, 2, self.sim.tensors[site + 1].shape[2])

        event = {
            "bond"           : site,
            "polarity"       : p_score,
            "effective_S_max": effective_S_max,
            "chi_before"     : len(s),
            "chi_after"      : chi_keep,
            "S_before"       : S_current,
            "S_after"        : S_after,
            "trunc_err"      : trunc_err,
        }
        self.pruning_log.append(event)

        if self.verbose:
            print(
                f"  [PRUNE] bond {site}: pol={p_score:.2f} "
                f"eff_ceil={effective_S_max:.3f}  "
                f"χ {len(s)}→{chi_keep}  "
                f"S {S_current:.3f}→{S_after:.3f}  "
                f"trunc={trunc_err:.2e}"
            )
        return True

    # ─────────────────────────────────────────────────────────────────────────
    # Monitoring
    # ─────────────────────────────────────────────────────────────────────────

    def polarity_dashboard(self):
        """Print entropy + polarity profile across all bonds."""
        entropies  = self.current_entropies()
        polarities = bond_polarities(self.sim)

        if not entropies:
            return

        print(f"\n{'─'*65}")
        print(f"  Semantic Polarity Dashboard  (S_max={self.S_max:.3f}  α={self.alpha})")
        print(f"  effective ceiling range: "
              f"{self.S_max:.3f} – {self.S_max*(1+self.alpha):.3f}")
        print(f"  Max polarity  : {max(polarities):.4f}")
        print(f"  Mean polarity : {np.mean(polarities):.4f}")
        print(f"  Min polarity  : {min(polarities):.4f}")
        print(f"  Prune events  : {len(self.pruning_log)}")
        print()
        print("  Bond profiles:")
        print("  [idx] S-bar          entropy  |  polarity-bar      polarity  eff_ceil")
        print()

        step = max(1, len(entropies) // 35)
        ceil_max = self.S_max * (1.0 + self.alpha)

        for i in range(0, len(entropies), step):
            S   = entropies[i]
            pol = polarities[i]
            eff = self.S_max * (1.0 + self.alpha * pol)

            s_bar = min(int(12 * S / max(ceil_max, 0.01)), 12)
            p_bar = min(int(12 * pol), 12)

            ceiling_warn = "⚠" if S > eff * 0.95 else " "
            print(
                f"  [{i:3d}] {'█'*s_bar+'░'*(12-s_bar)} {S:6.3f}  |  "
                f"{'▓'*p_bar+'░'*(12-p_bar)} {pol:6.3f}  eff={eff:.3f} {ceiling_warn}"
            )

        print(f"{'─'*65}")

    def polarity_summary(self) -> Dict:
        """Return aggregate polarity statistics."""
        pols = bond_polarities(self.sim)
        return {
            "mean_polarity": float(np.mean(pols)),
            "max_polarity" : float(max(pols)),
            "min_polarity" : float(min(pols)),
            "prune_events" : len(self.pruning_log),
            "alpha"        : self.alpha,
            "S_max_base"   : self.S_max,
        }
