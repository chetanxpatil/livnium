"""
Livnium → MPS Bridge
====================

Maps the Livnium Core lattice geometry (spec §1–§2) into the
SemanticPolarityGovernor's polarity signal.

The spec defines:
    - 3×3×3 cubic lattice, coords ∈ {-1,0,+1}³
    - 24 allowed cube rotations (R⁴ = I)
    - Symbolic Weight SW = 9f  (f = boundary coordinate count)
    - Semantic Polarity = cos(θ), θ between motion vector and observer direction

This module:
    1.  Implements the lattice + all 24 rotations correctly
    2.  Computes per-symbol semantic polarity after each rotation
    3.  Derives a single α signal (mean |polarity|) to feed into
        SemanticPolarityGovernor as the polarity reward

Nothing is assumed.  Everything is verified in test_bridge.py before use.
"""

import numpy as np
from typing import List, Tuple, Dict, Callable
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from mps_simulator import MPSSimulator
from polarity_governor import SemanticPolarityGovernor, bond_polarities


# ─────────────────────────────────────────────────────────────────────────────
# Lattice
# ─────────────────────────────────────────────────────────────────────────────

Coord = Tuple[int, int, int]

COORDS_3x3x3: List[Coord] = [
    (x, y, z)
    for x in range(-1, 2)
    for y in range(-1, 2)
    for z in range(-1, 2)
]


def boundary_exposure(coord: Coord) -> int:
    """
    f = number of coordinates equal to ±(N−1)/2.
    For N=3, (N−1)/2 = 1, so f counts how many of |x|,|y|,|z| equal 1.
    """
    return sum(1 for c in coord if abs(c) == 1)


def symbolic_weight(coord: Coord) -> int:
    """SW = 9f  (spec §1 A4)"""
    return 9 * boundary_exposure(coord)


class LivniumLattice:
    """
    3×3×3 Livnium lattice.

    State: dict[coord → symbol_id]
    Initial: symbol i lives at COORDS_3x3x3[i]  (arbitrary but fixed canonical order)

    Only the 24 cube rotations are permitted (spec §1 A5).
    """

    def __init__(self):
        self.n = 3
        self._coords = COORDS_3x3x3[:]
        # state: coord → symbol_id
        self.state: Dict[Coord, int] = {
            coord: i for i, coord in enumerate(self._coords)
        }

    def copy(self) -> "LivniumLattice":
        lat = LivniumLattice.__new__(LivniumLattice)
        lat.n = self.n
        lat._coords = self._coords
        lat.state = dict(self.state)
        return lat

    # ── Rotation primitives ───────────────────────────────────────────────

    def _apply_rotation(self, fn: Callable[[int,int,int], Coord]):
        new_state: Dict[Coord, int] = {}
        for (x, y, z), sym in self.state.items():
            new_state[fn(x, y, z)] = sym
        self.state = new_state

    def rotate_z90(self):
        """Rz(90°): (x,y,z) → (−y, x, z)"""
        self._apply_rotation(lambda x,y,z: (-y, x, z))

    def rotate_x90(self):
        """Rx(90°): (x,y,z) → (x, −z, y)"""
        self._apply_rotation(lambda x,y,z: (x, -z, y))

    def rotate_y90(self):
        """Ry(90°): (x,y,z) → (z, y, −x)"""
        self._apply_rotation(lambda x,y,z: (z, y, -x))

    # ── Conservation checks ───────────────────────────────────────────────

    def total_sw(self) -> int:
        """ΣSW — must be 486 for 3×3×3, invariant under all rotations."""
        return sum(symbolic_weight(coord) for coord in self.state)

    def class_counts(self) -> Dict[int, int]:
        """Returns {f: count} — must be {0:1, 1:6, 2:12, 3:8} for 3×3×3."""
        counts = {0: 0, 1: 0, 2: 0, 3: 0}
        for coord in self.state:
            counts[boundary_exposure(coord)] += 1
        return counts

    def is_bijection(self) -> bool:
        """Each symbol appears exactly once."""
        syms = list(self.state.values())
        return len(set(syms)) == len(syms) == 27

    def is_valid(self) -> Tuple[bool, str]:
        """Verify all conservation laws (spec §2 D5)."""
        sw = self.total_sw()
        if sw != 486:
            return False, f"ΣSW={sw} ≠ 486"
        cc = self.class_counts()
        expected = {0: 1, 1: 6, 2: 12, 3: 8}
        if cc != expected:
            return False, f"class counts {cc} ≠ {expected}"
        if not self.is_bijection():
            return False, "bijection violated"
        return True, "OK"

    # ── Semantic Polarity ─────────────────────────────────────────────────

    def semantic_polarity(
        self,
        old_coord: Coord,
        new_coord: Coord,
    ) -> float:
        """
        cos(θ) between the symbol's motion vector and its observer direction.

        motion       = new_coord − old_coord
        observer_dir = (0,0,0) − old_coord   (direction toward the Global Observer)

        Returns 0.0 if the symbol doesn't move or sits at the origin.
        (spec §1 A6)
        """
        m = np.array(new_coord, float) - np.array(old_coord, float)
        o = -np.array(old_coord, float)   # toward origin

        m_norm = np.linalg.norm(m)
        o_norm = np.linalg.norm(o)

        if m_norm < 1e-12 or o_norm < 1e-12:
            return 0.0

        return float(np.dot(m, o) / (m_norm * o_norm))

    def apply_rotation_with_polarity(
        self,
        rotation_fn: Callable,
    ) -> Dict:
        """
        Apply a rotation, compute semantic polarity for every symbol that moved,
        and return a summary dict.

        Returns:
            {
              "polarities": list of cos(θ) per moved symbol,
              "mean_abs"  : mean |cos θ|  ← the bridge signal for the MPS governor,
              "min"       : min polarity,
              "max"       : max polarity,
              "n_moved"   : number of symbols that changed position,
            }
        """
        # snapshot before
        old_pos: Dict[int, Coord] = {sym: coord for coord, sym in self.state.items()}

        rotation_fn(self)   # all rotation callables receive the lattice as first arg

        # snapshot after
        new_pos: Dict[int, Coord] = {sym: coord for coord, sym in self.state.items()}

        polarities = []
        for sym in range(27):
            old = old_pos[sym]
            new = new_pos[sym]
            if old != new:
                polarities.append(self.semantic_polarity(old, new))

        mean_abs = float(np.mean(np.abs(polarities))) if polarities else 0.0

        return {
            "polarities": polarities,
            "mean_abs"  : mean_abs,
            "min"       : float(min(polarities)) if polarities else 0.0,
            "max"       : float(max(polarities)) if polarities else 0.0,
            "n_moved"   : len(polarities),
        }


# ─────────────────────────────────────────────────────────────────────────────
# 24-rotation generator
# ─────────────────────────────────────────────────────────────────────────────

def generate_all_24_rotations() -> List[Callable]:
    """
    Generate all 24 distinct cube orientations as sequences of
    rotate_x90 / rotate_y90 / rotate_z90.

    Strategy: BFS from identity through the 3 generators.
    Two orientations are equal if they map every coord identically.
    """
    def _state_key(lat: LivniumLattice) -> tuple:
        return tuple(sorted(lat.state.items()))

    generators = [
        ("X", lambda l: l.rotate_x90()),
        ("Y", lambda l: l.rotate_y90()),
        ("Z", lambda l: l.rotate_z90()),
    ]

    seen = set()
    rotations = []   # list of (label, fn) pairs

    def identity(l): pass

    queue = [([], identity, LivniumLattice())]

    while queue:
        seq, fn, lat = queue.pop(0)
        key = _state_key(lat)
        if key in seen:
            continue
        seen.add(key)

        label = "".join(seq) if seq else "I"

        # Build a closed-over function for this specific sequence
        seq_copy = list(seq)

        def make_fn(s):
            def _fn(l):
                for axis in s:
                    if axis == "X":
                        l.rotate_x90()
                    elif axis == "Y":
                        l.rotate_y90()
                    elif axis == "Z":
                        l.rotate_z90()
            return _fn

        rotations.append((label, make_fn(seq_copy)))

        if len(rotations) == 24:
            break

        for axis, gen in generators:
            next_lat = lat.copy()
            gen(next_lat)
            queue.append((seq + [axis], make_fn(seq_copy + [axis]), next_lat))

    return rotations


# ─────────────────────────────────────────────────────────────────────────────
# Bridge: Livnium polarity → MPS governor alpha
# ─────────────────────────────────────────────────────────────────────────────

def livnium_polarity_signal(lattice: LivniumLattice, rotation_fn: Callable) -> float:
    """
    Apply rotation_fn(lattice) and return mean |cos θ| across all moved symbols.
    This is the α signal for SemanticPolarityGovernor.

    rotation_fn must accept the lattice as its first argument:
        def my_rot(l): l.rotate_z90()

    High α (≈ 1.0) → symbols move radially toward/away from origin → structured rotation
    Low α  (≈ 0.0) → symbols move tangentially → maximally neutral rotation
    """
    result = lattice.apply_rotation_with_polarity(rotation_fn)
    return result["mean_abs"]


class LivniumGovernedCircuit:
    """
    Runs an MPS quantum circuit where the entropy ceiling at each bond
    is modulated by the Livnium semantic polarity of the last lattice rotation.

    Usage:
        lat = LivniumLattice()
        circ = LivniumGovernedCircuit(n_qubits=20, s_max=np.log(8))
        alpha = circ.apply_livnium_rotation(lat, lat.rotate_z90)
        circ.hadamard(0)
        circ.cnot(0, 1)
        ...
    """

    def __init__(self, n_qubits: int, s_max: float = np.log(8)):
        self.sim = MPSSimulator(n_qubits=n_qubits, max_bond_dim=128)
        self.gov = SemanticPolarityGovernor(
            self.sim, S_max=s_max, alpha=0.5, verbose=False
        )
        self.last_polarity: float = 0.0
        self.rotation_log: List[Dict] = []

    def apply_livnium_rotation(
        self,
        lattice: LivniumLattice,
        rotation_fn: Callable,
    ) -> float:
        """
        Apply a Livnium rotation to the lattice, extract mean |polarity|,
        update the governor's alpha, and log the event.
        Returns mean |cos θ|.
        """
        result = lattice.apply_rotation_with_polarity(rotation_fn)
        alpha = result["mean_abs"]

        self.last_polarity = alpha
        self.gov.alpha = alpha

        ok, msg = lattice.is_valid()
        self.rotation_log.append({
            "mean_polarity" : alpha,
            "n_moved"       : result["n_moved"],
            "pol_min"       : result["min"],
            "pol_max"       : result["max"],
            "lattice_valid" : ok,
            "lattice_msg"   : msg,
        })
        return alpha

    def hadamard(self, q: int):
        self.gov.hadamard(q)

    def cnot(self, q_ctrl: int, q_tgt: int):
        self.gov.cnot(q_ctrl, q_tgt)

    def measure_all(self):
        return self.sim.measure_all()

    def summary(self):
        print(f"\n{'─'*55}")
        print(f"  LivniumGovernedCircuit summary")
        print(f"  Rotation events:  {len(self.rotation_log)}")
        print(f"  Governor prunes:  {len(self.gov.pruning_log)}")
        print(f"  MPS qubits:       {self.sim.n}")
        if self.rotation_log:
            alphas = [e["mean_polarity"] for e in self.rotation_log]
            print(f"  Mean α:  {np.mean(alphas):.4f}")
            print(f"  Max α:   {max(alphas):.4f}")
            print(f"  Min α:   {min(alphas):.4f}")
        self.gov.polarity_dashboard()
