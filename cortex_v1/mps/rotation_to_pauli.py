"""
SO(3) → SU(2):  Livnium Cube Rotations to Qubit Gates
======================================================

Every 3D rotation has an axis n̂ and an angle θ.
The corresponding qubit gate is:

    U(n̂, θ) = cos(θ/2) · I  −  i · sin(θ/2) · (n̂ · σ)

where σ = (σx, σy, σz) are the Pauli matrices.

This is the ONLY non-arbitrary way to map a cube rotation to a qubit gate.
The mapping is exact — no modulus operators, no index tricks.

The 24 cube rotations produce these angle classes:
    θ =   0°  →  1 identity        Tr(M) =  3
    θ =  90°  →  6 face rotations  Tr(M) =  1
    θ = 120°  →  8 vertex rotations Tr(M) =  0
    θ = 180°  →  9 rotations       Tr(M) = −1
    (total: 1+6+8+9 = 24, but actually: 1+6+3+8+6 = 24 — see below)

Actual breakdown:
    1  identity (θ = 0°)
    6  face quarter-turns (±90° about ±x,±y,±z)
    3  face half-turns  (180° about x,y,z axes)
    8  vertex rotations (±120° about 4 body diagonals)
    6  edge half-turns  (180° about 6 edge midpoint axes)
    Total: 1 + 6 + 3 + 8 + 6 = 24  ✓

SU(2) note:
    SU(2) double-covers SO(3): both U and −U map to the same SO(3) rotation.
    We pick a canonical branch: upper-left entry has non-negative real part.
"""

import numpy as np
from typing import Tuple, List, Dict

# ─────────────────────────────────────────────────────────────────────────────
# Pauli matrices
# ─────────────────────────────────────────────────────────────────────────────

I2  = np.eye(2, dtype=complex)
SX  = np.array([[0, 1], [1, 0]], dtype=complex)
SY  = np.array([[0, -1j], [1j, 0]], dtype=complex)
SZ  = np.array([[1, 0], [0, -1]], dtype=complex)

# ─────────────────────────────────────────────────────────────────────────────
# Generator 3×3 matrices  (verified from coordinate formulas in livnium_to_tensor.py)
# ─────────────────────────────────────────────────────────────────────────────
#
# Rx90: (x,y,z) → (x, −z, y)
#   M @ [1,0,0] = [1,0,0],  M @ [0,1,0] = [0,0,1],  M @ [0,0,1] = [0,−1,0]
#
# Ry90: (x,y,z) → (z, y, −x)
#   M @ [1,0,0] = [0,0,−1], M @ [0,1,0] = [0,1,0],  M @ [0,0,1] = [1,0,0]
#
# Rz90: (x,y,z) → (−y, x, z)
#   M @ [1,0,0] = [0,1,0],  M @ [0,1,0] = [−1,0,0], M @ [0,0,1] = [0,0,1]

Rx90_mat = np.array([[1, 0, 0],
                     [0, 0,-1],
                     [0, 1, 0]], dtype=float)

Ry90_mat = np.array([[0, 0, 1],
                     [0, 1, 0],
                     [-1,0, 0]], dtype=float)

Rz90_mat = np.array([[0,-1, 0],
                     [1, 0, 0],
                     [0, 0, 1]], dtype=float)

GENERATOR_MATRICES = {"X": Rx90_mat, "Y": Ry90_mat, "Z": Rz90_mat}


def compose_sequence(seq: str) -> np.ndarray:
    """
    Compose a string like "XYZ" into a single 3×3 rotation matrix.
    Applies left-to-right: X first, then Y, then Z.
    As matrix multiplication: M = Rz @ Ry @ Rx  (rightmost applied first).
    """
    M = np.eye(3)
    for axis in seq:
        M = GENERATOR_MATRICES[axis] @ M
    return M


# ─────────────────────────────────────────────────────────────────────────────
# Axis / angle extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_axis_angle(M: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Extract rotation axis n̂ and angle θ ∈ [0, π] from a 3×3 rotation matrix.

    Method:
        θ = arccos( (Tr(M) − 1) / 2 )

        For θ ∈ (0°, 180°):
            (M − Mᵀ) = 2 sinθ · [[0,−nz, ny],
                                   [nz, 0,−nx],
                                   [−ny,nx, 0]]
            → n = [M[2,1]−M[1,2], M[0,2]−M[2,0], M[1,0]−M[0,1]] / (2 sinθ)

        For θ = 180°:
            M = 2nnᵀ − I  →  nnᵀ = (M + I)/2
            Diagonal gives n², off-diagonal fixes signs.

        For θ = 0°:
            Identity — return z-axis by convention.

    Returns:
        (n̂, θ)  where n̂ is a unit vector, θ in radians.
    """
    trace    = np.trace(M)
    cos_t    = np.clip((trace - 1.0) / 2.0, -1.0, 1.0)
    theta    = float(np.arccos(cos_t))

    # ── Identity ──────────────────────────────────────────────────────────
    if theta < 1e-10:
        return np.array([0.0, 0.0, 1.0]), 0.0

    # ── 180° special case ─────────────────────────────────────────────────
    if abs(theta - np.pi) < 1e-10:
        NNT = (M + np.eye(3)) / 2.0
        n_sq = np.clip(np.diag(NNT), 0.0, None)
        n = np.sqrt(n_sq)
        # Fix signs from off-diagonal elements
        if n[0] > 1e-10:
            if NNT[0, 1] < -1e-10: n[1] = -n[1]
            if NNT[0, 2] < -1e-10: n[2] = -n[2]
        elif n[1] > 1e-10:
            if NNT[1, 2] < -1e-10: n[2] = -n[2]
        norm = np.linalg.norm(n)
        return (n / norm if norm > 1e-10 else np.array([1.0, 0.0, 0.0])), theta

    # ── General case ───────────────────────────────────────────────────────
    sin_t = np.sin(theta)
    # (M − Mᵀ)[i,j] = 2 sinθ × antisymmetric basis element
    diff = M - M.T
    nx = (diff[2, 1]) / (2.0 * sin_t)
    ny = (diff[0, 2]) / (2.0 * sin_t)
    nz = (diff[1, 0]) / (2.0 * sin_t)

    n = np.array([nx, ny, nz])
    norm = np.linalg.norm(n)
    return (n / norm if norm > 1e-10 else np.array([0.0, 0.0, 1.0])), theta


# ─────────────────────────────────────────────────────────────────────────────
# SU(2) gate construction
# ─────────────────────────────────────────────────────────────────────────────

def axis_angle_to_su2(n: np.ndarray, theta: float) -> np.ndarray:
    """
    Build the 2×2 SU(2) gate corresponding to rotation (n̂, θ):

        U = cos(θ/2) · I  −  i · sin(θ/2) · (n̂ · σ)

    where n̂ · σ = nx σx + ny σy + nz σz
                = [[nz,      nx − i·ny],
                   [nx + i·ny,  −nz  ]]

    Returns a 2×2 complex numpy array.
    Canonical branch: upper-left has non-negative real part.
    """
    nx, ny, nz = float(n[0]), float(n[1]), float(n[2])
    c = np.cos(theta / 2.0)
    s = np.sin(theta / 2.0)

    n_dot_sigma = np.array([
        [nz,        nx - 1j * ny],
        [nx + 1j * ny, -nz      ]
    ], dtype=complex)

    U = c * I2 - 1j * s * n_dot_sigma

    # Canonical branch: make upper-left real part non-negative
    if U[0, 0].real < -1e-14:
        U = -U

    return U


def rotation_matrix_to_su2(M: np.ndarray) -> np.ndarray:
    """
    Full pipeline: 3×3 rotation matrix → 2×2 SU(2) gate.
    """
    n, theta = extract_axis_angle(M)
    return axis_angle_to_su2(n, theta)


# ─────────────────────────────────────────────────────────────────────────────
# Generate all 24 (SO(3) matrix, SU(2) gate) pairs
# ─────────────────────────────────────────────────────────────────────────────

def generate_all_24_so3_su2() -> List[Dict]:
    """
    BFS over the 3 generators to enumerate all 24 distinct SO(3) matrices,
    then compute the corresponding SU(2) gate for each.

    Returns list of dicts:
        {
            "label"  : str (e.g. "XYZ"),
            "M_so3"  : 3×3 float ndarray,
            "U_su2"  : 2×2 complex ndarray,
            "axis"   : 3-vector,
            "theta"  : float (radians),
            "theta_deg": float,
        }
    """
    def mat_key(M):
        return tuple(np.round(M, 8).flatten())

    seen   = set()
    result = []
    queue  = [("I", np.eye(3))]

    gen_mats = {"X": Rx90_mat, "Y": Ry90_mat, "Z": Rz90_mat}

    while queue and len(result) < 24:
        label, M = queue.pop(0)
        key = mat_key(M)
        if key in seen:
            continue
        seen.add(key)

        n, theta = extract_axis_angle(M)
        U = axis_angle_to_su2(n, theta)

        result.append({
            "label"    : label,
            "M_so3"    : M,
            "U_su2"    : U,
            "axis"     : n,
            "theta"    : theta,
            "theta_deg": np.degrees(theta),
        })

        for axis, G in gen_mats.items():
            new_M   = G @ M
            new_label = label + axis if label != "I" else axis
            queue.append((new_label, new_M))

    return result
