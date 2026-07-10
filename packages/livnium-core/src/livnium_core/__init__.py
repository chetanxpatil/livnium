"""
Livnium Core — a conserved geometric state space.

A reversible symbolic system on an odd cubic lattice with:
  - a base-27 positional codec (lossless, carry-correct, binary-convertible)
  - a four-class exposure structure with a closed-form symbolic-weight law
  - dynamics restricted to the 24-element cube rotation group (all reversible)
  - an additive conservation ledger preserved under every operation

Everything exported here is pure mathematics, verified by the test suite.
No claims are made about machine-learning performance — see BENCHMARKS.md.
"""

from .base27 import (
    ALPHABET,
    base27_to_binary,
    base27_to_int,
    binary_to_base27,
    int_to_base27,
)
from .hierarchy import capacity, global_ledger, wreath_group_order
from .lattice import (
    SW,
    Field,
    MultipoleSignature,
    class_counts,
    dipole,
    exposure,
    mat_det,
    mat_frobenius,
    mat_trace,
    monopole,
    multipole_signature,
    quadrupole,
    rotate_field,
    sw_field,
    symbolic_weight_total,
    vec_norm,
)
from .layer_language import evaluate as ll_evaluate
from .layer_language import parse as ll_parse
from .moves import FACES, apply_sequence, face_permutation, solved_state
from .ping import (
    CELLS_26,
    Match,
    Ping,
    Prune,
    Signature,
    Step,
    classify,
    cos_path,
    cosine,
    meaning_match,
    path_signature,
    prune,
    turn_angles,
    world_direction,
    world_doorways,
    world_path,
)
from .rotations import ROT_X, ROT_Y, ROT_Z, rotation_group

__version__ = "0.1.0"
__all__ = [
    "int_to_base27",
    "base27_to_int",
    "base27_to_binary",
    "binary_to_base27",
    "ALPHABET",
    "class_counts",
    "symbolic_weight_total",
    "exposure",
    "SW",
    "Field",
    "sw_field",
    "monopole",
    "dipole",
    "quadrupole",
    "vec_norm",
    "mat_trace",
    "mat_frobenius",
    "mat_det",
    "rotate_field",
    "MultipoleSignature",
    "multipole_signature",
    "rotation_group",
    "ROT_X",
    "ROT_Y",
    "ROT_Z",
    "capacity",
    "global_ledger",
    "wreath_group_order",
    "face_permutation",
    "apply_sequence",
    "solved_state",
    "FACES",
    "ll_parse",
    "ll_evaluate",
    "CELLS_26",
    "Ping",
    "Step",
    "Match",
    "Prune",
    "cosine",
    "world_doorways",
    "world_direction",
    "cos_path",
    "world_path",
    "turn_angles",
    "Signature",
    "path_signature",
    "classify",
    "meaning_match",
    "prune",
]
