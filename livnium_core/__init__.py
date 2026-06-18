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
from .lattice import SW, class_counts, exposure, symbolic_weight_total
from .layer_language import evaluate as ll_evaluate
from .layer_language import parse as ll_parse
from .moves import FACES, apply_sequence, face_permutation, solved_state
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
]
