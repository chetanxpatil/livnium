r"""
hierarchy.py — multi-scale Livnium and the additive conservation ledger.

A macro lattice of odd size N may host a micro lattice of odd size M inside
each of its N^3 cells. Total symbol capacity is N^3 * M^3, and the symbolic-
weight ledger is strictly additive across scales:

    SW_global = (N^3 * SW_total(M))  +  SW_total(N)
                \-- all micro blocks --/   \-- macro --/

The combined symmetry is the wreath product  G_M wr G_N  (order 24 * 24^(N^3)
in the rotation case). Verified in tests/test_hierarchy.py.
"""

from __future__ import annotations

from .lattice import symbolic_weight_total


def capacity(N: int, M: int) -> int:
    """Total symbol capacity of a macro-N lattice hosting micro-M lattices."""
    return (N**3) * (M**3)


def global_ledger(N: int, M: int) -> int:
    """Additive global symbolic-weight ledger across both scales."""
    micro_sum = (N**3) * symbolic_weight_total(M)
    macro = symbolic_weight_total(N)
    return micro_sum + macro


def wreath_group_order(N: int, base_group_order: int = 24) -> int:
    """Order of the hierarchical symmetry group G_M wr G_N (rotation case)."""
    return base_group_order * (base_group_order ** (N**3))
