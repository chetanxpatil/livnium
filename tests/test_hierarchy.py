from livnium_core.hierarchy import capacity, global_ledger
from livnium_core.lattice import symbolic_weight_total


def test_capacity():
    assert capacity(3, 3) == 27 * 27
    assert capacity(3, 5) == 27 * 125


def test_ledger_is_additive():
    for N, M in [(3, 3), (3, 5), (5, 3), (5, 5)]:
        expected = (N**3) * symbolic_weight_total(M) + symbolic_weight_total(N)
        assert global_ledger(N, M) == expected


def test_known_global_ledger():
    # N=3 hosting M=3: 27 micro blocks * 486 + macro 486
    assert global_ledger(3, 3) == 27 * 486 + 486 == 13608
