from livnium_core.lattice import (
    brute_force_total,
    class_counts,
    symbolic_weight_total,
)


def test_class_counts_sum_to_N_cubed():
    for N in (3, 5, 7, 9):
        c = class_counts(N)
        assert c["core"] + c["center"] + c["edge"] + c["corner"] == N**3


def test_class_counts_known():
    assert class_counts(3) == {"core": 1, "center": 6, "edge": 12, "corner": 8}
    assert class_counts(5) == {"core": 27, "center": 54, "edge": 36, "corner": 8}
    assert class_counts(7) == {"core": 125, "center": 150, "edge": 60, "corner": 8}


def test_total_symbolic_weight_closed_form():
    assert symbolic_weight_total(3) == 486
    assert symbolic_weight_total(5) == 1350
    assert symbolic_weight_total(7) == 2646  # corrects the old 3024 typo
    assert symbolic_weight_total(9) == 4374


def test_closed_form_matches_brute_force():
    for N in (3, 5, 7, 9):
        assert symbolic_weight_total(N) == brute_force_total(N)
