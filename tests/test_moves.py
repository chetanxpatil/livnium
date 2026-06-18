import random

from livnium_core.lattice import SW, exposure
from livnium_core.moves import _CELLS, FACES, apply_sequence, face_permutation, solved_state


def _is_perm(p):
    return sorted(p.values()) == sorted(_CELLS)


def test_face_turns_are_valid_order4_perms():
    for name in FACES:
        p = face_permutation(name)
        assert _is_perm(p)
        q = {c: c for c in _CELLS}
        for _ in range(4):
            q = {c: p[q[c]] for c in _CELLS}
        assert all(q[c] == c for c in _CELLS)  # order 4


def test_core_and_centers_fixed():
    for name in FACES:
        p = face_permutation(name)
        assert p[(0, 0, 0)] == (0, 0, 0)  # Om fixed
        for c in _CELLS:
            if exposure(c, 3) == 1:  # face-centers
                # a face-center is fixed by its own face's turn; never leaves center class
                assert exposure(p[c], 3) == 1


def test_face_turns_class_preserving():
    for name in FACES:
        p = face_permutation(name)
        assert all(exposure(p[c], 3) == exposure(c, 3) for c in _CELLS)


def test_sigma_sw_conserved_under_random_scramble():
    random.seed(0)
    state = solved_state()
    moves = [random.choice(list(FACES)) for _ in range(5000)]
    state = apply_sequence(state, moves)
    # total energy carried by tokens is conserved
    assert sum(SW(state[c], 3) for c in _CELLS) == sum(SW(c, 3) for c in _CELLS) == 486
    # every slot still holds a token of its own exposure class
    assert all(exposure(state[c], 3) == exposure(c, 3) for c in _CELLS)


def test_reversible():
    random.seed(1)
    moves = [random.choice(list(FACES)) for _ in range(300)]
    state = apply_sequence(solved_state(), moves)
    for m in reversed(moves):  # inverse of a turn = 3 more turns
        state = apply_sequence(state, [m, m, m])
    assert state == solved_state()
