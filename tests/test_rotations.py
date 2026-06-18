import itertools

from livnium_core.lattice import SW, exposure
from livnium_core.rotations import (
    ROT_X,
    ROT_Y,
    ROT_Z,
    I,
    apply,
    matpow,
    rotation_group,
)


def test_group_order_is_24():
    assert len(rotation_group()) == 24


def test_generators_order_4():
    for g in (ROT_X, ROT_Y, ROT_Z):
        assert matpow(g, 4) == I


def test_rotations_preserve_exposure_and_weight():
    # Every rotation maps the N=3 lattice onto itself preserving class & weight.
    coords = list(itertools.product((-1, 0, 1), repeat=3))
    for rot in rotation_group():
        images = [apply(rot, c) for c in coords]
        assert sorted(images) == sorted(coords)  # permutation (reversible)
        for c in coords:
            assert exposure(apply(rot, c), 3) == exposure(c, 3)
            assert SW(apply(rot, c), 3) == SW(c, 3)
