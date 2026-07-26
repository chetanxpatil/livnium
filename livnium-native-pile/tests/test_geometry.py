import torch

from livnium_native.geometry import (
    INVERSE_ACTION,
    action_index_maps,
    action_permutations,
    reachable_indices,
    validate_geometry,
)


def test_observer_shifts_are_permutations_with_exact_inverses():
    validate_geometry()
    permutations = action_permutations(dtype=torch.int64)
    identity = torch.eye(27, dtype=torch.int64)
    for action, inverse in enumerate(INVERSE_ACTION):
        assert torch.equal(permutations[action] @ permutations[inverse], identity)


def test_cyclic_shifts_and_reorientations_have_expected_periods():
    maps = action_index_maps()
    starts = torch.arange(27)
    for action in range(6):
        positions = starts.clone()
        for _ in range(3):
            positions = maps[action, positions]
        assert torch.equal(positions, starts)
    for action in (6, 7):
        positions = starts.clone()
        for _ in range(4):
            positions = maps[action, positions]
        assert torch.equal(positions, starts)


def test_every_cell_is_reachable_from_every_start():
    for start in range(27):
        assert reachable_indices(start) == set(range(27))


def test_action_language_contains_non_commuting_operations():
    maps = action_index_maps()
    starts = torch.arange(27)
    pairs = []
    for first in range(maps.shape[0]):
        for second in range(maps.shape[0]):
            first_then_second = maps[second, maps[first, starts]]
            second_then_first = maps[first, maps[second, starts]]
            if not torch.equal(first_then_second, second_then_first):
                pairs.append((first, second))
    assert pairs
