"""Exact 3x3x3 Livnium observer geometry used by the memory block.

Rigid cube rotations preserve exposure class, so they cannot navigate all 27
cells: center, face, edge and corner cells are separate orbits.  A memory reader
therefore needs a distinct *observer-address* operation.

The neural part selects among eight fixed, exact and reversible observer
operations:

    Ox+, Ox-, Oy+, Oy-, Oz+, Oz-, Rx+, Rx-

These operations move only the observer address; they never rearrange or alter
the pile.  Coordinates cycle through ``-1 -> 0 -> 1 -> -1`` (and the inverse).
The final two operations reorient the observer frame around the x axis. Shifts
and reorientations do not all commute, so path order matters. Each operation is
represented as a 27x27 permutation matrix. A row-vector observer distribution
``p`` evolves as ``p_next = p @ permutation``.
"""

from __future__ import annotations

from itertools import product

import torch

Coord = tuple[int, int, int]

COORDINATES: tuple[Coord, ...] = tuple(product((-1, 0, 1), repeat=3))
COORD_TO_INDEX = {coord: index for index, coord in enumerate(COORDINATES)}

ACTION_NAMES: tuple[str, ...] = (
    "Ox+",
    "Ox-",
    "Oy+",
    "Oy-",
    "Oz+",
    "Oz-",
    "Rx+",
    "Rx-",
)
INVERSE_ACTION: tuple[int, ...] = (1, 0, 3, 2, 5, 4, 7, 6)
_AXIS_VALUES = (-1, 0, 1)


def _cyclic_shift(value: int, amount: int) -> int:
    index = _AXIS_VALUES.index(value)
    return _AXIS_VALUES[(index + amount) % len(_AXIS_VALUES)]


def shift_observer(coord: Coord, action: int) -> Coord:
    """Apply one exact reversible shift to an observer coordinate."""
    x, y, z = coord
    if action == 0:  # Ox+
        return (_cyclic_shift(x, 1), y, z)
    if action == 1:  # Ox-
        return (_cyclic_shift(x, -1), y, z)
    if action == 2:  # Oy+
        return (x, _cyclic_shift(y, 1), z)
    if action == 3:  # Oy-
        return (x, _cyclic_shift(y, -1), z)
    if action == 4:  # Oz+
        return (x, y, _cyclic_shift(z, 1))
    if action == 5:  # Oz-
        return (x, y, _cyclic_shift(z, -1))
    if action == 6:  # Rx+
        return (x, -z, y)
    if action == 7:  # Rx-
        return (x, z, -y)
    raise ValueError(f"action must be in [0, {len(ACTION_NAMES) - 1}], got {action}")


def action_index_maps() -> torch.Tensor:
    """Return ``maps[action, source_index] = destination_index``."""
    maps = torch.empty((len(ACTION_NAMES), len(COORDINATES)), dtype=torch.long)
    for action in range(len(ACTION_NAMES)):
        for source, coord in enumerate(COORDINATES):
            maps[action, source] = COORD_TO_INDEX[shift_observer(coord, action)]
    return maps


def action_permutations(dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Return exact permutation matrices with shape ``[actions, 27, 27]``."""
    maps = action_index_maps()
    matrices = torch.zeros(
        (len(ACTION_NAMES), len(COORDINATES), len(COORDINATES)),
        dtype=dtype,
    )
    sources = torch.arange(len(COORDINATES))
    for action in range(len(ACTION_NAMES)):
        matrices[action, sources, maps[action]] = 1
    return matrices


def apply_action_indices(indices: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    """Apply one hard action per item to integer observer positions."""
    if indices.shape != actions.shape:
        raise ValueError("indices and actions must have the same shape")
    maps = action_index_maps().to(indices.device)
    return maps[actions, indices]


def inverse_sequence(actions: torch.Tensor, pad_value: int = -1) -> torch.Tensor:
    """Return the exact inverse of padded action sequences."""
    if actions.ndim != 2:
        raise ValueError("actions must have shape [batch, steps]")
    inverse = torch.full_like(actions, pad_value)
    inverse_table = torch.tensor(INVERSE_ACTION, device=actions.device)
    for row in range(actions.shape[0]):
        active = actions[row][actions[row] != pad_value]
        if active.numel():
            inverse[row, : active.numel()] = inverse_table[active.flip(0)]
    return inverse


def reachable_indices(start: int) -> set[int]:
    """Return the complete orbit reachable from one observer address."""
    maps = action_index_maps()
    reached = {int(start)}
    frontier = [int(start)]
    while frontier:
        current = frontier.pop()
        for action in range(len(ACTION_NAMES)):
            destination = int(maps[action, current])
            if destination not in reached:
                reached.add(destination)
                frontier.append(destination)
    return reached


def validate_geometry() -> None:
    """Raise if any observer permutation, inverse or reachability law breaks."""
    matrices = action_permutations(dtype=torch.int64)
    ones = torch.ones(len(COORDINATES), dtype=torch.int64)
    if not torch.equal(matrices.sum(dim=1), ones.expand_as(matrices.sum(dim=1))):
        raise AssertionError("an observer shift does not have one source per destination")
    if not torch.equal(matrices.sum(dim=2), ones.expand_as(matrices.sum(dim=2))):
        raise AssertionError("an observer shift does not have one destination per source")

    identity = torch.eye(len(COORDINATES), dtype=torch.int64)
    for action, inverse in enumerate(INVERSE_ACTION):
        if not torch.equal(matrices[action] @ matrices[inverse], identity):
            raise AssertionError(f"{ACTION_NAMES[action]} inverse law failed")
    for start in range(len(COORDINATES)):
        if len(reachable_indices(start)) != len(COORDINATES):
            raise AssertionError(f"observer cannot reach every cell from index {start}")
