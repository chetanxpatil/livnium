import math

from livnium_core.ping import (
    CELLS_26,
    Ping,
    Step,
    accumulated_frames,
    classify,
    cos_path,
    cosine,
    meaning_match,
    norm,
    path_signature,
    prune,
    turn_angles,
    world_direction,
    world_doorways,
    world_path,
)
from livnium_core.rotations import ROT_X, ROT_Y, I, rotation_group


def test_26_doorways_exclude_core():
    assert len(CELLS_26) == 26
    assert (0, 0, 0) not in CELLS_26


def test_doorway_cannot_be_core():
    import pytest

    with pytest.raises(ValueError):
        Step((0, 0, 0))


def test_step_rejects_non_rotation_frame():
    import pytest

    scaling = ((2, 0, 0), (0, 2, 0), (0, 0, 2))
    with pytest.raises(ValueError, match="24 orientation-preserving"):
        Step((1, 0, 0), scaling)


def test_ping_rejects_core_as_local_direction():
    import pytest

    with pytest.raises(ValueError, match="26 non-core"):
        Ping(d_local=(0, 0, 0))


def test_cosine_basic():
    assert math.isclose(cosine((1, 0, 0), (1, 0, 0)), 1.0)
    assert math.isclose(cosine((1, 0, 0), (-1, 0, 0)), -1.0)
    assert math.isclose(cosine((1, 0, 0), (0, 1, 0)), 0.0)


def test_cosine_undefined_for_core():
    import pytest

    with pytest.raises(ValueError):
        cosine((0, 0, 0), (1, 0, 0))


def test_identity_frames_keep_local_direction():
    p = Ping(steps=(Step((1, 0, 0)), Step((0, 1, 0))), d_local=(0, 0, 1))
    assert world_direction(p) == (0, 0, 1)


def test_descent_conserves_magnitude_at_any_depth():
    # Every OM-frame is a rotation, so the inward descent never inflates or
    # loses magnitude — the conservation anchor that bounds the illusion space.
    group = rotation_group()
    for d_local in [(1, 0, 0), (1, 1, 0), (1, 1, 1)]:
        steps = tuple(Step(cell=c, frame=r) for c, r in zip(CELLS_26, group))
        p = Ping(steps=steps, d_local=d_local)
        assert math.isclose(norm(world_direction(p)), norm(d_local), rel_tol=1e-9)


def test_accumulated_frames_compose():
    p = Ping(steps=(Step((1, 0, 0), ROT_X), Step((0, 1, 0), ROT_Y)))
    frames = accumulated_frames(p)
    assert frames[0] == I
    assert len(frames) == p.depth + 1


def test_same_pattern_different_direction():
    # Identical doorways and local pattern, one child frame rotated -> the two
    # descents face different world directions (same shape, different orientation).
    base = Ping(steps=(Step((1, 0, 0)),), d_local=(1, 0, 0))
    turned = Ping(steps=(Step((1, 0, 0), ROT_Y),), d_local=(1, 0, 0))
    assert abs(cosine(world_direction(base), world_direction(turned)) - 1.0) > 1e-6


def test_cos_path_length_matches_depth():
    p = Ping(steps=(Step((1, 0, 0)), Step((0, 1, 0)), Step((0, 0, 1))))
    # one cosine per consecutive doorway pair, plus the final step into d_local
    assert len(cos_path(p)) == p.depth
    assert all(-1.0 - 1e-9 <= x <= 1.0 + 1e-9 for x in cos_path(p))


def test_cos_path_empty_for_depth_zero():
    assert cos_path(Ping()) == []


def test_meaning_match_prefix_then_diverge():
    p = Ping(steps=(Step((1, 0, 0)), Step((0, 1, 0)), Step((0, 0, 1))))
    q = Ping(steps=(Step((1, 0, 0)), Step((0, 1, 0)), Step((-1, 0, 0))))
    m = meaning_match(p, q)
    assert m.prefix_agreement == 2
    assert m.shared_depth == 3
    assert len(m.cos_levels) == 3
    assert math.isclose(m.cos_levels[0], 1.0)


def test_meaning_match_identical_is_full():
    p = Ping(steps=(Step((1, 0, 0)), Step((0, 1, 0))))
    m = meaning_match(p, p)
    assert m.prefix_agreement == p.depth
    assert math.isclose(m.score, 1.0)


def test_meaning_match_without_shared_depth_is_not_full_agreement():
    empty = Ping()
    nonempty = Ping(steps=(Step((1, 0, 0)),))

    assert meaning_match(empty, empty).score == 1.0
    assert meaning_match(empty, nonempty).score == 0.0
    assert meaning_match(nonempty, empty).score == 0.0


def test_world_doorways_count():
    p = Ping(steps=(Step((1, 0, 0)), Step((0, 1, 0))))
    assert len(world_doorways(p)) == p.depth


def test_signature_fields_consistent():
    p = Ping(steps=tuple(Step((1, 0, 0), ROT_Y) for _ in range(4)))
    sig = path_signature(p)
    assert sig.start == (0, 0, 0)
    assert sig.end == world_path(p)[-1]
    assert math.isclose(sig.closure_error, norm(sig.displacement))
    assert 0.0 <= sig.net_over_path <= 1.0
    assert len(sig.turn_angles) == len(turn_angles(p))


def test_classify_straight_is_transport():
    assert classify(Ping(steps=(Step((1, 0, 0)),) * 4)) == "straight"
    # a single inward step has no bend -> still straight
    assert classify(Ping(steps=(Step((1, 0, 0)),))) == "straight"


def test_classify_loop_returns_to_source():
    loop = Ping(steps=tuple(Step((1, 0, 0), ROT_Y) for _ in range(4)))
    assert classify(loop) == "loop"
    assert math.isclose(path_signature(loop).closure_error, 0.0, abs_tol=1e-9)


def test_classify_spiral_is_open_constant_curvature():
    spiral = Ping(steps=tuple(Step((1, 0, 0), ROT_Y) for _ in range(3)))
    assert classify(spiral) == "spiral"
    assert path_signature(spiral).turn_spread < 1e-6
    assert path_signature(spiral).closure_error > 0


def test_classify_broken_is_varying_curvature():
    broken = Ping(steps=(Step((1, 0, 0)), Step((1, 1, 0)), Step((1, 1, 1))))
    assert classify(broken) == "broken"
    assert path_signature(broken).turn_spread > 1e-6


def test_classify_point_for_empty():
    assert classify(Ping()) == "point"


def test_prune_collapses_branching_space():
    pr = prune(query=(1, 1, 1), depth=4, threshold=0.9)
    assert pr.full_space == 26**4
    assert 0 < pr.survivors < pr.full_space
    assert all(len(path) == 4 for path in pr.paths)


def test_prune_threshold_monotone():
    loose = prune(query=(1, 1, 1), depth=2, threshold=0.0)
    tight = prune(query=(1, 1, 1), depth=2, threshold=0.9)
    assert tight.survivors <= loose.survivors


def test_world_path_has_depth_plus_one_points():
    p = Ping(steps=(Step((1, 0, 0)), Step((0, 1, 0))))
    path = world_path(p)
    assert len(path) == p.depth + 1
    assert path[0] == (0, 0, 0)


def test_world_path_increments_are_transported_doorways():
    p = Ping(steps=(Step((1, 0, 0), ROT_X), Step((0, 1, 0), ROT_Y)))
    path = world_path(p)
    incs = world_doorways(p)
    for i, w in enumerate(incs):
        step = tuple(path[i + 1][k] - path[i][k] for k in range(3))
        assert step == w


def test_identity_frames_give_straight_path():
    # Always step (1,0,0) inward with no frame rotation -> a straight line.
    p = Ping(steps=(Step((1, 0, 0)),) * 4)
    assert world_path(p) == [(0, 0, 0), (1, 0, 0), (2, 0, 0), (3, 0, 0), (4, 0, 0)]
    assert all(abs(a) < 1e-9 for a in turn_angles(p))


def test_locally_straight_but_globally_curved():
    # SAME local program (always doorway (1,0,0)); rotating frames bend the path.
    straight = Ping(steps=(Step((1, 0, 0)),) * 4)
    curved = Ping(steps=tuple(Step((1, 0, 0), ROT_Y) for _ in range(4)))
    assert [s.cell for s in straight.steps] == [s.cell for s in curved.steps]
    assert all(abs(a) < 1e-9 for a in turn_angles(straight))
    assert all(a > 1e-6 for a in turn_angles(curved))


def test_curved_path_returns_to_start_after_full_turn():
    # ROT_Y has order 4, so four identical inward steps close the loop.
    curved = Ping(steps=tuple(Step((1, 0, 0), ROT_Y) for _ in range(4)))
    assert world_path(curved)[-1] == (0, 0, 0)


def test_prune_depth_zero_is_single_empty_path():
    pr = prune(query=(1, 0, 0), depth=0, threshold=0.5)
    assert pr.full_space == 1
    assert pr.survivors == 1
    assert pr.paths == ((),)
