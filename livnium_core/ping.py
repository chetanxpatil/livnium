r"""
ping.py — the inward ping: meaning as a path of cosines through nested origins.

The rest of Livnium Core describes a *static* reference frame: a cube with an
Om/core (`base27`), 26 outer cells in four exposure classes (`lattice`), the 24
rigid OM-frames (`rotations`), class-preserving rearrangement (`moves`), and a
conserved nested geometry (`hierarchy`). `layer_language` is explicit that it
computes STRUCTURE, not meaning.

This module adds the missing piece: a *direction that goes inward*. A ping does
not merely point from A to B in one space — at every level it chooses a doorway
(one of the 26 cells) into a child cube that carries its own OM-frame, and then
chooses again. Direction is therefore a DESCENT, and similarity between two pings
is not one flat cosine but how long two descents keep entering the same way.

Model
-----
  - 26 doorways: the non-core cells of the N=3 cube, CELLS_26.
  - an OM-frame: an element of the order-24 cube rotation group (rotations.py).
  - a Step = (doorway cell c_i, child OM-frame R_i): "enter cell c_i; the child
    cube it opens is oriented by R_i relative to the parent".
  - a Ping = a list of Steps + a final local pattern direction d_local (a cell).

Frame algebra (the honest version of the chat's "cosθ -> lo -> cosθ ...")
-------------------------------------------------------------------------
  F_{-1} = I
  F_i    = F_{i-1} · R_i                      # accumulated world orientation
  w_i    = F_{i-1} · c_i                      # doorway i expressed in WORLD coords
  world_direction = F_{n-1} · d_local         # where the descent finally faces

A single scalar cosθ cannot pin a 3-D direction; the real object is the product
of small relative frames, and the cosines fall out as a *readout* between
consecutive transported doorways:

  cos_path = [ cos(w_0, w_1), cos(w_1, w_2), ... , cos(w_{n-1}, world_direction) ]

The descent also traces a PATH:  P_{i+1} = P_i + w_i  (world_path). If every
local doorway is the same ("go inward"), the path is straight when the frames are
fixed and curved when they rotate — locally straight, globally curved. The path
is not bending; the viewer-frame is rotating. The bend per step is turn_angles().

Conservation anchor
-------------------
  Every R_i is a rotation, so |world_direction| == |d_local|: the descent never
  inflates or loses magnitude, however deep it goes. The frame is the anchor; the
  branching is the illusion space; pruning is what keeps the 26^depth space from
  ever being built. All four facts are checked in tests/test_ping.py and in the
  __main__ self-check below.

Honest scope
------------
This is geometry, not semantics. It defines HOW a direction nests, composes,
correlates, and prunes — a meaning-shaped *machine*, not meanings themselves.
Meaning would enter only as data choosing the doorways (which c_i, which R_i);
this module supplies the lawful space those choices live in. cf. layer_language.
"""

from __future__ import annotations

import itertools
import math
import statistics
from dataclasses import dataclass, field
from typing import List, Sequence, Tuple

from .rotations import I, Matrix, apply, matmul

Vec = Tuple[int, int, int]

# The 26 doorways: every cell of the N=3 cube except the Om/core (0,0,0).
CELLS_26: Tuple[Vec, ...] = tuple(
    c for c in itertools.product((-1, 0, 1), repeat=3) if c != (0, 0, 0)
)


# --------------------------------------------------------------------------- #
# direction primitives
# --------------------------------------------------------------------------- #
def dot(a: Vec, b: Vec) -> int:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def norm(a: Vec) -> float:
    return math.sqrt(dot(a, a))


def cosine(a: Vec, b: Vec) -> float:
    """Cosine of the angle between two non-zero integer vectors, in [-1, 1]."""
    na, nb = norm(a), norm(b)
    if na == 0 or nb == 0:
        raise ValueError("cosine is undefined for the zero vector (the Om/core)")
    return dot(a, b) / (na * nb)


# --------------------------------------------------------------------------- #
# the ping
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Step:
    """One inward step: enter doorway `cell`; its child cube is oriented `frame`."""

    cell: Vec
    frame: Matrix = I

    def __post_init__(self) -> None:
        if self.cell == (0, 0, 0):
            raise ValueError("a doorway cannot be the Om/core (0,0,0)")
        if self.cell not in CELLS_26:
            raise ValueError(f"{self.cell!r} is not one of the 26 doorways")


@dataclass(frozen=True)
class Ping:
    """A descent through nested origins, ending at a local pattern direction."""

    steps: Tuple[Step, ...] = field(default_factory=tuple)
    d_local: Vec = (1, 0, 0)

    @property
    def depth(self) -> int:
        return len(self.steps)


def accumulated_frames(ping: Ping) -> List[Matrix]:
    """F_{-1}=I, F_i = F_{i-1}·R_i — the world orientation entering each level."""
    frames = [I]
    for step in ping.steps:
        frames.append(matmul(frames[-1], step.frame))
    return frames  # length depth+1; frames[i] is F_{i-1}, frames[-1] is F_{n-1}


def world_doorways(ping: Ping) -> List[Vec]:
    """w_i = F_{i-1}·c_i — each doorway transported into world coordinates."""
    frames = accumulated_frames(ping)
    return [apply(frames[i], step.cell) for i, step in enumerate(ping.steps)]


def world_direction(ping: Ping) -> Vec:
    """F_{n-1}·d_local — where the whole inward descent finally faces."""
    frames = accumulated_frames(ping)
    return apply(frames[-1], ping.d_local)


def cos_path(ping: Ping) -> List[float]:
    """Cosines between consecutive transported doorways, then into d_local.

    This is the chat's `cosθ0, cosθ1, ...`: not one flat angle but the alignment
    the descent maintains from each doorway to the next, ending in the final
    pattern direction. Empty for a depth-0 ping.
    """
    pts = world_doorways(ping)
    if not pts:
        return []
    pts = pts + [world_direction(ping)]
    return [cosine(pts[i], pts[i + 1]) for i in range(len(pts) - 1)]


# --------------------------------------------------------------------------- #
# the traced path: locally straight, globally curved
# --------------------------------------------------------------------------- #
def world_path(ping: Ping, start: Vec = (0, 0, 0)) -> List[Vec]:
    """The descent traced as a polyline in world space:

        P_0 = start ,  P_{i+1} = P_i + F_{i-1}·c_i

    i.e. each step adds the doorway *as seen through the accumulated orientation*
    (the transported doorways from `world_doorways`). Returns depth+1 points.
    Rotation matrices are integer, so the path stays on the integer lattice.
    """
    pts: List[Vec] = [start]
    for w in world_doorways(ping):
        p = pts[-1]
        pts.append((p[0] + w[0], p[1] + w[1], p[2] + w[2]))
    return pts


def turn_angles(ping: Ping) -> List[float]:
    """Angle (radians) between consecutive world steps — the GLOBAL bend.

    All zeros == the path is globally straight. Nonzero == it curves, even when
    every local step is just "go inward": the path is not bending, the
    viewer-frame is rotating. Accumulated turn is the spiral's total twist.
    """
    incs = world_doorways(ping)
    out: List[float] = []
    for i in range(len(incs) - 1):
        c = max(-1.0, min(1.0, cosine(incs[i], incs[i + 1])))
        out.append(math.acos(c))
    return out


# --------------------------------------------------------------------------- #
# path signature: the traced descent reduced to a classifiable atom
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Signature:
    """A ping's traced structure reduced to measurable quantities.

    closure_error == |end - start|  (0.0 == the descent returned to its source).
    net_over_path  == |displacement| / total_length in [0, 1]: 1 = perfectly
    straight, 0 = closed loop. turn_spread is the stdev of the per-step bend —
    ~0 means constant curvature (a clean spiral), larger means the curvature
    itself wanders (a broken/noisy path).
    """

    start: Vec
    end: Vec
    displacement: Vec
    closure_error: float
    total_length: float
    net_over_path: float
    turn_angles: Tuple[float, ...]
    mean_turn: float
    turn_spread: float
    cos_path: Tuple[float, ...]


def path_signature(ping: Ping, start: Vec = (0, 0, 0)) -> Signature:
    """Reduce a ping's world path to the signature used by `classify`."""
    pts = world_path(ping, start)
    end = pts[-1]
    disp = (end[0] - start[0], end[1] - start[1], end[2] - start[2])
    incs = world_doorways(ping)
    total = sum(norm(w) for w in incs)
    closure = norm(disp)
    turns = tuple(turn_angles(ping))
    mean_turn = sum(turns) / len(turns) if turns else 0.0
    spread = statistics.pstdev(turns) if len(turns) >= 2 else 0.0
    return Signature(
        start=start,
        end=end,
        displacement=disp,
        closure_error=closure,
        total_length=total,
        net_over_path=(closure / total) if total > 0 else 0.0,
        turn_angles=turns,
        mean_turn=mean_turn,
        turn_spread=spread,
        cos_path=tuple(cos_path(ping)),
    )


def classify(
    ping: Ping,
    *,
    angle_tol: float = 1e-6,
    closure_tol: float = 1e-9,
    spread_tol: float = 1e-6,
) -> str:
    """Sort a ping by the shape it traces. Deterministic, threshold-documented:

        point    : no movement (depth 0)
        straight : never bends (every turn <= angle_tol)            -> transport
        loop     : bends but returns to source (closure <= closure_tol) -> memory/return
        spiral   : bends at constant curvature (turn_spread <= spread_tol) -> recursive descent
        broken   : bends at varying curvature                       -> divergence / noise

    The spiral/broken split is exactly your "twist is accumulated orientation"
    vs. "twist that wanders": a constant turn is structure, a varying turn is noise.
    """
    s = path_signature(ping)
    if s.total_length == 0.0:
        return "point"
    if all(a <= angle_tol for a in s.turn_angles):
        return "straight"
    if s.closure_error <= closure_tol:
        return "loop"
    if s.turn_spread <= spread_tol:
        return "spiral"
    return "broken"


# --------------------------------------------------------------------------- #
# meaning match: how long two descents keep entering the same way
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Match:
    prefix_agreement: int       # leading levels with identical (cell, frame)
    shared_depth: int           # min(depth_p, depth_q)
    cos_levels: Tuple[float, ...]  # cos(p.w_i, q.w_i) over shared depth
    score: float                # mean alignment over shared depth (1.0 if both empty)


def meaning_match(p: Ping, q: Ping) -> Match:
    """Nested similarity: meaning is not the surface vector, it is the depth to
    which two pings choose the same inner doors and stay aligned doing so."""
    pa = 0
    for sp, sq in zip(p.steps, q.steps):
        if sp == sq:
            pa += 1
        else:
            break

    wp, wq = world_doorways(p), world_doorways(q)
    d = min(len(wp), len(wq))
    cos_levels = tuple(cosine(wp[i], wq[i]) for i in range(d))
    score = 1.0 if d == 0 else sum(cos_levels) / d
    return Match(prefix_agreement=pa, shared_depth=d, cos_levels=cos_levels, score=score)


# --------------------------------------------------------------------------- #
# pruning: the 26^depth illusion space, collapsed to its aligned spine
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Prune:
    depth: int
    threshold: float
    full_space: int          # 26 ** depth — never actually built
    survivors: int           # aligned paths that remain
    paths: Tuple[Tuple[Vec, ...], ...]  # the surviving doorway sequences


def prune(query: Vec, depth: int, threshold: float) -> Prune:
    """Walk the 26-branching descent toward `query`, keeping only doorways whose
    direction aligns with the query (cos >= threshold) at every level.

    Demonstrates the core claim: the space is 26**depth, but only its aligned
    spine is ever instantiated, so a huge branching space collapses to a short
    confident path. (Inner OM-frames are held at I here so the cell-branching
    collapse is shown in isolation; frames multiply the space by 24 per level and
    prune the same way.)
    """
    if depth < 0:
        raise ValueError("depth must be >= 0")
    aligned = [c for c in CELLS_26 if cosine(c, query) >= threshold]
    paths: List[Tuple[Vec, ...]] = [()]
    for _ in range(depth):
        paths = [path + (c,) for path in paths for c in aligned]
    return Prune(
        depth=depth,
        threshold=threshold,
        full_space=26 ** depth,
        survivors=len(paths),
        paths=tuple(paths),
    )


# --------------------------------------------------------------------------- #
# self-check
# --------------------------------------------------------------------------- #
def _selfcheck() -> None:
    from .rotations import ROT_Y, rotation_group

    assert len(CELLS_26) == 26

    # 1) Conservation: the descent preserves magnitude however deep it goes.
    deep = Ping(steps=tuple(Step(cell=c, frame=r)
                            for c, r in zip(CELLS_26[:6], rotation_group()[:6])),
                d_local=(1, 0, 0))
    assert math.isclose(norm(world_direction(deep)), norm(deep.d_local), rel_tol=1e-9)
    print(f"1. conservation: |d_local|={norm(deep.d_local):.3f} == "
          f"|world_dir|={norm(world_direction(deep)):.3f}  (depth {deep.depth})")

    # 2) same pattern, different direction: identical doorways/local, one frame turned.
    base = Ping(steps=(Step((1, 0, 0)),), d_local=(1, 0, 0))
    turned = Ping(steps=(Step((1, 0, 0), ROT_Y),), d_local=(1, 0, 0))
    c = cosine(world_direction(base), world_direction(turned))
    assert abs(c - 1.0) > 1e-6
    print(f"2. same pattern != same direction: cos(world_base, world_turned)={c:+.3f}")

    # 3) meaning match: shared prefix then divergence.
    p = Ping(steps=(Step((1, 0, 0)), Step((0, 1, 0)), Step((0, 0, 1))))
    q = Ping(steps=(Step((1, 0, 0)), Step((0, 1, 0)), Step((-1, 0, 0))))
    m = meaning_match(p, q)
    assert m.prefix_agreement == 2 and m.shared_depth == 3
    print(f"3. meaning match: prefix_agreement={m.prefix_agreement}/{m.shared_depth}, "
          f"score={m.score:+.3f}")

    # 4) pruning: 26^depth illusion space collapses to its aligned spine.
    pr = prune(query=(1, 1, 1), depth=4, threshold=0.9)
    ratio = pr.survivors / pr.full_space
    assert pr.survivors < pr.full_space
    print(f"4. pruning: full={pr.full_space:,}  survivors={pr.survivors:,}  "
          f"kept={ratio:.2e}  (threshold {pr.threshold})")

    # 5) locally straight, globally curved: SAME local program ("always go (1,0,0)
    #    inward"), two frame choices. Identity frames -> straight; rotating frames
    #    -> the world path bends though the local doorway never changes.
    local = (Step((1, 0, 0)),) * 4
    straight = Ping(steps=local)
    curved = Ping(steps=tuple(Step((1, 0, 0), ROT_Y) for _ in range(4)))
    assert [s.cell for s in straight.steps] == [s.cell for s in curved.steps]
    ts, tc = turn_angles(straight), turn_angles(curved)
    assert all(abs(a) < 1e-9 for a in ts)
    assert all(a > 1e-6 for a in tc)
    print(f"5. straight vs curved (identical local doorways):")
    print(f"     straight path = {world_path(straight)}  turns={[round(a,3) for a in ts]}")
    print(f"     curved   path = {world_path(curved)}  turns={[round(a,3) for a in tc]}")

    # 6) path signature: classify the four canonical descent shapes.
    catalogue = {
        "straight": Ping(steps=(Step((1, 0, 0)),) * 4),
        "loop": Ping(steps=tuple(Step((1, 0, 0), ROT_Y) for _ in range(4))),
        "spiral": Ping(steps=tuple(Step((1, 0, 0), ROT_Y) for _ in range(3))),
        "broken": Ping(steps=(Step((1, 0, 0)), Step((1, 1, 0)), Step((1, 1, 1)))),
    }
    print("6. classify:")
    for expected, png in catalogue.items():
        got = classify(png)
        assert got == expected, f"{expected!r} misclassified as {got!r}"
        sig = path_signature(png)
        print(f"     {got:8s}  closure={sig.closure_error:.2f}  "
              f"net/path={sig.net_over_path:.2f}  turn_spread={sig.turn_spread:.3f}")

    print("all ping self-checks passed.")


if __name__ == "__main__":
    _selfcheck()
