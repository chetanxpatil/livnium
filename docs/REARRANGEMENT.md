# Rearrangement Dynamics — "the cube can change its own nodes"

*The math behind the intuition: "allowed change in the cube — it can rearrange
the nodes inside; the box around the core defines how each node processes energy."*
All claims below are validated in `tests/test_moves.py` (20/20 passing repo suite).

## Two move sets

| move set | what it does | size |
|---|---|---|
| **Rigid rotations** (`rotations.py`) | turn the *whole* cube | 24 |
| **Face turns** (`moves.py`) | turn *one layer* → nodes rearrange among themselves | generates the Rubik's group, ~4.3 × 10¹⁹ states |

A **face turn** is a rotation applied to a single layer (e.g. all cells with
`y = +1`), leaving the rest fixed. This is what lets the nodes move *relative to
each other* — the "allowed change" — instead of the whole frame turning together.

## What is conserved when you allow rearrangement (validated)

For every face turn, and any sequence of them:

1. **It is a valid permutation of order 4** (`T⁴ = I`).
2. **The core Om (0,0,0) stays fixed.** It is in no face layer, so no turn moves it.
3. **The 6 face-centers stay fixed.** (A center is the pivot of its own face.)
4. **Moves are class-preserving:** a corner can only land in a corner slot, an
   edge in an edge slot, a face-center in a face-center slot.
5. **Therefore ΣSW = 486 is invariant** under *any* scramble — confirmed after
   10,000 random face turns: every slot still holds a token of its own exposure
   class, total energy unchanged.
6. **Every move is reversible** (a turn's inverse is three more of the same turn).

## The precise statement of your intuition

> The **shell structure** ("the box around the core") assigns each node its energy
> by which slot-type it occupies: corner = 27, edge = 18, face = 9, core = 0.
> **Rearrangement** (face turns) moves nodes only between slots of the *same*
> type, so it redistributes *which* node sits where while conserving the total
> and the per-shell energy budget. The core never moves; it is the fixed point
> the whole structure is defined around.

## Honest boundary

This is a true and rather beautiful **mathematical** property: a conservation law
that survives a 4.3 × 10¹⁹-element rearrangement group. But conservation here
holds *by construction* (moves preserve class), so — like the rest of the core —
it constrains the **structure**, not any external task. See `LIMITS.md`. It is
real mathematics about the object; it is not, by itself, a learning mechanism.

## Hierarchical rearrangement ("26 expand inwards depth, each its own pyramid")

Let each of the 26 outer nodes expand inward into its **own nested cube**, each
with its own local core **LO**; the macro core is **Om**. Apply class-preserving
face turns at every level ("everything moves up in its own pyramid"). Validated:

- **Additive ledger conserved at every level:** macro = 486, each of the 26 micro
  cubes = 486, GLOBAL = 486 + 26·486 = **13122**.
- **Om stays fixed, and every LO stays fixed** — the core of each level is the
  invariant point it is built around (`Om ↔ LO`).

## Why the "correct pattern rules" are *necessary* (not optional)

- Rule-following rearrangement (class-preserving): every node's energy stays
  matched to its shell — structure intact.
- Arbitrary permutation (rules ignored): the total still sums to 486, but
  **17 / 27 nodes land in wrong-energy slots** — the structure is violated.

So: the *total* is conserved by any shuffle trivially, but the *meaningful*
invariant — each node's energy matching its shell, cores fixed, pyramids intact —
survives **only** under rule-following rearrangement. The pattern rules are a
proven requirement.

## Group sizes (the "rearrangement freedom")

```
rigid whole-cube rotations          : 24
class-preserving position perms      : 8! · 12! · 6! = 13,905,608,048,640,000
Rubik's cube group (with orientation): 43,252,003,274,489,856,000
```

ΣSW is invariant across all of them, because symbolic weight depends only on
exposure class, and every one of these groups fixes the classes setwise.
