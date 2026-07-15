# Livnium Core — All Formulas & Logic

Every formula in the system, with the reasoning behind it. All entries verified
by the test suite (`packages/livnium-core/tests/`) and re-derived from scratch. Notation: `N` is an
odd integer ≥ 3 (the lattice dimension); a "cell" is one unit of the lattice.

---

## 1. The lattice

$$\mathcal{L}_N = \left\{-\tfrac{N-1}{2},\dots,+\tfrac{N-1}{2}\right\}^3,\qquad |\mathcal{L}_N| = N^3$$

**Logic.** A cube of side `N` centered on the origin, integer coordinates. For
`N=3` the coordinates are `{-1,0,1}` → 27 cells. The half-width is
`h = (N-1)/2`.

Each cell holds one symbol of a 27-letter alphabet via a **reversible bijection**
Σ ↔ 𝓛_N. The center cell `(0,0,0)` is **Om**, the global observer/anchor.

---

## 2. Base-27 codec

Alphabet (index = digit value): `0 a b c … z` → values `0,1,…,26`. The `0`
symbol is Om.

**Decode (string → integer):**
$$\text{val}(s)=\sum_{i=0}^{L-1} d_i \cdot 27^{\,L-1-i}$$
where `d_i` is the value of the i-th character.

**Encode (integer → string):** repeatedly take `n mod 27` (digit) and `n // 27`
(carry), read digits in reverse.

**Binary bridge:** `base27 → int → bin` and back, both lossless.

**Carry law.** Standard positional carry in base 27:
$$26 + 1 = 27 \;\Rightarrow\; \texttt{z} + \texttt{a} = \texttt{a0}$$
$$728 + 1 = 729 \;\Rightarrow\; \texttt{zz} + \texttt{a} = \texttt{a00}$$

**Logic.** This is exactly a positional numeral system with radix 27, so it
inherits exact arithmetic and lossless conversion for free. Verified:
`base27_to_int(add(x,y)) == base27_to_int(x) + base27_to_int(y)` for random pairs.

---

## 3. Exposure and Symbolic Weight

**Exposure** of a cell `(x,y,z)` — how many coordinates sit on the outer boundary:
$$f(x,y,z) = [\,|x|=h\,] + [\,|y|=h\,] + [\,|z|=h\,] \in \{0,1,2,3\}$$

**Symbolic Weight:**
$$\boxed{\;\mathrm{SW} = 9f\;}$$

| class | f | SW | meaning |
|---|---|---|---|
| Core | 0 | 0 | interior / witness |
| Center (face) | 1 | 9 | stabilizer |
| Edge | 2 | 18 | connector |
| Corner | 3 | 27 | initiator |

**Logic.** `f` counts how many of the three axes are "maxed out" at the surface.
A corner maxes all three (f=3), an edge two, a face-center one, the deep interior
none. SW just scales that by 9.

---

## 4. Class counts (closed form)

$$\text{core} = (N-2)^3,\quad \text{center} = 6(N-2)^2,\quad \text{edge} = 12(N-2),\quad \text{corner} = 8$$

**Logic / derivation.**
- **Corners:** a cube always has `8` corners (`2³`).
- **Edges:** `12` edges, each with `(N-2)` non-corner cells → `12(N-2)`.
- **Face-centers:** `6` faces, each an `(N-2)×(N-2)` interior grid → `6(N-2)²`.
- **Core:** the fully interior `(N-2)³` block.

**Sum identity** (why it must total `N³`):
$$(N-2)^3 + 6(N-2)^2 + 12(N-2) + 8 = \big((N-2)+2\big)^3 = N^3$$
It is literally the binomial expansion of `(a+2)³` with `a = N-2`. The 1·6·12·8
pattern is the binomial coefficients `1,3,3,1` times `2^k`.

---

## 5. Total Symbolic Weight (the conserved ledger)

$$\boxed{\;\Sigma\mathrm{SW}(N) = 54(N-2)^2 + 216(N-2) + 216\;}$$

**Derivation.**
$$\Sigma\mathrm{SW} = 9\big[\,1\cdot\text{center} + 2\cdot\text{edge} + 3\cdot\text{corner}\,\big]
= 9\big[\,6(N-2)^2 + 24(N-2) + 24\,\big]$$
$$= 54(N-2)^2 + 216(N-2) + 216$$
(Core contributes 0 since SW=0 there.)

**Values:** `ΣSW(3)=486`, `ΣSW(5)=1350`, `ΣSW(7)=2646`, `ΣSW(9)=4374`.
*(Note: an early draft wrote `ΣSW(7)=3024`; that was a typo — the correct value is 2646.)*

---

## 6. Dynamics — the cube rotation group

Generators (90° rotations about X, Y, Z as integer matrices):

$$R_X=\begin{pmatrix}1&0&0\\0&0&-1\\0&1&0\end{pmatrix}\;
R_Y=\begin{pmatrix}0&0&1\\0&1&0\\-1&0&0\end{pmatrix}\;
R_Z=\begin{pmatrix}0&-1&0\\1&0&0\\0&0&1\end{pmatrix}$$

**Group:** closing `{R_X,R_Y,R_Z}` under composition gives exactly
$$|G| = 24,\qquad G \cong S_4$$
and each generator satisfies `R⁴ = I`.

**Logic.** These are the orientation-preserving symmetries of a cube — 6 faces ×
4 rotations = 24. A 90° turn done four times returns to start, hence `R⁴=I`.
Applying a rotation to a coordinate is `c ↦ R·c`.

---

## 7. Conservation theorem

> Under any sequence of allowed rotations, the following are **invariant**:
> total symbolic weight `ΣSW`, the four class counts, the symbol count `N³`,
> and the symbol↔coordinate bijection.

**Logic.** Every rotation is a **reversible permutation** of the lattice (it maps
cells onto cells, one-to-one). A rotation sends a corner to a corner, an edge to
an edge, a face to a face — because it preserves how many coordinates are at the
boundary (`f` is rotation-invariant). If class membership is preserved cell-by-cell,
then class counts and the SW sum cannot change. Verified for all 24 elements on
`N=3`.

---

## 8. Orientation independence (Theorem T13)

> A rotation's effect on the exposure structure is **identical regardless of the
> lattice's current orientation**. Consequently the full simulation reduces to
> **24 precomputed constants** (≈497× speedup, bit-identical output).

**Logic.** Rotations form a group, so composing a rotation `g` with any starting
orientation `o` is just another group element `g·o` that permutes the lattice
bijectively. The exposure-response pattern therefore depends only on `g`, not on
`o`. Verified: across all 24 starting orientations there is exactly **1** distinct
exposure-response pattern.

---

## 9. Polarity (observer-relative meaning)

$$\text{Polarity} = \cos\theta \in [-1,+1]$$
where `θ` is the angle between a symbol's motion vector and the vector pointing
toward the active observer (Om, or a temporary Local Observer).

| Polarity | angle | meaning |
|---|---|---|
| +1 | 0° | approach / intent |
| 0 | 90° | neutral / orthogonal |
| −1 | 180° | recession / negation |

**Logic.** Meaning is encoded as *direction relative to an observer* — moving
toward Om = affirmation, away = negation. This is a modeling definition (not a
theorem); its usefulness for tasks is an empirical question (see `BENCHMARKS.md`).

---

## 10. Hierarchy (multi-scale "world")

A macro lattice of size `N` hosts a micro lattice of size `M` inside every cell.

**Capacity:**
$$\text{capacity}(N,M) = N^3 \cdot M^3$$

**Additive ledger (conservation across scales):**
$$\Sigma\mathrm{SW}_{\text{global}} = \underbrace{N^3 \cdot \Sigma\mathrm{SW}(M)}_{\text{all micro blocks}} + \underbrace{\Sigma\mathrm{SW}(N)}_{\text{macro}}$$

**Symmetry — wreath product:**
$$G_M \wr G_N,\qquad |G_M \wr G_N| = 24 \cdot 24^{\,N^3}\ \text{(rotation case)}$$

**Logic.** Each micro block conserves its own ledger; the macro lattice conserves
its own; the global total is just the sum. The wreath product captures "rotate
inside blocks **and** rotate the blocks themselves." Verified: e.g. `N=M=3` →
`27·486 + 486 = 13608`.

---

## 11. One-line summary

> Livnium Core is a reversible symbolic system on an odd cubic lattice: a base-27
> codec assigns symbols to coordinates; exposure `f` gives weight `SW=9f`; the
> four classes follow `(a+2)³` binomial counts; the total `ΣSW=54(N-2)²+216(N-2)+216`
> is conserved under the 24-element cube rotation group (all reversible); the
> response is orientation-independent (24 constants); and lattices nest under a
> wreath product with a strictly additive ledger.

*Every boxed/closed-form result above is checked in
`packages/livnium-core/tests/`. See `docs/history/FINDINGS.md` for the complete
win/partial/falsified inventory, and `docs/results/BENCHMARKS.md` for the honest
NLI results.*
