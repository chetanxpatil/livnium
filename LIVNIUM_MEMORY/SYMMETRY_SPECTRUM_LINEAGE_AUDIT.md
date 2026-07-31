# Symmetry Spectrum Lineage Audit

**Session:** S15  
**Audit date:** 2026-07-26  
**Historical source tree:** `/Users/chetanpatil/Desktop/test`  
**Canonical reading copy:**  
`/Users/chetanpatil/Desktop/test/_ORGANIZED/02_Experiments/Symmetry`  
**Historical root mirror:** `/Users/chetanpatil/Desktop/test`  
**Independent probe:** `SYMMETRY_SPECTRUM_AUDIT_PROBE.py` in this memory

## Executive verdict

The complete Symmetry family is four files copied exactly between the organized
folder and the archive root. It contains one 7x7x7 grid-Laplacian experiment,
one JSON result, one figure, and one verdict.

The central numerical observation is correct:

- the 343 eigenvalues of the unweighted 7x7x7 grid Laplacian occupy 70 distinct
  levels at tolerance `1e-6`;
- only seven levels are simple;
- the saved multiplicity histogram
  `{1:7, 3:36, 6:20, 15:6, 18:1}` is exact; and
- ten connected random graph controls with the same node/edge counts have 343
  distinct eigenvalues.

The attribution in the historical verdict is too simple. The spectrum is not
only a fingerprint of the 24 proper cube rotations:

1. the operator is the Cartesian graph product `P7 □ P7 □ P7`;
2. all three axis operators are identical, producing coordinate-permutation
   degeneracy;
3. all 24 improper signed-axis/reflection symmetries also commute with the
   Laplacian, giving a 48-element full cubic symmetry;
4. the 1-D path spectrum obeys exact complementary-pair identities; and
5. several different unordered mode triples therefore collide at the same
   eigenvalue.

The multiplicities 15 and 18 are **not** dimensions of irreducible
representations of the proper cube group, whose irreducible dimensions are 1,
1, 2, 3, and 3. They are sums of several permutation-orbit multiplicities that
share an eigenvalue.

The best honest conclusion is:

> The isotropic Cartesian cube grid has an exact, highly degenerate separable
> spectrum. Cubic symmetries preserve each eigenspace, while equal-axis product
> structure and path-eigenvalue arithmetic determine the observed 70 levels and
> their larger degeneracies. This establishes a structural spectral signature,
> not a computational task advantage.

This remains a useful exact artifact. It should be retained as a spectral unit
test and a starting point for representation decomposition, not cited as proof
of protected computation, conserved physical quantities, meaning, compression,
or learning advantage.

## Source identity and preservation

All four files in the organized folder have same-named byte-for-byte duplicates
at the archive root. They are preservation copies, not independent replications.
Both historical copies remain untouched.

| File | Bytes | SHA-256 |
|---|---:|---|
| `SYMMETRY_SPECTRUM_VERDICT.md` | 2,673 | `9410a55b795c69b1570877b82d7ee8e903e6c0a56ca11599554f1935ba9c7539` |
| `livnium_symmetry_spectrum.json` | 148 | `47688ee06669e4848ffeab3811a99c898e1559b6e2bda178011261b63d1b9585` |
| `livnium_symmetry_spectrum.png` | 88,716 | `5c91d368b135dad7fd1629a71d4e39e5a554eb769ef62fc3588d4936b4315373` |
| `livnium_symmetry_spectrum.py` | 6,041 | `cedf0eeb611bedae4f0982a4ba61ed4488df4e7c4bd6c0abc17cab70750d3102` |

No learned model, checkpoint, downstream task, perturbation study, eigenvector
artifact, representation decomposition, or cross-size replication accompanies
the family.

## What the source actually computes

The source constructs the combinatorial graph Laplacian of the 343 lattice
points:

```text
{−3,…,3} × {−3,…,3} × {−3,…,3}
```

Nearest neighbors differ by one along one coordinate. The graph has:

- 343 vertices;
- 882 undirected edges;
- boundary degrees from 3 through 6; and
- exact product form `P7 □ P7 □ P7`.

It compares this with one random graph containing the same number of vertices
and edges. It groups sorted eigenvalues when adjacent values differ by at most
`1e-6`.

Calling each eigenvector a “vein” and each eigenvalue a “pull speed” is a
metaphor. Mathematically these are Laplacian modes and eigenvalues. Under
continuous diffusion `dx/dt = -Lx`, larger eigenvalues decay faster; under the
earlier Tikhonov operator `I + λL`, eigenvalues determine attenuation. The
source does not itself simulate either time evolution.

## Exact analytic spectrum

The path graph `P7` has eigenvalues:

```text
μk = 2 − 2 cos(kπ/7),  k=0,…,6
```

The Cartesian product spectrum is:

```text
λijk = μi + μj + μk
```

for all 343 ordered triples `(i,j,k)`.

Fresh comparison between these analytic sums and the numerical eigensolver has
maximum absolute error `3.55e-14`. The archived result is therefore not a noisy
empirical curiosity; it is an exact separable-spectrum calculation.

## Where 70 distinct levels come from

### Coordinate-permutation structure

If only axis permutations are considered, 343 ordered triples reduce to:

```text
C(7+3−1, 3) = C(9,3) = 84
```

unordered triples.

Their ordinary permutation-orbit sizes are:

- 1 when all three indices are equal;
- 3 when two are equal; and
- 6 when all three differ.

If every unordered triple had a distinct eigenvalue, there would be 84 spectral
levels with multiplicities only 1, 3, and 6.

### Extra path-spectrum identities

The path eigenvalues satisfy:

```text
μ1 + μ6 = μ2 + μ5 = μ3 + μ4 = 4
```

This creates seven larger collisions.

The unique 18-dimensional level merges three six-permutation families:

```text
(0,1,6), (0,2,5), (0,3,4)
```

Each of the six 15-dimensional levels merges one three-permutation family and
two six-permutation families. For example:

```text
(1,1,6)  -> 3 permutations
(1,2,5)  -> 6 permutations
(1,3,4)  -> 6 permutations
total       15
```

The same construction occurs with leading index 2 through 6.

Merging three unordered-triple levels into one removes two levels each time:

```text
84 − 7×2 = 70
```

This exactly explains the archived count.

## Symmetry boundary

Fresh exhaustive permutation checks show:

| Symmetry class | Elements checked | Maximum `||PᵀLP−L||` |
|---|---:|---:|
| Proper signed-axis rotations | 24 | 0 |
| Improper signed-axis/reflection operations | 24 | 0 |

The operator therefore has the full 48-element cubic signed-permutation
symmetry, not only the 24-element orientation-preserving subgroup.

Commutation means every Laplacian eigenspace is invariant under the symmetry
action. Representation theory can decompose each eigenspace into irreducible
components. It does **not** mean that each observed eigenspace multiplicity is
itself an irrep dimension.

For the proper octahedral group, irreducible dimensions are 1, 1, 2, 3, and 3.
The archived multiplicities 6, 15, and 18 must contain multiple irreducible
components and, for 15/18, multiple separable mode families sharing one
eigenvalue.

Therefore these sentences are safe:

- cubic symmetry preserves and organizes the eigenspaces;
- equal-axis isotropy forces coordinate-related modes to share eigenvalues;
- the exact multiplicity pattern is a signature of this particular finite
  product grid.

This sentence is not safe without a character decomposition:

- “the multiplicities 3, 6, 15, and 18 are the cube group’s irreps.”

## Random comparator audit

The archived random graph reports 339 distinct levels and maximum multiplicity
five. Fresh inspection shows that graph has five connected components.

For every graph Laplacian, zero-eigenvalue multiplicity equals the number of
connected components. The random graph’s fivefold level is therefore the zero
eigenvalue from disconnectedness—not a small accidental symmetry multiplet.
Its remaining 338 nonzero levels are numerically simple.

Ten fresh connected random controls were built from a random spanning tree plus
additional random edges to reach the same 882-edge budget. Every seed has:

```text
343 distinct eigenvalues of 343
```

This strengthens the generic-spectrum control while correcting the archived
disconnectedness.

The comparator is still only edge-count matched. It does not match:

- degree sequence;
- boundary structure;
- locality;
- diameter;
- clustering;
- Cartesian product form; or
- automorphism group.

Those mismatches do not invalidate the existence of the cube-grid degeneracy.
They only prevent attributing every difference specifically to one symmetry
property.

## Anisotropy control

The fresh probe keeps the separable path modes but weights the three axes
differently:

```text
μi + √2 μj + π μk
```

All 343 sums are distinct at tolerance `1e-6`.

This control shows that separability alone is not enough. Equality/isotropy of
the three axis operators is load-bearing for the coordinate-permutation
degeneracies. The 70-level pattern belongs to the isotropic cube grid, not to
every rectangular or separable operator.

The converse is also important: spectrum alone does not uniquely identify
spatial cube structure. Any orthogonal conjugate `QLQᵀ` has exactly the same
eigenvalues while generally becoming dense and losing the original coordinate
locality. A spectral signature is an invariant, not a complete structural
fingerprint.

## Claim decisions

### Verified

- All four organized/root pairs are exact duplicates.
- The 7x7x7 grid has exactly 70 distinct Laplacian levels at tolerance `1e-6`.
- The saved multiplicity histogram is exact.
- The analytic Cartesian-product spectrum matches numerical eigenvalues to
  `3.55e-14`.
- Proper and improper cubic signed-coordinate operations commute with the
  Laplacian.
- Generic connected random controls at the same node/edge counts have simple
  spectra in all ten fresh seeds.
- Breaking axis equality with generic weights removes all detected degeneracy.
- The source explicitly leaves task advantage open.

### Narrowed

- “Cube rotation symmetry causes the pattern” → cubic symmetry, identical-axis
  Cartesian product structure, and exact path-spectrum identities jointly
  determine it.
- “Random max multiplicity five is accidental” → it is exactly the number of
  connected components at eigenvalue zero.
- “336 veins are symmetry-locked” → 336 modes belong to nonsimple eigenspaces,
  but larger degeneracies include arithmetic collisions among separate mode
  families.
- “The multiplicities are the cube group’s irreps” → eigenspaces carry
  representations; their full dimensions are not individual irrep dimensions.
- “Pattern the random operator cannot have” → this random graph lacks it;
  another operator can be cospectral, and many other symmetric graphs have
  structured degeneracies.

### Still untested

- task accuracy;
- sample efficiency;
- compression;
- robustness under perturbation;
- protected computation;
- selection rules for a specified task/operator;
- conserved quantities beyond standard diffusion invariants;
- cross-size behavior;
- eigenvector localization;
- whether the relevant task respects the same group action; and
- advantage over standard equivariant architectures.

## What should be preserved

1. The exact product-spectrum construction.
2. Exhaustive commutator tests for all 48 signed-axis symmetries.
3. Multiplicity histograms as regression tests.
4. The distinction between convergence speed and spectral structure.
5. The honest statement that structural existence and task usefulness are
   different claims.
6. Connected and anisotropic controls from the fresh probe.
7. The complementary-mode identity `μk + μ7−k = 4`.

## Recommended next experiment

Do not ask whether degeneracy merely exists again; that is settled.

Choose a task whose target is explicitly equivariant or invariant under the
cube action. Compare, with matched data and parameter budgets:

1. unconstrained linear/MLP baseline;
2. data augmentation;
3. group averaging;
4. a fully equivariant model;
5. the same architecture with anisotropic axis weights; and
6. random orthogonal features preserving the spectrum but destroying locality.

Predeclare:

- underlying-world-disjoint train/test identities;
- task accuracy and sample-efficiency curves;
- perturbations that preserve versus break symmetry;
- full seed distribution;
- parameter and compute counts; and
- exact equivariance residuals.

The decisive question is:

> Does respecting the known representation decomposition improve
> generalization or robustness on a group-compatible task beyond ordinary
> augmentation and matched generic priors?

## Preservation action

- Historical organized folder: unchanged.
- Historical root duplicate: unchanged.
- New report and probe: written only into Livnium memory.
- Probe imports disable bytecode; no source cache was created.

## Final S15 classification

**Status:** incorporated as a complete four-file historical family.

**Strongest result:** exact 70-level spectrum of the isotropic Cartesian cube
grid and exact multiplicity arithmetic.

**Important correction:** multiplicities 15 and 18 are merged separable
eigenspaces, not cube-group irrep dimensions; the random fivefold level is graph
disconnectedness.

**Research value:** high as a clean mathematical/unit-test artifact, open as a
computational advantage.

**Do not cite as established:** task advantage, protected modes for a task,
novel conserved quantities, physical meaning, compression advantage, or a
unique spectral fingerprint of cube geometry.

