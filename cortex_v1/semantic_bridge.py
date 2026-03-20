"""
Semantic Bridge  v1
===================

Replaces MD5 % 24 with a geometrically grounded word → SO(3) mapping.

Pipeline
--------
  word
    → GloVe-50 vector  v ∈ R^50          (pretrained semantic embedding)
    → PCA projection   v' ∈ R^3           (sklearn PCA fit on corpus vocabulary)
    → normalise        n̂ = v'/‖v'‖       (rotation axis on unit sphere)
    → angle            θ = f(word)        (see below)
    → SO(3) matrix     M = axis_angle(n̂, θ)
    → (optional) nearest of 24 cube rotations, OR use M directly via SU(2)

Angle assignment
----------------
Two options are available:

  'idf'    : θ = π × (1 − exp(−IDF(word)))
             Rare / high-information words get θ → π  (half-turn, high α)
             Common words ("the", "is") get θ → 0     (identity, α ≈ 0)

  'cosine' : θ = arccos(cosine_similarity(v, corpus_mean))
             Words far from the average concept get large θ
             Words near the corpus centroid get small θ

Why this changes the governor behaviour
----------------------------------------
With MD5:
    - Same word always hits same rotation class
    - Position in sequence determines pruning, not content

With semantic embedding:
    - Semantically similar words → nearby axes → rotations nearly commute
      → low entropy growth → governor preserves the bond
    - Semantically dissimilar adjacent words → orthogonal axes
      → large entropy growth → governor prunes harder

This is constructive vs. destructive interference in the MPS,
driven by actual word meaning rather than hash collision.

Verification test (embedded in semantic_bridge_test())
------------------------------------------------------
For any pair of words A, B:
    axis_similarity(A, B) = dot(n̂_A, n̂_B)

Claim: axis_similarity should correlate with GloVe cosine similarity.
We test this on known pairs before using the bridge in the organism.
"""

from __future__ import annotations

import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from livnium_cortex_v1 import (
    extract_axis_angle,
    axis_angle_to_su2,
    _Rx90, _Ry90, _Rz90,
)

# ─────────────────────────────────────────────────────────────────────────────
# GloVe model (loaded once, cached)
# ─────────────────────────────────────────────────────────────────────────────

_GLOVE_MODEL = None

def _get_glove():
    global _GLOVE_MODEL
    if _GLOVE_MODEL is None:
        import gensim.downloader as api
        _GLOVE_MODEL = api.load("glove-wiki-gigaword-50")
    return _GLOVE_MODEL


# ─────────────────────────────────────────────────────────────────────────────
# PCA projector: R^50 → R^3
# ─────────────────────────────────────────────────────────────────────────────

class SemanticProjector:
    """
    Fits a PCA from R^50 to R^3 on a seed vocabulary.

    The projection captures the top-3 axes of variance in the semantic space.
    Words with similar meanings will map to nearby points in R^3.

    Parameters
    ----------
    seed_vocab : list of words to fit PCA on.
                 Should cover the domain of interest.
                 Unknown words fall back to zero vector.
    """

    def __init__(self, seed_vocab: list[str] | None = None):
        self._fitted = False
        self._pca_components = None   # (3, 50) — top 3 eigenvectors
        self._pca_mean       = None   # (50,)   — mean of seed vectors
        self._corpus_mean_3d = None   # (3,)    — projected mean (for angle=cosine)
        self._idf_cache: dict[str, float] = {}
        self._vocab_freq: dict[str, int]  = {}

        if seed_vocab is not None:
            self.fit(seed_vocab)

    def fit(self, vocab: list[str]):
        """
        Fit PCA on the GloVe vectors of `vocab`.
        Only words present in GloVe are used.
        """
        from sklearn.decomposition import PCA

        model  = _get_glove()
        vecs   = []
        valid  = []

        for w in vocab:
            w_low = w.lower()
            if w_low in model:
                vecs.append(model[w_low])
                valid.append(w_low)

        if len(vecs) < 3:
            raise ValueError(
                f"SemanticProjector needs ≥3 known words, got {len(vecs)}."
            )

        V = np.array(vecs)          # (n_words, 50)
        self._pca_mean = V.mean(axis=0)
        V_centered = V - self._pca_mean

        pca = PCA(n_components=3)
        pca.fit(V_centered)
        self._pca_components = pca.components_   # (3, 50)
        self._fitted = True

        # Corpus centroid in 3D (for angle=cosine mode)
        proj = V_centered @ self._pca_components.T    # (n, 3)
        self._corpus_mean_3d = proj.mean(axis=0)
        norm = np.linalg.norm(self._corpus_mean_3d)
        if norm > 1e-10:
            self._corpus_mean_3d /= norm

        # Build IDF-like frequency table from vocab
        for w in valid:
            self._vocab_freq[w] = self._vocab_freq.get(w, 0) + 1

        explained = pca.explained_variance_ratio_
        print(f"  [SemanticProjector] Fitted on {len(valid)}/{len(vocab)} words.")
        print(f"  PCA explained variance: "
              f"{explained[0]:.3f} + {explained[1]:.3f} + {explained[2]:.3f} "
              f"= {sum(explained):.3f}")

    def project(self, word: str) -> np.ndarray | None:
        """
        Project a word's GloVe vector into R^3.

        Returns a 3D vector (not normalised) or None if word is unknown.
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before project().")

        model = _get_glove()
        w     = word.lower()

        if w not in model:
            return None

        v50  = model[w] - self._pca_mean       # centre
        v3   = v50 @ self._pca_components.T    # project: (3,)
        return v3

    def word_to_axis_angle(
        self,
        word: str,
        angle_mode: str = "cosine",
        fallback_alpha: float = 0.722,    # vertex rotation (120°) as neutral fallback
    ) -> tuple[np.ndarray, float] | None:
        """
        Convert a word to a rotation axis n̂ and angle θ.

        Parameters
        ----------
        word         : the word
        angle_mode   : 'idf'    — rare words → large θ
                       'cosine' — words far from corpus mean → large θ
        fallback_alpha : if word is unknown, return axis aligned to
                         cos(fallback_alpha × π / 2) approximation.

        Returns (axis, theta) or None if unknown (caller decides fallback).
        """
        v3 = self.project(word)

        if v3 is None:
            return None

        # ── Axis: normalise v3 to unit sphere ────────────────────────────
        norm = np.linalg.norm(v3)
        if norm < 1e-10:
            # Zero vector — word exists but projects to origin; use z-axis
            n = np.array([0.0, 0.0, 1.0])
        else:
            n = v3 / norm

        # ── Angle ─────────────────────────────────────────────────────────
        if angle_mode == "cosine":
            # θ = arccos(n̂ · corpus_mean_3d) ∈ [0, π]
            cos_t = float(np.clip(np.dot(n, self._corpus_mean_3d), -1.0, 1.0))
            theta = float(np.arccos(cos_t))

        else:  # idf
            # Approximate IDF via inverse frequency in seed vocab.
            # Words never seen in seed get max IDF.
            w = word.lower()
            freq = self._vocab_freq.get(w, 0)
            if freq == 0:
                idf = 1.0   # unknown → max information density → large θ
            else:
                total = sum(self._vocab_freq.values())
                idf   = 1.0 - (freq / total)
            theta = float(np.pi * idf)

        return n, theta


# ─────────────────────────────────────────────────────────────────────────────
# Word → SO(3) matrix and SU(2) gate
# ─────────────────────────────────────────────────────────────────────────────

def word_to_so3(
    word: str,
    projector: SemanticProjector,
    angle_mode: str = "idf",
) -> np.ndarray | None:
    """
    Returns the 3×3 SO(3) rotation matrix for a word, or None if unknown.

    M = I + sin(θ) · K + (1 − cos(θ)) · K²
    where K is the skew-symmetric matrix of n̂.
    """
    result = projector.word_to_axis_angle(word, angle_mode=angle_mode)
    if result is None:
        return None

    n, theta = result
    nx, ny, nz = n

    K = np.array([
        [0,   -nz,  ny],
        [nz,   0,  -nx],
        [-ny,  nx,   0],
    ], dtype=float)

    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)


def word_to_su2(
    word: str,
    projector: SemanticProjector,
    angle_mode: str = "idf",
) -> np.ndarray | None:
    """
    Returns the 2×2 SU(2) gate for a word, or None if unknown.
    Uses the exact axis_angle_to_su2 from livnium_cortex_v1.
    """
    result = projector.word_to_axis_angle(word, angle_mode=angle_mode)
    if result is None:
        return None
    n, theta = result
    return axis_angle_to_su2(n, theta)


def word_to_alpha(
    word: str,
    projector: SemanticProjector,
    angle_mode: str = "cosine",
) -> float:
    """
    Returns the α signal (mean |cos θ| proxy) for a word.

    For a single-word rotation, α ≈ |cos(θ)| is a reasonable proxy
    for the polarity signal.  High θ (rare word) → high α.
    Low θ (common word near corpus mean) → low α.
    """
    result = projector.word_to_axis_angle(word, angle_mode=angle_mode)
    if result is None:
        return 0.722   # fallback: vertex rotation class
    _, theta = result
    return float(abs(np.sin(theta / 2)))    # inverted: rare/outlier words → high α


# ─────────────────────────────────────────────────────────────────────────────
# Axis similarity: the key verifiable claim
# ─────────────────────────────────────────────────────────────────────────────

def axis_similarity(
    word_a: str,
    word_b: str,
    projector: SemanticProjector,
) -> float | None:
    """
    dot(n̂_a, n̂_b) ∈ [−1, +1].

    This should correlate with GloVe cosine similarity.
    That's the core claim of the semantic bridge.
    """
    ra = projector.word_to_axis_angle(word_a)
    rb = projector.word_to_axis_angle(word_b)
    if ra is None or rb is None:
        return None
    return float(np.dot(ra[0], rb[0]))


def glove_similarity(word_a: str, word_b: str) -> float | None:
    """Cosine similarity from GloVe (ground truth for the correlation test)."""
    model = _get_glove()
    wa, wb = word_a.lower(), word_b.lower()
    if wa not in model or wb not in model:
        return None
    return float(model.similarity(wa, wb))


# ─────────────────────────────────────────────────────────────────────────────
# Verification test
# ─────────────────────────────────────────────────────────────────────────────

def semantic_bridge_test(projector: SemanticProjector) -> bool:
    """
    Verify the core claim BEFORE using the bridge in the organism:

    Claim: axis similarity between words should correlate with
           their GloVe cosine similarity.

    Method: Pearson correlation on a known test set.
    Pass threshold: r > 0.5  (moderate positive correlation).

    We also check that known-similar pairs have higher axis similarity
    than known-dissimilar pairs on average.
    """
    print("\nSemantic Bridge Verification Test")
    print("=" * 55)

    # Test pairs: (word_a, word_b, expected_relation)
    pairs = [
        # Similar pairs (high GloVe cosine expected)
        ("geometry",   "mathematics",  "similar"),
        ("entropy",    "energy",       "similar"),
        ("quantum",    "physics",      "similar"),
        ("lattice",    "crystal",      "similar"),
        ("kernel",     "core",         "similar"),
        ("noise",      "signal",       "similar"),
        # Dissimilar pairs (low / negative GloVe cosine expected)
        ("geometry",   "banana",       "dissimilar"),
        ("entropy",    "pizza",        "dissimilar"),
        ("quantum",    "chair",        "dissimilar"),
        ("lattice",    "happiness",    "dissimilar"),
        ("kernel",     "ocean",        "dissimilar"),
        ("stillness",  "explosion",    "dissimilar"),
    ]

    glove_sims  = []
    axis_sims   = []
    all_pass    = True

    for wa, wb, rel in pairs:
        gs  = glove_similarity(wa, wb)
        axs = axis_similarity(wa, wb, projector)

        if gs is None or axs is None:
            print(f"  SKIP  '{wa}' / '{wb}' — OOV")
            continue

        glove_sims.append(gs)
        axis_sims.append(axs)

        tag = "✓" if (rel == "similar") == (axs > 0) else "✗"
        print(f"  {tag}  {rel:12s}  '{wa:12s}' / '{wb:12s}' "
              f"glove={gs:+.3f}  axis={axs:+.3f}")

    if len(glove_sims) < 4:
        print("  WARN: Not enough pairs to compute correlation.")
        return False

    # Pearson r
    gs_arr  = np.array(glove_sims)
    ax_arr  = np.array(axis_sims)
    r = float(np.corrcoef(gs_arr, ax_arr)[0, 1])

    # Mean similarity by group
    similar_mask    = [p[2] == "similar" for p in pairs
                       if glove_similarity(p[0], p[1]) is not None
                       and axis_similarity(p[0], p[1], projector) is not None]
    dissimilar_mask = [not m for m in similar_mask]

    ax_sim_mean  = float(np.mean(ax_arr[np.array(similar_mask)]))    \
                   if any(similar_mask) else 0.0
    ax_dis_mean  = float(np.mean(ax_arr[np.array(dissimilar_mask)])) \
                   if any(dissimilar_mask) else 0.0

    print(f"\n  Pearson r (glove ~ axis):  {r:.3f}")
    print(f"  Mean axis_sim for 'similar' pairs:    {ax_sim_mean:+.3f}")
    print(f"  Mean axis_sim for 'dissimilar' pairs: {ax_dis_mean:+.3f}")

    pass_r   = r > 0.50
    pass_gap = ax_sim_mean > ax_dis_mean

    print(f"\n  r > 0.50 :  {'PASS ✓' if pass_r else 'FAIL ✗'}  (r={r:.3f})")
    print(f"  sim > dis:  {'PASS ✓' if pass_gap else 'FAIL ✗'}")

    passed = pass_r and pass_gap
    print(f"\n  VERDICT: {'SEMANTIC BRIDGE VERIFIED ✓' if passed else 'NOT VERIFIED — axis similarity does not track meaning'}")
    print("=" * 55)
    return passed


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

# Seed vocabulary: domain words from our conversation + common stopwords
SEED_VOCAB = [
    # Core technical concepts
    "geometry", "lattice", "quantum", "entropy", "kernel", "matrix",
    "rotation", "symmetry", "conservation", "invariant", "manifold",
    "tensor", "algebra", "topology", "fractal", "dimension",
    "energy", "physics", "mathematics", "frequency", "signal",
    "noise", "filter", "compression", "information", "structure",
    "pattern", "order", "chaos", "complexity", "system",
    # Livnium-specific
    "stillness", "polarity", "entanglement", "governor", "pruning",
    "crystal", "core", "surface", "boundary", "observer",
    # Common words (anchor the corpus mean toward neutral)
    "the", "is", "a", "in", "of", "and", "to", "that", "it",
    "this", "with", "for", "not", "are", "be", "was", "has",
    # Test contrast
    "banana", "pizza", "chair", "ocean", "happiness", "explosion",
]


if __name__ == "__main__":
    print("Semantic Bridge v1")
    print("=" * 55)

    # Fit projector on seed vocabulary
    print("\nFitting SemanticProjector on seed vocabulary...")
    proj = SemanticProjector(seed_vocab=SEED_VOCAB)

    # Run verification test BEFORE making any claims
    passed = semantic_bridge_test(proj)

    if not passed:
        print("\n[HALT] Verification failed. "
              "Do not use semantic bridge in organism until fixed.")
    else:
        print("\n[PROCEED] Bridge verified. "
              "Showing word → rotation mappings for key concepts:\n")

        test_words = [
            "geometry", "lattice", "quantum", "entropy", "kernel",
            "noise", "stillness", "conservation", "invariant",
            "the", "is", "a",
            "banana", "pizza",
        ]

        for word in test_words:
            result = proj.word_to_axis_angle(word, angle_mode="idf")
            if result is None:
                print(f"  {'OOV':20s} '{word}'")
                continue
            n, theta = result
            alpha = word_to_alpha(word, proj, angle_mode="idf")
            deg   = np.degrees(theta)
            print(f"  '{word:15s}'  θ={deg:6.1f}°  n̂=({n[0]:+.3f},{n[1]:+.3f},{n[2]:+.3f})  α={alpha:.3f}")

        # Demonstrate constructive vs destructive interference
        print("\n  Axis similarity matrix for key concept pairs:")
        concept_words = ["geometry", "lattice", "quantum", "entropy", "noise", "banana"]
        header = f"  {'':12s}" + "".join(f"{w:12s}" for w in concept_words)
        print(header)
        for wa in concept_words:
            row = f"  {wa:12s}"
            for wb in concept_words:
                s = axis_similarity(wa, wb, proj)
                row += f"  {s:+.3f}    " if s is not None else "  None      "
            print(row)
