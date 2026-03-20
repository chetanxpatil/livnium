"""
SemanticProjector X-Ray  v1.0
==============================

Audits the raw α scores being fed to the governor so we can confirm
whether the sensor is producing semantically correct signal or inverted
logic (filler words outranking content words).

Run from livnium_extension/:
    python sensor_xray.py

What to look for
----------------
CORRECT signal:  "Shor", "1994", "RSA", "qubits" → α ≥ 0.70
                 "and",  "the",  "in",  "a"      → α ≤ 0.35

INVERTED signal: filler words appear at the TOP of the ranking.
                 This indicates PCA micro-universe distortion —
                 the corpus is too small so "and" looks like a
                 geometric outlier while nouns cluster at the centre.

Root cause (if inverted)
------------------------
  SemanticProjector.fit() was called on a tiny seed vocabulary (≈89 words).
  In that micro-universe the corpus centroid (corpus_mean_3d) is pulled toward
  frequent filler words. Under angle_mode='cosine', θ = arccos(n̂ · centroid),
  so words NEAR the centroid (common fillers) get θ ≈ 0 → α = |sin(0/2)| ≈ 0.
  But if the centroid is distorted by the tiny corpus, the mapping inverts.

  Under angle_mode='idf', α ∝ rarity in the *seed vocab*, not in English.
  A seed vocab of 89 words will show every word as "rare" (freq=1/89)
  → everything maps to α ≈ 0.99, killing all discrimination.
"""

import sys
import os

# ── Path setup ────────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# ── Benchmark document (same as benchmark_retrieval_triage.py) ───────────────
DOCUMENT = """
In 1994 Peter Shor discovered a quantum algorithm that can factor large integers exponentially
faster than the best known classical algorithm . This breakthrough known as Shor algorithm
demonstrated that a theoretical quantum computer could break RSA encryption . However building
such a device requires overcoming significant decoherence and maintaining high-fidelity qubits .
"""

GROUND_TRUTH_FACTS = {
    "1994", "peter", "shor", "algorithm", "factor", "integers",
    "rsa", "encryption", "decoherence", "qubits",
}


# ═══════════════════════════════════════════════════════════════════════════════
# §1  LOAD ALPHA BACK-END
# ═══════════════════════════════════════════════════════════════════════════════

def load_alpha_fn(mode: str = "cosine"):
    """
    Returns (alpha_fn, backend_name) where alpha_fn(word) -> float.

    Tries to import the real SemanticProjector from semantic_bridge.py.
    If gensim / sklearn are absent, falls back to the mock table so the
    x-ray can at least show what mock logic would produce.
    """
    try:
        from semantic_bridge import SemanticProjector, word_to_alpha, SEED_VOCAB

        print(f"  [x-ray] SemanticProjector found. Fitting on seed vocab …")
        proj = SemanticProjector(seed_vocab=SEED_VOCAB)
        n_seed = len(SEED_VOCAB)
        print(f"  [x-ray] Fitted on {n_seed} seed words. angle_mode='{mode}'\n")

        def alpha_fn(word: str) -> float:
            return word_to_alpha(word.rstrip(".,;:"), proj, angle_mode=mode)

        return alpha_fn, f"SemanticProjector (angle_mode={mode}, seed={n_seed})"

    except ImportError as e:
        print(f"  [x-ray] Import failed ({e}). Falling back to mock α.\n")
        return _mock_alpha, "MOCK (hand-crafted table)"


def _mock_alpha(word: str) -> float:
    HIGH = {
        "shor", "algorithm", "rsa", "encryption", "qubits", "decoherence",
        "peter", "1994", "factor", "integers", "exponentially", "breakthrough",
        "theoretical", "fidelity", "quantum", "classical",
    }
    LOW = {
        "in", "a", "that", "can", "than", "the", "this", "as", "could",
        "such", "and", "an", "of", "to", "is", "it", "for", "are", "be",
        "was", "has", "however", "known", "demonstrated", "overcoming",
        "significant", "maintaining", "high", "large", "faster", "best",
        "requires", "would", "break",
    }
    w = word.lower().rstrip(".,;:")
    if w in HIGH:  return 0.95
    if w in LOW:   return 0.25
    return 0.50


# ═══════════════════════════════════════════════════════════════════════════════
# §2  X-RAY SCAN
# ═══════════════════════════════════════════════════════════════════════════════

def run_xray(alpha_fn, backend_name: str):
    tokens = sorted(set(w.rstrip(".,;:") for w in DOCUMENT.split() if w.strip()))

    print("─" * 55)
    print(f"  SemanticProjector X-Ray")
    print(f"  Backend  : {backend_name}")
    print(f"  Tokens   : {len(tokens)} unique")
    print("─" * 55)

    results = []
    for t in tokens:
        a = alpha_fn(t)
        is_fact = t.lower() in GROUND_TRUTH_FACTS
        results.append((t, a, is_fact))

    results.sort(key=lambda x: x[1], reverse=True)

    # ── Full table ────────────────────────────────────────────────────────────
    print(f"\n  {'RANK':<5} {'WORD':<18} {'α':>7}   {'FACT?'}")
    print(f"  {'────':<5} {'────':<18} {'─'*7}   {'─────'}")
    for rank, (word, alpha, is_fact) in enumerate(results, 1):
        tag = "★ FACT" if is_fact else ""
        print(f"  {rank:<5} {word:<18} {alpha:>7.4f}   {tag}")

    # ── Top-10 / Bottom-10 summary ────────────────────────────────────────────
    print(f"\n  TOP 10 (what the governor considers MOST valuable):")
    for word, alpha, is_fact in results[:10]:
        flag = " ← ★ FACT" if is_fact else ""
        print(f"    {word:<18} α={alpha:.4f}{flag}")

    print(f"\n  BOTTOM 10 (what the governor will EVICT first):")
    for word, alpha, is_fact in results[-10:]:
        flag = " ← ★ FACT" if is_fact else ""
        print(f"    {word:<18} α={alpha:.4f}{flag}")

    # ── Verdict ───────────────────────────────────────────────────────────────
    facts_in_results = [(w, a) for w, a, f in results if f]
    filler_words     = {"in", "a", "that", "can", "the", "this", "and", "an",
                        "of", "to", "as", "such", "could"}
    fillers_in_top10 = [(w, a) for w, a, _ in results[:10]
                        if w.lower() in filler_words]
    facts_in_bottom  = [(w, a) for w, a, f in results[-10:] if f]

    mean_fact_alpha   = sum(a for _, a, f in results if f) / max(len(facts_in_results), 1)
    mean_filler_alpha = sum(a for w, a, _ in results
                            if w.lower() in filler_words) / max(
                            len([w for w, _, _ in results if w.lower() in filler_words]), 1)

    print(f"\n  DIAGNOSIS")
    print(f"  {'─'*51}")
    print(f"  Mean α for ★ FACTS  : {mean_fact_alpha:.4f}")
    print(f"  Mean α for fillers  : {mean_filler_alpha:.4f}")
    print(f"  Filler words in top-10 : {len(fillers_in_top10)}  {[w for w,_ in fillers_in_top10]}")
    print(f"  FACTS in bottom-10     : {len(facts_in_bottom)}  {[w for w,_ in facts_in_bottom]}")
    print()

    if mean_fact_alpha > mean_filler_alpha + 0.10:
        print(f"  SENSOR OK — facts outrank fillers by "
              f"{mean_fact_alpha - mean_filler_alpha:.4f}")
        print(f"      The α signal is semantically correct.")
    elif mean_fact_alpha > mean_filler_alpha:
        print(f"  WEAK SIGNAL — facts barely outrank fillers "
              f"(Δ={mean_fact_alpha - mean_filler_alpha:.4f})")
        print(f"      Discrimination is insufficient for reliable triage.")
    else:
        print(f"  INVERTED — fillers outrank facts by "
              f"{mean_filler_alpha - mean_fact_alpha:.4f}")
        print(f"      Root cause: PCA micro-universe distortion.")
        print(f"      Fix: expand seed vocab OR switch to IDF-from-corpus scoring.")

    print("─" * 55)
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# §3  COMPARE BOTH ANGLE MODES  (cosine vs idf)
# ═══════════════════════════════════════════════════════════════════════════════

def compare_modes():
    """Run x-ray under both angle_mode='cosine' and angle_mode='idf'."""
    for mode in ("cosine", "idf"):
        print(f"\n{'═'*55}")
        print(f"  ANGLE MODE: {mode.upper()}")
        print(f"{'═'*55}")
        alpha_fn, backend = load_alpha_fn(mode=mode)
        run_xray(alpha_fn, backend)


# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="X-ray the SemanticProjector α output for the benchmark document."
    )
    parser.add_argument(
        "--mode", choices=["cosine", "idf", "both"], default="both",
        help="angle_mode for SemanticProjector. 'both' compares side-by-side. Default: both"
    )
    args = parser.parse_args()

    if args.mode == "both":
        compare_modes()
    else:
        alpha_fn, backend = load_alpha_fn(mode=args.mode)
        run_xray(alpha_fn, backend)
