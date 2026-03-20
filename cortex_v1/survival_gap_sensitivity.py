"""
LIVNIUM Survival Gap Sensitivity Analysis
==========================================
Sweeps S_max from 4.0 nats down to 0.3 nats on a fixed nutrient stream
of tagged content and function words.

For each S_max value, measures:
  P(content_word_survives) — fraction of content words that pass without pruning
  P(function_word_survives) — fraction of function words that pass without pruning
  Δ_survival = P(content) − P(function)

The "sweet spot" S* is the S_max where Δ_survival peaks.

Usage
-----
  python survival_gap_sensitivity.py

Output
------
  survival_gap_results.json   — full sweep data
  survival_gap_report.txt     — human-readable table + conclusion
"""

import sys
import os
import json
import numpy as np
from pathlib import Path

QE_DIR = Path(__file__).parent
sys.path.insert(0, str(QE_DIR))

from semantic_bridge import SemanticProjector, SEED_VOCAB
from livnium_cortex_v1 import LivniumGovernedCircuit, SemanticPolarityGovernor


# ══════════════════════════════════════════════════════════════════════════
# §1  Fixed Nutrient Stream
# ══════════════════════════════════════════════════════════════════════════

# 60 words: 30 content (high-θ, high-α) + 30 function (low-θ, low-α)
# Interleaved so both groups face identical bond-pressure history.
# This is the same stream for every S_max value.

CONTENT_WORDS = [
    "lattice", "invariant", "manifold", "entropy", "tensor",
    "quantum", "geometry", "kernel", "topology", "eigenvalue",
    "wavefunction", "compression", "polarity", "fractal", "algebra",
    "entanglement", "symmetry", "gradient", "divergence", "rotation",
    "dimension", "frequency", "signal", "stillness", "crystal",
    "boundary", "observer", "pattern", "complexity", "conservation",
]

FUNCTION_WORDS = [
    "the", "is", "a", "in", "of",
    "and", "to", "that", "it", "this",
    "with", "for", "not", "are", "be",
    "was", "has", "an", "by", "at",
    "from", "or", "but", "on", "as",
    "if", "so", "we", "he", "she",
]

# Interleave: content[0], function[0], content[1], function[1], ...
NUTRIENT_STREAM = []
WORD_LABELS: dict[str, str] = {}
for c, f in zip(CONTENT_WORDS, FUNCTION_WORDS):
    NUTRIENT_STREAM.extend([c, f])
    WORD_LABELS[c] = "content"
    WORD_LABELS[f] = "function"


# ══════════════════════════════════════════════════════════════════════════
# §2  Single Run
# ══════════════════════════════════════════════════════════════════════════

def run_organism(proj: SemanticProjector,
                 s_max: float,
                 n_qubits: int = 15,
                 max_bond_dim: int = 64) -> dict:
    """
    Feed NUTRIENT_STREAM into a fresh organism at the given S_max.
    Returns per-word survival records.
    """
    circuit = LivniumGovernedCircuit(
        n_qubits=n_qubits,
        s_max=s_max,
        max_bond_dim=max_bond_dim,
    )

    records = []
    for idx, word in enumerate(NUTRIENT_STREAM):
        qubit   = idx % n_qubits
        next_q  = (qubit + 1) % n_qubits

        # Semantic rotation
        result = proj.word_to_axis_angle(word, angle_mode="cosine")
        if result is not None:
            _, theta = result
            alpha = float(abs(np.sin(theta / 2)))
        else:
            alpha = 0.5

        prunes_before = len(circuit.gov.pruning_log)

        circuit.rx(qubit, alpha * np.pi)
        circuit.cnot(qubit, next_q)

        prunes_after = len(circuit.gov.pruning_log)
        pruned = prunes_after > prunes_before

        records.append({
            "word":    word,
            "label":   WORD_LABELS.get(word, "unknown"),
            "alpha":   round(alpha, 4),
            "qubit":   qubit,
            "pruned":  pruned,
        })

    return {"s_max": s_max, "records": records}


# ══════════════════════════════════════════════════════════════════════════
# §3  Survival Gap Calculation
# ══════════════════════════════════════════════════════════════════════════

def compute_gap(run: dict) -> dict:
    records = run["records"]
    content_total   = sum(1 for r in records if r["label"] == "content")
    function_total  = sum(1 for r in records if r["label"] == "function")
    content_pruned  = sum(1 for r in records if r["label"] == "content"  and r["pruned"])
    function_pruned = sum(1 for r in records if r["label"] == "function" and r["pruned"])

    p_content_survives  = 1.0 - (content_pruned  / content_total)  if content_total  else 0.0
    p_function_survives = 1.0 - (function_pruned / function_total) if function_total else 0.0
    delta               = p_content_survives - p_function_survives

    total_pruned = content_pruned + function_pruned
    total_words  = content_total + function_total

    return {
        "s_max":               round(run["s_max"], 4),
        "p_content_survives":  round(p_content_survives,  4),
        "p_function_survives": round(p_function_survives, 4),
        "delta_survival":      round(delta, 4),
        "content_pruned":      content_pruned,
        "function_pruned":     function_pruned,
        "total_pruned":        total_pruned,
        "total_words":         total_words,
        "prune_rate":          round(total_pruned / total_words, 4),
    }


# ══════════════════════════════════════════════════════════════════════════
# §4  Sweep
# ══════════════════════════════════════════════════════════════════════════

def sweep(s_max_values: list[float], proj: SemanticProjector) -> list[dict]:
    results = []
    for i, s_max in enumerate(s_max_values):
        print(f"  [{i+1}/{len(s_max_values)}] S_max = {s_max:.2f} nats …", flush=True)
        run = run_organism(proj, s_max)
        gap = compute_gap(run)
        results.append(gap)
        print(f"         Δ_survival = {gap['delta_survival']:+.4f}  "
              f"(content={gap['p_content_survives']:.3f}, "
              f"function={gap['p_function_survives']:.3f}, "
              f"prune_rate={gap['prune_rate']:.3f})")
    return results


# ══════════════════════════════════════════════════════════════════════════
# §5  Report
# ══════════════════════════════════════════════════════════════════════════

def generate_report(results: list[dict]) -> str:
    lines = []
    lines.append("=" * 72)
    lines.append("  LIVNIUM Survival Gap Sensitivity Analysis")
    lines.append(f"  Nutrient stream: {len(NUTRIENT_STREAM)} words "
                 f"({len(CONTENT_WORDS)} content / {len(FUNCTION_WORDS)} function)")
    lines.append("=" * 72)
    lines.append("")
    lines.append(f"{'S_max':>8}  {'P(content)':>11}  {'P(function)':>12}  "
                 f"{'Δ_survival':>11}  {'prune_rate':>11}  regime")
    lines.append("-" * 72)

    best = max(results, key=lambda r: r["delta_survival"])

    for r in results:
        marker = " ← S*" if r["s_max"] == best["s_max"] else ""
        if r["prune_rate"] < 0.05:
            regime = "lazy"
        elif r["prune_rate"] > 0.80:
            regime = "starvation"
        else:
            regime = "active"
        lines.append(
            f"{r['s_max']:>8.2f}  {r['p_content_survives']:>11.4f}  "
            f"{r['p_function_survives']:>12.4f}  {r['delta_survival']:>+11.4f}  "
            f"{r['prune_rate']:>11.4f}  {regime}{marker}"
        )

    lines.append("")
    lines.append(f"Peak Δ_survival = {best['delta_survival']:+.4f} "
                 f"at S_max = {best['s_max']:.2f} nats")
    lines.append("")

    # Verdict
    lines.append("VERDICT:")
    if best["delta_survival"] > 0.10:
        lines.append(f"  GOVERNOR IS A FUNCTIONAL FILTER at S* = {best['s_max']:.2f} nats.")
        lines.append(f"  Content words survive {best['delta_survival']*100:.1f}pp more often than")
        lines.append(f"  function words under equal entanglement pressure.")
        lines.append(f"  The α signal is load-bearing, not decorative.")
    elif best["delta_survival"] > 0.0:
        lines.append(f"  WEAK DISCRIMINATION: peak Δ = {best['delta_survival']:+.4f}.")
        lines.append(f"  The governor shows directional preference but effect size")
        lines.append(f"  is too small to claim meaningful semantic filtering.")
        lines.append(f"  Lower S_max further or increase nutrient stream length.")
    else:
        lines.append(f"  NO DISCRIMINATION: Δ_survival ≤ 0 across entire sweep.")
        lines.append(f"  The α signal is not strong enough to overcome entropy growth.")
        lines.append(f"  The governor is a statistical observer, not an active filter.")

    lines.append("")
    lines.append("HONEST CAVEATS:")
    lines.append("  · 'Pruned' = triggered at least one bond truncation event.")
    lines.append("    It does not mean the word is erased — bond may partially survive.")
    lines.append("  · Interleaved stream means function words follow content words.")
    lines.append("    Position effects are not fully controlled.")
    lines.append("  · This test uses a fixed 60-word stream, not a natural corpus.")
    lines.append("    Results at S* should be re-validated on Reuters/Inspec.")
    lines.append("=" * 72)

    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════
# §6  Main
# ══════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("LIVNIUM Survival Gap Sensitivity Sweep")
    print(f"Stream: {len(NUTRIENT_STREAM)} words  "
          f"({len(CONTENT_WORDS)} content / {len(FUNCTION_WORDS)} function)")
    print("=" * 60)

    # S_max sweep: 4.0 nats down to 0.3 nats in 15 steps
    # log(2) ≈ 0.693 nats = 1 ebit (natural reference point)
    s_max_values = [round(v, 2) for v in np.linspace(4.0, 0.3, 15)]

    print("\nFitting SemanticProjector …", flush=True)
    proj = SemanticProjector(seed_vocab=SEED_VOCAB)
    print("Ready.\n")

    print("Running sweep …")
    results = sweep(s_max_values, proj)

    # Save JSON
    out_dir = QE_DIR
    json_path = out_dir / "survival_gap_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults → {json_path}")

    # Save + print report
    report = generate_report(results)
    report_path = out_dir / "survival_gap_report.txt"
    with open(report_path, "w") as f:
        f.write(report)
    print("\n" + report)
    print(f"\nReport → {report_path}")

    # Copy to deliverables folder
    import shutil
    for src in [json_path, report_path]:
        dst = Path("/sessions/clever-gracious-fermat/mnt/livnium") / src.name
        shutil.copy(src, dst)


if __name__ == "__main__":
    main()
