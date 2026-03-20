"""
Output Decoder  v1
==================

Reads the final MPS state of any organism and produces a
"Structural Persistence Report" — which words contributed to the
most stable (Area-Law) bonds.

What "structurally persistent" means
-------------------------------------
High bond polarity at bond i = the entanglement between qubit i and i+1
is far below the theoretical bipartition maximum.  That means the words
routed through those qubits built STRUCTURED entanglement (Area Law),
not maximal noise (Volume Law).

What it does NOT mean
----------------------
"The system found these words meaningful."
The structure is a property of how those particular SU(2) gates interacted
with the entropy ceiling — not of semantic understanding.
Words that happened to produce similar axes and land on uncrowded qubits
will score high.  Words that arrived after zone overflow will score low
regardless of their semantic importance.

Use this report as a diagnostic of the routing policy, not as a readout
of the conversation's meaning.
"""

from __future__ import annotations

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from livnium_cortex_v1 import bond_entropies, bond_polarities


def decode_organism(
    organism,
    label: str = "Organism",
    top_n: int = 10,
) -> dict:
    """
    Produce a structural persistence report for any organism that has:
        organism.circ.sim     : MPSSimulator
        organism.survive_log  : list of dicts with 'qubit', 'word'
        organism.prune_log    : list of dicts with 'qubit', 'word'

    Returns a dict with the full report.
    """
    sim  = organism.circ.sim
    pols = bond_polarities(sim)
    ents = bond_entropies(sim)

    # Build qubit → [words] map from both logs
    qubit_words: dict[int, list[tuple[str, str]]] = {}  # qubit → [(word, status)]
    for r in organism.survive_log:
        q = r["qubit"]
        qubit_words.setdefault(q, []).append((r["word"], "kept"))
    for r in organism.prune_log:
        q = r["qubit"]
        qubit_words.setdefault(q, []).append((r["word"], "pruned"))

    # For each bond i (between qubit i and i+1), collect words from both sides
    bond_reports = []
    for i, (pol, ent) in enumerate(zip(pols, ents)):
        words_left  = [w for w, _ in qubit_words.get(i, [])]
        words_right = [w for w, _ in qubit_words.get(i + 1, [])]
        bond_reports.append({
            "bond"       : i,
            "polarity"   : pol,
            "entropy"    : ent,
            "words_left" : words_left,
            "words_right": words_right,
        })

    # Rank bonds by polarity — highest = most structurally stable
    ranked = sorted(bond_reports, key=lambda r: -r["polarity"])

    # Collect words that appear in top-N high-polarity bonds
    # (appears on either side of a high-polarity bond = "resonant")
    resonant_words: dict[str, list[float]] = {}
    for b in ranked[:top_n]:
        for w in b["words_left"] + b["words_right"]:
            if w:
                resonant_words.setdefault(w, []).append(b["polarity"])

    # Score = mean polarity of bonds the word participated in
    word_scores = {
        w: float(np.mean(scores))
        for w, scores in resonant_words.items()
    }
    top_words = sorted(word_scores.items(), key=lambda x: -x[1])

    # Identify collision qubits (more than one word → same qubit)
    collision_qubits = {
        q: [w for w, _ in ws]
        for q, ws in qubit_words.items()
        if len(ws) > 1
    }

    report = dict(
        label             = label,
        n_bonds           = len(pols),
        mean_polarity     = float(np.mean(pols)),
        max_polarity      = float(max(pols)),
        min_polarity      = float(min(pols)),
        mean_entropy      = float(np.mean(ents)),
        top_words         = top_words[:10],
        collision_qubits  = collision_qubits,
        bond_reports      = ranked,
    )
    return report


def print_report(report: dict):
    print(f"\n{'═'*60}")
    print(f"  STRUCTURAL PERSISTENCE REPORT: {report['label']}")
    print(f"{'═'*60}")
    print(f"  Bonds:         {report['n_bonds']}")
    print(f"  Mean polarity: {report['mean_polarity']:.3f}")
    print(f"  Max polarity:  {report['max_polarity']:.3f}")
    print(f"  Min polarity:  {report['min_polarity']:.3f}")
    print(f"  Mean entropy:  {report['mean_entropy']:.3f} nats")

    print(f"\n  Top words by bond polarity (structurally persistent):")
    print(f"  {'word':20s}  mean_polarity  interpretation")
    for w, score in report["top_words"]:
        interp = "Area-Law bond neighbor" if score > 0.7 \
                 else "moderate structure" if score > 0.4 \
                 else "low structure"
        print(f"  {w:20s}  {score:.3f}          {interp}")

    print(f"\n  Collision qubits (zone overflow — lower fidelity):")
    if report["collision_qubits"]:
        for q, words in sorted(report["collision_qubits"].items()):
            print(f"    q{q:2d}: {words}")
    else:
        print(f"    None — no qubit reuse")

    print(f"\n  Bond polarity profile:")
    for b in sorted(report["bond_reports"], key=lambda x: x["bond"]):
        bar = "█" * int(b["polarity"] * 15)
        left  = ",".join(b["words_left"][:2])  or "-"
        right = ",".join(b["words_right"][:2]) or "-"
        print(f"    [{b['bond']:2d}] {bar:<15s} {b['polarity']:.3f}  "
              f"{left[:12]:12s} ↔ {right[:12]:12s}")

    print(f"\n  CAVEAT: 'structurally persistent' ≠ 'semantically important'.")
    print(f"  High polarity = these words built Area-Law bonds.")
    print(f"  Low polarity  = these bonds hit the entropy ceiling.")
    print(f"  Zone overflow qubits ({len(report['collision_qubits'])} total) have")
    print(f"  lower fidelity because two words compete for the same state.")
    print(f"{'═'*60}")


def compare_decoders(organisms: list[tuple[str, object]]):
    """
    Run the decoder on multiple organisms and print a comparison table.
    """
    reports = [(label, decode_organism(org, label=label))
               for label, org in organisms]

    print(f"\n{'═'*65}")
    print(f"  DECODER COMPARISON")
    print(f"  {'Config':18s}  {'mean_pol':10s}  {'max_pol':9s}  "
          f"{'mean_S':8s}  {'collisions':12s}")
    print(f"  {'─'*60}")
    for label, r in reports:
        print(f"  {label:18s}  {r['mean_polarity']:8.3f}    "
              f"{r['max_polarity']:7.3f}    "
              f"{r['mean_entropy']:6.3f}    "
              f"{len(r['collision_qubits'])} qubits")

    print(f"\n  Top-5 structurally persistent words by config:")
    for label, r in reports:
        words = [w for w, _ in r["top_words"][:5]]
        print(f"  {label:18s}  {words}")
    print(f"{'═'*65}")

    # Cross-config consensus: words that appear in top-5 of all configs
    all_top = [set(w for w, _ in r["top_words"][:5]) for _, r in reports]
    consensus = set.intersection(*all_top) if all_top else set()
    print(f"\n  Words in top-5 across ALL configs: {sorted(consensus)}")
    print(f"  (These are robust regardless of routing policy.)")
    print(f"{'═'*65}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from semantic_bridge import SemanticProjector, SEED_VOCAB, semantic_bridge_test
    from organism_seed import ChatOrganism
    from semantic_organism_test import SemanticOrganism
    from contextual_router import TopicalOrganism

    print("Output Decoder v1")
    print("=" * 60)

    proj = SemanticProjector(seed_vocab=SEED_VOCAB)
    if not semantic_bridge_test(proj):
        raise SystemExit("Bridge verification failed.")

    nutrients = [
        "Geometry is the Kernel",
        "Talk in stillness",
        "The spec is verified",
        "Entropy is the budget of the mind",
        "Lattice Quantum conservation invariant",
        "dont trust anything test first",
    ]

    org_a = ChatOrganism(n_qubits=15, s_max_bits=1.2, max_bond_dim=32)
    org_b = SemanticOrganism(projector=proj, n_qubits=15,
                             s_max_bits=1.2, max_bond_dim=32)
    org_c = TopicalOrganism(projector=proj, n_tech=7, n_func=8,
                            s_max_bits=1.2, max_bond_dim=32)

    for msg in nutrients:
        org_a.feed(msg, silent=True)
        org_b.feed(msg, silent=True)
        org_c.feed(msg, silent=True)

    # Individual reports
    for label, org in [("A: MD5-seq", org_a), ("B: sem-seq", org_b),
                       ("C: sem-top", org_c)]:
        print_report(decode_organism(org, label=label))

    # Comparison table
    compare_decoders([
        ("A: MD5-seq",  org_a),
        ("B: sem-seq",  org_b),
        ("C: sem-top",  org_c),
    ])
