"""
Organism Seed  v1  —  Honest Edition
=====================================

What this actually is:
    A text-driven lattice+MPS state machine.

    Words → MD5 hash → one of 24 Livnium rotations
    Rotation → α signal (mean |cos θ|) → updates governor ceiling
    Governor applies a qubit gate and enforces entropy budget
    Pruning log = words that triggered a bond truncation

What it is NOT:
    - A semantic memory  (the MPS state is a quantum state, not a word store)
    - An identity model  (MD5 hash has no relationship to word meaning)
    - A bridge between messages  (entanglement is between qubits, not concepts)

What is genuinely interesting:
    - High-α words (map to 180° rotations) relax the entropy ceiling → survive
    - Low-α words (map to 90° rotations) tighten the ceiling → pruned harder
    - The prune/survive split is real and auditable
    - The final MPS state is a unique signature of the sequence of rotations
      (order-dependent, deterministic given the same word sequence)

The MD5→rotation mapping is arbitrary in the same way % 3 was arbitrary.
It produces a valid experiment but no semantic claims can be made from it.
A non-arbitrary mapping would require embedding words in SO(3) geometry
(e.g. word2vec vectors projected onto the unit sphere → axis-angle → SU(2)).
That is a real next step and is noted in grow_next_steps().
"""

import numpy as np
import hashlib
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from livnium_cortex_v1 import (
    LivniumGovernedCircuit,
    LivniumLattice,
    generate_all_24_rotations,
    livnium_polarity_signal,
    von_neumann_entropy,
    bond_entropies,
)


# ─────────────────────────────────────────────────────────────────────────────
# Pre-compute all 24 (label, fn, α) tuples once at module load
# ─────────────────────────────────────────────────────────────────────────────

_ROTATIONS = generate_all_24_rotations()   # [(label, fn), ...]
_ALPHAS    = [
    livnium_polarity_signal(LivniumLattice(), fn)
    for _, fn in _ROTATIONS
]

# α by rotation class:
#   Identity  (1): α = 0.000
#   90°       (6): α ≈ 0.595
#   120°      (8): α ≈ 0.722
#   180°      (9): α ≈ 0.836–0.841
ROTATION_TABLE = [
    {"label": label, "fn": fn, "alpha": alpha}
    for (label, fn), alpha in zip(_ROTATIONS, _ALPHAS)
]


def word_to_rotation_idx(word: str) -> int:
    """
    Hash a word to one of the 24 Livnium rotations.

    This mapping is:
        DETERMINISTIC  — same word always → same rotation
        ARBITRARY      — no geometric/semantic relationship to word meaning
        UNIFORM-ish    — MD5 distributes roughly evenly across 24 buckets

    A semantically grounded mapping would project a word embedding
    onto the unit sphere in R³ and use that as an axis-angle for SO(3).
    That is grow_next_steps() item #1.
    """
    h = hashlib.md5(word.lower().strip().encode()).hexdigest()
    return int(h, 16) % 24


def rotation_class(alpha: float) -> str:
    """Human-readable class for a rotation given its α value."""
    if alpha < 0.01:   return "identity"
    if alpha < 0.65:   return "quarter-turn (90°)"
    if alpha < 0.78:   return "vertex (120°)"
    return "half-turn (180°)"


# ─────────────────────────────────────────────────────────────────────────────
# ChatOrganism
# ─────────────────────────────────────────────────────────────────────────────

class ChatOrganism:
    """
    Text-driven Livnium+MPS state machine.

    Each word in each message:
        1. Maps to a rotation index via MD5 hash
        2. Updates the Livnium lattice (changes geometry)
        3. Extracts α = mean |cos θ| for all moved symbols
        4. Updates the governor's polarity ceiling
        5. Applies the corresponding SU(2) gate to a qubit
           (qubit = word_position % n_qubits)
        6. Governor enforces entropy budget after the gate

    The prune log records every bond truncation: word, qubit, entropy drop.
    The survive log records words that integrated without triggering a prune.

    Parameters
    ----------
    n_qubits     : MPS width (recommend 12–20 for tractable simulation)
    s_max_bits   : entropy ceiling in bits (converted to nats internally)
    max_bond_dim : hard χ cap
    """

    def __init__(
        self,
        n_qubits: int     = 15,
        s_max_bits: float = 1.5,
        max_bond_dim: int = 32,
    ):
        self.n_qubits = n_qubits
        self.lattice  = LivniumLattice()
        self.circ     = LivniumGovernedCircuit(
            n_qubits     = n_qubits,
            s_max        = s_max_bits * np.log(2),
            max_bond_dim = max_bond_dim,
            verbose      = False,
        )

        self.survive_log: list = []   # words that passed without pruning
        self.prune_log:   list = []   # words that triggered a prune
        self.word_count   = 0
        self.message_count = 0

    # ── Core: consume one word ────────────────────────────────────────────────

    def _consume_word(self, word: str, qubit: int, message_idx: int):
        rot_idx  = word_to_rotation_idx(word)
        rot      = ROTATION_TABLE[rot_idx]

        prunes_before = len(self.circ.gov.pruning_log)

        # Update lattice + governor ceiling
        alpha = self.circ.apply_livnium_rotation(self.lattice, rot["fn"])

        # Apply the SU(2) rotation gate to the current qubit
        self.circ.rx(qubit, alpha * np.pi)

        # CNOT to adjacent qubit — this is what creates entanglement.
        # Without this, every qubit evolves independently, bond entropy
        # stays zero, and the governor has nothing to govern.
        next_qubit = (qubit + 1) % self.n_qubits
        self.circ.cnot(qubit, next_qubit)

        prunes_after = len(self.circ.gov.pruning_log)
        pruned = prunes_after > prunes_before

        record = dict(
            word        = word,
            qubit       = qubit,
            rot_label   = rot["label"],
            alpha       = alpha,
            rot_class   = rotation_class(alpha),
            message_idx = message_idx,
            word_idx    = self.word_count,
        )

        if pruned:
            record["trunc_err"] = sum(
                e["trunc_err"] for e in self.circ.gov.pruning_log[prunes_before:]
            )
            self.prune_log.append(record)
        else:
            self.survive_log.append(record)

        self.word_count += 1
        return record, pruned

    # ── Feed a message ────────────────────────────────────────────────────────

    def feed(self, message: str, silent: bool = False) -> dict:
        """
        Consume all words in a message.

        Returns a summary dict with per-word results.
        """
        words = [w for w in message.split() if w.strip()]
        results = []

        if not silent:
            print(f"\n[MSG {self.message_count}] '{message[:60]}'")

        for i, word in enumerate(words):
            qubit = (self.word_count) % self.n_qubits
            record, pruned = self._consume_word(word, qubit, self.message_count)
            results.append((word, pruned, record["alpha"], record["rot_class"]))

            if not silent:
                status = "PRUNED" if pruned else "kept  "
                print(f"  {status}  '{word:20s}'  "
                      f"rot={record['rot_label']:6s}  "
                      f"α={record['alpha']:.3f}  "
                      f"class={record['rot_class']}")

        self.message_count += 1

        summary = dict(
            message    = message,
            n_words    = len(words),
            n_pruned   = sum(1 for _, p, _, _ in results if p),
            n_survived = sum(1 for _, p, _, _ in results if not p),
            mean_alpha = float(np.mean([alpha for _, _, alpha, _ in results])) if results else 0.0,
            results    = results,
        )
        return summary

    # ── Audit ─────────────────────────────────────────────────────────────────

    def audit(self):
        """Print a full audit of the organism's current state."""
        total = self.word_count
        n_p   = len(self.prune_log)
        n_s   = len(self.survive_log)

        print(f"\n{'═'*60}")
        print(f"  ORGANISM AUDIT")
        print(f"  Messages consumed : {self.message_count}")
        print(f"  Total words       : {total}")
        print(f"  Survived          : {n_s}  ({100*n_s/max(total,1):.1f}%)")
        print(f"  Pruned            : {n_p}  ({100*n_p/max(total,1):.1f}%)")
        print(f"  Governor prunes   : {len(self.circ.gov.pruning_log)}")
        print(f"  Max χ used        : {self.circ.sim.max_bond_dim_used}")
        print(f"  MPS memory        : {self.circ.sim.memory_bytes:,} bytes")

        if self.survive_log:
            print(f"\n  Words that survived (sample, sorted by α desc):")
            sorted_s = sorted(self.survive_log, key=lambda r: -r["alpha"])
            for r in sorted_s[:15]:
                print(f"    α={r['alpha']:.3f}  {r['rot_class']:20s}  '{r['word']}'")

        if self.prune_log:
            print(f"\n  Words that triggered pruning (sample):")
            for r in self.prune_log[:15]:
                print(f"    err={r.get('trunc_err', 0):.2e}  "
                      f"α={r['alpha']:.3f}  '{r['word']}'")

        # Entropy profile
        ents = bond_entropies(self.circ.sim)
        if ents:
            print(f"\n  Bond entropy profile ({len(ents)} bonds):")
            print(f"    mean={np.mean(ents):.3f}  "
                  f"max={max(ents):.3f}  "
                  f"min={min(ents):.3f} nats")

        print(f"{'═'*60}")
        print()
        print("  HONEST CAVEATS:")
        print("  ─────────────────────────────────────────────────────")
        print("  • Prune/survive is driven by bond entropy, not meaning.")
        print("  • MD5→rotation is arbitrary (no word↔geometry link).")
        print("  • The MPS state does NOT store word history — it is a")
        print("    quantum state that evolved through the rotation sequence.")
        print("  • 'Survived' words are not semantically important;")
        print("    they mapped to rotations that kept entropy below ceiling.")
        print("  • To make this semantic: embed words in R³ via word2vec,")
        print("    project to unit sphere, use as axis-angle → SO(3).")
        print(f"{'═'*60}")

    def grow_next_steps(self):
        """Print what genuine next steps look like (vs. metaphor)."""
        print("""
  GENUINE NEXT STEPS
  ══════════════════════════════════════════════════════

  1. SEMANTIC SO(3) EMBEDDING  (replaces MD5 % 24)
     Load a word2vec / GloVe model.
     Each word vector v ∈ R³⁰⁰ → project to R³ (e.g. first 3 PCA components).
     Normalise to unit sphere → axis n̂.
     Assign θ = π × (1 − cosine_similarity(v, corpus_mean)).
     Now the rotation is geometrically grounded in word meaning.

  2. CONTEXTUAL ROUTING  (replaces position % n_qubits)
     Track topic clusters via TF-IDF or BM25.
     Route words from the same topic to the same qubit region.
     Entanglement then genuinely reflects topical co-occurrence.

  3. READOUT THAT MEANS SOMETHING
     Instead of measuring qubits (collapses state, loses context),
     compute bond polarities across the MPS after ingestion.
     High-polarity bonds = qubits that formed Area-Law structure.
     Map those back to the word clusters that built them.
     That IS an auditable semantic signature.

  4. SCALE CAREFULLY
     15 qubits, χ=32: fast, characterised.
     50 qubits, χ=64: ~10 seconds per message, tractable.
     260 qubits, χ=64: untested, potentially minutes per word.
     Profile before committing to deep-drill scale.
""")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Organism Seed v1 — honest edition")
    print("=" * 60)

    # Check α distribution of the 24 rotations
    print("\nRotation catalogue (MD5 hash pool):")
    for i, r in enumerate(ROTATION_TABLE):
        print(f"  [{i:2d}] label={r['label']:6s}  α={r['alpha']:.3f}  "
              f"class={rotation_class(r['alpha'])}")

    # Show which rotation class our test words land in
    test_words = ["Lattice", "Quantum", "Bro", "stillness", "conservation",
                  "entropy", "geometry", "kernel", "noise", "the", "is", "a"]
    print(f"\nWord → rotation mapping (MD5 % 24):")
    for w in test_words:
        idx = word_to_rotation_idx(w)
        r   = ROTATION_TABLE[idx]
        print(f"  '{w:15s}' → idx={idx:2d}  label={r['label']:6s}  "
              f"α={r['alpha']:.3f}  {rotation_class(r['alpha'])}")

    # Run the organism on the seed nutrients
    print(f"\n{'─'*60}")
    print("Growing organism on seed nutrients...")

    seed = ChatOrganism(n_qubits=15, s_max_bits=1.2, max_bond_dim=32)

    nutrients = [
        "Geometry is the Kernel",
        "Talk in stillness",
        "The spec is verified",
        "Entropy is the budget of the mind",
        "Lattice Quantum conservation invariant",
        "dont trust anything test first",
    ]

    for msg in nutrients:
        seed.feed(msg)

    seed.audit()
    seed.grow_next_steps()
