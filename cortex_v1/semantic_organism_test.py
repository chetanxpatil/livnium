"""
Semantic Organism Test
======================

Runs the same seed nutrients through two organism configurations:

  Config A: MD5 hash  (arbitrary, current baseline)
  Config B: Semantic embedding  (GloVe axes + cosine-distance angles)

Metrics compared:
  - Prune rate (% of words triggering a pruning event)
  - Mean truncation error per pruned word
  - Distribution of α values (do technical words get higher α than function words?)
  - Entropy profile across bonds after ingestion

The key question: does the semantic embedding produce
  (a) lower α for function words ("the", "is") vs. content words?
  (b) different prune rate vs. MD5?
  (c) lower entropy variance (more coherent state)?

We do NOT claim the MPS "understands" the text.
We only claim:
  semantic bridge → axis similarity correlates with word meaning (verified r=0.752)
  → adjacent semantically-similar words produce smaller entropy growth
  → governor pruning pattern reflects semantic coherence, not timing
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from livnium_cortex_v1 import (
    LivniumGovernedCircuit, LivniumLattice,
    generate_all_24_rotations, livnium_polarity_signal,
    bond_entropies, axis_angle_to_su2,
)
from semantic_bridge import (
    SemanticProjector, SEED_VOCAB,
    word_to_su2, word_to_alpha,
    semantic_bridge_test,
)
from organism_seed import (
    ChatOrganism, word_to_rotation_idx, ROTATION_TABLE,
)


# ─────────────────────────────────────────────────────────────────────────────
# Semantic organism: same structure as ChatOrganism, semantic word→gate
# ─────────────────────────────────────────────────────────────────────────────

class SemanticOrganism:
    """
    Same as ChatOrganism but replaces MD5→rotation with
    GloVe vector → SO(3) axis + cosine-distance angle → SU(2) gate.
    """

    def __init__(
        self,
        projector: SemanticProjector,
        n_qubits: int     = 15,
        s_max_bits: float = 1.2,
        max_bond_dim: int = 32,
    ):
        self.proj     = projector
        self.n_qubits = n_qubits
        self.lattice  = LivniumLattice()
        self.circ     = LivniumGovernedCircuit(
            n_qubits     = n_qubits,
            s_max        = s_max_bits * np.log(2),
            max_bond_dim = max_bond_dim,
            verbose      = False,
        )
        self.survive_log: list = []
        self.prune_log:   list = []
        self.word_count   = 0
        self.message_count = 0

        # Track α values by word class for comparison
        self.alpha_by_word: dict[str, float] = {}

    def _consume_word(self, word: str, qubit: int, message_idx: int):
        prunes_before = len(self.circ.gov.pruning_log)

        # Get semantic SU(2) gate
        gate  = word_to_su2(word, self.proj, angle_mode="cosine")
        alpha = word_to_alpha(word, self.proj, angle_mode="cosine")

        if gate is None:
            # OOV fallback: use identity-like gate (near-zero rotation)
            gate  = np.eye(2, dtype=complex)
            alpha = 0.0

        # Update governor α with this word's semantic weight
        self.circ.gov.alpha = alpha

        # Apply semantic gate + CNOT
        self.circ.apply_gate(qubit, gate)
        next_q = (qubit + 1) % self.n_qubits
        self.circ.cnot(qubit, next_q)

        prunes_after = len(self.circ.gov.pruning_log)
        pruned       = prunes_after > prunes_before

        self.alpha_by_word[word] = alpha

        record = dict(
            word=word, qubit=qubit, alpha=alpha,
            message_idx=message_idx, word_idx=self.word_count,
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

    def feed(self, message: str, silent: bool = False) -> dict:
        words   = [w for w in message.split() if w.strip()]
        results = []
        if not silent:
            print(f"\n  [SEM MSG {self.message_count}] '{message[:60]}'")
        for word in words:
            qubit = self.word_count % self.n_qubits
            record, pruned = self._consume_word(word, qubit, self.message_count)
            results.append((word, pruned, record["alpha"]))
            if not silent:
                status = "PRUNED" if pruned else "kept  "
                print(f"    {status}  '{word:20s}'  α={record['alpha']:.3f}")
        self.message_count += 1
        return dict(
            n_words=len(words),
            n_pruned=sum(1 for _, p, _ in results if p),
            mean_alpha=float(np.mean([a for _, _, a in results])) if results else 0.0,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Comparison runner
# ─────────────────────────────────────────────────────────────────────────────

def compare_organisms(nutrients: list[str], projector: SemanticProjector):
    """Run both organisms on the same nutrient list and compare."""

    # ── Config A: MD5 organism ────────────────────────────────────────────────
    print("\n" + "═"*60)
    print("  CONFIG A: MD5 hash → rotation (current baseline)")
    print("═"*60)

    org_md5 = ChatOrganism(n_qubits=15, s_max_bits=1.2, max_bond_dim=32)
    for msg in nutrients:
        org_md5.feed(msg)

    # ── Config B: Semantic organism ───────────────────────────────────────────
    print("\n" + "═"*60)
    print("  CONFIG B: GloVe → SO(3) axis + cosine angle (semantic)")
    print("═"*60)

    org_sem = SemanticOrganism(
        projector=projector, n_qubits=15, s_max_bits=1.2, max_bond_dim=32
    )
    for msg in nutrients:
        org_sem.feed(msg)

    # ── Side-by-side comparison ───────────────────────────────────────────────
    print("\n" + "═"*60)
    print("  COMPARISON")
    print("═"*60)

    # Prune rates
    total_md5  = org_md5.word_count
    total_sem  = org_sem.word_count
    prune_md5  = len(org_md5.prune_log)
    prune_sem  = len(org_sem.prune_log)
    rate_md5   = 100 * prune_md5  / max(total_md5, 1)
    rate_sem   = 100 * prune_sem  / max(total_sem, 1)

    print(f"\n  Prune rate:  MD5={rate_md5:.1f}%   Semantic={rate_sem:.1f}%")

    # Entropy profiles
    ents_md5 = bond_entropies(org_md5.circ.sim)
    ents_sem = bond_entropies(org_sem.circ.sim)
    print(f"  Bond entropy (MD5):      mean={np.mean(ents_md5):.3f}  "
          f"max={max(ents_md5):.3f}  std={np.std(ents_md5):.3f}")
    print(f"  Bond entropy (Semantic): mean={np.mean(ents_sem):.3f}  "
          f"max={max(ents_sem):.3f}  std={np.std(ents_sem):.3f}")

    # α distribution: function words vs. content words
    function_words = {"the", "is", "a", "in", "of", "and", "to", "that", "it"}
    sem_alpha = org_sem.alpha_by_word

    func_alphas    = [v for k, v in sem_alpha.items() if k.lower() in function_words]
    content_alphas = [v for k, v in sem_alpha.items() if k.lower() not in function_words]

    if func_alphas and content_alphas:
        print(f"\n  α by word class (semantic organism):")
        print(f"    Function words ('the','is','a',...): mean α = {np.mean(func_alphas):.3f}")
        print(f"    Content words  (technical terms):    mean α = {np.mean(content_alphas):.3f}")
        better = "✓  Content words get higher α" \
                 if np.mean(content_alphas) > np.mean(func_alphas) \
                 else "✗  No α separation (expected content > function)"
        print(f"    {better}")

    # Which survived in each organism
    survived_md5 = {r["word"] for r in org_md5.survive_log}
    survived_sem = {r["word"] for r in org_sem.survive_log}
    pruned_md5   = {r["word"] for r in org_md5.prune_log}
    pruned_sem   = {r["word"] for r in org_sem.prune_log}

    print(f"\n  Words pruned by MD5 but NOT by semantic: "
          f"{sorted(pruned_md5 - pruned_sem)}")
    print(f"  Words pruned by semantic but NOT by MD5: "
          f"{sorted(pruned_sem - pruned_md5)}")

    # Check the core claim: do function words survive more often in semantic?
    func_pruned_sem = [r["word"] for r in org_sem.prune_log
                       if r["word"].lower() in function_words]
    func_pruned_md5 = [r["word"] for r in org_md5.prune_log
                       if r["word"].lower() in function_words]

    print(f"\n  Function words pruned:")
    print(f"    MD5:      {func_pruned_md5}")
    print(f"    Semantic: {func_pruned_sem}")

    print(f"\n  VERDICT")
    print(f"  {'─'*50}")
    claim1 = len(func_alphas) > 0 and len(content_alphas) > 0 and \
             np.mean(func_alphas) < np.mean(content_alphas)
    claim2 = len(func_pruned_sem) <= len(func_pruned_md5)
    print(f"  Content words get higher α than function words: "
          f"{'✓' if claim1 else '✗'}")
    print(f"  Function words pruned ≤ as often in semantic:  "
          f"{'✓' if claim2 else '✗'}")
    print(f"{'═'*60}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Semantic Organism Test")
    print("=" * 60)

    # Step 1: fit projector and verify bridge
    print("\nStep 1: Fitting SemanticProjector and verifying bridge...")
    proj = SemanticProjector(seed_vocab=SEED_VOCAB)
    passed = semantic_bridge_test(proj)

    if not passed:
        print("\n[HALT] Bridge verification failed. Fix before proceeding.")
        sys.exit(1)

    # Step 2: show α distribution before running organism
    print("\nStep 2: α distribution by word class (cosine mode)")
    print("  (This is what drives governor ceiling, not just rotation class)")
    domain_words = ["geometry", "lattice", "quantum", "entropy", "kernel",
                    "invariant", "conservation", "stillness", "polarity"]
    function_words = ["the", "is", "a", "in", "of", "and", "to"]
    noise_words    = ["banana", "pizza", "chair", "ocean"]

    print(f"  {'domain/technical':20s}", end="")
    for w in domain_words:
        a = word_to_alpha(w, proj)
        print(f"  {w}={a:.3f}", end="")
    print()
    print(f"  {'function/stopwords':20s}", end="")
    for w in function_words:
        a = word_to_alpha(w, proj)
        print(f"  {w}={a:.3f}", end="")
    print()
    print(f"  {'unrelated content':20s}", end="")
    for w in noise_words:
        a = word_to_alpha(w, proj)
        print(f"  {w}={a:.3f}", end="")
    print()

    # Step 3: run comparison
    print("\nStep 3: Running organisms on seed nutrients...")
    nutrients = [
        "Geometry is the Kernel",
        "Talk in stillness",
        "The spec is verified",
        "Entropy is the budget of the mind",
        "Lattice Quantum conservation invariant",
        "dont trust anything test first",
    ]
    compare_organisms(nutrients, proj)
