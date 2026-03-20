"""
Contextual Router  v1
=====================

Routes words to qubit zones based on semantic weight (α).

  Technical words (α ≥ threshold) → qubits 0 … N_TECH-1
  Function words  (α <  threshold) → qubits N_TECH … N_TECH+N_FUNC-1

CNOT links stay within zone: tech-to-tech, func-to-func.
This means semantically similar words entangle with each other,
not with grammatically adjacent but semantically unrelated words.

Hypothesis
----------
  Tech zone: similar GloVe axes → SU(2) gates nearly commute
             → lower entropy growth → governor preserves bonds more

  Func zone: common words have diverse GloVe axes (they appear in all
             contexts, so their PCA projections scatter widely)
             → higher entropy growth → governor prunes harder

  Net effect: technical words survive at a higher rate than with
              sequential routing; function words pruned harder.

This is the testable claim. The comparison runs three configs:

  A: Sequential routing, MD5 hash gates          (baseline)
  B: Sequential routing, semantic gates          (v1.1 calibrated)
  C: Topical routing,    semantic gates          (this module)
"""

from __future__ import annotations

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from livnium_cortex_v1 import (
    LivniumGovernedCircuit, LivniumLattice,
    bond_entropies,
)
from semantic_bridge import (
    SemanticProjector, SEED_VOCAB,
    word_to_su2, word_to_alpha,
    semantic_bridge_test,
)


# ─────────────────────────────────────────────────────────────────────────────
# Router
# ─────────────────────────────────────────────────────────────────────────────

class TopicalRouter:
    """
    Assigns each word to a qubit in the appropriate zone and returns
    the CNOT target (next qubit in the same zone).

    Parameters
    ----------
    n_tech      : number of qubits in the technical zone  (default 7)
    n_func      : number of qubits in the functional zone (default 8)
    alpha_thresh: α threshold separating technical from functional words
    """

    def __init__(
        self,
        n_tech: int   = 7,
        n_func: int   = 8,
        alpha_thresh: float = 0.5,
    ):
        self.n_tech       = n_tech
        self.n_func       = n_func
        self.alpha_thresh = alpha_thresh
        self._tech_ptr    = 0
        self._func_ptr    = 0

    @property
    def n_qubits(self) -> int:
        return self.n_tech + self.n_func

    def route(self, alpha: float) -> tuple[int, int]:
        """
        Given a word's α, return (qubit, cnot_target).
        Both qubit and cnot_target are in the same zone.
        """
        if alpha >= self.alpha_thresh:
            q   = self._tech_ptr % self.n_tech
            nxt = (self._tech_ptr + 1) % self.n_tech
            self._tech_ptr += 1
        else:
            q   = self.n_tech + (self._func_ptr % self.n_func)
            nxt = self.n_tech + ((self._func_ptr + 1) % self.n_func)
            self._func_ptr += 1
        return q, nxt

    def zone(self, qubit: int) -> str:
        return "TECH" if qubit < self.n_tech else "func"

    def reset(self):
        self._tech_ptr = 0
        self._func_ptr = 0


# ─────────────────────────────────────────────────────────────────────────────
# Topical organism
# ─────────────────────────────────────────────────────────────────────────────

class TopicalOrganism:
    """
    Semantic organism with topical qubit routing.

    Architecture:
        word → GloVe → α  (semantic weight)
             → TopicalRouter → (qubit, cnot_target)
             → SU(2) gate on qubit
             → CNOT(qubit → cnot_target)  [within zone]
             → SemanticPolarityGovernor enforces entropy ceiling
    """

    def __init__(
        self,
        projector: SemanticProjector,
        n_tech: int   = 7,
        n_func: int   = 8,
        s_max_bits: float = 1.2,
        max_bond_dim: int = 32,
    ):
        self.proj   = projector
        self.router = TopicalRouter(n_tech=n_tech, n_func=n_func)
        self.circ   = LivniumGovernedCircuit(
            n_qubits     = self.router.n_qubits,
            s_max        = s_max_bits * np.log(2),
            max_bond_dim = max_bond_dim,
            verbose      = False,
        )

        self.survive_log: list = []
        self.prune_log:   list = []
        self.word_count   = 0
        self.message_count = 0
        self.alpha_by_word: dict[str, float] = {}

    def _consume_word(self, word: str, message_idx: int):
        alpha = word_to_alpha(word, self.proj)
        gate  = word_to_su2(word, self.proj)

        if gate is None:
            gate  = np.eye(2, dtype=complex)
            alpha = 0.0

        qubit, cnot_tgt = self.router.route(alpha)

        prunes_before = len(self.circ.gov.pruning_log)

        self.circ.gov.alpha = alpha
        self.circ.apply_gate(qubit, gate)
        self.circ.cnot(qubit, cnot_tgt)

        prunes_after = len(self.circ.gov.pruning_log)
        pruned       = prunes_after > prunes_before

        self.alpha_by_word[word] = alpha

        record = dict(
            word=word, qubit=qubit, cnot_tgt=cnot_tgt,
            alpha=alpha, zone=self.router.zone(qubit),
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
            print(f"\n  [TOP MSG {self.message_count}] '{message[:60]}'")
        for word in words:
            record, pruned = self._consume_word(word, self.message_count)
            results.append((word, pruned, record["alpha"], record["zone"]))
            if not silent:
                status = "PRUNED" if pruned else "kept  "
                print(f"    {status}  '{word:20s}'  "
                      f"q={record['qubit']:2d}  "
                      f"α={record['alpha']:.3f}  "
                      f"{record['zone']}")
        self.message_count += 1
        return dict(
            n_words=len(words),
            n_pruned=sum(1 for _, p, _, _ in results if p),
            mean_alpha=float(np.mean([a for _, _, a, _ in results])) if results else 0.0,
        )

    def zone_audit(self) -> dict:
        """Separate survival/prune statistics by zone."""
        tech_survive = [r for r in self.survive_log if r["zone"] == "TECH"]
        tech_prune   = [r for r in self.prune_log   if r["zone"] == "TECH"]
        func_survive = [r for r in self.survive_log if r["zone"] == "func"]
        func_prune   = [r for r in self.prune_log   if r["zone"] == "func"]

        def rate(s, p):
            total = len(s) + len(p)
            return 100 * len(p) / total if total else 0.0

        ents = bond_entropies(self.circ.sim)
        tech_ents = ents[:self.router.n_tech - 1]
        func_ents = ents[self.router.n_tech - 1:]

        return dict(
            tech_prune_rate   = rate(tech_survive, tech_prune),
            func_prune_rate   = rate(func_survive, func_prune),
            tech_survived     = [r["word"] for r in tech_survive],
            tech_pruned       = [r["word"] for r in tech_prune],
            func_survived     = [r["word"] for r in func_survive],
            func_pruned       = [r["word"] for r in func_prune],
            tech_mean_entropy = float(np.mean(tech_ents)) if tech_ents else 0.0,
            func_mean_entropy = float(np.mean(func_ents)) if func_ents else 0.0,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Three-way comparison
# ─────────────────────────────────────────────────────────────────────────────

def three_way_comparison(nutrients: list[str], projector: SemanticProjector):
    """
    Config A: sequential routing + MD5 gates
    Config B: sequential routing + semantic gates
    Config C: topical routing   + semantic gates
    """
    from organism_seed import ChatOrganism
    from semantic_organism_test import SemanticOrganism

    print("\n" + "═"*65)
    print("  THREE-WAY COMPARISON")
    print("  A: sequential + MD5  |  B: sequential + semantic  |  C: topical + semantic")
    print("═"*65)

    # ── A: MD5 sequential ────────────────────────────────────────────────────
    org_a = ChatOrganism(n_qubits=15, s_max_bits=1.2, max_bond_dim=32)
    for msg in nutrients:
        org_a.feed(msg, silent=True)

    # ── B: semantic sequential ───────────────────────────────────────────────
    org_b = SemanticOrganism(projector=projector, n_qubits=15,
                             s_max_bits=1.2, max_bond_dim=32)
    for msg in nutrients:
        org_b.feed(msg, silent=True)

    # ── C: semantic topical ───────────────────────────────────────────────────
    print("\n  Running Config C (topical routing):")
    org_c = TopicalOrganism(projector=projector, n_tech=7, n_func=8,
                            s_max_bits=1.2, max_bond_dim=32)
    for msg in nutrients:
        org_c.feed(msg)

    # ── Summary table ─────────────────────────────────────────────────────────
    def stats(org, label):
        total   = org.word_count
        n_prune = len(org.prune_log)
        ents    = bond_entropies(org.circ.sim)
        return dict(
            label      = label,
            prune_rate = 100 * n_prune / max(total, 1),
            mean_ent   = float(np.mean(ents)) if ents else 0.0,
            max_ent    = float(max(ents)) if ents else 0.0,
            std_ent    = float(np.std(ents)) if ents else 0.0,
            survived   = {r["word"] for r in org.survive_log},
            pruned     = {r["word"] for r in org.prune_log},
        )

    sa = stats(org_a, "A: MD5-seq")
    sb = stats(org_b, "B: sem-seq")
    sc = stats(org_c, "C: sem-top")

    print(f"\n  {'Config':15s}  {'prune%':8s}  {'mean S':8s}  {'max S':8s}  {'std S':8s}")
    print(f"  {'─'*55}")
    for s in [sa, sb, sc]:
        print(f"  {s['label']:15s}  {s['prune_rate']:6.1f}%   "
              f"{s['mean_ent']:6.3f}   {s['max_ent']:6.3f}   {s['std_ent']:6.3f}")

    # ── Zone breakdown for C ──────────────────────────────────────────────────
    za = org_c.zone_audit()
    print(f"\n  Config C zone breakdown:")
    print(f"    TECH zone: prune_rate={za['tech_prune_rate']:.1f}%  "
          f"mean_S={za['tech_mean_entropy']:.3f}")
    print(f"    func zone: prune_rate={za['func_prune_rate']:.1f}%  "
          f"mean_S={za['func_mean_entropy']:.3f}")
    print(f"    TECH survived: {sorted(za['tech_survived'])}")
    print(f"    TECH pruned:   {sorted(za['tech_pruned'])}")
    print(f"    func survived: {sorted(za['func_survived'])}")
    print(f"    func pruned:   {sorted(za['func_pruned'])}")

    # ── Verdict ───────────────────────────────────────────────────────────────
    print(f"\n  VERDICT")
    print(f"  {'─'*55}")

    claim1 = za["tech_prune_rate"] < za["func_prune_rate"]
    claim2 = za["tech_mean_entropy"] <= za["func_mean_entropy"]
    claim3 = sc["prune_rate"] <= sb["prune_rate"]

    print(f"  Tech zone prunes less than func zone:      "
          f"{'✓' if claim1 else '✗'}  "
          f"({za['tech_prune_rate']:.0f}% vs {za['func_prune_rate']:.0f}%)")
    print(f"  Tech zone lower entropy than func zone:    "
          f"{'✓' if claim2 else '✗'}  "
          f"({za['tech_mean_entropy']:.3f} vs {za['func_mean_entropy']:.3f})")
    print(f"  Topical routing ≤ prune rate of sequential: "
          f"{'✓' if claim3 else '✗'}  "
          f"({sc['prune_rate']:.0f}% vs {sb['prune_rate']:.0f}%)")

    all_pass = claim1 and claim2 and claim3
    print(f"\n  {'ALL CLAIMS VERIFIED ✓' if all_pass else 'PARTIAL — see above'}")
    print(f"{'═'*65}")

    return all_pass


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Contextual Router v1 — Three-Way Comparison")
    print("=" * 65)

    proj = SemanticProjector(seed_vocab=SEED_VOCAB)

    # Verify bridge first — don't run organism on unverified bridge
    print("\nVerifying semantic bridge...")
    if not semantic_bridge_test(proj):
        print("[HALT] Bridge verification failed.")
        raise SystemExit(1)

    nutrients = [
        "Geometry is the Kernel",
        "Talk in stillness",
        "The spec is verified",
        "Entropy is the budget of the mind",
        "Lattice Quantum conservation invariant",
        "dont trust anything test first",
    ]

    three_way_comparison(nutrients, proj)
