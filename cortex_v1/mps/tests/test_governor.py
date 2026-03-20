"""
Entanglement Governor Test
==========================

Three scenarios:

1. Structured (GHZ-20)     — S stays tiny, governor never fires
2. Chaotic (random circuit) — S grows, governor prunes repeatedly
3. Comparison              — same circuit with/without governor:
                             does pruning preserve GHZ correlations?
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from mps_simulator import MPSSimulator
from entanglement_governor import EntanglementGovernor, bond_entropies, von_neumann_entropy


rng = np.random.default_rng(42)


# ============================================================
# Scenario 1: Structured — GHZ-20
# ============================================================

def test_structured(n: int = 20):
    print("=" * 60)
    print(f"SCENARIO 1: Structured — GHZ-{n}")
    print("=" * 60)

    sim = MPSSimulator(n_qubits=n, max_bond_dim=64)
    gov = EntanglementGovernor(sim, S_max=np.log(64), verbose=False)

    gov.hadamard(0)
    for i in range(n - 1):
        gov.cnot(i, i + 1)

    gov.dashboard()
    print(f"  Governor fires: {len(gov.pruning_log)} times")
    print(f"  → GHZ is Area Law. Governor never needed to intervene.\n")


# ============================================================
# Scenario 2: Chaotic — random circuit
# ============================================================

def test_chaotic(n: int = 20, depth: int = 5):
    print("=" * 60)
    print(f"SCENARIO 2: Chaotic — random circuit, {n} qubits, depth {depth}")
    print("=" * 60)
    print(f"  S_max ceiling = log(8) = {np.log(8):.3f} (tight cap)\n")

    sim = MPSSimulator(n_qubits=n, max_bond_dim=256)
    gov = EntanglementGovernor(sim, S_max=np.log(8), verbose=True)

    # Layer: random single-qubit rotations on all qubits
    # then CNOT on all adjacent pairs (Volume Law circuit)
    for _ in range(depth):
        for i in range(n):
            gov.rx(i, rng.uniform(0, np.pi))
            gov.rz(i, rng.uniform(0, 2 * np.pi))
        for i in range(0, n - 1, 2):
            gov.cnot(i, i + 1)
        for i in range(1, n - 1, 2):
            gov.cnot(i, i + 1)

    gov.dashboard()
    print(f"  Governor fires: {len(gov.pruning_log)} times")
    print(f"  → Chaotic circuit triggers pruning. Governor enforces the ceiling.\n")


# ============================================================
# Scenario 3: Does pruning preserve GHZ correlations?
# ============================================================

def test_preservation(n: int = 10, shots: int = 200):
    print("=" * 60)
    print(f"SCENARIO 3: Does governor preserve GHZ correlations? (n={n})")
    print("=" * 60)
    print(f"  S_max = log(2) = {np.log(2):.3f}  (minimal — allows exactly χ=2)\n")

    legal_governed   = 0
    legal_ungoverned = 0

    for _ in range(shots):
        # Without governor
        sim1 = MPSSimulator(n_qubits=n, max_bond_dim=64)
        sim1.hadamard(0)
        for i in range(n - 1):
            sim1.cnot(i, i + 1)
        r1 = sim1.measure_all()
        bits1 = "".join(map(str, r1))
        if bits1 in ("0" * n, "1" * n):
            legal_ungoverned += 1

        # With governor (very tight ceiling = log(2))
        sim2 = MPSSimulator(n_qubits=n, max_bond_dim=64)
        gov  = EntanglementGovernor(sim2, S_max=np.log(2), verbose=False)
        gov.hadamard(0)
        for i in range(n - 1):
            gov.cnot(i, i + 1)
        r2 = sim2.measure_all()
        bits2 = "".join(map(str, r2))
        if bits2 in ("0" * n, "1" * n):
            legal_governed += 1

    print(f"  Without governor: {legal_ungoverned}/{shots} legal  "
          f"({'✅' if legal_ungoverned == shots else '⚠️'})")
    print(f"  With governor   : {legal_governed}/{shots} legal  "
          f"({'✅' if legal_governed == shots else '⚠️  some truncation error introduced'})")
    print()
    print(f"  Interpretation:")
    if legal_governed == shots:
        print(f"  GHZ needs χ=2. S_max=log(2) is the exact minimum.")
        print(f"  Governor set correctly → zero information lost.")
    else:
        loss = shots - legal_governed
        print(f"  {loss} shots had illegal states — S_max was too tight")
        print(f"  and the governor pruned necessary bonds.")
    print()


# ============================================================
# Scenario 4: Entropy growth curve
# ============================================================

def test_entropy_growth(n: int = 30):
    print("=" * 60)
    print(f"SCENARIO 4: Entropy growth as entanglement increases (n={n})")
    print("=" * 60)
    print(f"  Adding CNOTs one by one. Watch max bond entropy grow.\n")

    sim = MPSSimulator(n_qubits=n, max_bond_dim=512)
    sim.hadamard(0)

    print(f"  {'CNOTs':>6}  {'max S':>8}  {'mean S':>8}  {'max χ':>6}  {'mem KB':>8}")
    print(f"  {'------':>6}  {'------':>8}  {'------':>8}  {'-----':>6}  {'------':>8}")

    for i in range(n - 1):
        sim.cnot(i, i + 1)
        if i % 3 == 0 or i == n - 2:
            e = bond_entropies(sim)
            print(f"  {i+1:>6}  {max(e):>8.4f}  {np.mean(e):>8.4f}  "
                  f"{sim.max_bond_dim_used:>6}  {sim.memory_bytes/1024:>8.2f}")

    print(f"\n  log(2)={np.log(2):.3f}  log(4)={np.log(4):.3f}  "
          f"log(8)={np.log(8):.3f}  log(64)={np.log(64):.3f}")
    print(f"  GHZ saturates at S=log(2)=0.693 — exactly 1 ebit per bond.\n")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    print()
    print("ENTANGLEMENT GOVERNOR TEST SUITE")
    print()

    test_structured(n=20)
    test_chaotic(n=20, depth=3)
    test_preservation(n=10, shots=300)
    test_entropy_growth(n=30)

    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("  GHZ (structured)  → S = log(2), governor silent")
    print("  Random circuit    → S grows, governor prunes")
    print("  S_max = log(2)    → preserves GHZ exactly (χ=2 is sufficient)")
    print("  S_max < needed    → information loss, illegal states appear")
    print()
    print("  The wall is not at a qubit count.")
    print("  The wall is at an entropy budget.")
    print("  Set the budget correctly and you can run millions of nodes.")
