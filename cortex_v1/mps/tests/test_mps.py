"""
MPS Test Suite
==============

Three tests:

1. Correctness  — GHZ chain of 10 qubits. Only |00...0> and |11...1> allowed.
2. Scale        — GHZ chain of 2500 qubits. Measures memory and bond dims.
3. Crash test   — Incrementally increase entanglement (bond dim) until
                  the MPS can't compress anymore and memory explodes.
                  This is where the wall actually lives for YOUR use case.
"""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from mps_simulator import MPSSimulator


# ============================================================
# Test 1: Correctness on GHZ-10
# ============================================================

def test_ghz_correctness(n: int = 10, shots: int = 500):
    print("=" * 65)
    print(f"TEST 1: GHZ correctness — {n} qubits, {shots} shots")
    print("=" * 65)

    legal   = 0
    illegal = 0

    for _ in range(shots):
        sim = MPSSimulator(n_qubits=n, max_bond_dim=64)

        # GHZ circuit: H on qubit 0, then chain of CNOTs
        sim.hadamard(0)
        for i in range(n - 1):
            sim.cnot(i, i + 1)

        results = sim.measure_all()
        bits = "".join(map(str, results))

        if bits == "0" * n or bits == "1" * n:
            legal += 1
        else:
            illegal += 1

    print(f"  Legal   (all-0 / all-1): {legal}/{shots}")
    print(f"  Illegal (anything else): {illegal}/{shots}")
    print(f"  {'✅ PASS' if illegal == 0 else '❌ FAIL — illegal states found'}")
    return illegal == 0


# ============================================================
# Test 2: Scale — GHZ-2500
# ============================================================

def test_ghz_scale(n: int = 2500):
    print()
    print("=" * 65)
    print(f"TEST 2: Scale — GHZ chain of {n} qubits")
    print("=" * 65)

    sim = MPSSimulator(n_qubits=n, max_bond_dim=2)

    sim.hadamard(0)
    for i in range(n - 1):
        sim.cnot(i, i + 1)

    bond_dims = sim.bond_dims
    mem = sim.memory_bytes

    print(f"  Qubits          : {n}")
    print(f"  Max bond dim    : {sim.max_bond_dim_used}")
    print(f"  Memory used     : {mem:,} bytes  ({mem / 1024:.1f} KB)")
    print(f"  Dense equivalent: ~{2**min(n,40) * 16:.2e} bytes  (impossible)")
    print(f"  Truncation error: {sum(sim.truncation_errors):.2e}")

    # Spot-check: measure first and last qubit — should match
    results = sim.measure_all()
    first, last = results[0], results[-1]
    match = (first == last)
    print(f"  Qubit 0 = {first}, Qubit {n-1} = {last}  {'✅ correlated' if match else '⚠️ mismatch'}")
    return mem


# ============================================================
# Test 3: Crash test — increase χ until memory explodes
# ============================================================

def test_crash(n: int = 50, max_chi_values=None):
    """
    Build a GHZ-like circuit but add random rotations to inject entanglement.
    As entanglement grows, bond dimension χ must grow to represent it.
    Watch memory climb.
    """
    if max_chi_values is None:
        max_chi_values = [2, 4, 8, 16, 32, 64, 128, 256, 512]

    print()
    print("=" * 65)
    print(f"TEST 3: Crash test — {n} qubits, increasing bond dimension χ")
    print("=" * 65)
    print(f"  {'χ_max':>6}  {'mem (KB)':>10}  {'trunc_err':>12}  {'bond_dims sample'}")
    print(f"  {'-'*6}  {'-'*10}  {'-'*12}  {'-'*20}")

    rng = np.random.default_rng(42)

    for chi in max_chi_values:
        sim = MPSSimulator(n_qubits=n, max_bond_dim=chi)

        # GHZ base
        sim.hadamard(0)
        for i in range(n - 1):
            sim.cnot(i, i + 1)

        # Inject entanglement: random Rz + Rx on every qubit
        for i in range(n):
            sim.rz(i, rng.uniform(0, 2 * np.pi))
            sim.rx(i, rng.uniform(0, np.pi))

        # More CNOTs after rotations — forces χ to grow
        for i in range(0, n - 1, 2):
            sim.cnot(i, i + 1)

        mem_kb = sim.memory_bytes / 1024
        trunc  = sum(sim.truncation_errors)
        bd     = sim.bond_dims
        bd_sample = bd[n//4 : n//4 + 4]  # middle section

        mem_warning = "  ⚠️  GROWING" if mem_kb > 500 else ""
        print(f"  {chi:>6}  {mem_kb:>10.1f}  {trunc:>12.4f}  {bd_sample}{mem_warning}")

        if mem_kb > 100_000:   # 100 MB — stop before laptop melts
            print(f"\n  🛑 STOPPING: memory hit {mem_kb:.0f} KB ({mem_kb/1024:.1f} MB)")
            print(f"     At χ={chi} this circuit can no longer be compressed.")
            break

    print()
    print("  Interpretation:")
    print("  - Low χ + low trunc_err  → state IS compressible (GHZ-like, structured)")
    print("  - High χ + high trunc_err → state needs full 2^n representation")
    print("  - The wall moves depending on HOW MUCH entanglement your circuit uses")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    print()
    print("MPS QUANTUM SIMULATOR — TEST SUITE")
    print("Tensor Network simulation: correct quantum mechanics, scalable for")
    print("low-entanglement states, honest about limits for high entanglement.")
    print()

    # 1. Correctness
    passed = test_ghz_correctness(n=10, shots=500)

    # 2. Scale
    mem = test_ghz_scale(n=2500)

    # 3. Crash test
    test_crash(n=50)

    print()
    print("=" * 65)
    print("SUMMARY")
    print("=" * 65)
    print(f"  GHZ correctness (10 qubits)  : {'✅ PASS' if passed else '❌ FAIL'}")
    print(f"  GHZ scale (2500 qubits)      : {mem:,} bytes  ({mem/1024:.1f} KB)")
    print(f"  Crash test                   : see table above")
    print()
    print("  Key insight:")
    print("  GHZ is maximally compressible (χ=2 is exact, memory = O(n)).")
    print("  Random highly-entangled circuits are NOT compressible (χ→2^(n/2)).")
    print("  The wall is not at a fixed qubit count — it's at a fixed entanglement level.")
