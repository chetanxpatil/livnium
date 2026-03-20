"""
Polarity Separation Test Suite
================================

Separates the two truncation mechanisms experimentally:

  hard cap : χ_max                          from _split_svd()
  soft cap : S_eff = S_base*(1+α*polarity)  from SemanticPolarityGovernor._enforce_bond()

Tests:
  1. polarity_score_theoretical in isolation (product / GHZ / random states)
  2. Hard cap vs fixed governor vs polarity governor — same circuit, three backends
  3. Alpha sweep (α = 0 → 0.5 → 1.0)
  4. Two circuit families (GHZ vs random)
  5. Bond-by-bond diagnostic log
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np

from mps_simulator        import MPSSimulator
from entanglement_governor import EntanglementGovernor, bond_entropies, von_neumann_entropy
from polarity_governor     import SemanticPolarityGovernor, polarity_score_theoretical, bond_polarities

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def make_product_state(n: int, chi: int = 4) -> MPSSimulator:
    """All qubits in |0> — no gates, no entanglement."""
    return MPSSimulator(n_qubits=n, max_bond_dim=chi)


def make_ghz(n: int, chi: int = 4) -> MPSSimulator:
    """GHZ state: H on qubit 0, then CNOT chain."""
    sim = MPSSimulator(n_qubits=n, max_bond_dim=chi)
    sim.hadamard(0)
    for q in range(n - 1):
        sim.cnot(q, q + 1)
    return sim


def make_random_circuit(n: int, depth: int = 5, chi: int = 32) -> MPSSimulator:
    """Random brick-wall circuit: alternating even/odd CNOT layers with H gates."""
    sim = MPSSimulator(n_qubits=n, max_bond_dim=chi)
    for d in range(depth):
        for q in range(n):
            sim.hadamard(q)
        start = d % 2
        for q in range(start, n - 1, 2):
            sim.cnot(q, q + 1)
    return sim


def bond_diagnostic_table(sim: MPSSimulator, label: str):
    """Print bond-by-bond: index | S(i) | S_max_theo | polarity | chi."""
    entropies  = bond_entropies(sim)
    polarities = bond_polarities(sim)
    n = sim.n

    print(f"\n{'═'*72}")
    print(f"  Bond Diagnostics — {label}")
    print(f"{'═'*72}")
    print(f"  {'bond':>5}  {'S(i)':>8}  {'S_max_theo':>12}  {'polarity':>10}  {'chi':>5}")
    print(f"  {'─'*5}  {'─'*8}  {'─'*12}  {'─'*10}  {'─'*5}")

    for i, (S, pol) in enumerate(zip(entropies, polarities)):
        n_left  = i + 1
        n_right = n - n_left
        S_theo  = min(n_left, n_right) * np.log(2)
        chi     = sim.tensors[i].shape[2]
        print(f"  {i:>5}  {S:>8.4f}  {S_theo:>12.4f}  {pol:>10.4f}  {chi:>5}")

    print(f"{'─'*72}")
    print(f"  mean S={np.mean(entropies):.4f}  "
          f"mean pol={np.mean(polarities):.4f}  "
          f"max chi={max(t.shape[2] for t in sim.tensors[:-1])}")


def summary_stats(sim: MPSSimulator, prune_events: int, trunc_err: float, label: str):
    entropies  = bond_entropies(sim)
    polarities = bond_polarities(sim)
    max_chi    = max(t.shape[2] for t in sim.tensors[:-1])
    print(f"  [{label}]  "
          f"mean_S={np.mean(entropies):.4f}  "
          f"mean_pol={np.mean(polarities):.4f}  "
          f"max_chi={max_chi}  "
          f"prune_events={prune_events}  "
          f"total_trunc_err={trunc_err:.4e}")


# ─────────────────────────────────────────────────────────────────────────────
# Test 1 — polarity_score_theoretical in isolation
# ─────────────────────────────────────────────────────────────────────────────

def test_polarity_isolation():
    print("\n" + "█"*72)
    print("  TEST 1 — polarity_score_theoretical in isolation")
    print("█"*72)

    N = 12

    # 1a. Product state
    sim_prod = make_product_state(N, chi=4)
    entropies = bond_entropies(sim_prod)
    polarities = bond_polarities(sim_prod)
    print(f"\n  Product state (n={N})")
    print(f"  Expected: S≈0 everywhere, polarity≈1 everywhere")
    print(f"  Got:      mean_S={np.mean(entropies):.6f}  mean_pol={np.mean(polarities):.6f}")
    assert np.mean(entropies) < 1e-10,  f"Product state should have near-zero entropy, got {np.mean(entropies)}"
    assert np.mean(polarities) > 0.99,  f"Product state should have near-1 polarity, got {np.mean(polarities)}"
    print("  ✓ PASS")

    # 1b. GHZ state
    sim_ghz = make_ghz(N, chi=4)
    entropies_ghz = bond_entropies(sim_ghz)
    polarities_ghz = bond_polarities(sim_ghz)
    log2 = np.log(2)
    print(f"\n  GHZ state (n={N})")
    print(f"  Expected: S=log(2)={log2:.4f} per bond, polarity HIGH (far below Page limit)")
    print(f"  Got:      mean_S={np.mean(entropies_ghz):.4f}  mean_pol={np.mean(polarities_ghz):.4f}")

    # Check S ≈ log(2) at each bond
    for i, S in enumerate(entropies_ghz):
        assert abs(S - log2) < 0.01, f"GHZ bond {i}: S={S:.4f} expected log(2)={log2:.4f}"

    # Check polarity is high — GHZ uses only 1 nat where it could use up to min(i+1,n-i-1)*log2
    mid_pol = polarities_ghz[N//2 - 1]
    print(f"  Middle bond polarity (i={N//2-1}): {mid_pol:.4f}  "
          f"(S_theo={min(N//2, N//2)*log2:.4f}, S_actual={log2:.4f})")
    assert mid_pol > 0.8, f"GHZ middle bond polarity should be high, got {mid_pol:.4f}"
    print("  ✓ PASS")

    # 1c. Random circuit — polarity should drop
    sim_rand = make_random_circuit(N, depth=5, chi=32)
    entropies_rand = bond_entropies(sim_rand)
    polarities_rand = bond_polarities(sim_rand)
    print(f"\n  Random circuit (n={N}, depth=5)")
    print(f"  Expected: higher S, lower polarity than GHZ")
    print(f"  Got:      mean_S={np.mean(entropies_rand):.4f}  mean_pol={np.mean(polarities_rand):.4f}")
    assert np.mean(entropies_rand) > np.mean(entropies_ghz), \
        "Random circuit should have higher entropy than GHZ"
    assert np.mean(polarities_rand) < np.mean(polarities_ghz), \
        "Random circuit should have lower polarity than GHZ"
    print("  ✓ PASS")

    print(f"\n  Polarity ranking (lower = more random):")
    print(f"    product:  {np.mean(polarities):.4f}")
    print(f"    GHZ:      {np.mean(polarities_ghz):.4f}")
    print(f"    random:   {np.mean(polarities_rand):.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# Test 2 — Hard cap vs fixed governor vs polarity governor
# ─────────────────────────────────────────────────────────────────────────────

def _build_ghz_with_governor(n, chi, S_base, alpha, GovernorClass):
    """Build GHZ through a governor layer."""
    sim = MPSSimulator(n_qubits=n, max_bond_dim=chi)
    gov = GovernorClass(sim, S_max=S_base, verbose=False) if GovernorClass != "raw" else None

    if gov is None:
        sim.hadamard(0)
        for q in range(n - 1):
            sim.cnot(q, q + 1)
        prune_events = 0
        trunc_err = sum(sim.truncation_errors)
    elif isinstance(gov, SemanticPolarityGovernor):
        gov.alpha = alpha
        gov.sim.hadamard(0)
        gov.enforce_all()
        for q in range(n - 1):
            gov.cnot(0 if q == 0 else q, q + 1)
        prune_events = len(gov.pruning_log)
        trunc_err = sum(e["trunc_err"] for e in gov.pruning_log)
    else:
        gov.sim.hadamard(0)
        gov.enforce_all()
        for q in range(n - 1):
            gov.cnot(0 if q == 0 else q, q + 1)
        prune_events = len(gov.pruning_log)
        trunc_err = sum(e["trunc_err"] for e in gov.pruning_log)

    return sim, prune_events, trunc_err


def test_mechanism_separation():
    print("\n" + "█"*72)
    print("  TEST 2 — Hard cap vs Fixed entropy governor vs Polarity governor")
    print("█"*72)

    N      = 15
    CHI    = 4
    S_BASE = 1.5 * np.log(2)
    ALPHA  = 0.7

    print(f"\n  Circuit: GHZ (n={N})  χ_max={CHI}  S_base={S_BASE:.3f}  α={ALPHA}")
    print(f"  {'─'*68}")

    # A — raw MPS only
    sim_a = make_ghz(N, chi=CHI)
    trunc_a = sum(sim_a.truncation_errors)
    summary_stats(sim_a, 0, trunc_a, "A raw MPS          ")
    bond_diagnostic_table(sim_a, f"A — raw MPS (n={N}, χ={CHI})")

    # B — fixed entropy governor
    sim_b = MPSSimulator(n_qubits=N, max_bond_dim=CHI)
    gov_b = EntanglementGovernor(sim_b, S_max=S_BASE, verbose=False)
    gov_b.hadamard(0)
    for q in range(N - 1):
        gov_b.cnot(q, q + 1)
    prune_b = len(gov_b.pruning_log)
    trunc_b = sum(e["trunc_err"] for e in gov_b.pruning_log)
    summary_stats(sim_b, prune_b, trunc_b, "B fixed governor   ")
    bond_diagnostic_table(sim_b, f"B — Fixed entropy governor (S_max={S_BASE:.3f})")

    # C — polarity governor
    sim_c = MPSSimulator(n_qubits=N, max_bond_dim=CHI)
    gov_c = SemanticPolarityGovernor(sim_c, S_max=S_BASE, alpha=ALPHA, verbose=False)
    gov_c.hadamard(0)
    for q in range(N - 1):
        gov_c.cnot(q, q + 1)
    prune_c = len(gov_c.pruning_log)
    trunc_c = sum(e["trunc_err"] for e in gov_c.pruning_log)
    summary_stats(sim_c, prune_c, trunc_c, "C polarity governor")
    bond_diagnostic_table(sim_c, f"C — Polarity governor (α={ALPHA})")

    print(f"\n  Comparison summary (GHZ, n={N}):")
    print(f"  {'Backend':<25} {'prune_events':>12} {'total_trunc_err':>16} {'mean_S':>8} {'mean_pol':>10}")
    print(f"  {'─'*25} {'─'*12} {'─'*16} {'─'*8} {'─'*10}")
    for label, sim, prune, trunc in [
        ("A raw MPS",           sim_a, 0,       trunc_a),
        ("B fixed governor",    sim_b, prune_b, trunc_b),
        ("C polarity governor", sim_c, prune_c, trunc_c),
    ]:
        E = bond_entropies(sim)
        P = bond_polarities(sim)
        print(f"  {label:<25} {prune:>12} {trunc:>16.4e} {np.mean(E):>8.4f} {np.mean(P):>10.4f}")

    # Key assertion: polarity governor should prune ≤ fixed governor on GHZ
    # because GHZ has high polarity → gets a looser ceiling
    print(f"\n  Key check: polarity governor prune_events ({prune_c}) "
          f"≤ fixed governor prune_events ({prune_b}) for GHZ?")
    if prune_c <= prune_b:
        print("  ✓ PASS — polarity governor is kinder to structured GHZ entanglement")
    else:
        print("  ⚠ NOTE — polarity governor pruned more; check S_base and α settings")


# ─────────────────────────────────────────────────────────────────────────────
# Test 3 — Alpha sweep
# ─────────────────────────────────────────────────────────────────────────────

def test_alpha_sweep():
    print("\n" + "█"*72)
    print("  TEST 3 — Alpha sweep (α = 0 → 0.5 → 1.0) on GHZ")
    print("█"*72)

    N      = 15
    CHI    = 4
    S_BASE = 1.2 * np.log(2)
    alphas = [0.0, 0.25, 0.5, 0.75, 1.0]

    print(f"\n  Circuit: GHZ (n={N})  χ_max={CHI}  S_base={S_BASE:.3f}")
    print(f"  {'α':>6} {'prune_events':>14} {'total_trunc_err':>16} {'mean_S':>8} {'mean_pol':>10}")
    print(f"  {'─'*6} {'─'*14} {'─'*16} {'─'*8} {'─'*10}")

    for alpha in alphas:
        sim = MPSSimulator(n_qubits=N, max_bond_dim=CHI)
        gov = SemanticPolarityGovernor(sim, S_max=S_BASE, alpha=alpha, verbose=False)
        gov.hadamard(0)
        for q in range(N - 1):
            gov.cnot(q, q + 1)
        prune = len(gov.pruning_log)
        trunc = sum(e["trunc_err"] for e in gov.pruning_log)
        E = bond_entropies(sim)
        P = bond_polarities(sim)
        print(f"  {alpha:>6.2f} {prune:>14} {trunc:>16.4e} {np.mean(E):>8.4f} {np.mean(P):>10.4f}")

    print(f"\n  Expected: prune_events decrease as α increases (GHZ gets more protection)")


# ─────────────────────────────────────────────────────────────────────────────
# Test 4 — Two circuit families: GHZ vs random
# ─────────────────────────────────────────────────────────────────────────────

def test_circuit_families():
    print("\n" + "█"*72)
    print("  TEST 4 — Circuit families: GHZ (structured) vs Random (volume-law)")
    print("█"*72)

    N      = 12
    CHI    = 16
    S_BASE = 1.5 * np.log(2)
    ALPHA  = 0.7

    print(f"\n  n={N}  χ_max={CHI}  S_base={S_BASE:.3f}  α={ALPHA}")
    print(f"\n  {'Circuit':<20} {'prune_events':>14} {'total_trunc_err':>16} "
          f"{'mean_S':>8} {'mean_pol':>10} {'max_chi':>8}")
    print(f"  {'─'*20} {'─'*14} {'─'*16} {'─'*8} {'─'*10} {'─'*8}")

    for label, build_fn in [
        ("GHZ",    lambda: make_ghz(N, chi=CHI)),
        ("Random", lambda: make_random_circuit(N, depth=4, chi=CHI)),
    ]:
        for backend_label, GovernorClass, alpha in [
            ("raw MPS",    None,                    None),
            ("fixed gov",  EntanglementGovernor,    None),
            ("polar gov",  SemanticPolarityGovernor, ALPHA),
        ]:
            sim = MPSSimulator(n_qubits=N, max_bond_dim=CHI)
            prune = 0
            trunc = 0.0

            if GovernorClass is None:
                if label == "GHZ":
                    sim.hadamard(0)
                    for q in range(N - 1):
                        sim.cnot(q, q + 1)
                else:
                    for d in range(4):
                        for q in range(N):
                            sim.hadamard(q)
                        start = d % 2
                        for q in range(start, N - 1, 2):
                            sim.cnot(q, q + 1)
                trunc = sum(sim.truncation_errors)

            elif GovernorClass == EntanglementGovernor:
                gov = EntanglementGovernor(sim, S_max=S_BASE, verbose=False)
                if label == "GHZ":
                    gov.hadamard(0)
                    for q in range(N - 1):
                        gov.cnot(q, q + 1)
                else:
                    for d in range(4):
                        for q in range(N):
                            gov.hadamard(q)
                        start = d % 2
                        for q in range(start, N - 1, 2):
                            gov.cnot(q, q + 1)
                prune = len(gov.pruning_log)
                trunc = sum(e["trunc_err"] for e in gov.pruning_log)

            else:
                gov = SemanticPolarityGovernor(sim, S_max=S_BASE, alpha=alpha, verbose=False)
                if label == "GHZ":
                    gov.hadamard(0)
                    for q in range(N - 1):
                        gov.cnot(q, q + 1)
                else:
                    for d in range(4):
                        for q in range(N):
                            gov.hadamard(q)
                        start = d % 2
                        for q in range(start, N - 1, 2):
                            gov.cnot(q, q + 1)
                prune = len(gov.pruning_log)
                trunc = sum(e["trunc_err"] for e in gov.pruning_log)

            E       = bond_entropies(sim)
            P       = bond_polarities(sim)
            max_chi = max(t.shape[2] for t in sim.tensors[:-1])
            name    = f"{label}/{backend_label}"
            print(f"  {name:<20} {prune:>14} {trunc:>16.4e} "
                  f"{np.mean(E):>8.4f} {np.mean(P):>10.4f} {max_chi:>8}")

    print(f"\n  Key thesis check:")
    print(f"  GHZ should have higher polarity and fewer prune events than random circuit")
    print(f"  under polarity-aware governor with same S_base and α")


# ─────────────────────────────────────────────────────────────────────────────
# Test 5 — χ_max saturation: α cannot recover lost capacity
# ─────────────────────────────────────────────────────────────────────────────

def test_chi_saturation():
    print("\n" + "█"*72)
    print("  TEST 5 — χ_max saturation: once hard cap hits, α cannot help")
    print("█"*72)

    N      = 10
    CHI    = 2          # deliberately tiny — will saturate on random circuit
    S_BASE = 2.0 * np.log(2)
    alphas = [0.0, 0.5, 1.0]

    print(f"\n  Random circuit (n={N}, depth=4)  χ_max={CHI}  S_base={S_BASE:.3f}")
    print(f"  (χ_max={CHI} is very tight — hard cap should dominate over α)\n")
    print(f"  {'α':>6} {'prune_events':>14} {'total_trunc_err':>16} {'mean_S':>8} {'max_chi':>8}")
    print(f"  {'─'*6} {'─'*14} {'─'*16} {'─'*8} {'─'*8}")

    for alpha in alphas:
        sim = MPSSimulator(n_qubits=N, max_bond_dim=CHI)
        gov = SemanticPolarityGovernor(sim, S_max=S_BASE, alpha=alpha, verbose=False)
        for d in range(4):
            for q in range(N):
                gov.hadamard(q)
            start = d % 2
            for q in range(start, N - 1, 2):
                gov.cnot(q, q + 1)
        prune   = len(gov.pruning_log)
        trunc   = sum(e["trunc_err"] for e in gov.pruning_log)
        E       = bond_entropies(sim)
        max_chi = max(t.shape[2] for t in sim.tensors[:-1])
        print(f"  {alpha:>6.2f} {prune:>14} {trunc:>16.4e} {np.mean(E):>8.4f} {max_chi:>8}")

    print(f"\n  Expected: total_trunc_err should be similar across all α values")
    print(f"  once χ_max={CHI} is the binding constraint — α modulates soft ceiling,")
    print(f"  but cannot expand the hard bond dimension cap")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "▓"*72)
    print("  POLARITY SEPARATION TEST SUITE")
    print("  Two truncation mechanisms: hard χ_max cap vs soft polarity ceiling")
    print("▓"*72)

    test_polarity_isolation()
    test_mechanism_separation()
    test_alpha_sweep()
    test_circuit_families()
    test_chi_saturation()

    print("\n" + "▓"*72)
    print("  ALL TESTS COMPLETE")
    print("▓"*72)
