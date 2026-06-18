"""
cortex_v2.test_regressions — guards against two bugs the original selftest missed.

R1: long-range CNOT (swap network) must preserve the state norm. The selftest
    only exercised adjacent CNOTs (GHZ), so the leftward swap-back norm collapse
    went undetected.
R2: word_to_rotation must be deterministic ACROSS processes, not just within one
    interpreter run (the old hash()-based version was salted per-process).

Run: python cortex_v2/test_regressions.py
"""

from __future__ import annotations
import subprocess
import sys

import numpy as np

from mps import MPS, ghz
from lattice import word_to_rotation


def _dense(m: MPS) -> np.ndarray:
    psi = m.tensors[0]
    for i in range(1, m.n):
        psi = np.tensordot(psi, m.tensors[i], axes=([-1], [0]))
    return psi.reshape(-1)


def _ref_h_cnot(n: int, c: int, t: int) -> np.ndarray:
    """Exact statevector for H(c) then CNOT(c, t)."""
    Hm = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
    psi = np.zeros([2] * n, dtype=complex)
    psi[(0,) * n] = 1.0
    psi = np.moveaxis(np.tensordot(Hm, psi, axes=([1], [c])), 0, c)
    sl = [slice(None)] * n
    sl[c] = 1
    psi[tuple(sl)] = np.flip(psi[tuple(sl)], axis=(t if t < c else t - 1))
    return psi.reshape(-1)


def test_long_range_cnot_norm_and_correctness():
    cases = [(0, 3, 4), (0, 2, 4), (1, 4, 5), (3, 0, 4), (4, 1, 5)]
    for c, t, n in cases:
        m = MPS(n)
        m.hadamard(c)
        m.cnot(c, t)
        v = _dense(m)
        norm = float(np.vdot(v, v).real)
        assert abs(norm - 1.0) < 1e-9, f"CNOT({c},{t}) norm={norm:.4f} (expected 1.0)"
        err = float(np.max(np.abs(v - _ref_h_cnot(n, c, t))))
        assert err < 1e-9, f"CNOT({c},{t}) max_err_vs_exact={err:.2e}"
    print("R1 OK  long-range CNOT preserves norm and matches exact statevector")


def test_ghz_still_valid():
    for n in (4, 6, 9):
        outs = [tuple(ghz(n).measure_all(np.random.default_rng(s))) for s in range(30)]
        assert all(len(set(o)) == 1 for o in outs), f"GHZ({n}) produced mixed bits"
    print("R1b OK  GHZ measurements remain all-0 / all-1")


def test_word_to_rotation_cross_process():
    words = ["cat", "dog", "livnium", "entropy", "lattice"]
    here = {w: word_to_rotation(w) for w in words}
    assert all(1 <= v <= 23 for v in here.values()), "rotation index out of 1..23"
    code = (
        "import sys; sys.path.insert(0,'.'); from lattice import word_to_rotation as w;"
        "import json;print(json.dumps({x:w(x) for x in %r}))" % words
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, cwd="."
    )
    import json

    other = json.loads(out.stdout)
    assert other == here, f"mapping differs across processes: {here} vs {other}"
    print("R2 OK  word_to_rotation is deterministic across separate processes")


if __name__ == "__main__":
    test_long_range_cnot_norm_and_correctness()
    test_ghz_still_valid()
    test_word_to_rotation_cross_process()
    print("\n3/3 regression guards pass.")
