"""
validate_compression.py — how much data can the MPS representation compress?

An MPS stores a length-2^n vector as n small tensors of bond dim <= chi.
  full size      = 2^n complex numbers
  compressed     ~ sum of tensor sizes (n * 2 * chi^2 worst case)
Compression is LOSSLESS only when the data is low-"entanglement"; random data
does not compress. We measure ratio AND reconstruction error on real signals.

Run from repo root:  python cortex_v2/validate_compression.py
"""
import numpy as np

def compress(vec, max_chi):
    """Left-to-right SVD into an MPS with bond cap max_chi.
    Returns (tensors, rel_L2_error, n_params_stored)."""
    n = int(round(np.log2(len(vec))))
    assert 2**n == len(vec), "length must be a power of 2"
    psi = vec.astype(complex) / np.linalg.norm(vec)
    tensors = []
    chi_l = 1
    resid = psi.reshape(1, -1)
    for i in range(n - 1):
        resid = resid.reshape(chi_l * 2, -1)
        U, s, Vh = np.linalg.svd(resid, full_matrices=False)
        k = min(max_chi, (s > 1e-14).sum() or 1)
        U, s, Vh = U[:, :k], s[:k], Vh[:k]
        tensors.append(U.reshape(chi_l, 2, k))
        resid = s[:, None] * Vh
        chi_l = k
    tensors.append(resid.reshape(chi_l, 2, 1))
    # reconstruct
    rec = tensors[0]
    for t in tensors[1:]:
        rec = np.tensordot(rec, t, axes=([-1], [0]))
    rec = rec.reshape(-1)
    err = float(np.linalg.norm(rec - psi) / np.linalg.norm(psi))
    nparams = sum(t.size for t in tensors)
    return tensors, err, nparams

def report(name, vec, max_chi=16):
    n = int(round(np.log2(len(vec))))
    full = len(vec)
    _, err, nparams = compress(vec, max_chi)
    ratio = full / nparams
    print(f"  {name:22}  n={n:2d}  full={full:>6}  stored={nparams:>6}  "
          f"ratio={ratio:6.1f}x  rel_err={err:.2e}  {'LOSSLESS' if err<1e-9 else 'lossy'}")

n = 14
x = np.linspace(0, 1, 2**n)
print("=" * 92)
print(f"MPS COMPRESSION of length-2^{n} = {2**n} arrays, bond cap chi=16")
print("=" * 92)
report("random noise",      np.random.default_rng(0).standard_normal(2**n))
report("constant (all ones)", np.ones(2**n))
report("single spike",      np.eye(2**n)[2**n // 3])
report("smooth sine",       np.sin(2 * np.pi * 5 * x))
report("low-freq sum",      sum(np.sin(2*np.pi*f*x) for f in (1,2,3)))
report("linear ramp",       x.copy())
report("exp decay",         np.exp(-8 * x))
report("step function",     (x > 0.5).astype(float))
report("2-level (GHZ-like)", np.array([1,0]*(2**(n-1)-1) + [0,1], float))

print()
print("now sweep the bond cap on a smooth signal (accuracy vs compression):")
sig = sum(np.sin(2*np.pi*f*x) for f in (1,2,3,5,8))
for chi in [1, 2, 4, 8, 16, 32]:
    _, err, nparams = compress(sig, chi)
    print(f"  chi={chi:3d}  stored={nparams:>6}  ratio={2**n/nparams:7.1f}x  rel_err={err:.2e}")

print()
print("READING: compression ratio is data-dependent. Structured/smooth data")
print("compresses 100x+ losslessly; random data does not compress at all.")
print("This is standard tensor-network behavior, not a universal compressor.")
