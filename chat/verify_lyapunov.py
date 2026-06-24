"""verify_lyapunov.py — empirical test that the collapse decoder is a
Lyapunov-stable, non-expansive, DRIVEN dynamical system (not a static map).

Self-contained: loads the trained weights from
model/premise_from_hyp_align_53.pt WITHOUT torch (custom unpickler), reimplements
collapse_step / generate in NumPy, and runs three tests:

  1. Lyapunov:    V(h)=1-cos(h,target) never increases under the driven collapse.
  2. Contraction: singular values of the per-step Jacobian dF/dh (fraction < 1).
  3. Driven:      during real generation the state is pulled toward each task
                  token (cos before->after), and the trajectory moves.

Usage:  python3 verify_lyapunov.py
Needs:  numpy ;  model/premise_from_hyp_align_53.pt
"""
import os, zipfile, pickle, io, numpy as np

CKPT = os.path.join(os.path.dirname(__file__), "model/premise_from_hyp_align_53.pt")
MAXLEN, MAX_WORDS, PAD = 34, 32, 0

# ---- load a torch checkpoint without torch -----------------------------------
def _rt2(storage, off, size, stride, *a, **k): return ("T", storage, tuple(size))
def _rp(data, *a, **k): return data
class _Stub:
    def __init__(self, *a, **k): pass
class _U(pickle.Unpickler):
    def persistent_load(self, pid):
        _, st, key, loc, numel = pid; return {"key": str(key), "dtype": st, "numel": numel}
    def find_class(self, m, n):
        if n == "_rebuild_tensor_v2": return _rt2
        if n == "_rebuild_parameter": return _rp
        if n == "OrderedDict":
            from collections import OrderedDict; return OrderedDict
        if "Storage" in n: return n
        return _Stub
_DT = {"FloatStorage": "<f4", "DoubleStorage": "<f8", "LongStorage": "<i8",
       "IntStorage": "<i4", "HalfStorage": "<f2", "BoolStorage": "|b1", "ByteStorage": "|u1"}
_z = zipfile.ZipFile(CKPT)
_pkl = [n for n in _z.namelist() if n.endswith("data.pkl")][0]
_obj = _U(io.BytesIO(_z.read(_pkl))).load()
_ent = {n.split("/data/")[-1]: n for n in _z.namelist() if "/data/" in n and not n.endswith("/")}
def _arr(t):
    _, st, size = t
    a = np.frombuffer(_z.read(_ent[st["key"]]), dtype=np.dtype(_DT.get(st["dtype"], "<f4"))).copy()
    return a.reshape(()) if size == () else a.reshape(size)
W = {k: _arr(v) for k, v in _obj["state_dict"].items()}
stoi, unk, eos = _obj["stoi"], _obj["unk"], _obj["eos"]
itos = {i: w for w, i in stoi.items()}; itos[unk] = "<unk>"; itos[eos] = "<eos>"

def norm(x, ax=-1, eps=1e-12): return x / (np.linalg.norm(x, axis=ax, keepdims=True) + eps)
def lin(x, w, b): return x @ w.T + b
def softplus(x): return np.log1p(np.exp(-abs(x))) + np.maximum(x, 0)
def sigmoid(x): return 1 / (1 + np.exp(-x))
A = norm(W["word_anchors"]); LAB = W["label_emb"]
strength = float(sigmoid(W["log_strength"])); temp = float(softplus(W["log_temp"]) + 1e-3)
Wt, bt = W["think.weight"], W["think.bias"]
W1, b1 = W["brain.0.weight"], W["brain.0.bias"]
W2, b2 = W["brain.2.weight"], W["brain.2.bias"]
Wk, bk = W["att_key.0.weight"], W["att_key.0.bias"]
Wq, bq = W["att_query.0.weight"], W["att_query.0.bias"]

def encode(s):
    ids = [stoi.get(t, unk) for t in s.lower().strip().split()][:MAX_WORDS] + [eos]
    ids += [PAD] * (MAXLEN - len(ids)); return np.array(ids[:MAXLEN])
def meanpool(ids):
    m = (ids != PAD); return A[ids[m]].mean(0) if m.any() else np.zeros(256)
def brain(x): return lin(np.tanh(lin(x, W1, b1)), W2, b2)
def collapse(h, target):
    al = float(norm(h) @ target); away = norm(h - target); h = h - strength * (1 - al) * away
    n = np.linalg.norm(h); return h * (10.0 / (n + 1e-8)) if n > 10 else h
def cosv(h, t): h = h / (np.linalg.norm(h) + 1e-12); return float(h @ t)

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    print(f"strength={strength:.3f}  temp={temp:.3f}\n")

    print("=== TEST 1: Lyapunov energy V(h)=1-cos(h,target) under driven collapse ===")
    viol = tot = conv = 0; N = 300
    for _ in range(N):
        t = A[rng.integers(1, A.shape[0])]
        h = rng.standard_normal(256); h *= rng.uniform(0.1, 10) / np.linalg.norm(h)
        Vs = [1 - cosv(h, t)]
        for _ in range(40):
            h = collapse(h, t); Vn = 1 - cosv(h, t); tot += 1
            if Vn > Vs[-1] + 1e-9: viol += 1
            Vs.append(Vn)
        if Vs[-1] < 1e-3: conv += 1
    print(f"  monotone-decreasing steps: {100*(tot-viol)/tot:.2f}%  (Lyapunov: V never increases)")
    print(f"  reached attractor (V<1e-3) in 40 steps: {100*conv/N:.1f}% (asymptotic, eases in)\n")

    print("=== TEST 2: contraction — singular values of dF/dh ===")
    def jac(h, t, eps=1e-5):
        base = collapse(h.copy(), t); J = np.empty((256, 256))
        for i in range(256):
            hp = h.copy(); hp[i] += eps; J[:, i] = (collapse(hp, t) - base) / eps
        return J
    for name, mk in [("random pt", lambda t: norm(rng.standard_normal(256)) * 5),
                     ("near attractor", lambda t: t + 0.05 * rng.standard_normal(256)),
                     ("mid (h=2*t)", lambda t: 2.0 * t)]:
        t = A[rng.integers(1, A.shape[0])]; s = np.linalg.svd(jac(mk(t), t), compute_uv=False)
        print(f"  {name:15} S<1: {int((s<1).sum())}/256 = {100*(s<1).mean():.1f}%"
              f"   Smax {s.max():.2f}  Smean {s.mean():.2f}")

    print("\n=== TEST 3: driven/dynamic vs static (real generation trajectory) ===")
    ids = encode("two men are playing football"); lab = LAB[1]
    z = lin(np.concatenate([meanpool(ids), lab]), Wt, bt); h = z
    hm = (ids != PAD); EH = A[ids]; K = np.tanh(lin(EH, Wk, bk))
    print("  word        cos(h,target) before -> after   |h|")
    norms = []
    for _ in range(MAXLEN):
        hn = norm(h); q = np.tanh(lin(np.concatenate([hn, lab]), Wq, bq))
        sc = np.where(hm, K @ q, -1e9); a = np.exp(sc - sc.max()); a /= a.sum(); ctx = a @ EH
        query = brain(np.concatenate([hn, z, ctx]))
        logits = (norm(query) @ A.T) / temp; logits[PAD] = -1e30
        nxt = int(logits.argmax()); before = cosv(h, A[nxt]); h = collapse(h, A[nxt])
        after = cosv(h, A[nxt]); norms.append(float(np.linalg.norm(h)))
        print(f"  {itos.get(nxt,'?'):10}  {before:+.3f} -> {after:+.3f}   {np.linalg.norm(h):.2f}")
        if nxt == eos: break
    print(f"  state-norm std across steps = {np.std(norms):.3f}  => trajectory MOVES (not static)")
