"""Cross-implementation test of Livnium vision levels 1 and 2.

Loads the trained .pt checkpoints WITHOUT torch (raw zip + pickle + numpy),
re-implements both forward passes independently, and checks:
  L1: canonical color anchors classify to themselves; random RGB accuracy
      vs nearest-anchor ground truth
  L2: full pipeline (numpy L1 labels -> fovea encode -> decode) reproduces
      the recon accuracies the Mac's torch run reported (0.560/0.278/0.510)
Agreement = both levels verified end to end by an independent implementation.
"""
import io
import math
import pickle
import zipfile

import numpy as np

DTYPES = {"FloatStorage": np.float32, "DoubleStorage": np.float64,
          "LongStorage": np.int64, "IntStorage": np.int32,
          "ByteStorage": np.uint8, "CharStorage": np.int8,
          "BoolStorage": np.bool_, "HalfStorage": np.float16,
          "ShortStorage": np.int16}


def load_pt(path):
    """torch.save zip format -> dict of numpy arrays. No torch needed."""
    zf = zipfile.ZipFile(path)
    pkl = [n for n in zf.namelist() if n.endswith("/data.pkl")][0]
    prefix = pkl[: -len("data.pkl")]

    def rebuild(storage, offset, size, stride, *rest):
        buf, dt = storage
        a = np.frombuffer(buf, dtype=dt)
        if len(size) == 0:
            return a[offset].copy()
        n = int(np.prod(size))
        # verify contiguous stride (detach().cpu() tensors are)
        exp = []
        acc = 1
        for s in reversed(size):
            exp.insert(0, acc)
            acc *= s
        assert tuple(stride) == tuple(exp), f"non-contiguous save {stride} vs {exp}"
        return a[offset:offset + n].reshape(size).copy()

    class Stub:
        def __init__(self, *a, **k):
            pass

    class U(pickle.Unpickler):
        def find_class(self, mod, name):
            if name == "_rebuild_tensor_v2":
                return rebuild
            if mod == "collections" and name == "OrderedDict":
                return dict
            if name.endswith("Storage"):
                return name          # marker string
            return Stub

        def persistent_load(self, pid):
            typ, key = pid[1], pid[2]
            name = typ if isinstance(typ, str) else type(typ).__name__
            return (zf.read(prefix + "data/" + key), DTYPES[name])

    return U(io.BytesIO(zf.read(pkl))).load()


def norm(v, ax=-1):
    return v / (np.linalg.norm(v, axis=ax, keepdims=True) + 1e-12)


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-float(x)))


def softplus(x):
    return math.log1p(math.exp(float(x)))


# ======================= LEVEL 1: pixel_color_pure =========================
def l1_forward(ck, rgb):
    """rgb (N,3) in [0,1] -> label indices. Verbatim label_pixels physics."""
    s = sigmoid(ck["log_strength"])
    A = norm(ck["ch_wells"].astype(np.float64))
    cw = norm(ck["color_wells"].astype(np.float64))
    h = np.broadcast_to(ck["start"].astype(np.float64),
                        (rgb.shape[0], A.shape[1])).copy()
    for c in range(3):
        t, v = A[c], rgb[:, c:c + 1]
        align = (norm(h) * t).sum(-1, keepdims=True)
        h = h - v * s * (1.0 - align) * norm(h - t)
        n = np.linalg.norm(h, axis=-1, keepdims=True)
        h = np.where(n > 10.0, h * (10.0 / (n + 1e-8)), h)
    return (norm(h) @ cw.T).argmax(1)


def test_level1(l1, colors):
    names = list(l1["names"])
    anchors = np.array([colors[n] for n in names])           # (13,3) in [0,1]
    print(f"L1 checkpoint: {len(names)} colors, dim {l1['ch_wells'].shape[1]}, "
          f"strength {sigmoid(l1['log_strength']):.3f}")
    # 1. anchors classify to themselves?
    pred = l1_forward(l1, anchors)
    ok = (pred == np.arange(len(names))).sum()
    print(f"  anchors -> self: {ok}/{len(names)}", "PASS" if ok == len(names)
          else f"FAIL ({[(names[i], names[p]) for i, p in enumerate(pred) if p != i]})")
    # 2. random RGBs vs nearest-anchor ground truth (the training objective)
    rng = np.random.default_rng(0)
    x = rng.random((20000, 3))
    gt = ((x[:, None, :] - anchors[None]) ** 2).sum(-1).argmin(1)
    pred = l1_forward(l1, x)
    acc = (pred == gt).mean()
    print(f"  20k random RGBs vs nearest-anchor: acc {acc:.3f}",
          "PASS" if acc > 0.80 else "MARGINAL" if acc > 0.6 else "FAIL")
    return acc


# ========================= LEVEL 2: img_fovea ==============================
def polar(S, R, A, focal):
    c = (S - 1) / 2.0
    y, x = np.mgrid[0:S, 0:S].astype(np.float64)
    dx, dy = (x - c).ravel(), (y - c).ravel()
    radius = np.hypot(dx, dy)
    r_norm = radius / radius.max()
    r_bin = np.clip((r_norm * R).astype(int), 0, R - 1)
    a_bin = np.clip(((np.arctan2(dy, dx) + math.pi) / (2 * math.pi) * A).astype(int),
                    0, A - 1)
    rays = np.stack([dx, dy, np.full_like(dx, focal)], -1)
    rays = norm(rays)
    return r_bin, a_bin, rays, r_norm


def test_level2(ck, labels_np, reported):
    cfg = ck["config"]
    S, D, R, A = cfg["size"], cfg["dim"], cfg["rings"], cfg["angles"]
    P, SS = cfg["pixel_chunk"], S * S
    r_bin, a_bin, rays, r_norm = polar(S, R, A, cfg["focal"])
    print(f"L2 checkpoint: S={S} dim={D} bind={cfg.get('bind')} "
          f"scan={cfg.get('scan')} rings={R} angles={A}")

    ring = ck["ring_wells"].astype(np.float64)
    ang = ck["angle_wells"].astype(np.float64)
    rayw = ck["ray_wells"].astype(np.float64)
    colw = ck["color_wells2"].astype(np.float64)
    if cfg.get("bind", "add") == "mul":
        pos = ring[r_bin] * ang[a_bin] * D ** 0.5 + rays @ rayw
    else:
        pos = ring[r_bin] + ang[a_bin] + rays @ rayw

    # address quality of the TRAINED model (the drift question)
    pn = norm(pos)
    rng = np.random.default_rng(1)
    i, j = rng.integers(0, SS, 8000), rng.integers(0, SS, 8000)
    keep = i != j
    xt = np.abs((pn[i[keep]] * pn[j[keep]]).sum(-1)).mean()
    print(f"  trained addresses: mean|cos| {xt:.3f} (free-well ref "
          f"{1 / D ** 0.5:.3f}) {'<- drifted' if xt > 2 / D ** 0.5 else 'ok'}")

    if cfg.get("scan") == "spiral":
        perm = np.argsort(r_norm * (2 * A) + a_bin / A, kind="stable")
    elif cfg.get("scan") == "shuffled":
        raise SystemExit("shuffled scan: need torch RNG parity, skip")
    else:
        perm = np.arange(SS)

    s = sigmoid(ck["log_strength"])
    g = np.exp(-softplus(ck["log_falloff"]) * r_norm)[perm]
    start = ck["start"].astype(np.float64)
    M = norm(pos[:, None, :] + colw[None, :, :])             # (SS,13,D)

    accs = []
    for k, lab in enumerate(labels_np):
        T = norm(pos + colw[lab])[perm]
        h = start.copy()
        for c0 in range(0, SS, P):
            t = T[c0:c0 + P]
            align = (norm(h[None]) * t).sum(-1)
            away = norm(h[None] - t)
            h = h - ((s * g[c0:c0 + P] * (1.0 - align))[:, None] * away).sum(0)
            n = np.linalg.norm(h)
            if n > 10.0:
                h = h * (10.0 / n)
        pred = (M @ norm(h)).argmax(-1)
        acc = (pred == lab).mean()
        accs.append(acc)
        flag = ""
        if reported and k < len(reported):
            flag = (f"   torch said {reported[k]:.3f}  "
                    f"{'MATCH' if abs(acc - reported[k]) < 0.02 else 'MISMATCH'}")
        print(f"  img {k}: numpy recon acc {acc:.3f}  "
              f"pred-colors {len(set(pred))}/13{flag}")
    return accs


if __name__ == "__main__":
    import sys
    import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from pixel_color_pure import COLORS   # anchor definitions only, no torch

    print("=" * 72)
    l1 = load_pt("vision/model/pixel_color_pure.pt")
    test_level1(l1, COLORS)

    print("=" * 72)
    cache = load_pt("vision/model/img_cache_rgb_64.pt")
    imgs = cache["imgs"]                                      # (N,64,64,3) uint8
    n_test = 3
    rgb = imgs[:n_test].reshape(-1, 3).astype(np.float64) / 255.0
    print(f"labeling {n_test} images with numpy L1 ...")
    labels = l1_forward(l1, rgb).reshape(n_test, -1)

    l2 = load_pt("vision/model/img_fovea.pt")
    test_level2(l2, labels, reported=[0.560, 0.278, 0.510])
    print("=" * 72)
