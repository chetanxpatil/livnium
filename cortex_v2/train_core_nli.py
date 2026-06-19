"""
train_core_nli.py — REAL (scaled) test: wire Livnium Core into the NLI collapse model.

Model (pure NumPy, trained end-to-end):
  word embeddings -> mean-pool premise u, hypothesis v
  feature f = [u - v, u * v]
  project p = W f  (81-dim, the cube-anchor space)
  logits = cosine(p, anchor_k) / tau   for k in {Entail, Neutral, Contradict}
  softmax + cross-entropy.  Anchors are FIXED (not learned).

We train twice, identical except the anchors:
  (1) CUBE anchors  — built from Livnium Core rotations (Core actually powers it)
  (2) RANDOM anchors — control
So the delta isolates whether Core's geometry helps.

Scaled for CPU: subset of SNLI, small dim, few epochs. NOT the full 20-epoch run,
so absolute numbers are below the 68.92% headline — what matters is cube vs random
and clearing the majority baseline (~33%).
"""
import sys, os, json, re, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from lattice import PERMS, COORDS, SW

DATA = "/sessions/loving-great-ramanujan/mnt/test/data-bank/snli_1.0_train_cleaned.jsonl"
N_TRAIN, N_DEV = 40000, 5000
DIM, VOCAB, EPOCHS, BATCH, LR, TAU = 48, 15000, 4, 256, 0.5, 0.15
LABELS = {"entailment": 0, "neutral": 1, "contradiction": 2}
rng = np.random.default_rng(0)

def unit(v): return v / (np.linalg.norm(v) + 1e-9)
tok = lambda s: re.findall(r"[a-z]+", s.lower())

# ---- cube anchors (Livnium Core) ----
def cube_anchor(rot):
    moved = COORDS[PERMS[rot]].astype(float)
    return unit((SW[:, None] * moved).reshape(-1))   # 81-dim
import itertools
CUBE = np.array([cube_anchor(r) for r in range(24)])
best, trip = 9, None
for t in itertools.combinations(range(24), 3):
    c = CUBE[list(t)]; m = max((c@c.T)[i,j] for i in range(3) for j in range(i+1,3))
    if m < best: best, trip = m, t
CUBE3 = CUBE[list(trip)]
RAND3 = np.array([unit(rng.standard_normal(81)) for _ in range(3)])

# ---- load data ----
print("loading SNLI subset ...", flush=True)
rows = []
with open(DATA) as f:
    for line in f:
        d = json.loads(line); g = d.get("gold_label")
        if g in LABELS:
            rows.append((tok(d["sentence1"]), tok(d["sentence2"]), LABELS[g]))
        if len(rows) >= N_TRAIN + N_DEV: break
rng.shuffle(rows)
train, dev = rows[:N_TRAIN], rows[N_TRAIN:N_TRAIN+N_DEV]

from collections import Counter
cnt = Counter(w for p,h,_ in train for w in p+h)
vocab = {w:i+1 for i,(w,_) in enumerate(cnt.most_common(VOCAB-1))}  # 0 = unk/pad
def ids(ws): return [vocab.get(w,0) for w in ws] or [0]
train = [(ids(p), ids(h), y) for p,h,y in train]
dev   = [(ids(p), ids(h), y) for p,h,y in dev]
print(f"train={len(train)} dev={len(dev)} vocab={len(vocab)} dim={DIM}", flush=True)

def run(anchors, name):
    E = rng.standard_normal((VOCAB, DIM)) * 0.1
    W = rng.standard_normal((81, 2*DIM)) * (1/np.sqrt(2*DIM))
    A = anchors
    def feats(batch):
        U = np.array([E[p].mean(0) for p,_,_ in batch])
        V = np.array([E[h].mean(0) for _,h,_ in batch])
        return U, V, np.concatenate([U-V, U*V], 1)
    for ep in range(EPOCHS):
        rng.shuffle(train)
        for b in range(0, len(train), BATCH):
            batch = train[b:b+BATCH]; B = len(batch)
            U,V,f = feats(batch)
            p = f @ W.T                       # B x 81
            n = np.linalg.norm(p,axis=1,keepdims=True)+1e-9
            g = p/n
            logits = (g @ A.T)/TAU            # B x 3
            logits -= logits.max(1,keepdims=True)
            sm = np.exp(logits); sm /= sm.sum(1,keepdims=True)
            y = np.array([t[2] for t in batch])
            dlog = sm.copy(); dlog[np.arange(B),y] -= 1; dlog /= (B*TAU)
            dg = dlog @ A                     # B x 81
            dp = (dg - g*(g*dg).sum(1,keepdims=True))/n
            dW = dp.T @ f
            df = dp @ W                       # B x 2D
            d_uv, d_uvprod = df[:,:DIM], df[:,DIM:]
            dU = d_uv + d_uvprod*V
            dV = -d_uv + d_uvprod*U
            W -= LR*dW
            for i,(pp,hh,_) in enumerate(batch):
                E[pp] -= LR*dU[i]/len(pp); E[hh] -= LR*dV[i]/len(hh)
        # dev eval
        correct=0
        for b in range(0,len(dev),BATCH):
            batch=dev[b:b+BATCH]
            _,_,f=feats(batch); p=f@W.T; g=p/(np.linalg.norm(p,axis=1,keepdims=True)+1e-9)
            pred=np.argmax((g@A.T),1); correct+=int(np.sum(pred==[t[2] for t in batch]))
        print(f"  [{name}] epoch {ep+1}: dev acc = {100*correct/len(dev):.2f}%", flush=True)
    return 100*correct/len(dev)

print("\n=== training (cube anchors = Livnium Core) ===", flush=True)
acc_cube = run(CUBE3, "cube")
print("\n=== training (random anchors = control) ===", flush=True)
acc_rand = run(RAND3, "random")
print(f"\nRESULT: cube={acc_cube:.2f}%  random={acc_rand:.2f}%  "
      f"(majority baseline ~33.3%, hypothesis-only ~61.5%)")
print("delta (cube - random) =", round(acc_cube-acc_rand,2), "points")
