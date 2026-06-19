"""
train_core_nli_scaled.py — scaled-up cube-anchor NLI (vectorized NumPy).

Same model as train_core_nli.py but bigger: more data, dim 128, richer features
[u, v, u-v, u*v], more epochs, padded-batch vectorization for speed. Cube anchors
(Livnium Core) drive the 3 wells. Goal: see how close Core gets to the 68.92%.
"""
import sys, os, json, re, itertools
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from lattice import PERMS, COORDS, SW

DATA = "/sessions/loving-great-ramanujan/mnt/test/data-bank/snli_1.0_train_cleaned.jsonl"
N_TRAIN, N_DEV = 100000, 8000
DIM, VOCAB, EPOCHS, BATCH, LR, TAU, MAXLEN = 128, 25000, 6, 512, 0.5, 0.12, 32
LABELS = {"entailment": 0, "neutral": 1, "contradiction": 2}
rng = np.random.default_rng(0)
unit = lambda v: v/(np.linalg.norm(v)+1e-9)
tok = lambda s: re.findall(r"[a-z]+", s.lower())

CUBE = np.array([unit((SW[:,None]*COORDS[PERMS[r]].astype(float)).reshape(-1)) for r in range(24)])
best, trip = 9, None
for t in itertools.combinations(range(24),3):
    c=CUBE[list(t)]; m=max((c@c.T)[i,j] for i in range(3) for j in range(i+1,3))
    if m<best: best,trip=m,t
A = CUBE[list(trip)]

print("loading ...", flush=True)
rows=[]
with open(DATA) as f:
    for line in f:
        d=json.loads(line); g=d.get("gold_label")
        if g in LABELS: rows.append((tok(d["sentence1"]),tok(d["sentence2"]),LABELS[g]))
        if len(rows)>=N_TRAIN+N_DEV: break
rng.shuffle(rows)
train, dev = rows[:N_TRAIN], rows[N_TRAIN:N_TRAIN+N_DEV]
from collections import Counter
cnt=Counter(w for p,h,_ in train for w in p+h)
vocab={w:i+1 for i,(w,_) in enumerate(cnt.most_common(VOCAB-1))}
def pack(data):
    P=np.zeros((len(data),MAXLEN),np.int32); H=np.zeros((len(data),MAXLEN),np.int32)
    lp=np.zeros(len(data)); lh=np.zeros(len(data)); Y=np.zeros(len(data),np.int64)
    for i,(p,h,y) in enumerate(data):
        pi=[vocab.get(w,0) for w in p][:MAXLEN] or [0]; hi=[vocab.get(w,0) for w in h][:MAXLEN] or [0]
        P[i,:len(pi)]=pi; H[i,:len(hi)]=hi; lp[i]=len(pi); lh[i]=len(hi); Y[i]=y
    return P,H,lp,lh,Y
Ptr,Htr,lptr,lhtr,Ytr=pack(train); Pdv,Hdv,lpdv,lhdv,Ydv=pack(dev)
print(f"train={len(train)} dev={len(dev)} vocab={len(vocab)} dim={DIM}", flush=True)

E=rng.standard_normal((VOCAB,DIM))*0.1
W=rng.standard_normal((81,4*DIM))*(1/np.sqrt(4*DIM))

def pool(P,lens):
    m=(P>0)|(np.arange(MAXLEN)==0)  # keep at least slot0
    emb=E[P]*(np.arange(MAXLEN)<lens[:,None])[...,None]
    return emb.sum(1)/lens[:,None]

def forward(P,H,lp,lh):
    U=pool(P,lp); V=pool(H,lh)
    f=np.concatenate([U-V,U*V,U,V],1)
    p=f@W.T; n=np.linalg.norm(p,axis=1,keepdims=True)+1e-9; g=p/n
    return U,V,f,p,n,g,(g@A.T)/TAU

def evald():
    correct=0
    for b in range(0,len(dev),BATCH):
        sl=slice(b,b+BATCH)
        *_,g,_=forward(Pdv[sl],Hdv[sl],lpdv[sl],lhdv[sl])
        pred=np.argmax(g@A.T,1); correct+=int(np.sum(pred==Ydv[sl]))
    return 100*correct/len(dev)

idx=np.arange(len(train))
for ep in range(EPOCHS):
    rng.shuffle(idx)
    for b in range(0,len(train),BATCH):
        s=idx[b:b+BATCH]
        P,H,lp,lh,y=Ptr[s],Htr[s],lptr[s],lhtr[s],Ytr[s]; B=len(s)
        U,V,f,p,n,g,logits=forward(P,H,lp,lh)
        logits-=logits.max(1,keepdims=True); sm=np.exp(logits); sm/=sm.sum(1,keepdims=True)
        dlog=sm.copy(); dlog[np.arange(B),y]-=1; dlog/=(B*TAU)
        dg=dlog@A; dp=(dg-g*(g*dg).sum(1,keepdims=True))/n
        dW=dp.T@f; df=dp@W
        if os.environ.get("LEARN_ANCHORS")=="1":
            A=A-LR*(dlog.T@g); A=A/(np.linalg.norm(A,axis=1,keepdims=True)+1e-9)
        d1,d2,d3,d4=df[:,:DIM],df[:,DIM:2*DIM],df[:,2*DIM:3*DIM],df[:,3*DIM:]
        dU=d1+d2*V+d3; dV=-d1+d2*U+d4
        W-=LR*dW
        # scatter embedding grads (mean-pool): each token gets dU/len
        gp=(dU/lp[:,None])[:,None,:]*((np.arange(MAXLEN)<lp[:,None])[...,None])
        gh=(dV/lh[:,None])[:,None,:]*((np.arange(MAXLEN)<lh[:,None])[...,None])
        np.add.at(E,P.reshape(-1),-LR*gp.reshape(-1,DIM))
        np.add.at(E,H.reshape(-1),-LR*gh.reshape(-1,DIM))
    print(f"  epoch {ep+1:2d}: dev acc = {evald():.2f}%", flush=True)
print(f"\nFINAL cube-anchor dev acc = {evald():.2f}%  (full learned model = 68.92%, majority 33.3%)", flush=True)
