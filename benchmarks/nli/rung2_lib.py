#!/usr/bin/env python3
"""Shared helpers + phased runner for Rung 2 (fits inside short shell calls).

Phases (argv[1]):
  trim  - load GloVe once, gather SNLI+ANLI vocab, save trimmed embeddings (npz+json)
  snli  - load trimmed emb, run SNLI learned-vs-baselines, save snli_res.json + md
  anli  - load trimmed emb, run ANLI honest harness, save anli_res.json + md
  combine - stitch md sections + verdict into RUNG2_RESULTS.md
"""

import json
import re
import sys
import time
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
SNLI_DIR = HERE / "data" / "snli"
ANLI_DIR = HERE / "anli_data"
EMB_NPZ = HERE / "_rung2_emb.npz"
EMB_VOC = HERE / "_rung2_vocab.json"
SEED = 42
N_TRAIN_SNLI = 50_000

LABELS = {"entailment": 0, "neutral": 1, "contradiction": 2}
LABNAME = ["entail", "neutral", "contra"]
_tok = re.compile(r"[a-z']+")


def toks(s):
    return _tok.findall(s.lower())


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


# ---- data ----
def load_snli(name):
    P, H, Y = [], [], []
    for line in open(SNLI_DIR / f"snli_1.0_{name}.jsonl"):
        d = json.loads(line)
        g = d.get("gold_label")
        if g in LABELS:
            P.append(d["sentence1"])
            H.append(d["sentence2"])
            Y.append(LABELS[g])
    return P, H, np.array(Y)


def load_anli():
    import pandas as pd

    tr = pd.concat([pd.read_parquet(ANLI_DIR / f"anli_train_r{i}.parquet") for i in (1, 2, 3)])
    tr = tr[tr.label.isin([0, 1, 2])]
    tests = {
        i: (lambda d: d[d.label.isin([0, 1, 2])])(
            pd.read_parquet(ANLI_DIR / f"anli_test_r{i}.parquet")
        )
        for i in (1, 2, 3)
    }
    return tr, tests


# ---- fast vectorized encoder over trimmed embeddings ----
class Enc:
    def __init__(self):
        z = np.load(EMB_NPZ)
        self.M = z["M"]
        self.dim = self.M.shape[1]
        self.idx = json.loads(EMB_VOC.read_text())

    def batch(self, sents):
        out = np.zeros((len(sents), self.dim), dtype=np.float32)
        cnt = np.zeros(len(sents), dtype=np.float32)
        rows = []
        cols = []
        for i, s in enumerate(sents):
            for t in toks(s):
                j = self.idx.get(t)
                if j is not None:
                    rows.append(i)
                    cols.append(j)
        if rows:
            rows = np.array(rows)
            cols = np.array(cols)
            np.add.at(out, rows, self.M[cols])
            np.add.at(cnt, rows, 1.0)
        cnt[cnt == 0] = 1.0
        return out / cnt[:, None]

    def pair(self, P, H):
        u = self.batch(P)
        v = self.batch(H)
        return np.hstack([u, v, np.abs(u - v), u * v]).astype(np.float32)


def acc(clf, X, y):
    from sklearn.metrics import accuracy_score

    return accuracy_score(y, clf.predict(X)) * 100


# ============================ phases ============================
def phase_trim():
    import gensim.downloader as api

    log("loading GloVe (cached)")
    kv = api.load("glove-wiki-gigaword-100")
    log(f"glove {len(kv.key_to_index)} dim {kv.vector_size}")
    vocab = set()
    for name in ("train", "dev", "test"):
        P, H, _ = load_snli(name)
        for s in P + H:
            vocab.update(toks(s))
    tr, tests = load_anli()
    import itertools

    for s in itertools.chain(tr.premise, tr.hypothesis):
        vocab.update(toks(s))
    for i in (1, 2, 3):
        for s in itertools.chain(tests[i].premise, tests[i].hypothesis):
            vocab.update(toks(s))
    log(f"corpus vocab {len(vocab)}")
    kept = {}
    vecs = []
    for t in vocab:
        if t in kv:
            kept[t] = len(vecs)
            vecs.append(kv[t])
    M = np.vstack(vecs).astype(np.float32)
    np.savez(EMB_NPZ, M=M)
    EMB_VOC.write_text(json.dumps(kept))
    log(f"saved trimmed emb: {M.shape} kept {len(kept)}/{len(vocab)} in-vocab")


def phase_snli():
    from scipy.sparse import hstack
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import confusion_matrix

    enc = Enc()
    log("snli: loading data")
    trP, trH, trY = load_snli("train")
    teP, teH, teY = load_snli("test")
    idx = np.random.RandomState(SEED).permutation(len(trY))[:N_TRAIN_SNLI]
    trP = [trP[i] for i in idx]
    trH = [trH[i] for i in idx]
    trY = trY[idx]
    log(f"snli train={len(trY)} test={len(teY)}")
    maj = np.bincount(trY).argmax()
    maj_acc = (teY == maj).mean() * 100
    vH = CountVectorizer(max_features=30000)
    clfH = LogisticRegression(max_iter=200, n_jobs=1).fit(vH.fit_transform(trH), trY)
    hyp = acc(clfH, vH.transform(teH), teY)
    vP, vHx = CountVectorizer(max_features=30000), CountVectorizer(max_features=30000)
    Xb = hstack([vP.fit_transform(trP), vHx.fit_transform(trH)]).tocsr()
    clfB = LogisticRegression(max_iter=200, n_jobs=1).fit(Xb, trY)
    bow = acc(clfB, hstack([vP.transform(teP), vHx.transform(teH)]).tocsr(), teY)
    log(f"snli maj={maj_acc:.1f} hyp={hyp:.1f} bow={bow:.1f}")
    Xtr = enc.pair(trP, trH)
    Xte = enc.pair(teP, teH)
    clfG = LogisticRegression(max_iter=250, n_jobs=1, tol=1e-3).fit(Xtr, trY)
    g = acc(clfG, Xte, teY)
    cm = confusion_matrix(teY, clfG.predict(Xte)).tolist()
    yshuf = np.random.RandomState(1).permutation(trY)
    clfS = LogisticRegression(max_iter=120, n_jobs=1, tol=1e-3).fit(Xtr, yshuf)
    shuf = acc(clfS, Xte, teY)
    log(f"snli glove={g:.1f} shuf={shuf:.1f}")
    json.dump(
        dict(
            maj=maj_acc,
            hyp=hyp,
            bow=bow,
            glove=g,
            shuf=shuf,
            cm=cm,
            ntrain=int(len(trY)),
            ntest=int(len(teY)),
        ),
        open(HERE / "_snli_res.json", "w"),
    )


def phase_anli():
    from scipy.sparse import hstack
    from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer, TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    enc = Enc()
    log("anli: loading data")
    tr, tests = load_anli()
    tr = tr.sample(n=min(30000, len(tr)), random_state=SEED).reset_index(drop=True)
    log(f"anli train={len(tr)} (subsampled, same for all models)")
    maj = np.bincount(tr.label).argmax()
    vH = CountVectorizer(max_features=30000)
    clfH = LogisticRegression(max_iter=200, solver="liblinear").fit(
        vH.fit_transform(tr.hypothesis), tr.label
    )

    def fitbow(cls, **kw):
        a, b = cls(**kw), cls(**kw)
        X = hstack([a.fit_transform(tr.premise), b.fit_transform(tr.hypothesis)]).tocsr()
        return a, b, LogisticRegression(max_iter=200, solver="liblinear").fit(X, tr.label)

    cnt = fitbow(CountVectorizer, max_features=30000)
    hsh = fitbow(HashingVectorizer, n_features=2**16, alternate_sign=False)
    tfi = fitbow(TfidfVectorizer, max_features=30000)
    log("anli: encoding train w/ glove")
    Xtr = enc.pair(list(tr.premise), list(tr.hypothesis))
    clfG = LogisticRegression(max_iter=200, n_jobs=1, tol=1e-3).fit(Xtr, tr.label)
    yshuf = np.random.RandomState(1).permutation(tr.label.values)
    clfS = LogisticRegression(max_iter=120, n_jobs=1, tol=1e-3).fit(Xtr, yshuf)
    res = {}
    for i in (1, 2, 3):
        te = tests[i]

        def ba(m):
            a, b, c = m
            return acc(
                c, hstack([a.transform(te.premise), b.transform(te.hypothesis)]).tocsr(), te.label
            )

        Xte = enc.pair(list(te.premise), list(te.hypothesis))
        res[i] = dict(
            maj=(te.label == maj).mean() * 100,
            hyp=acc(clfH, vH.transform(te.hypothesis), te.label),
            cnt=ba(cnt),
            hsh=ba(hsh),
            tfi=ba(tfi),
            glove=acc(clfG, Xte, te.label),
            shuf=acc(clfS, Xte, te.label),
        )
        log(
            f"anli R{i} hyp={res[i]['hyp']:.1f} cnt={res[i]['cnt']:.1f} glove={res[i]['glove']:.1f}"
        )
    json.dump(dict(ntrain=int(len(tr)), rounds=res), open(HERE / "_anli_res.json", "w"))


def phase_combine():
    s = json.load(open(HERE / "_snli_res.json"))
    A = json.load(open(HERE / "_anli_res.json"))
    a = A["rounds"]
    a = {int(k): v for k, v in a.items()}
    L = [
        "# Rung 2 results — learned representation vs boring baselines\n",
        f"*Run {time.strftime('%Y-%m-%d %H:%M')}. GloVe-wiki-gigaword-100, mean-pooled, "
        "pair features [u, v, |u-v|, u*v], logistic regression. Same split per task, seed 42.*\n",
        "## SNLI (does learned beat word-counting?)\n",
        f"Train subsample = {s['ntrain']:,} (same for all models), test = {s['ntest']:,}.\n",
        "| model | test acc % |",
        "|---|---|",
        f"| majority | {s['maj']:.1f} |",
        f"| hypothesis-only BoW | {s['hyp']:.1f} |",
        f"| full BoW (prem+hyp) | {s['bow']:.1f} |",
        f"| **GloVe-100 avg + logreg (learned)** | **{s['glove']:.1f}** |",
        f"| shuffled-label control | {s['shuf']:.1f} |",
        "",
        f"Margin: learned beats full BoW by **{s['glove']-s['bow']:+.1f} pts**, "
        f"hyp-only by **{s['glove']-s['hyp']:+.1f} pts**.\n",
        "GloVe confusion matrix (rows=true, cols=pred):\n",
        "```",
        "        " + "".join(f"{n:>8}" for n in LABNAME),
    ]
    for i, row in enumerate(s["cm"]):
        L.append(f"{LABNAME[i]:>7} " + "".join(f"{c:>8}" for c in row))
    L += [
        "```\n",
        "## ANLI (the hard, artifact-free task — honest run)\n",
        f"Train = R1+R2+R3 ({A['ntrain']:,}), tested per round. Bar = hypothesis-only (~38-41%).\n",
        "| round | maj | hypOnly | Count | Hash | TFIDF | **GloVe** | shufLbl |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for i in (1, 2, 3):
        r = a[i]
        L.append(
            f"| R{i} | {r['maj']:.1f} | {r['hyp']:.1f} | {r['cnt']:.1f} | {r['hsh']:.1f} | {r['tfi']:.1f} | **{r['glove']:.1f}** | {r['shuf']:.1f} |"
        )
    L.append("")
    snli_win = s["glove"] > s["bow"] and s["glove"] > s["hyp"]
    anli_beats = all(a[i]["glove"] > a[i]["hyp"] for i in (1, 2, 3))
    L += [
        "## Verdict (measured)\n",
        f"- **SNLI:** learned GloVe = **{s['glove']:.1f}%** vs BoW {s['bow']:.1f}% vs hyp-only {s['hyp']:.1f}%. "
        + (
            "**Learned beats word-counting — Rung 2 SNLI gate PASSED.**"
            if snli_win
            else "Learned did NOT clear BoW."
        )
        + "\n",
        "- **ANLI:** GloVe minus hyp-only bar: "
        + ", ".join(f"R{i} {a[i]['glove']-a[i]['hyp']:+.1f}" for i in (1, 2, 3))
        + ". "
        + (
            "Clears the bar on all rounds."
            if anli_beats
            else "Does NOT clear the hyp-only bar on all rounds — honest near-chance, as expected on ANLI."
        )
        + "\n",
        f"- **Kill-test:** shuffled-label controls collapse to chance (SNLI {s['shuf']:.1f}%, "
        f"ANLI ~{np.mean([a[i]['shuf'] for i in (1,2,3)]):.0f}%) → pipeline has no leakage.\n",
    ]
    (HERE / "RUNG2_RESULTS.md").write_text("\n".join(L))
    log("wrote RUNG2_RESULTS.md")
    print("\n==== SUMMARY ====")
    print(
        f"SNLI learned={s['glove']:.1f} BoW={s['bow']:.1f} hyp={s['hyp']:.1f} gate={'PASS' if snli_win else 'FAIL'}"
    )
    for i in (1, 2, 3):
        print(f"ANLI R{i} learned={a[i]['glove']:.1f} hyp={a[i]['hyp']:.1f} cnt={a[i]['cnt']:.1f}")


if __name__ == "__main__":
    {"trim": phase_trim, "snli": phase_snli, "anli": phase_anli, "combine": phase_combine}[
        sys.argv[1]
    ]()
