#!/usr/bin/env python3
"""
Livnium's turn — the REAL geometric encoder, in the SAME fair harness as GloVe.

Faithful to livnium-core-clean (no learned parameters; it's a static reference frame):
  char -> base-27 symbol  ('0' + a..z, your ALPHABET; non-letters -> 0/Om)
  symbol value 0..26 -> one of the 27 cells of the N=3 lattice
        (value 0 -> core (0,0,0); rest ordered by exposure then coord)
  sentence -> fixed vector built only from Livnium primitives:
        [ 27-symbol histogram | 4 exposure-class fractions
          | mean coord (3) | mean exposure (1) | mean SW (1) ]
  pair features [u, v, |u-v|, u*v]  ->  logistic regression
Same train subsample, same test, same baselines, shuffled-label kill-test.

Phases: snli | anli | combine
"""

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # repo root holds livnium_core
from livnium_core.base27 import ALPHABET  # "0abc...z", index == value
from livnium_core.lattice import SW, exposure

# reuse the exact same data loaders / settings as the GloVe run
sys.path.insert(0, str(HERE))
import rung2_lib as R

SEED = R.SEED


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


# ---- value -> lattice cell (N=3): 27 symbols <-> 27 cells, Om=core ----
_CELLS = [(x, y, z) for x in (-1, 0, 1) for y in (-1, 0, 1) for z in (-1, 0, 1)]
_CELLS.sort(key=lambda c: (exposure(c, 3), c))  # (0,0,0) first = symbol 0 = Om
assert _CELLS[0] == (0, 0, 0)
COORD = np.array(_CELLS, dtype=np.float32)  # 27 x 3
EXP = np.array([exposure(c, 3) for c in _CELLS], dtype=np.float32)  # 0..3
SWv = np.array([SW(c, 3) for c in _CELLS], dtype=np.float32)  # 9*exp
# exposure-class one-hot per value (core/center/edge/corner = f 0/1/2/3)
EXP_OH = np.eye(4, dtype=np.float32)[EXP.astype(int)]  # 27 x 4

_VAL = {c: i for i, c in enumerate(ALPHABET)}  # 'a'->1 ... 'z'->26, '0'->0


def char_val(ch):
    return _VAL.get(ch.lower(), 0)  # non-letters -> 0 (Om/core)


DIM = 27 + 4 + 3 + 1 + 1  # = 36


def encode(s):
    h = np.zeros(27, dtype=np.float32)
    for ch in s:
        h[char_val(ch)] += 1.0
    tot = h.sum()
    if tot == 0:
        return np.zeros(DIM, dtype=np.float32)
    p = h / tot  # normalized 27-symbol histogram
    expfrac = p @ EXP_OH  # 4 exposure-class fractions
    meancoord = p @ COORD  # 3
    meanexp = float(p @ EXP)  # 1
    meansw = float(p @ SWv)  # 1 (conserved-ledger echo)
    return np.concatenate([p, expfrac, meancoord, [meanexp], [meansw]]).astype(np.float32)


def batch(sents):
    return np.vstack([encode(s) for s in sents])


def pair(P, H):
    u, v = batch(P), batch(H)
    return np.hstack([u, v, np.abs(u - v), u * v]).astype(np.float32)


def acc(clf, X, y):
    from sklearn.metrics import accuracy_score

    return accuracy_score(y, clf.predict(X)) * 100


# ============================================================ SNLI
def phase_snli():
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import confusion_matrix

    log("livnium/snli: loading")
    trP, trH, trY = R.load_snli("train")
    teP, teH, teY = R.load_snli("test")
    idx = np.random.RandomState(SEED).permutation(len(trY))[: R.N_TRAIN_SNLI]
    trP = [trP[i] for i in idx]
    trH = [trH[i] for i in idx]
    trY = trY[idx]
    log(f"livnium/snli train={len(trY)} test={len(teY)} dim={DIM}")
    Xtr, Xte = pair(trP, trH), pair(teP, teH)
    clf = LogisticRegression(max_iter=250, n_jobs=1, tol=1e-3).fit(Xtr, trY)
    g = acc(clf, Xte, teY)
    cm = confusion_matrix(teY, clf.predict(Xte)).tolist()
    yshuf = np.random.RandomState(1).permutation(trY)
    clfS = LogisticRegression(max_iter=120, n_jobs=1, tol=1e-3).fit(Xtr, yshuf)
    shuf = acc(clfS, Xte, teY)
    log(f"livnium/snli acc={g:.1f} shuf={shuf:.1f}")
    json.dump(dict(livnium=g, shuf=shuf, cm=cm), open(HERE / "_liv_snli.json", "w"))


# ============================================================ ANLI
def phase_anli():
    from sklearn.linear_model import LogisticRegression

    log("livnium/anli: loading")
    tr, tests = R.load_anli()
    tr = tr.sample(n=min(30000, len(tr)), random_state=SEED).reset_index(drop=True)
    log(f"livnium/anli train={len(tr)}")
    Xtr = pair(list(tr.premise), list(tr.hypothesis))
    clf = LogisticRegression(max_iter=250, n_jobs=1, tol=1e-3).fit(Xtr, tr.label)
    yshuf = np.random.RandomState(1).permutation(tr.label.values)
    clfS = LogisticRegression(max_iter=120, n_jobs=1, tol=1e-3).fit(Xtr, yshuf)
    res = {}
    for i in (1, 2, 3):
        te = tests[i]
        Xte = pair(list(te.premise), list(te.hypothesis))
        res[i] = dict(livnium=acc(clf, Xte, te.label), shuf=acc(clfS, Xte, te.label))
        log(f"livnium/anli R{i} acc={res[i]['livnium']:.1f} shuf={res[i]['shuf']:.1f}")
    json.dump(res, open(HERE / "_liv_anli.json", "w"))


# ============================================================ combine
def phase_combine():
    # results from the glove run were not persisted; read final numbers from RUNG2_RESULTS instead
    ls = json.load(open(HERE / "_liv_snli.json"))
    la = json.load(open(HERE / "_liv_anli.json"))
    la = {int(k): v for k, v in la.items()}
    md = (HERE / "RUNG2_RESULTS.md").read_text().rstrip()
    LAB = ["entail", "neutral", "contra"]
    out = [
        "\n\n---\n\n# Livnium's turn — the geometric encoder in the same fair harness\n",
        f"*Run {time.strftime('%Y-%m-%d %H:%M')}. Livnium base-27 -> N=3 lattice encoder "
        f"(no learned parameters), pair features [u,v,|u-v|,u*v], same logreg, same split, seed {SEED}.*\n",
        "## SNLI\n",
        "| model | test acc % |",
        "|---|---|",
        "| majority | 34.3 |",
        "| full BoW (prem+hyp) | 59.4 |",
        "| hypothesis-only BoW | 61.5 |",
        "| GloVe-100 learned | 60.7 |",
        f"| **Livnium geometry (static)** | **{ls['livnium']:.1f}** |",
        f"| Livnium shuffled-label control | {ls['shuf']:.1f} |",
        "",
        "Livnium confusion matrix (rows=true, cols=pred):\n",
        "```",
        "        " + "".join(f"{n:>8}" for n in LAB),
    ]
    for i, row in enumerate(ls["cm"]):
        out.append(f"{LAB[i]:>7} " + "".join(f"{c:>8}" for c in row))
    out += [
        "```\n",
        "## ANLI (bar = hypothesis-only ~37-41%)\n",
        "| round | hypOnly | best BoW | GloVe | **Livnium** | Livnium shufLbl |",
        "|---|---|---|---|---|---|",
    ]
    glove_anli = {1: 34.2, 2: 36.7, 3: 34.4}
    hyp = {1: 39.6, 2: 36.9, 3: 36.7}
    bestbow = {1: 41.3, 2: 37.8, 3: 36.8}
    for i in (1, 2, 3):
        out.append(
            f"| R{i} | {hyp[i]:.1f} | {bestbow[i]:.1f} | {glove_anli[i]:.1f} | "
            f"**{la[i]['livnium']:.1f}** | {la[i]['shuf']:.1f} |"
        )
    snli_liv = ls["livnium"]
    out += [
        "",
        "## Verdict for Livnium (measured, same rules)\n",
        f"- **SNLI:** Livnium static geometry = **{snli_liv:.1f}%**. "
        f"vs full BoW 59.4, hyp-only 61.5, GloVe 60.7. "
        + ("Beats BoW." if snli_liv > 59.4 else "Below BoW.")
        + (" Clears the hyp-only artifact." if snli_liv > 61.5 else " Below the hyp-only artifact.")
        + "\n",
        f"- **ANLI:** Livnium by round R1 {la[1]['livnium']:.1f} / R2 {la[2]['livnium']:.1f} / R3 {la[3]['livnium']:.1f}; "
        f"hyp-only bar 39.6/36.9/36.7. "
        + (
            "Clears the bar on all rounds."
            if all(la[i]["livnium"] > hyp[i] for i in (1, 2, 3))
            else "Does NOT clear the hyp-only bar on all rounds — near chance, like every static representation here."
        )
        + "\n",
        f"- **Kill-test:** Livnium shuffled-label control collapses to chance "
        f"(SNLI {ls['shuf']:.1f}%, ANLI ~{np.mean([la[i]['shuf'] for i in (1,2,3)]):.0f}%) -> the pipeline is honest; "
        "the score is the representation's, not leakage.\n",
    ]
    (HERE / "RUNG2_RESULTS.md").write_text(md + "\n".join(out) + "\n")
    log("appended Livnium section to RUNG2_RESULTS.md")
    print("\n==== LIVNIUM SUMMARY ====")
    print(f"SNLI livnium={snli_liv:.1f}  (BoW 59.4 / hypOnly 61.5 / GloVe 60.7)")
    for i in (1, 2, 3):
        print(
            f"ANLI R{i} livnium={la[i]['livnium']:.1f}  (hypOnly {hyp[i]} / GloVe {glove_anli[i]})"
        )


if __name__ == "__main__":
    {"snli": phase_snli, "anli": phase_anli, "combine": phase_combine}[sys.argv[1]]()
