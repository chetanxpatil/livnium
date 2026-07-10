"""common.py — shared plumbing for the matched-baseline harness.

Design rule: every artifact records the SHA-256 of the frozen corpus and the
shared vocab it was built from. If two results carry different hashes they are
not comparable, and report.py refuses to put them in the same table.
"""

import hashlib
import json
import os
import platform
import subprocess
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.join(HERE, "work")          # corpus, vocab, models, results
DATA = os.path.join(HERE, "data")          # cached eval datasets

# word-similarity datasets, "w1 w2 score" per line (Faruqui eval-word-vectors)
_RAW = "https://raw.githubusercontent.com/mfaruqui/eval-word-vectors/master/data/word-sim"
EVAL_SETS = {
    "simlex999": f"{_RAW}/EN-SIMLEX-999.txt",
    "ws353": f"{_RAW}/EN-WS-353-ALL.txt",
    "men3000": f"{_RAW}/EN-MEN-TR-3k.txt",
}
# noun-only SimLex (official file has POS) — comparable to the published 0.362
SIMLEX_ZIP = "https://fh295.github.io/SimLex-999.zip"


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                return h.hexdigest()
            h.update(b)


def save_json(path, obj):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def corpus_lines(corpus_txt):
    """Yield token lists from the frozen corpus. NO further cleaning anywhere."""
    with open(corpus_txt, encoding="utf-8") as f:
        for line in f:
            toks = line.split()
            if toks:
                yield toks


def load_vocab(work=WORK):
    v = load_json(os.path.join(work, "vocab.json"))
    v["stoi"] = {w: int(i) for w, i in v["stoi"].items()}
    return v


def stamp(work=WORK):
    """Lineage stamp every artifact must carry."""
    man = load_json(os.path.join(work, "corpus_manifest.json"))
    voc = load_json(os.path.join(work, "vocab.json"))
    return {"corpus_sha256": man["sha256"], "vocab_sha256": voc["vocab_sha256"]}


def caffeinate():
    """On macOS, keep the machine awake for the life of this process."""
    if platform.system() == "Darwin" and not os.environ.get("_CAFFEINATED"):
        try:
            subprocess.Popen(["caffeinate", "-i", "-w", str(os.getpid())])
            os.environ["_CAFFEINATED"] = "1"
            print("[caffeinate -i armed for this run]", flush=True)
        except FileNotFoundError:
            pass


def fetch(url, dest):
    if not os.path.exists(dest):
        os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
        print(f"downloading {url}", flush=True)
        urllib.request.urlretrieve(url, dest)
    return dest


def load_eval_set(name):
    """Return list of (w1, w2, gold) for a dataset name in EVAL_SETS or
    'simlex_nouns' (official SimLex, POS == N — matches chat/embed_eval.py)."""
    if name == "simlex_nouns":
        import zipfile
        z = fetch(SIMLEX_ZIP, os.path.join(DATA, "simlex.zip"))
        d = os.path.join(DATA, "SimLex-999")
        if not os.path.exists(os.path.join(d, "SimLex-999.txt")):
            with zipfile.ZipFile(z) as f:
                f.extractall(DATA)
        pairs = []
        with open(os.path.join(d, "SimLex-999.txt"), encoding="utf-8") as f:
            head = f.readline().rstrip("\n").split("\t")
            i1, i2, ip, isim = (head.index(c) for c in ("word1", "word2", "POS", "SimLex999"))
            for line in f:
                p = line.rstrip("\n").split("\t")
                if p[ip] == "N":
                    pairs.append((p[i1].lower(), p[i2].lower(), float(p[isim])))
        return pairs
    path = fetch(EVAL_SETS[name], os.path.join(DATA, f"{name}.txt"))
    pairs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            p = line.split()
            if len(p) >= 3:
                pairs.append((p[0].lower(), p[1].lower(), float(p[2])))
    return pairs


def spearman(a, b):
    """Tie-aware Spearman (the only correlation this harness reports)."""
    from scipy.stats import spearmanr
    return float(spearmanr(a, b).statistic)


def die(msg):
    sys.exit(f"error: {msg}")
