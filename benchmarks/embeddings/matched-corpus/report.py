"""report.py — aggregate raw results into the comparison table.

- Refuses to mix results with different corpus/vocab hashes.
- Recomputes rho on the strict pair INTERSECTION covered by ALL model
  families (equal coverage), per dataset — reported next to own-coverage rho.
- Aggregates across seeds: mean ± population std (and per-seed values in JSON).
- collapse_v1 (legacy objective) and collapse_v2 (masked negatives) are
  separate families and never pooled.

Outputs: work/BASELINE_RESULTS.md, work/results/aggregate.json,
         work/results/aggregate.csv
"""

import csv
import glob
import os
import re
import statistics
from collections import defaultdict

from common import WORK, die, load_json, save_json, spearman

DATASETS = ["simlex999", "simlex_nouns", "ws353", "men3000"]
FAMILY_RE = re.compile(r"^(collapse_v1|collapse_v2|sgns|ppmi_svd)_seed(\d+)$")


def read_pairs(path):
    out = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            key = (r["word1"], r["word2"], float(r["gold"]))
            out[key] = float(r["model_cosine"]) if r["covered"] == "1" else None
    return out


def main():
    rdir = os.path.join(WORK, "results")
    runs = {}
    for p in sorted(glob.glob(os.path.join(rdir, "*.json"))):
        if os.path.basename(p) == "aggregate.json":
            continue
        s = load_json(p)
        m = FAMILY_RE.match(s["model"])
        if m:
            runs[s["model"]] = (m.group(1), int(m.group(2)), s)
    if not runs:
        die("no results in work/results — run evaluate.py first")

    # ---- lineage: every run must share one corpus+vocab hash
    hashes = {(s.get("meta", {}).get("corpus_sha256"),
               s.get("meta", {}).get("vocab_sha256")) for _, _, s in runs.values()}
    if len(hashes) > 1:
        die(f"mixed corpus/vocab hashes across results: {hashes} — not comparable")
    corpus_sha, vocab_sha = next(iter(hashes))

    # ---- equal-coverage: intersection of covered pairs across ALL runs
    shared_rho = defaultdict(dict)       # dataset -> tag -> rho on intersection
    shared_n = {}
    for ds in DATASETS:
        per_run = {}
        for tag in runs:
            path = os.path.join(rdir, f"{tag}.{ds}.pairs.csv")
            if os.path.exists(path):
                per_run[tag] = read_pairs(path)
        if not per_run:
            continue
        common_keys = set.intersection(
            *({k for k, v in pr.items() if v is not None} for pr in per_run.values()))
        shared_n[ds] = len(common_keys)
        keys = sorted(common_keys)
        for tag, pr in per_run.items():
            if len(keys) >= 10:
                shared_rho[ds][tag] = spearman([pr[k] for k in keys],
                                               [k[2] for k in keys])

    # ---- aggregate per family
    agg = defaultdict(lambda: defaultdict(lambda: {"own": [], "shared": [], "seeds": []}))
    for tag, (fam, seed, s) in runs.items():
        for ds, d in s["datasets"].items():
            cell = agg[fam][ds]
            if d["rho_own_coverage"] is not None:
                cell["own"].append(d["rho_own_coverage"])
            if tag in shared_rho.get(ds, {}):
                cell["shared"].append(shared_rho[ds][tag])
            cell["seeds"].append(seed)
            cell["coverage"] = f'{d["covered"]}/{d["total"]}'

    def ms(vals):
        if not vals:
            return "—"
        m = statistics.mean(vals)
        sd = statistics.pstdev(vals) if len(vals) > 1 else 0.0
        return f"{m:.4f} ± {sd:.4f}" if len(vals) > 1 else f"{m:.4f}"

    # ---- markdown
    fams = [f for f in ["collapse_v1", "collapse_v2", "sgns", "ppmi_svd"] if f in agg]
    lines = [
        "# Matched-baseline results",
        "",
        f"Corpus SHA-256: `{corpus_sha}`  ",
        f"Vocab SHA-256: `{vocab_sha}`  ",
        "Identical corpus, preprocessing, vocabulary, noun targets, ±5 window, "
        "256 dims. Tie-aware Spearman. `shared` = rho on the pair intersection "
        "covered by every model family (equal coverage); `own` = each model's "
        "full coverage. collapse_v1 = legacy objective (no false-negative mask); "
        "collapse_v2 = masked negatives. Never pooled.",
        "",
    ]
    for ds in DATASETS:
        if not any(ds in agg[f] for f in fams):
            continue
        lines += [f"## {ds}  (shared pairs: {shared_n.get(ds, 0)})", "",
                  "| model | seeds | rho (shared coverage) | rho (own coverage) | own coverage |",
                  "|---|---|---|---|---|"]
        for f in fams:
            c = agg[f].get(ds)
            if c:
                lines.append(f"| {f} | {len(set(c['seeds']))} | {ms(c['shared'])} "
                             f"| {ms(c['own'])} | {c.get('coverage', '—')} |")
        lines.append("")
    md = "\n".join(lines)
    with open(os.path.join(WORK, "BASELINE_RESULTS.md"), "w") as f:
        f.write(md)

    # ---- raw exports
    save_json(os.path.join(rdir, "aggregate.json"), {
        "corpus_sha256": corpus_sha, "vocab_sha256": vocab_sha,
        "shared_pairs": shared_n,
        "families": {f: {ds: {"own": c["own"], "shared": c["shared"],
                              "seeds": sorted(set(c["seeds"]))}
                         for ds, c in agg[f].items()} for f in fams}})
    with open(os.path.join(rdir, "aggregate.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["family", "dataset", "seed_count", "rho_shared_mean", "rho_shared_std",
                    "rho_own_mean", "rho_own_std", "shared_pairs"])
        for fam in fams:
            for ds, c in agg[fam].items():
                w.writerow([
                    fam, ds, len(set(c["seeds"])),
                    statistics.mean(c["shared"]) if c["shared"] else "",
                    statistics.pstdev(c["shared"]) if len(c["shared"]) > 1 else "",
                    statistics.mean(c["own"]) if c["own"] else "",
                    statistics.pstdev(c["own"]) if len(c["own"]) > 1 else "",
                    shared_n.get(ds, "")])
    print(md)
    print(f"\nwritten: work/BASELINE_RESULTS.md, results/aggregate.json, results/aggregate.csv")


if __name__ == "__main__":
    main()
