#!/usr/bin/env bash
# run_all.sh — the whole matched-baseline experiment, resumable.
#
#   ./run_all.sh /path/to/enwiki-...multistream.xml.bz2
#
# Every stage skips itself if its output already exists, so re-running after
# an interruption continues where it stopped (train_collapse also resumes
# mid-pass from its own checkpoint). On macOS the process keeps the machine
# awake via caffeinate.
set -euo pipefail
cd "$(dirname "$0")"

DATA="${1:?usage: ./run_all.sh <wiki dump or txt corpus>}"
SEEDS="${SEEDS:-0 1 2 3 4}"           # >= 5 seeds where randomness applies
MAX_LINES="${MAX_LINES:-5000000}"     # the published subset size
MAX_OCC="${MAX_OCC:-0}"               # occurrence budget for collapse (0 = full pass)

if [[ "$(uname)" == "Darwin" && -z "${_CAFFEINATED:-}" ]]; then
  export _CAFFEINATED=1
  exec caffeinate -i "$0" "$@"
fi

echo "== stage 1: freeze corpus =="
[[ -f work/corpus.txt ]] || python3 freeze_corpus.py --data "$DATA" --max-lines "$MAX_LINES"

echo "== stage 2: shared vocabulary =="
[[ -f work/vocab.json ]] || python3 build_vocab.py

echo "== stage 3: train (resumable; skips finished models) =="
for s in $SEEDS; do
  python3 train_collapse.py --variant v1 --seed "$s" --max-occ "$MAX_OCC"
  python3 train_collapse.py --variant v2 --seed "$s" --max-occ "$MAX_OCC"
  python3 train_sgns.py     --seed "$s"
  python3 train_ppmi_svd.py --seed "$s"
done

echo "== stage 4: evaluate =="
for m in work/models/*.npz; do
  python3 evaluate.py --model "$m"
done

echo "== stage 5: report =="
python3 report.py
