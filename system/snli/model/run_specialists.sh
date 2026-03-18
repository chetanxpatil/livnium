#!/usr/bin/env bash
# =============================================================================
# run_specialists.sh — 3-Specialist Training Pipeline for Livnium
# =============================================================================
#
# Stage 1: Train 3 binary specialists on frozen BERT embeddings
#   Each specialist learns a clean 2-class geometry for one NLI label.
#   The primary anchor (anchor_entail) ends up shaped for that label.
#
# Stage 2: Train the full 3-class model, seeding its 3 anchors from
#   the 3 specialists' primary anchors. Better starting geometry =
#   faster convergence and higher accuracy.
#
# Usage:
#   Edit the CONFIG section below, then:
#     chmod +x run_specialists.sh
#     ./run_specialists.sh
#
# Expected total time: ~3 × 15min (specialists) + ~60min (final) ≈ 2h
# =============================================================================

set -e

# ── CONFIG ────────────────────────────────────────────────────────────────────
SNLI_TRAIN="/path/to/data/snli/snli_1.0_train.jsonl"
SNLI_DEV="/path/to/data/snli/snli_1.0_dev.jsonl"
BERT_GGUF="/path/to/bert-base-uncased-Q8_0.gguf"
EMBED_CACHE="/path/to/bert_snli_cache.pt"   # pre-computed by precompute_embeddings.py
OUTPUT_BASE="/path/to/outputs"

# Shared hyperparameters
BATCH=64
SPEC_EPOCHS=5         # specialists need fewer epochs (simpler binary task)
FULL_EPOCHS=15
SPEC_LR=3e-4
FULL_LR=3e-4
NUM_LAYERS=6
# ─────────────────────────────────────────────────────────────────────────────

SPEC_DIR="$OUTPUT_BASE/specialists"
FULL_DIR="$OUTPUT_BASE/full_model"
mkdir -p "$SPEC_DIR/entailment" "$SPEC_DIR/contradiction" "$SPEC_DIR/neutral" "$FULL_DIR"

TRAIN_PY="$(dirname "$0")/train.py"

# ── Stage 1: Train specialists ────────────────────────────────────────────────
echo "======================================================================"
echo " Stage 1a — Entailment specialist"
echo "======================================================================"
python3 "$TRAIN_PY" \
    --snli-train "$SNLI_TRAIN" \
    --snli-dev   "$SNLI_DEV" \
    --encoder-type llamacpp \
    --llamacpp-model "$BERT_GGUF" \
    --embed-cache "$EMBED_CACHE" \
    --binary-label entailment \
    --epochs "$SPEC_EPOCHS" \
    --lr "$SPEC_LR" \
    --batch-size "$BATCH" \
    --num-layers "$NUM_LAYERS" \
    --barrier 0.38 \
    --output-dir "$SPEC_DIR/entailment"

echo ""
echo "======================================================================"
echo " Stage 1b — Contradiction specialist"
echo "======================================================================"
python3 "$TRAIN_PY" \
    --snli-train "$SNLI_TRAIN" \
    --snli-dev   "$SNLI_DEV" \
    --encoder-type llamacpp \
    --llamacpp-model "$BERT_GGUF" \
    --embed-cache "$EMBED_CACHE" \
    --binary-label contradiction \
    --epochs "$SPEC_EPOCHS" \
    --lr "$SPEC_LR" \
    --batch-size "$BATCH" \
    --num-layers "$NUM_LAYERS" \
    --barrier 0.38 \
    --output-dir "$SPEC_DIR/contradiction"

echo ""
echo "======================================================================"
echo " Stage 1c — Neutral specialist"
echo "======================================================================"
python3 "$TRAIN_PY" \
    --snli-train "$SNLI_TRAIN" \
    --snli-dev   "$SNLI_DEV" \
    --encoder-type llamacpp \
    --llamacpp-model "$BERT_GGUF" \
    --embed-cache "$EMBED_CACHE" \
    --binary-label neutral \
    --epochs "$SPEC_EPOCHS" \
    --lr "$SPEC_LR" \
    --batch-size "$BATCH" \
    --num-layers "$NUM_LAYERS" \
    --barrier 0.38 \
    --output-dir "$SPEC_DIR/neutral"

echo ""
echo "======================================================================"
echo " Stage 2 — Full 3-class model, seeded from specialists"
echo "======================================================================"
# --init-from-specialists expects: E_ckpt,C_ckpt,N_ckpt (in that order)
python3 "$TRAIN_PY" \
    --snli-train "$SNLI_TRAIN" \
    --snli-dev   "$SNLI_DEV" \
    --encoder-type llamacpp \
    --llamacpp-model "$BERT_GGUF" \
    --embed-cache "$EMBED_CACHE" \
    --init-from-specialists \
        "$SPEC_DIR/entailment/best_model.pt,$SPEC_DIR/contradiction/best_model.pt,$SPEC_DIR/neutral/best_model.pt" \
    --epochs "$FULL_EPOCHS" \
    --lr "$FULL_LR" \
    --batch-size "$BATCH" \
    --num-layers "$NUM_LAYERS" \
    --barrier 0.38 \
    --lambda-rep 0.1 \
    --margin-rep 0.3 \
    --output-dir "$FULL_DIR"

echo ""
echo "======================================================================"
echo " All done!  Best model: $FULL_DIR/best_model.pt"
echo "======================================================================"
