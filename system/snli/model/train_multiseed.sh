#!/usr/bin/env bash
# ============================================================================
# train_multiseed.sh  —  True multi-seed training from random init
# ============================================================================
# Trains 3 independent Livnium-BERT runs from scratch (no warm start).
# Each run uses a different random seed, so BERT fine-tuning, anchor
# initialization, and MLP weights are all independently initialized.
#
# After training, automatically runs:
#   1. test_gradient_collapse.py  (beta sweep, full dev set)
#   2. ablation_experiment.py     (5-condition, lambda sweep)
#
# Output layout:
#   pretrained/multiseed/seed42/
#   pretrained/multiseed/seed123/
#   pretrained/multiseed/seed999/
#       best_model.pt
#       grad_v_results.txt
#       ablation_results.json
#
# Usage:
#   cd system/snli/model
#   bash train_multiseed.sh
#
# Requirements:
#   - SNLI data at: ../../../data/snli/snli_1.0_{train,dev}.jsonl
#   - Sufficient disk space (~1GB per run)
#   - ~4-6 hours per seed on MPS/GPU (less on CUDA)
# ============================================================================

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$SCRIPT_DIR/../../.."
DATA_DIR="$REPO_ROOT/data/snli"
PRETRAINED_DIR="$REPO_ROOT/pretrained/multiseed"

SEEDS=(42 123 999)
EPOCHS=10
BATCH_SIZE=32
LR=2e-5
DIM=768
LAYERS=6

TRAIN_DATA="$DATA_DIR/snli_1.0_train.jsonl"
DEV_DATA="$DATA_DIR/snli_1.0_dev.jsonl"

mkdir -p "$PRETRAINED_DIR"

echo "============================================================"
echo "  Livnium True Multi-Seed Training"
echo "  Seeds: ${SEEDS[*]}"
echo "  Epochs: $EPOCHS   Batch: $BATCH_SIZE   LR: $LR"
echo "  Output: $PRETRAINED_DIR"
echo "============================================================"
echo ""

# ── Training loop ─────────────────────────────────────────────────────────────
for SEED in "${SEEDS[@]}"; do
    OUT_DIR="$PRETRAINED_DIR/seed${SEED}"
    mkdir -p "$OUT_DIR"

    echo "------------------------------------------------------------"
    echo "  Training seed=${SEED}  →  $OUT_DIR"
    echo "------------------------------------------------------------"

    # Train from scratch: NO --resume, NO warm start
    python3 "$SCRIPT_DIR/train.py" \
        --train-data   "$TRAIN_DATA" \
        --dev-data     "$DEV_DATA" \
        --save-dir     "$OUT_DIR" \
        --epochs       "$EPOCHS" \
        --batch-size   "$BATCH_SIZE" \
        --lr           "$LR" \
        --dim          "$DIM" \
        --num-layers   "$LAYERS" \
        --seed         "$SEED" \
        2>&1 | tee "$OUT_DIR/train.log"

    # Find best checkpoint
    BEST_CKPT=$(ls "$OUT_DIR"/best_model.pt 2>/dev/null \
                || ls "$OUT_DIR"/epoch_*.pt 2>/dev/null | sort | tail -1)

    if [ -z "$BEST_CKPT" ]; then
        echo "  ERROR: No checkpoint found in $OUT_DIR — skipping eval."
        continue
    fi

    echo ""
    echo "  Best checkpoint: $BEST_CKPT"
    echo ""

    # ── Grad-V evaluation (canonical beta sweep) ───────────────────────────────
    echo "  Running grad-V beta sweep (full dev set)..."
    python3 "$SCRIPT_DIR/test_gradient_collapse.py" \
        --checkpoint "$BEST_CKPT" \
        --snli-dev   "$DEV_DATA" \
        --n-samples  -1 \
        --beta-sweep \
        2>&1 | tee "$OUT_DIR/grad_v_results.txt"

    echo ""
    echo "  Running 5-condition ablation..."
    python3 "$SCRIPT_DIR/ablation_experiment.py" \
        --checkpoint  "$BEST_CKPT" \
        --snli-dev    "$DEV_DATA" \
        --n-samples   -1 \
        --beta        2.0 \
        --alpha       0.05 \
        --svd-rank    16 \
        --lambda-sweep \
        2>&1 | tee "$OUT_DIR/ablation_results.txt"

    echo ""
    echo "  ✓ Seed $SEED complete."
    echo ""
done

# ── Summary ───────────────────────────────────────────────────────────────────
echo "============================================================"
echo "  MULTI-SEED SUMMARY"
echo "============================================================"
echo ""
for SEED in "${SEEDS[@]}"; do
    OUT_DIR="$PRETRAINED_DIR/seed${SEED}"
    echo "  Seed $SEED:"
    if [ -f "$OUT_DIR/grad_v_results.txt" ]; then
        # Pull out Full and best Grad-V lines from the results
        grep -E "^  (full|Best grad-V)" "$OUT_DIR/grad_v_results.txt" 2>/dev/null \
            | head -4 | sed 's/^/    /'
    else
        echo "    (no results)"
    fi
    echo ""
done

echo "  All ablation results saved to:"
for SEED in "${SEEDS[@]}"; do
    echo "    $PRETRAINED_DIR/seed${SEED}/ablation_results.json"
done
echo ""
echo "  Done."
