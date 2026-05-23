#!/bin/bash
# ======================================================================
# SIRA-CIRR: End-to-end pipeline for CIRR dataset
#
# Usage:
#     bash run_scripts/run_sira_cirr.sh              # Full pipeline
#     bash run_scripts/run_sira_cirr.sh precompute    # Step 1 only
#     bash run_scripts/run_sira_cirr.sh train          # Step 2 only
#     bash run_scripts/run_sira_cirr.sh eval           # Step 3 only
#     bash run_scripts/run_sira_cirr.sh submit         # Step 4 only
# ======================================================================

set -e

# ---- Configuration ----
CIRR_ROOT="/home/otw/chiennhm/data/CIRR"
CACHE_DIR="./cache/cirr_cls"
CLIP_MODEL="ViT-B/32"
DEVICE="cuda"

# Training hyperparameters
EPOCHS=20
BATCH_SIZE=256
LR=5e-4
D_SYNERGY=64
GATE_RANK=16
LAMBDA_RET=1.0
LAMBDA_SYN=0.3
LAMBDA_ORTH=0.1
SYN_EPOCH=5
ORTH_EPOCH=10

# Output
OUTPUT_DIR="./checkpoints/sira_cirr_v3"
SUBMISSION_DIR="./submissions/sira_cirr_v3"

# ---- Determine which steps to run ----
STEP=${1:-"all"}

# ---- Step 1: Pre-compute CLS embeddings ----
if [[ "$STEP" == "all" || "$STEP" == "precompute" ]]; then
    echo ""
    echo "===== Step 1: Pre-computing CLS-level CLIP embeddings ====="
    echo ""

    python cirr/precompute_cirr_cls.py \
        --cirr-root "$CIRR_ROOT" \
        --output-dir "$CACHE_DIR" \
        --split train \
        --clip-model "$CLIP_MODEL" \
        --batch-size "$BATCH_SIZE" \
        --device "$DEVICE"

    python cirr/precompute_cirr_cls.py \
        --cirr-root "$CIRR_ROOT" \
        --output-dir "$CACHE_DIR" \
        --split val \
        --clip-model "$CLIP_MODEL" \
        --batch-size "$BATCH_SIZE" \
        --device "$DEVICE"

    echo ""
    echo "===== Pre-computation complete ====="
    echo ""
fi

# ---- Step 2: Train SIRA-CIRR ----
if [[ "$STEP" == "all" || "$STEP" == "train" ]]; then
    echo ""
    echo "===== Step 2: Training SIRA-CIRR ====="
    echo ""

    python train_sira_cirr.py \
        --cache-dir "$CACHE_DIR" \
        --d-synergy "$D_SYNERGY" \
        --gate-rank "$GATE_RANK" \
        --epochs "$EPOCHS" \
        --batch-size "$BATCH_SIZE" \
        --lr "$LR" \
        --lambda-ret "$LAMBDA_RET" \
        --lambda-syn "$LAMBDA_SYN" \
        --lambda-orth "$LAMBDA_ORTH" \
        --syn-epoch "$SYN_EPOCH" \
        --orth-epoch "$ORTH_EPOCH" \
        --output-dir "$OUTPUT_DIR" \
        --device "$DEVICE"

    echo ""
    echo "===== Training complete ====="
    echo ""
fi

# ---- Step 3: Evaluate ----
if [[ "$STEP" == "all" || "$STEP" == "eval" ]]; then
    echo ""
    echo "===== Step 3: Evaluating SIRA-CIRR ====="
    echo ""

    python eval_sira_cirr.py \
        --checkpoint "$OUTPUT_DIR/sira_cirr_best.pt" \
        --cache-dir "$CACHE_DIR" \
        --split val \
        --device "$DEVICE" \
        --output "./results/sira_cirr_eval_val.json"

    echo ""
    echo "===== Evaluation complete ====="
    echo ""
fi

# ---- Step 4: Generate test1 submission ----
if [[ "$STEP" == "submit" ]]; then
    echo ""
    echo "===== Step 4: Generating CIRR test1 submission ====="
    echo ""

    python generate_sira_cirr_submission.py \
        --checkpoint "$OUTPUT_DIR/sira_cirr_best.pt" \
        --cirr-root "$CIRR_ROOT" \
        --clip-model "$CLIP_MODEL" \
        --output-dir "$SUBMISSION_DIR" \
        --device "$DEVICE"

    echo ""
    echo "===== Submission generated ====="
    echo "  Upload to: https://cirr.cecs.anu.edu.au/test_process/"
    echo ""
fi
