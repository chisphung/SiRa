#!/bin/bash
# Evaluate fine-tuned SIRA (projection strategy) on HL dataset.

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT_DIR="$(dirname "$(dirname "$DIR")")"
cd "$ROOT_DIR"

source /home/otw/chisphung/.venv/bin/activate
python Organized_Synergistic/eval_hl.py \
    --sira-ckpt checkpoints/sira/sira_best.pt \
    --clip-model ViT-B/32 \
    --device cuda:0
