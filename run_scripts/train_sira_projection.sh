#!/bin/bash
# Train SIRA with projection head unfreezing.

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT_DIR="$(dirname "$(dirname "$DIR")")"
cd "$ROOT_DIR"

source /home/otw/chisphung/.venv/bin/activate
python Organized_Synergistic/train_sira.py \
    --dataset-type hl \
    --epochs 30 \
    --batch-size 256 \
    --device cuda:0 \
    --unfreeze-strategy projection \
    --output-dir ./checkpoints/sira
