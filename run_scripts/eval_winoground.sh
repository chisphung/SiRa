#!/bin/bash
# Evaluate SIRA on the Winoground benchmark.

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT_DIR="$(dirname "$(dirname "$DIR")")"
cd "$ROOT_DIR"

source /home/otw/chisphung/.venv/bin/activate
python Organized_Synergistic/eval_sira_winoground.py \
    --checkpoint checkpoints/sira/sira_best.pt \
    --clip-model ViT-B/32 \
    --device cuda:0 \
    --data-dir /home/otw/chisphung/Synergistic/winoground/data
