#!/bin/bash
# Run SIRA gate-stuck diagnostic tool on a checkpoint.

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT_DIR="$(dirname "$(dirname "$DIR")")"
cd "$ROOT_DIR"

source /home/otw/chisphung/.venv/bin/activate
python Organized_Synergistic/diagnose.py \
    --checkpoint checkpoints/sira/sira_best.pt \
    --winoground-data /home/otw/chisphung/Synergistic/winoground/data \
    --device cuda:0
