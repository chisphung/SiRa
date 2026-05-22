#!/bin/bash
# Run HL/Winoground/Diagnose evaluations immediately.
# Use: ./eval_all.sh [CKPT_PATH]

CKPT=${1:-"/home/otw/chisphung/Synergistic/checkpoints/sira/sira_best.pt"}
DEVICE="cuda:0"
CLIP_MODEL="ViT-B/32"

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT_DIR="$(dirname "$(dirname "$DIR")")"
cd "$ROOT_DIR"

echo "Activating virtual environment..."
source /home/otw/chisphung/.venv/bin/activate

echo "========================================================================"
echo " 1. Running Winoground Evaluation"
echo "========================================================================"
python Organized_Synergistic/eval_sira_winoground.py --checkpoint "$CKPT" --clip-model "$CLIP_MODEL" --device "$DEVICE" --data-dir /home/otw/chisphung/Synergistic/winoground/data

echo "========================================================================"
echo " 2. Running HL Evaluation"
echo "========================================================================"
python Organized_Synergistic/eval_hl.py --sira-ckpt "$CKPT" --clip-model "$CLIP_MODEL" --device "$DEVICE"

echo "========================================================================"
echo " 3. Running Diagnostic Script"
echo "========================================================================"
python Organized_Synergistic/diagnose.py --checkpoint "$CKPT" --winoground-data /home/otw/chisphung/Synergistic/winoground/data --device "$DEVICE"
