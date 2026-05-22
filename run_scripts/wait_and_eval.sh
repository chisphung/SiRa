#!/bin/bash
# Wait for training PID and run HL/Winoground/Diagnose evaluations.
# Use: ./wait_and_eval.sh <PID> [CKPT_PATH]

PID=$1
CKPT=${2:-"/home/otw/chisphung/Synergistic/checkpoints/sira/sira_best.pt"}
DEVICE="cuda:0"
CLIP_MODEL="ViT-B/32"

if [ -z "$PID" ]; then
    echo "Usage: $0 <PID> [CKPT_PATH]"
    exit 1
fi

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT_DIR="$(dirname "$(dirname "$DIR")")"
cd "$ROOT_DIR"

echo "Waiting for SIRA training process (PID: $PID) to complete..."

while kill -0 $PID 2>/dev/null; do
    sleep 10
done

echo "Training process $PID has completed."
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
