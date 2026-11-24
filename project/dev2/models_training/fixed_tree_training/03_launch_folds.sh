#!/usr/bin/env bash
# Launch several folds of one network in the background (survives closing the terminal).
# Usage: bash 03_launch_folds.sh <net> [gpus] [folds]
#   gpus:  space-separated GPU ids, assigned round-robin to the folds (default "0 1 2 3")
#   folds: space-separated folds (default "0 1 2 3 4")
# Examples:
#   bash 03_launch_folds.sh nxb16                      # 5 folds on GPUs 0-3
#   bash 03_launch_folds.sh nxbsupersynth "2 3" "0 1"  # folds 0,1 on GPUs 2,3
#   EXTRA="--c" bash 03_launch_folds.sh nxb16          # resume
#   (no --npz by default, see 02_train.sh)
# Follow: tail -f models/<net>/logs/train_*.log     Stop: pkill -f "nnUNetv2_train <dataset_id>"
set -euo pipefail
cd "$(dirname "$0")"
source env.sh "${1:?net required}"
read -ra GPUS <<< "${2:-0 1 2 3}"
read -ra FOLDS <<< "${3:-0 1 2 3 4}"

mkdir -p "$LOG_DIR"
for i in "${!FOLDS[@]}"; do
    FOLD=${FOLDS[$i]}
    GPU=${GPUS[$((i % ${#GPUS[@]}))]}
    LOG=$LOG_DIR/train_${PLANS}_${TRAINER}_fold${FOLD}.log
    CUDA_VISIBLE_DEVICES=$GPU nohup nnUNetv2_train "$DATASET_ID" 3d_fullres "$FOLD" -p "$PLANS" -tr "$TRAINER" ${EXTRA:-} \
        > "$LOG" 2>&1 &
    echo "$NET fold $FOLD -> GPU $GPU (pid $!, log $LOG)"
done
