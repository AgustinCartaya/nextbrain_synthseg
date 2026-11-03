#!/usr/bin/env bash
# Usage: bash 02_train_v2.sh [gpus] [plans]
#   Launches folds 1-4 in parallel in the background (fold 0 is launched separately with 02_train.sh).
#   gpus: space-separated GPU ids, assigned round-robin to the folds (default "2 3" -> folds 1,3 on GPU 2, folds 2,4 on GPU 3)
#   plans: must match the planner used in 01 (nnUNetResEncUNetMPlans, nnUNetResEncUNetLPlans, nnUNetPlans)
# Add --c to resume interrupted runs:  EXTRA="--c" bash 02_train_v2.sh "2 3"
set -euo pipefail
cd "$(dirname "$0")"
source env.sh
read -ra GPUS <<< "${1:-2 3}"
PLANS=${2:-nnUNetResEncUNetMPlans}
TRAINER=${TRAINER:-nnUNetTrainer}
FOLDS=(0 1 2 3 4)

mkdir -p logs
for i in "${!FOLDS[@]}"; do
    FOLD=${FOLDS[$i]}
    GPU=${GPUS[$((i % ${#GPUS[@]}))]}
    LOG=logs/train_${PLANS}_${TRAINER}_fold${FOLD}.log
    # CUDA_VISIBLE_DEVICES=$GPU nohup nnUNetv2_train $DATASET_ID 3d_fullres $FOLD -p $PLANS -tr $TRAINER --npz ${EXTRA:-} \
    #     > $LOG 2>&1 &
    CUDA_VISIBLE_DEVICES=$GPU nohup nnUNetv2_train $DATASET_ID 2d $FOLD -p $PLANS -tr $TRAINER --npz ${EXTRA:-} \
        > $LOG 2>&1 &
    echo "fold $FOLD -> GPU $GPU (pid $!, log $LOG)"
done
