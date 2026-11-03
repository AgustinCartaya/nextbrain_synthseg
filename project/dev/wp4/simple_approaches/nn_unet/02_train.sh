#!/usr/bin/env bash
# Usage: bash 02_train.sh <fold|all> [gpu] [plans]
#   fold: 0-4 (each holds out one template, see nnUNet_preprocessed/*/splits_templates.json) or "all"
#   plans: must match the planner used in 01 (nnUNetResEncUNetMPlans, nnUNetResEncUNetLPlans, nnUNetPlans)
# Add --c to resume an interrupted run:  EXTRA="--c" bash 02_train.sh 0 1
set -euo pipefail
cd "$(dirname "$0")"
source env.sh
FOLD=${1:?fold required}
GPU=${2:-0}
PLANS=${3:-nnUNetResEncUNetMPlans}
TRAINER=${TRAINER:-nnUNetTrainer}

mkdir -p logs
CUDA_VISIBLE_DEVICES=$GPU nnUNetv2_train $DATASET_ID 3d_fullres $FOLD -p $PLANS -tr $TRAINER --npz ${EXTRA:-} \
    2>&1 | tee logs/train_${PLANS}_${TRAINER}_fold${FOLD}.log

