#!/usr/bin/env bash
# Usage: bash 02_train.sh <fold|all> [gpu] [plans]
#   fold: 0-4 (fold i: the generator never sees template i, validation = images of template i,
#         see nnUNet_preprocessed/*/nxgen.json) or "all" (every template, validation on all of them)
#   gpu: one id ("1"), or two ids ("1,2": training on the first, set NXGEN_DEVICE=cuda:1 to
#        generate on the second)
#   plans: must match the planner used in 01 (nnUNetResEncUNetMPlans, nnUNetResEncUNetLPlans, nnUNetPlans)
# Resume an interrupted run:   EXTRA="--c" bash 02_train.sh 0 1
# Short test run:              NXGEN_NUM_EPOCHS=2 NXGEN_ITERS_PER_EPOCH=10 bash 02_train.sh 0 1
set -euo pipefail
cd "$(dirname "$0")"
source env.sh
FOLD=${1:?fold required}
GPU=${2:-0}
PLANS=${3:-nnUNetResEncUNetMPlans}
TRAINER=${TRAINER:-nnUNetTrainerNxGen}

mkdir -p logs
CUDA_VISIBLE_DEVICES=$GPU nnUNetv2_train $DATASET_ID 3d_fullres $FOLD -p $PLANS -tr $TRAINER --npz ${EXTRA:-} \
    2>&1 | tee logs/train_${PLANS}_${TRAINER}_fold${FOLD}.log
