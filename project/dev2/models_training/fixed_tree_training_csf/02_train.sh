#!/usr/bin/env bash
# Train one fold of one network in the foreground.
# Usage: bash 02_train.sh <net> <fold|all> [gpu]
#   fold: 0-4 holds out one template (see models/<net>/nnUNet_preprocessed/<dataset>/splits_templates.json),
#         "all" trains on every image
# Resume an interrupted run:  EXTRA="--c" bash 02_train.sh nxb18 0 1
# No --npz by default: the validation probabilities are only needed to ensemble several
# configurations and take ~2.4 GB/case with 17 labels (~13x more with 231). Add EXTRA="--npz" if needed.
set -euo pipefail
cd "$(dirname "$0")"
source env.sh "${1:?net required}"
FOLD=${2:?fold required}
GPU=${3:-0}

mkdir -p "$LOG_DIR"
CUDA_VISIBLE_DEVICES=$GPU nnUNetv2_train "$DATASET_ID" 3d_fullres "$FOLD" -p "$PLANS" -tr "$TRAINER" ${EXTRA:-} \
    2>&1 | tee "$LOG_DIR/train_${PLANS}_${TRAINER}_fold${FOLD}.log"
