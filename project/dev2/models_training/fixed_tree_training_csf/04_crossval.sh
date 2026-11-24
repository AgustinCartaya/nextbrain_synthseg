#!/usr/bin/env bash
# Cross-validation summary + postprocessing of one network, once its 5 folds have finished.
# Usage: bash 04_crossval.sh <net> [folds]      folds default "0 1 2 3 4"
# Writes, in models/<net>/nnUNet_results/<dataset>/:
#   inference_information.json, inference_instructions.txt   CV Dice and how to predict
#   <trainer>__<plans>__3d_fullres/crossval_results_folds_0_1_2_3_4/
#     summary.json                             Dice of the 1000 validation predictions
#     postprocessing.pkl                       postprocessing found to help (or nothing)
# Then remove the files only needed for ensembling: bash 05_cleanup.sh <net>
set -euo pipefail
cd "$(dirname "$0")"
source env.sh "${1:?net required}"
read -ra FOLDS <<< "${2:-0 1 2 3 4}"
NP=${NP:-32}

mkdir -p "$LOG_DIR"
# a single configuration: nothing to ensemble, so the .npz probabilities are not used
nnUNetv2_find_best_configuration "$DATASET_ID" -c 3d_fullres -p "$PLANS" -tr "$TRAINER" \
    -f "${FOLDS[@]}" -np "$NP" --disable_ensembling \
    2>&1 | tee "$LOG_DIR/crossval_${PLANS}_${TRAINER}.log"
