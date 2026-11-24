#!/usr/bin/env bash
# Free the disk used by the validation probabilities of one network, after 04_crossval.sh.
# Usage: bash 05_cleanup.sh <net>
# Removes fold_*/validation/*.npz and *.pkl: softmax probabilities saved by --npz, only
# used to ensemble several configurations (~2.4 GB per case for 17 labels). Kept: checkpoints,
# validation predictions (*.nii.gz) and summary.json, crossval_results_*, postprocessing.pkl,
# i.e. everything nnUNetv2_predict and the evaluation need.
set -euo pipefail
cd "$(dirname "$0")"
source env.sh "${1:?net required}"
RES="$nnUNet_results/$DATASET_NAME/${TRAINER}__${PLANS}__3d_fullres"

ls -d "$RES"/crossval_results_folds_* > /dev/null 2>&1 \
    || { echo "no crossval_results_folds_* in $RES: run 04_crossval.sh $NET first" >&2; exit 1; }
for f in "$RES"/fold_*/validation; do
    [ -f "$f/summary.json" ] || { echo "$f has no summary.json (fold not finished?)" >&2; exit 1; }
done
echo "before: $(du -sh "$RES" | cut -f1)"
find "$RES"/fold_*/validation -maxdepth 1 \( -name '*.npz' -o -name '*.pkl' \) -delete
echo "after:  $(du -sh "$RES" | cut -f1)"
