#!/usr/bin/env bash
# Segment new images with the final network (ensemble of the 5 folds + postprocessing).
# Usage: bash 06_predict.sh <net> <input_dir> <output_dir> [gpu] [folds]
#   input_dir: images named <case>_0000.nii.gz (any resolution/orientation: nnU-Net resamples
#              to 0.4 mm and the prediction comes back on the input grid)
#   folds:     default "0 1 2 3 4" (the 5-fold ensemble); "all" if trained with fold all
# Needs 04_crossval.sh (postprocessing.pkl). Output:
#   <output_dir>/indices/   raw network output, label indices 0..N-1
#   <output_dir>/pp/        after postprocessing (indices)
#   <output_dir>/*.nii.gz   final segmentation in raw label values (label_space.json)
set -euo pipefail
cd "$(dirname "$0")"
source env.sh "${1:?net required}"
IN=$(realpath "${2:?input_dir required}")
OUT=$(realpath -m "${3:?output_dir required}")
GPU=${4:-0}
read -ra FOLDS <<< "${5:-0 1 2 3 4}"
NP=${NP:-8}

RES="$nnUNet_results/$DATASET_NAME/${TRAINER}__${PLANS}__3d_fullres"
# postprocessing found by 04_crossval.sh for these folds, else the 5-fold one (fold "all" or a subset)
PP="$RES/crossval_results_folds_$(IFS=_; echo "${FOLDS[*]}")/postprocessing.pkl"
[ -f "$PP" ] || PP="$RES/crossval_results_folds_0_1_2_3_4/postprocessing.pkl"
[ -f "$PP" ] || { echo "no $PP: run 04_crossval.sh $NET first" >&2; exit 1; }

CUDA_VISIBLE_DEVICES=$GPU nnUNetv2_predict -i "$IN" -o "$OUT/indices" -d "$DATASET_ID" -c 3d_fullres \
    -p "$PLANS" -tr "$TRAINER" -f "${FOLDS[@]}" -npp "$NP" -nps "$NP"
nnUNetv2_apply_postprocessing -i "$OUT/indices" -o "$OUT/pp" -pp_pkl_file "$PP" -np "$NP" \
    -plans_json "$RES/plans.json" -dataset_json "$RES/dataset.json"
python map_labels.py --input "$OUT/pp" --output "$OUT"
