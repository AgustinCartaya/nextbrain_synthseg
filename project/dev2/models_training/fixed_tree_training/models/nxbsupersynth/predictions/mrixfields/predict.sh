#!/usr/bin/env bash
# Segment the MRIxFields test images with the final nxbsupersynth network: ensemble of the 5 folds
# (3d_fullres, ResEnc M, checkpoint_final), optionally followed by the postprocessing chosen by
# cross-validation (04_crossval.sh; for nxbsupersynth it only improves a few labels by <0.0002 Dice, so it is off by
# default). The output mirrors the input tree:
#
#   data_shared/test/mrixfields/images/<contrast>/<field>/<case>_0000.nii.gz
#   -> models/nxbsupersynth/predictions/mrixfields/<contrast>/<field>/<case>.nii.gz   (raw label values)
#
# Usage (runs in the background, survives closing the terminal):
#   bash predict.sh                  # GPUs 0 1 2 3, one worker per GPU
#   bash predict.sh "1 3"            # only GPUs 1 and 3
#   bash predict.sh --postprocessing # also apply the postprocessing (any position, e.g. "1 3" --postprocessing)
#   bash predict.sh --fields 7T --contrasts T1w,FLAIR   # only some field strengths / contrasts
#                                    (comma-separated; default: every <contrast>/<field> folder)
#   NPP=2 NPS=2 bash predict.sh      # fewer CPU workers per GPU (default 4 + 4)
#   FOLDS="0" bash predict.sh        # quick look with a single fold instead of the ensemble
# Follow: tail -f logs/predict_*.log logs/gpu*.log     Stop: pkill -f "mrixfields/predict.sh"; pkill -f "nnUNetv2_predict -i .*mrixfields"
# Resume after a stop or a crash: run the same command again (finished cases are skipped).
#
# Each GPU worker predicts its share (-num_parts/-part_id) of every folder into
# _work/indices/; when all are done, the mapping to raw label values (after the postprocessing
# into _work/pp/ with --postprocessing) gives the final files. _work/ can be deleted at the end.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
TRAIN_ROOT=$(cd "$HERE/../../../.." && pwd)                 # fixed_tree_training
IN=${IN:-$TRAIN_ROOT/data_shared/test/mrixfields/images}
OUT=${OUT:-$HERE}
WORK=$OUT/_work
LOGS=$OUT/logs
mkdir -p "$LOGS"

# ---- relaunch itself in the background
if [ -z "${_PREDICT_BG:-}" ]; then
    log="$LOGS/predict_$(date +%Y%m%d_%H%M%S).log"
    _PREDICT_BG=1 setsid nohup bash "$0" "$@" > "$log" 2>&1 < /dev/null &
    echo "PID $! -> $log"
    exit 0
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate nnunet
source "$TRAIN_ROOT/env.sh" nxbsupersynth                              # nnUNet_* paths, DATASET_*, PLANS, TRAINER

POSTPROCESSING=0
SEL_FIELDS=""          # e.g. 3T,7T
SEL_CONTRASTS=""       # e.g. T1w,T2w,FLAIR
POS=()
while [ $# -gt 0 ]; do
    case "$1" in
        --postprocessing) POSTPROCESSING=1 ;;
        --fields)         SEL_FIELDS=${2:?--fields needs a value, e.g. 3T,7T}; shift ;;
        --fields=*)       SEL_FIELDS=${1#*=} ;;
        --contrasts)      SEL_CONTRASTS=${2:?--contrasts needs a value, e.g. T1w,FLAIR}; shift ;;
        --contrasts=*)    SEL_CONTRASTS=${1#*=} ;;
        -*) echo "unknown option $1" >&2; exit 1 ;;
        *) POS+=("$1") ;;
    esac
    shift
done
read -ra GPUS <<< "${POS[0]:-0 1 2 3}"
read -ra FOLD_LIST <<< "${FOLDS:-0 1 2 3 4}"
NPP=${NPP:-4}
NPS=${NPS:-4}
RES="$nnUNet_results/$DATASET_NAME/${TRAINER}__${PLANS}__3d_fullres"
PP="$RES/crossval_results_folds_0_1_2_3_4/postprocessing.pkl"
if [ $POSTPROCESSING -eq 1 ] && [ ! -f "$PP" ]; then echo "missing $PP: run 04_crossval.sh nxbsupersynth first" >&2; exit 1; fi

# input folders = every <contrast>/<field> directory holding *_0000.nii.gz, relative to IN
mapfile -t ALL_DIRS < <(cd "$IN" && find -L . -name '*_0000.nii.gz' -printf '%h\n' | sed 's|^\./||' | sort -u)
in_list() { [ -z "$2" ] || [[ ",$2," == *",$1,"* ]]; }      # empty list = everything
DIRS=()
for d in "${ALL_DIRS[@]}"; do
    in_list "${d%%/*}" "$SEL_CONTRASTS" && in_list "${d#*/}" "$SEL_FIELDS" && DIRS+=("$d")
done
[ ${#DIRS[@]} -gt 0 ] || { echo "no folder under $IN matches contrasts='$SEL_CONTRASTS' fields='$SEL_FIELDS' (available: ${ALL_DIRS[*]})" >&2; exit 1; }
n_in() { find -L "$IN/$1" -maxdepth 1 -name '*_0000.nii.gz' | wc -l; }

echo "model:   $RES (folds ${FOLD_LIST[*]})"
echo "input:   $IN"
echo "output:  $OUT"
echo "GPUs:    ${GPUS[*]}   (NPP=$NPP NPS=$NPS per GPU)"
echo "postprocessing: $([ $POSTPROCESSING -eq 1 ] && echo "yes ($PP)" || echo no)"
for d in "${DIRS[@]}"; do echo "  $d: $(n_in "$d") images"; done
echo "start:   $(date)"

# ---- 1. prediction, one worker per GPU, each one part of every folder
for k in "${!GPUS[@]}"; do
    g=${GPUS[$k]}
    (
        for d in "${DIRS[@]}"; do
            echo "=== $(date +%H:%M:%S) $d (part $k/${#GPUS[@]}, GPU $g)"
            CUDA_VISIBLE_DEVICES=$g nnUNetv2_predict -i "$IN/$d" -o "$WORK/indices/$d" \
                -d "$DATASET_ID" -c 3d_fullres -p "$PLANS" -tr "$TRAINER" -f "${FOLD_LIST[@]}" \
                -num_parts ${#GPUS[@]} -part_id "$k" -npp "$NPP" -nps "$NPS" --continue_prediction
        done
        echo "=== $(date +%H:%M:%S) worker $k done"
    ) > "$LOGS/gpu${g}_part${k}.log" 2>&1 &
    echo "worker $k on GPU $g: pid $!, log $LOGS/gpu${g}_part${k}.log"
done
wait

missing=0
for d in "${DIRS[@]}"; do
    n=$(find "$WORK/indices/$d" -maxdepth 1 -name '*.nii.gz' 2>/dev/null | wc -l)
    echo "  $d: $n / $(n_in "$d") predicted"
    [ "$n" -eq "$(n_in "$d")" ] || missing=1
done
[ $missing -eq 0 ] || { echo "some cases are missing (see logs/gpu*.log): run the same command again" >&2; exit 1; }

# ---- 2. (postprocessing +) raw label values -> final tree
for d in "${DIRS[@]}"; do
    src="$WORK/indices/$d"
    if [ $POSTPROCESSING -eq 1 ]; then
        echo "=== $(date +%H:%M:%S) postprocessing $d"
        nnUNetv2_apply_postprocessing -i "$src" -o "$WORK/pp/$d" -pp_pkl_file "$PP" \
            -np $((NPS * ${#GPUS[@]})) -plans_json "$RES/plans.json" -dataset_json "$RES/dataset.json"
        src="$WORK/pp/$d"
    fi
    python "$TRAIN_ROOT/map_labels.py" --input "$src" --output "$OUT/$d"
done
echo "finished: $(date). Final segmentations in $OUT/<contrast>/<field>/; $WORK can be deleted."
