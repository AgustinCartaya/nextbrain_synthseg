#!/usr/bin/env bash
# Segment the MRIxFields BIDS rawdata with NextBrain and write the results to
# derivatives/nextbrain, in one go: left + right hemisphere -> merge -> resample
# onto the input grid.
#
# Usage:
#   bash segment_nextbrain.sh [options]
#
# Options:
#   --gpus 0,1,2,3        GPU indices to use (default: 0)
#   --jobs-per-gpu K      cases running at the same time on each GPU (default: 1)
#   --contrasts LIST      comma-separated: T1w,T2w,FLAIR (default: all)
#   --fields LIST         comma-separated: 0p1T,1p5T,3T,5T,7T (default: all;
#                         0.1T / 1.5T / 3.0T ... are also accepted)
#   --overwrite           re-segment cases whose segmentation already exists
#                         (their whole <contrast>/ folder is deleted first).
#                         Without it, existing segmentations are skipped.
#   --threads T           CPU threads per NextBrain run (default: NextBrain's)
#   --dry-run             only list what would be processed
#   -h, --help            show this help
#
# Examples:
#   bash segment_nextbrain.sh --dry-run
#   bash segment_nextbrain.sh --gpus 0,1,2,3 --fields 7T --contrasts T1w
#   bash segment_nextbrain.sh --gpus 2,3 --jobs-per-gpu 2
#
# Input : /home/Data/MRIXFIELDS/rawdata/sub-*/ses-*/anat/sub-*_ses-*_<C>.nii.gz
# Output: /home/Data/MRIXFIELDS/derivatives/nextbrain/sub-*/ses-*/anat/<C>/
#           sub-*_ses-*_acq-<C>_desc-nextbrain_dseg.nii.gz   final segmentation (input grid)
#           source_data/   NextBrain outputs: seg.left/right.nii.gz, vols.*.csv, lut.txt,
#                          SuperSynth/, seg.combined.nii.gz, seg.sides.nii.gz,
#                          segment.<side>.log, merge.log, resample.log, .done.* markers
#         /home/Data/MRIXFIELDS/derivatives/nextbrain/logs/<timestamp>/
#           worker_<k>.log, failed.txt
#
# A case counts as done when its final dseg exists (it is written atomically, at
# the very end). An interrupted case is resumed: steps with a .done.<step>
# marker in source_data/ are not repeated.
# Requires FreeSurfer (mri_histo_atlas_segment_fireants, mri_vol2vol) in PATH and
# a Python >= 3.10 with nibabel + scipy (NB_PYTHON) for the hemisphere merge.


# cd /home/agustin/phd/synthseg_nextbrain/project/dev2/other_codes/segment_mrixfields/with_nextbrain
# bash segment_nextbrain.sh --fields 7T --gpus 0,1,2,3


set -uo pipefail

RAW=/home/Data/MRIXFIELDS/rawdata
OUT=/home/Data/MRIXFIELDS/derivatives/nextbrain
NB_PYTHON=${NB_PYTHON:-/home/agustin/miniconda3/bin/python3}
NB_UTILS=${NB_UTILS:-/home/agustin/phd/synthseg_nextbrain/project/dev2/libraries/nextbrain-utils}
export PYTHONPATH="$NB_UTILS${PYTHONPATH:+:$PYTHONPATH}"
MODE=invivo

# ---------------------------------------------------------------- arguments
GPUS=0; JOBS_PER_GPU=1; CONTRASTS=T1w,T2w,FLAIR; FIELDS=0p1T,1p5T,3T,5T,7T
OVERWRITE=0; DRY_RUN=0; THREADS=""

usage() { sed -n '2,/^$/s/^# \{0,1\}//p' "$0"; exit "${1:-0}"; }
while (($#)); do
    case $1 in
        --gpus)          GPUS=$2; shift 2 ;;
        --jobs-per-gpu)  JOBS_PER_GPU=$2; shift 2 ;;
        --contrasts)     CONTRASTS=$2; shift 2 ;;
        --fields)        FIELDS=$2; shift 2 ;;
        --threads)       THREADS=$2; shift 2 ;;
        --overwrite)     OVERWRITE=1; shift ;;
        --dry-run)       DRY_RUN=1; shift ;;
        -h|--help)       usage 0 ;;
        *) echo "Unknown option: $1" >&2; usage 1 ;;
    esac
done

norm_contrast() {
    case ${1,,} in
        t1w) echo T1w ;; t2w) echo T2w ;; flair|t2flair) echo FLAIR ;;
        *) echo "Unknown contrast: $1" >&2; exit 1 ;;
    esac
}
norm_field() {
    case ${1,,} in
        0p1t|0.1t) echo 0p1T ;; 1p5t|1.5t) echo 1p5T ;; 3t|3.0t) echo 3T ;;
        5t|5.0t) echo 5T ;; 7t|7.0t) echo 7T ;;
        *) echo "Unknown field: $1" >&2; exit 1 ;;
    esac
}
IFS=',' read -ra GPU_LIST <<< "$GPUS"
IFS=',' read -ra C_LIST <<< "$CONTRASTS"
IFS=',' read -ra F_LIST <<< "$FIELDS"
for i in "${!C_LIST[@]}"; do C_LIST[$i]=$(norm_contrast "${C_LIST[$i]}") || exit 1; done
for i in "${!F_LIST[@]}"; do F_LIST[$i]=$(norm_field "${F_LIST[$i]}") || exit 1; done
EXTRA=(); [[ -n $THREADS ]] && EXTRA=(--threads "$THREADS")

# ---------------------------------------------------------------- checks
for cmd in mri_histo_atlas_segment_fireants mri_vol2vol; do
    command -v "$cmd" >/dev/null || { echo "$cmd not found (source FreeSurfer first)" >&2; exit 1; }
done
"$NB_PYTHON" -c "import nibabel, scipy, nextbrain_utils.combine_hemis" 2>/dev/null || {
    echo "nibabel/scipy/nextbrain_utils not importable with $NB_PYTHON (set NB_PYTHON/NB_UTILS)" >&2; exit 1; }

# ---------------------------------------------------------------- helpers
case_dir() {   # <image> -> derivatives/nextbrain/sub-X/ses-Y/anat/<C>
    local img=$1 rel base
    rel=${img#"$RAW"/}; base=$(basename "$img" .nii.gz)
    echo "$OUT/$(dirname "$rel")/${base##*_}"
}
dseg_path() {  # <image> -> final segmentation path
    local img=$1 base
    base=$(basename "$img" .nii.gz)              # sub-R0329_ses-3T_T1w
    echo "$(case_dir "$img")/${base%_*}_acq-${base##*_}_desc-nextbrain_dseg.nii.gz"
}

merge_hemis() {  # <left> <right> <combined> <sides>
    # NextBrain writes left/right on grids whose voxel sizes differ slightly, which
    # makes combine_hemis crash; the right side is first resampled (nearest) onto
    # the left voxel grid, extended to cover the whole right FOV.
    "$NB_PYTHON" - "$@" <<'PY'
import sys
from itertools import product
import nibabel as nb
import numpy as np
from scipy.ndimage import map_coordinates
from nextbrain_utils.combine_hemis import combine_hemis

left_f, right_f, out_f, sides_f = sys.argv[1:]
left, right = nb.load(left_f), nb.load(right_f)
r2l = np.linalg.inv(left.affine) @ right.affine
corners = np.array(list(product([0, 1], repeat=3))) * (np.array(right.shape[:3]) - 1)
corners = corners @ r2l[:3, :3].T + r2l[:3, 3]
lo, hi = np.floor(corners.min(0)).astype(int), np.ceil(corners.max(0)).astype(int)
shape = tuple(hi - lo + 1)
affine = left.affine.copy()
affine[:3, 3] = left.affine[:3, :3] @ lo + left.affine[:3, 3]
v2v = np.linalg.inv(right.affine) @ affine
grid = np.indices(shape, dtype=np.float32).reshape(3, -1)
data = np.asarray(right.dataobj)
res = map_coordinates(data, v2v[:3, :3] @ grid + v2v[:3, 3:4], order=0, mode="constant", cval=0)
right_rs = nb.Nifti1Image(res.reshape(shape).astype(data.dtype), affine, left.header.copy())
print(f"left {left.shape}, right {right.shape} -> resampled {right_rs.shape}")
combine_hemis(left, right_rs, out_f, sides_f)
print(f"saved {out_f} and {sides_f}")
PY
}

ts() { date '+%F %T'; }

process_case() {  # <n> <image> <gpu>
    local n=$1 img=$2 gpu=$3
    local dir src dseg side start
    dir=$(case_dir "$img"); src=$dir/source_data; dseg=$(dseg_path "$img")
    echo "$(ts) [$n/$N] ${dir#"$OUT"/} on GPU $gpu"

    if ((OVERWRITE)) && [[ -d $dir ]]; then
        [[ $dir == "$OUT"/sub-*/ses-*/anat/* ]] || { echo "    refusing to delete $dir"; return 1; }
        rm -rf "$dir"
    fi
    mkdir -p "$src"

    for side in left right; do
        [[ -f $src/.done.$side ]] && { echo "    $side: already done"; continue; }
        start=$(date +%s)
        CUDA_VISIBLE_DEVICES=$gpu mri_histo_atlas_segment_fireants \
            --i "$img" --o "$src" --device cuda --side "$side" --mode "$MODE" "${EXTRA[@]}" \
            > "$src/segment.$side.log" 2>&1 \
            || { echo "    $side: FAILED (see $src/segment.$side.log)"; return 1; }
        touch "$src/.done.$side"
        echo "    $side: ok ($(( $(date +%s) - start ))s)"
    done

    if [[ ! -f $src/.done.merged ]]; then
        merge_hemis "$src/seg.left.nii.gz" "$src/seg.right.nii.gz" \
                    "$src/seg.combined.nii.gz" "$src/seg.sides.nii.gz" > "$src/merge.log" 2>&1 \
            || { echo "    merge: FAILED (see $src/merge.log)"; return 1; }
        touch "$src/.done.merged"
        echo "    merge: ok"
    fi

    # resample onto the input grid (header only, nearest); written atomically
    local tmp=${dseg%.nii.gz}.part.nii.gz
    mri_vol2vol --mov "$src/seg.combined.nii.gz" --targ "$img" --regheader --interp nearest \
                --o "$tmp" > "$src/resample.log" 2>&1 && mv "$tmp" "$dseg" \
        || { rm -f "$tmp"; echo "    resample: FAILED (see $src/resample.log)"; return 1; }
    echo "    resample: ok -> $(basename "$dseg")"
}

# ---------------------------------------------------------------- collect cases
shopt -s nullglob
TODO=(); declare -A N_ALL N_SKIP
for f in "${F_LIST[@]}"; do
    for c in "${C_LIST[@]}"; do
        for img in "$RAW"/sub-*/ses-"$f"/anat/sub-*_ses-"${f}"_"${c}".nii.gz; do
            N_ALL[$f/$c]=$(( ${N_ALL[$f/$c]:-0} + 1 ))
            if [[ -f $(dseg_path "$img") ]]; then
                N_SKIP[$f/$c]=$(( ${N_SKIP[$f/$c]:-0} + 1 ))
                ((OVERWRITE)) || continue
            fi
            TODO+=("$img")
        done
    done
done
N=${#TODO[@]}

printf '%-6s %-6s %6s %8s %8s\n' field contr images existing to_do
for f in "${F_LIST[@]}"; do for c in "${C_LIST[@]}"; do
    a=${N_ALL[$f/$c]:-0}; s=${N_SKIP[$f/$c]:-0}
    printf '%-6s %-6s %6d %8d %8d\n' "$f" "$c" "$a" "$s" $((OVERWRITE ? a : a - s))
done; done
echo "Total to segment: $N   (overwrite: $OVERWRITE)"

if ((DRY_RUN)); then printf '  %s\n' "${TODO[@]}"; exit 0; fi
((N == 0)) && { echo "Nothing to do."; exit 0; }

# ---------------------------------------------------------------- run
WORKER_GPU=()
for ((j = 0; j < JOBS_PER_GPU; j++)); do WORKER_GPU+=("${GPU_LIST[@]}"); done
W=${#WORKER_GPU[@]}; ((W > N)) && W=$N

LOG_DIR=$OUT/logs/$(date '+%Y%m%d_%H%M%S')
mkdir -p "$LOG_DIR"; FAILED=$LOG_DIR/failed.txt; : > "$FAILED"
echo "Running $W worker(s) on GPU(s) ${WORKER_GPU[*]:0:$W}   logs: $LOG_DIR"

worker() {  # <k>: cases k, k+W, k+2W, ...
    local k=$1 i
    for ((i = k; i < N; i += W)); do
        process_case $((i + 1)) "${TODO[$i]}" "${WORKER_GPU[$k]}" || echo "${TODO[$i]}" >> "$FAILED"
    done
    echo "$(ts) worker $k finished"
}

trap 'echo "Interrupted, stopping workers"; kill 0; exit 130' INT TERM
for ((k = 0; k < W; k++)); do
    worker "$k" > "$LOG_DIR/worker_$k.log" 2>&1 &
    echo "  worker $k (pid $!) -> GPU ${WORKER_GPU[$k]}"
done
wait

nfail=$(wc -l < "$FAILED")
echo "$(ts) Finished: $((N - nfail)) ok, $nfail failed (see $FAILED)"
((nfail == 0))
