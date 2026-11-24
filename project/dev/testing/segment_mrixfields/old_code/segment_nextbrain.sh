#!/usr/bin/env bash
# Segment all MRIxFields inputs with NextBrain (mri_histo_atlas_segment_fireants),
# both hemispheres, then merge them into one segmentation.
#
# Usage: bash segment_nextbrain.sh [devices] [mode] [sides]
#   devices: comma-separated list, default cuda:3   (e.g. cuda:0,cuda:3)
#   mode:    default invivo
#   sides:   default "left right"  (merge runs only when both are done)
# Environment options:
#   LIMIT=N          process only the first N images (quick test)
#   JOBS_PER_GPU=K   subjects run in parallel on each device (default 1)
#   THREADS=T        --threads passed to NextBrain (default: NextBrain's own default)
#
# Examples:
#   LIMIT=3 bash segment_nextbrain.sh
#   JOBS_PER_GPU=2 bash segment_nextbrain.sh cuda:0,cuda:3     # 4 subjects at a time
#
# Parallelism: #workers = #devices x JOBS_PER_GPU. Images are distributed round-robin
# over the workers; each worker processes its subjects one after another, and for each
# subject runs left -> right -> merge sequentially (the second side reuses SuperSynth).
#
# Input : <IN_ROOT>/<contrast>/<field>/<case>_0000.nii.gz
# Output: <OUT_ROOT>/<contrast>/<field>/<case>/
#           seg.left.nii.gz, seg.right.nii.gz   NextBrain outputs (+ vols.<side>.csv, lut.txt)
#           seg.combined.nii.gz                 both hemispheres (NextBrain labels)
#           seg.sides.nii.gz                    1=left, 2=right
#           segment.<side>.log, merge.log
#         <OUT_ROOT>/logs/worker_<k>.log        progress of each worker
# Hemispheres are combined with combine_hemis_safe.py (nextbrain-utils combine_hemis
# after resampling right onto the left grid). It needs Python >= 3.10 (fspython is too
# old) -> NB_PYTHON; the library is used from NB_UTILS via PYTHONPATH (no install).
# Finished steps leave .done.<side> / .done.merged markers and are skipped on re-runs.

set -uo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
IN_ROOT=/home/agustin/phd/synthseg_nextbrain/project/dev/wp4/simple_approaches/nn_unet/data/inputs/mrixfields
OUT_ROOT=$HERE/data/outputs
NB_PYTHON=${NB_PYTHON:-/home/agustin/miniconda3/bin/python3}
NB_UTILS=${NB_UTILS:-/home/agustin/phd/synthseg_nextbrain/project/dev/wp1/allen_ontology/nextbrain-utils}
export PYTHONPATH="$NB_UTILS${PYTHONPATH:+:$PYTHONPATH}"

IFS=',' read -ra DEVICES <<< "${1:-cuda:3}"
MODE=${2:-invivo}
read -ra SIDES <<< "${3:-left right}"
JOBS_PER_GPU=${JOBS_PER_GPU:-1}
EXTRA=()
[[ -n "${THREADS:-}" ]] && EXTRA=(--threads "$THREADS")

command -v mri_histo_atlas_segment_fireants >/dev/null || {
    echo "mri_histo_atlas_segment_fireants not found in PATH (source FreeSurfer first)" >&2; exit 1; }
"$NB_PYTHON" -c "import nextbrain_utils.combine_hemis" 2>/dev/null || {
    echo "nextbrain_utils not importable with $NB_PYTHON (set NB_PYTHON)" >&2; exit 1; }

mapfile -t IMAGES < <(find -L "$IN_ROOT" -name '*.nii.gz' | sort)
if [[ -n "${LIMIT:-}" ]]; then IMAGES=("${IMAGES[@]:0:$LIMIT}"); fi
N=${#IMAGES[@]}

# one worker slot per (device, job) pair
WORKER_DEV=()
for ((j = 0; j < JOBS_PER_GPU; j++)); do
    for d in "${DEVICES[@]}"; do WORKER_DEV+=("$d"); done
done
W=${#WORKER_DEV[@]}
(( W > N )) && W=$N

LOG_DIR=$OUT_ROOT/logs
mkdir -p "$LOG_DIR"
FAILED_FILE=$LOG_DIR/failed.txt
: > "$FAILED_FILE"

echo "Found $N images under $IN_ROOT (sides: ${SIDES[*]})"
echo "Running $W worker(s) on: ${WORKER_DEV[*]:0:$W}   logs: $LOG_DIR/worker_<k>.log"

ts() { date '+%F %T'; }

process_case() {   # <index> <image> <device>
    local i=$1 img=$2 dev=$3
    local rel=${img#"$IN_ROOT"/}            # T1W/3T/R_T1W_3T_0329_0000.nii.gz
    local sub; sub=$(dirname "$rel")        # T1W/3T
    local case; case=$(basename "$rel" .nii.gz)
    case=${case%_0000}                      # R_T1W_3T_0329
    local out="$OUT_ROOT/$sub/$case"
    mkdir -p "$out"
    echo "$(ts) [$((i+1))/$N] $sub/$case on $dev"

    local side start
    for side in "${SIDES[@]}"; do
        if [[ -f "$out/.done.$side" ]]; then echo "    $side: skip (done)"; continue; fi
        start=$(date +%s)
        if mri_histo_atlas_segment_fireants \
            --i "$img" \
            --o "$out" \
            --device "$dev" \
            --side "$side" \
            --mode "$MODE" "${EXTRA[@]}" > "$out/segment.$side.log" 2>&1; then
            touch "$out/.done.$side"
            echo "    $side: ok ($(( $(date +%s) - start ))s)"
        else
            echo "$sub/$case ($side)" >> "$FAILED_FILE"
            echo "    $side: FAILED - see $out/segment.$side.log"
        fi
    done

    if [[ -f "$out/.done.left" && -f "$out/.done.right" && ! -f "$out/.done.merged" ]]; then
        if "$NB_PYTHON" "$HERE/combine_hemis_safe.py" \
                "$out/seg.left.nii.gz" "$out/seg.right.nii.gz" \
                "$out/seg.combined.nii.gz" "$out/seg.sides.nii.gz" > "$out/merge.log" 2>&1; then
            touch "$out/.done.merged"; echo "    merged: ok"
        else
            echo "$sub/$case (merge)" >> "$FAILED_FILE"
            echo "    merged: FAILED - see $out/merge.log"
        fi
    fi
}

worker() {   # <k>: processes images k, k+W, k+2W, ...
    local k=$1 dev=${WORKER_DEV[$1]} i
    for ((i = k; i < N; i += W)); do
        process_case "$i" "${IMAGES[$i]}" "$dev"
    done
    echo "$(ts) worker $k done"
}

# kill all workers (and their NextBrain processes) on Ctrl-C / kill
trap 'echo "Interrupted, stopping workers"; kill 0; exit 130' INT TERM

for ((k = 0; k < W; k++)); do
    worker "$k" > "$LOG_DIR/worker_$k.log" 2>&1 &
    echo "  worker $k (pid $!) -> ${WORKER_DEV[$k]}"
done
wait

nfail=$(wc -l < "$FAILED_FILE")
echo "$(ts) Finished. $nfail failures (listed in $FAILED_FILE)."
cat "$FAILED_FILE"
