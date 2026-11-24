#!/usr/bin/env bash
# Resample every merged NextBrain segmentation (seg.combined.nii.gz, produced by
# segment_nextbrain.sh) onto the voxel grid of its input image, using only the
# headers (no registration) and nearest-neighbour interpolation.
#
# Usage: bash resample_nextbrain.sh
# Environment options:
#   JOBS=N     cases resampled in parallel (default 8)
#   FORCE=1    redo cases whose seg_registered.nii.gz already exists
#
# Input : <OUT_ROOT>/<contrast>/<field>/<case>/seg.combined.nii.gz   (NextBrain, ~0.4 mm)
#         <IN_ROOT>/<contrast>/<field>/<case>_0000.nii.gz            (target grid)
# Output: <OUT_ROOT>/<contrast>/<field>/<case>/seg_registered.nii.gz
#         <OUT_ROOT>/<contrast>/<field>/<case>/resample.log
#         <OUT_ROOT>/logs/resample_failed.txt

set -uo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
IN_ROOT=/home/agustin/phd/synthseg_nextbrain/project/dev/wp4/simple_approaches/nn_unet/data/inputs/mrixfields
OUT_ROOT=$HERE/data/outputs
JOBS=${JOBS:-8}
FORCE=${FORCE:-0}

command -v mri_vol2vol >/dev/null || {
    echo "mri_vol2vol not found in PATH (source FreeSurfer first)" >&2; exit 1; }

FAILED_FILE=$OUT_ROOT/logs/resample_failed.txt
mkdir -p "$(dirname "$FAILED_FILE")"
: > "$FAILED_FILE"
export IN_ROOT OUT_ROOT FORCE FAILED_FILE

resample_case() {   # <path to seg.combined.nii.gz>
    local seg=$1
    local dir; dir=$(dirname "$seg")         # .../T1W/3T/R_T1W_3T_0329
    local rel=${dir#"$OUT_ROOT"/}            # T1W/3T/R_T1W_3T_0329
    local targ="$IN_ROOT/${rel}_0000.nii.gz"
    local out="$dir/seg_registered.nii.gz"

    if [[ "$FORCE" != 1 && -f "$out" ]]; then echo "skip  $rel"; return; fi
    if [[ ! -f "$targ" ]]; then
        echo "$rel (input not found: $targ)" >> "$FAILED_FILE"
        echo "FAIL  $rel: input not found"; return
    fi
    if mri_vol2vol --mov "$seg" --targ "$targ" --regheader --interp nearest \
                   --o "$out" > "$dir/resample.log" 2>&1; then
        echo "ok    $rel"
    else
        rm -f "$out"
        echo "$rel (mri_vol2vol, see $dir/resample.log)" >> "$FAILED_FILE"
        echo "FAIL  $rel: see $dir/resample.log"
    fi
}
export -f resample_case

mapfile -t SEGS < <(find "$OUT_ROOT" -name seg.combined.nii.gz | sort)
echo "Found ${#SEGS[@]} merged segmentations under $OUT_ROOT (JOBS=$JOBS)"

printf '%s\0' "${SEGS[@]}" | xargs -0 -n 1 -P "$JOBS" bash -c 'resample_case "$1"' _

nfail=$(wc -l < "$FAILED_FILE")
echo "Finished. $nfail failures (listed in $FAILED_FILE)."
cat "$FAILED_FILE"
