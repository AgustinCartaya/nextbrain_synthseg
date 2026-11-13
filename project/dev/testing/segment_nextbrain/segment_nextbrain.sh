#!/usr/bin/env bash
# Segment one image with NextBrain (both hemispheres) and resample the result
# onto the input image grid. Wrapper around segment_nextbrain.py.
#
# Usage: bash segment_nextbrain.sh <image> <output_dir> [device] [mode]
#   device: default cuda:0
#   mode:   default invivo
# Environment options:
#   THREADS=T   --threads passed to NextBrain
#   FORCE=1     redo steps whose outputs already exist
#   PYTHON=...  python with numpy/scipy/nibabel (default: nxb env)
#   FREESURFER_HOME=...  sourced if NextBrain is not already in PATH
#
# Example:
#   bash segment_nextbrain.sh /path/sub01_T1w.nii.gz /path/out/sub01 cuda:3
#
# Output: <output_dir>/seg_registered.nii.gz  (final, on the input grid)
#         plus seg.left/right/combined/sides.nii.gz and logs



# mri_histo_atlas_segment_fireants \
# --i /home/Data/Synthetic/NextBrain/Dataset001_NextBrain_L1/imagesTr/synth_0000_0000.nii.gz \
# --o /home/agustin/phd/synthseg_nextbrain/project/dev/testing/segment_nextbrain/data/outputs/synth_0000_0000 \
# --device cuda:3 \
# --side left \
# --mode invivo

# mri_vol2vol \
# --mov /home/agustin/phd/synthseg_nextbrain/project/dev/testing/segment_nextbrain/data/outputs/synth_0000_0000/seg.left.nii.gz \
# --targ /home/Data/Synthetic/NextBrain/Dataset001_NextBrain_L1/imagesTr/synth_0000_0000.nii.gz \
# --regheader --interp nearest \
# --o /home/agustin/phd/synthseg_nextbrain/project/dev/testing/segment_nextbrain/data/outputs/synth_0000_0000/seg_resampled.nii.gz


# python /home/agustin/phd/synthseg_nextbrain/project/dev/wp1/allen_ontology/tree_segmentation.py \
# --i /home/agustin/phd/synthseg_nextbrain/project/dev/testing/segment_nextbrain/data/outputs/synth_0000_0000/seg_resampled.nii.gz \
# --o /home/agustin/phd/synthseg_nextbrain/project/dev/testing/segment_nextbrain/data/outputs/synth_0000_0000/seg_registered_merged.nii.gz \
# --v brainstem_l1 csf_ventricles_l1 cerebellar_gm_l1 cerebellar_wm_l1 cerebral_gm_l1 cerebral_wm_l1 deep_gm_l1 other_l1 \
# --tree /home/agustin/phd/synthseg_nextbrain/project/dev/wp1/create_nextbraintree/data/outputs/nextbrain_tree_v1.json



set -euo pipefail

if [[ $# -lt 2 ]]; then
    sed -n '2,18p' "$0" | sed 's/^# \{0,1\}//'
    exit 1
fi

HERE=$(cd "$(dirname "$0")" && pwd)
IMAGE=$1
OUT=$2
DEVICE=${3:-cuda:0}
MODE=${4:-invivo}
PYTHON=${PYTHON:-/home/agustin/miniconda3/envs/nxb/bin/python3}

if ! command -v mri_histo_atlas_segment_fireants >/dev/null; then
    export FREESURFER_HOME=${FREESURFER_HOME:-/home/code/freesurfer/freesurfer-nextbrain}
    # shellcheck disable=SC1091
    set +eu; source "$FREESURFER_HOME/SetUpFreeSurfer.sh" >/dev/null; set -eu
fi

ARGS=(--device "$DEVICE" --mode "$MODE")
[[ -n "${THREADS:-}" ]] && ARGS+=(--threads "$THREADS")
[[ "${FORCE:-0}" == 1 ]] && ARGS+=(--force)

exec "$PYTHON" "$HERE/segment_nextbrain.py" "$IMAGE" "$OUT" "${ARGS[@]}"
