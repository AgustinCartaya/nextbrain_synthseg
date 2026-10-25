# Source this file before any nnU-Net command:  source env.sh
NNUNET_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export nnUNet_raw="$NNUNET_ROOT/nnUNet_raw"
export nnUNet_preprocessed="$NNUNET_ROOT/nnUNet_preprocessed"
export nnUNet_results="$NNUNET_ROOT/nnUNet_results"
# 128 cores available; keep data loading fast but don't hog the shared machine
export nnUNet_n_proc_DA=${nnUNet_n_proc_DA:-16}

export DATASET_ID=1
export DATASET_NAME=Dataset001_NextBrain_L1
export SOURCE_DATASET=/home/agustin/phd/synthseg_nextbrain/project/dev/testing/generating_merged_seg_images/data/outputs/Dataset001_NextBrain_L1
