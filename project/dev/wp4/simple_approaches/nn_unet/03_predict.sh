#!/usr/bin/env bash
# Usage: bash 03_predict.sh <input_dir> <output_dir> [folds] [gpu] [plans]
#   input files must be named <case>_0000.nii.gz
set -euo pipefail
cd "$(dirname "$0")"
source env.sh
IN=${1:?input dir}; OUT=${2:?output dir}
FOLDS=${3:-0}; GPU=${4:-0}; PLANS=${5:-nnUNetResEncUNetMPlans}
CUDA_VISIBLE_DEVICES=$GPU nnUNetv2_predict -i "$IN" -o "$OUT" -d $DATASET_ID -c 3d_fullres -p $PLANS -f $FOLDS
