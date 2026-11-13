d#!/usr/bin/env bash
# Usage: bash 03_predict.sh <input_dir> <output_dir> [folds] [gpu] [plans]

# or directly
# nnUNetv2_predict -d Dataset001_NextBrain_L1 \
#  -i /home/agustin/phd/synthseg_nextbrain/project/dev/wp4/simple_approaches/nn_unet/data/inputs/mrixfields/T2/3T \
#  -o /home/agustin/phd/synthseg_nextbrain/project/dev/wp4/simple_approaches/nn_unet/data/outputs/mrixfields/T2/3T \
# -f  0 1 2 3 4 -tr nnUNetTrainer -c 3d_fullres -p nnUNetResEncUNetMPlans -device cuda


# CUDA_VISIBLE_DEVICES=2 nnUNetv2_predict -d Dataset001_NextBrain_L1 \
#  -i /home/agustin/phd/synthseg_nextbrain/project/dev/wp4/simple_approaches/nn_unet/data/inputs/mrixfields/T2FLAIR/3T \
#  -o /home/agustin/phd/synthseg_nextbrain/project/dev/wp4/simple_approaches/nn_unet/data/outputs/mrixfields/T2FLAIR/3T \
#   -f 0 1 2 3 4 -tr nnUNetTrainer -c 3d_fullres -p nnUNetResEncUNetMPlans -device cuda


#   input files must be named <case>_0000.nii.gz
set -euo pipefail
cd "$(dirname "$0")"
source env.sh
IN=${1:?input dir}; OUT=${2:?output dir}
FOLDS=${3:-0}; GPU=${4:-0}; PLANS=${5:-nnUNetResEncUNetMPlans}
CUDA_VISIBLE_DEVICES=$GPU nnUNetv2_predict -i "$IN" -o "$OUT" -d $DATASET_ID -c 3d_fullres -p $PLANS -f $FOLDS
