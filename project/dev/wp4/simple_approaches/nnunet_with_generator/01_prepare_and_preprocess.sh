#!/usr/bin/env bash
# Usage: bash 01_prepare_and_preprocess.sh [planner] [gpu]
#   planner: nnUNetPlannerResEncM (default), nnUNetPlannerResEncL, ExperimentPlanner (classic UNet)
#   N_VAL=4 validation volumes per template (fixed, generated once)
set -euo pipefail
cd "$(dirname "$0")"
source env.sh
PLANNER=${1:-nnUNetPlannerResEncM}
GPU=${2:-0}
NP=${NP:-8}

# 1. fixed validation set (full volumes) + dataset.json + generator settings
CUDA_VISIBLE_DEVICES=$GPU python prepare_dataset.py raw --n-per-template ${N_VAL:-4} --device cuda:0
# 2. plan + preprocess (only 3d_fullres: the generator makes isotropic 3D patches)
nnUNetv2_plan_and_preprocess -d $DATASET_ID -c 3d_fullres -pl $PLANNER -np $NP -npfp $NP --verify_dataset_integrity
# 3. leave-one-template-out splits, generator settings next to the plans
python prepare_dataset.py splits
