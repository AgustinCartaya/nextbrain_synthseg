#!/usr/bin/env bash
# Usage: bash 01_prepare_and_preprocess.sh [planner]
#   planner: nnUNetPlannerResEncM (default), nnUNetPlannerResEncL, ExperimentPlanner (classic UNet)
set -euo pipefail
cd "$(dirname "$0")"
source env.sh
PLANNER=${1:-nnUNetPlannerResEncM}
NP=${NP:-16}

python prepare_dataset.py
# Only 3d_fullres: data is isotropic 0.6 mm, ~190x340x260 voxels
nnUNetv2_plan_and_preprocess -d $DATASET_ID -c 3d_fullres -pl $PLANNER -np $NP -npfp $NP --verify_dataset_integrity
