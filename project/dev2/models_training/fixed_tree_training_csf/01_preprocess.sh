#!/usr/bin/env bash
# Plan + preprocess one dataset and write its leave-one-template-out split.
# Usage: bash 01_preprocess.sh <net>            net: nxb18 | nxbsimplified_csf
#        PLANNER=nnUNetPlannerResEncL PLANS=nnUNetResEncUNetLPlans bash 01_preprocess.sh nxb18
# Run it once the generator has finished (it checks numTraining against labelsTr).
set -euo pipefail
cd "$(dirname "$0")"
source env.sh "${1:?net required}"
NP=${NP:-16}

mkdir -p "$LOG_DIR"
{
    # Only 3d_fullres: data is isotropic 0.4 mm
    nnUNetv2_plan_and_preprocess -d "$DATASET_ID" -c 3d_fullres -pl "$PLANNER" -np "$NP" -npfp "$NP" \
        --verify_dataset_integrity
    # Keep one copy of the images for all the nets (GT stays per net)
    python share_images.py
    # After planning, so the custom split is never overwritten
    python make_splits.py
} 2>&1 | tee "$LOG_DIR/preprocess_${PLANS}.log"
