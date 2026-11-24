# Source before any nnU-Net command:  source env.sh <net>
#   net: nxb18 | nxbsimplified_csf
# The networks share the same images and differ only in the ground truth (only nxb18 is trained
# for now; nxbsimplified_csf has its raw dataset ready).
TRAIN_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GEN_DATA="$TRAIN_ROOT/data_shared/generation"     # written by generate_data/generate.py

NET=${1:?"usage: source env.sh <nxb18|nxbsimplified_csf>"}
case "$NET" in
    nxbsimplified_csf) export DATASET_ID=101 DATASET_NAME=Dataset101_nxbsimplified_csf ;;
    nxb18)             export DATASET_ID=102 DATASET_NAME=Dataset102_nxb18 ;;
    *) echo "unknown net '$NET' (nxb18 | nxbsimplified_csf)" >&2; return 1 ;;
esac
export NET

# Every net has its own nnU-Net folders in models/<net>/. Its raw dataset only links to
# data_shared/generation (imagesTr: hard links to images/, labelsTr: link to labels/<net>/)
export MODEL_DIR="$TRAIN_ROOT/models/$NET"
export nnUNet_raw="$MODEL_DIR/nnUNet_raw"
export nnUNet_preprocessed="$MODEL_DIR/nnUNet_preprocessed"
export nnUNet_results="$MODEL_DIR/nnUNet_results"
export LOG_DIR="$MODEL_DIR/logs"
# Preprocessed images shared by the nets (each net links to them, see share_images.py)
export SHARED_PREPROCESSED="$TRAIN_ROOT/data_shared/shared_preprocessed"
export PLANS_DIR="$GEN_DATA/plans"      # per-image JSON, records the template used

# Planner and plans name must match (ResEnc M -> nnUNetResEncUNetMPlans,
# ResEnc L -> nnUNetResEncUNetLPlans, ExperimentPlanner -> nnUNetPlans)
export PLANNER=${PLANNER:-nnUNetPlannerResEncM}
export PLANS=${PLANS:-nnUNetResEncUNetMPlans}
export TRAINER=${TRAINER:-nnUNetTrainer}

# 128 cores shared by the trainings: keep data augmentation workers moderate
export nnUNet_n_proc_DA=${nnUNet_n_proc_DA:-12}
