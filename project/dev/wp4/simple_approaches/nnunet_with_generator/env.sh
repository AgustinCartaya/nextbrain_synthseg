# Source this file before any command:  source env.sh
NXNN_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --- nnU-Net -----------------------------------------------------------------
export nnUNet_raw="$NXNN_ROOT/nnUNet_raw"
export nnUNet_preprocessed="$NXNN_ROOT/nnUNet_preprocessed"
export nnUNet_results="$NXNN_ROOT/nnUNet_results"
# nnU-Net looks here for trainers it does not ship (nnUNetTrainerNxGen)
export nnUNet_extTrainer="$NXNN_ROOT/trainers"
# only the validation loader uses worker processes (training batches come from the generator)
export nnUNet_n_proc_DA=${nnUNet_n_proc_DA:-8}

export DATASET_ID=101
export DATASET_NAME=Dataset101_NextBrain_L1_nxgen

# --- generator (nxgen) -------------------------------------------------------
export NXGEN_ROOT="$(cd "$NXNN_ROOT/../../../wp3/generator_v2_modular" && pwd)"
export PYTHONPATH="$NXGEN_ROOT${PYTHONPATH:+:$PYTHONPATH}"
# generator config (L1 tree, 0.6 mm output, 0.2 mm HR templates)
export NXGEN_CONFIG="$NXNN_ROOT/configs/nextbrain_l1_0.6mm.json"
# ground-truth L1 label maps (case-1..case-5, 0.2 mm)
export NXGEN_SEG_DIR="$NXGEN_ROOT/__data_test__/inputs/L1/changed_recon_0.2mm_level1"
# HR templates built from them (00_build_templates.sh)
export NXGEN_TEMPLATES="$NXNN_ROOT/templates"

# Runtime options of the trainer (read by trainers/nnUNetTrainerNxGen.py):
#   NXGEN_DEVICE    GPU of the generator, as seen inside the training process (default cuda:0 = training GPU)
#   NXGEN_SEED      run seed of the training stream (default 0)
#   NXGEN_PREFETCH  batches generated ahead in a background thread (default 2, 0 = synchronous)
#   NXGEN_NUM_EPOCHS, NXGEN_ITERS_PER_EPOCH   override nnU-Net's 1000 epochs x 250 iterations
