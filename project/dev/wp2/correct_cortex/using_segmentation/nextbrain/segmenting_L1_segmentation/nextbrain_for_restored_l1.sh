
#!/bin/bash

set -e

INPUT_DIR="/home/agustin/phd/synthseg_nextbrain/project/dev/wp2/fill_holes/data/outputs/restored_manual/restored/L1"

OUTPUT_DIR="/home/agustin/phd/synthseg_nextbrain/project/dev/wp2/correct_cortex/data/outputs/nextbrain_for_restored_l1"

mkdir -p "$OUTPUT_DIR"

for i in {1..5}; do

    INPUT="${INPUT_DIR}/case-${i}.nii.gz"
    OUTPUT="${OUTPUT_DIR}/case-${i}.nii.gz"

    echo "======================================"
    echo "Processing case-${i}"
    echo "======================================"

    if [[ ! -f "$INPUT" ]]; then
        echo "WARNING: Input not found: $INPUT"
        continue
    fi

    mri_histo_atlas_segment_fireants \
        --i "$INPUT" \
        --o "$OUTPUT" \
        --device cuda:3 \
        --side left \
        --mode exvivo

    echo "Finished case-${i}"

done

echo "All cases processed!"
