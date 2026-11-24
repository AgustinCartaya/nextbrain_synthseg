
#!/bin/bash

set -e

MODE="exvivo" #exvivo or invivo
INPUT_DIR="/home/agustin/phd/synthseg_nextbrain/project/dev2/generator/test_generators/test_tree_generator/data/outputs/deterministic/images"

OUTPUT_DIR="/home/agustin/phd/synthseg_nextbrain/project/dev2/generator/test_generators/test_tree_generator/data/outputs/deterministic/segmentation_nextbrain_${MODE}"

# Voxels of zero padding added on every side before running NextBrain.
# Needed because NextBrain's cropping (cropLabelVolTorch in my_functions.py) clamps
# the Y/Z crop to the X size when the brain is within 5 voxels of the +Y/+Z border,
# which cuts the hemisphere and ends in "Could not initialize label group N".
PAD=10

DEVICE="cuda:3"

mkdir -p "$OUTPUT_DIR"

for i in {1..5}; do

    INPUT="${INPUT_DIR}/case-${i}.nii.gz"
    OUTPUT="${OUTPUT_DIR}/case-${i}.nii.gz"
    PADDED="${OUTPUT}/input_padded.nii.gz"

    echo "======================================"
    echo "Processing case-${i}"
    echo "======================================"

    if [[ ! -f "$INPUT" ]]; then
        echo "WARNING: Input not found: $INPUT"
        continue
    fi

    mkdir -p "$OUTPUT"

    # Zero-pad the input, shifting the affine so world coordinates are unchanged
    fspython - "$INPUT" "$PADDED" "$PAD" <<'EOF'
import sys
import numpy as np
import nibabel as nib

src, dst, pad = sys.argv[1], sys.argv[2], int(sys.argv[3])
img = nib.load(src)
data = np.asarray(img.dataobj)
padded = np.pad(data, [(pad, pad)] * 3 + [(0, 0)] * (data.ndim - 3))
affine = img.affine.copy()
affine[:3, 3] -= affine[:3, :3] @ np.full(3, pad)
out = nib.Nifti1Image(padded.astype(data.dtype), affine)
out.header.set_xyzt_units(*img.header.get_xyzt_units())
nib.save(out, dst)
print(f"Padded {data.shape} -> {padded.shape}: {dst}")
EOF

    # Run SuperSynth first (NextBrain reuses it) to find out which hemisphere the image contains
    if [[ ! -f "$OUTPUT/SuperSynth/segmentation.mgz" ]]; then
        mri_super_synth \
            --i "$PADDED" \
            --o "$OUTPUT/SuperSynth/" \
            --mode "${MODE}" \
            --threads "$(nproc)" \
            --device "$DEVICE"
    fi

    # Side = hemisphere with the larger cerebral white matter + cortex volume (labels 2,3 vs 41,42)
    SIDE=$(python3 - "$OUTPUT/SuperSynth/volumes.csv" <<'PYEOF'
import csv, sys
names, vals = list(csv.reader(open(sys.argv[1])))[:2]
vol = lambda labels: sum(float(v) for n, v in zip(names, vals) if any(n.endswith(f"({l})") for l in labels))
print("left" if vol([2, 3]) > vol([41, 42]) else "right")
PYEOF
)
    echo "Detected hemisphere: ${SIDE}"

    mri_histo_atlas_segment_fireants \
        --i "$PADDED" \
        --o "$OUTPUT" \
        --device "$DEVICE" \
        --side "$SIDE" \
        --mode "${MODE}" \
        --skip_bf

    # Resample back onto the original (unpadded) input grid
    mri_vol2vol \
        --mov "$OUTPUT/seg.${SIDE}.nii.gz" \
        --targ "$INPUT" \
        --regheader --interp nearest \
        --o "$OUTPUT/seg.${SIDE}_resampled.nii.gz"

    mri_vol2vol \
        --mov "$OUTPUT/SuperSynth/segmentation.mgz" \
        --targ "$INPUT" \
        --regheader --interp nearest \
        --o "$OUTPUT/SuperSynth/segmentation_resampled.nii.gz"

    rm -f "$PADDED"

    echo "Finished case-${i}"

done

echo "All cases processed!"
