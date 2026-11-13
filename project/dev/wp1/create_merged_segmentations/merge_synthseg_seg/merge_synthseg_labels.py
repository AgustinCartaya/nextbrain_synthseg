"""Merge SynthSeg (FreeSurfer ASeg) labels into the L1 tissue classes.

L1 classes:
    0 background, 1 brainstem, 2 csf_ventricles, 3 cerebellar_gm,
    4 cerebellar_wm, 5 cerebral_gm, 6 cerebral_wm, 7 deep_gm

Grouping follows the NextBrain ontology (L1_tissue): hippocampus and amygdala
go to cerebral_gm; ventral DC (hypothalamus / subthalamus) goes to deep_gm.
Cortical parcellation labels (1000-1035 / 2000-2035, from SynthSeg --parc)
are also mapped to cerebral_gm.

Usage:
    python merge_synthseg_labels.py -i seg_synthseg.nii.gz -o out_dir/
    python merge_synthseg_labels.py -i synthseg_folder/ -o out_dir/
"""
import argparse
import os
import sys

import nibabel as nib
import numpy as np

L1_LABELS = {
    "background": 0,
    "brainstem": 1,
    "csf_ventricles": 2,
    "cerebellar_gm": 3,
    "cerebellar_wm": 4,
    "cerebral_gm": 5,
    "cerebral_wm": 6,
    "deep_gm": 7,
}

# SynthSeg / ASeg label id -> L1 class name
SYNTHSEG_TO_L1 = {
    0: "background",            # Unknown
    2: "cerebral_wm",           # Left-Cerebral-White-Matter
    3: "cerebral_gm",           # Left-Cerebral-Cortex
    4: "csf_ventricles",        # Left-Lateral-Ventricle
    5: "csf_ventricles",        # Left-Inf-Lat-Vent
    7: "cerebellar_wm",         # Left-Cerebellum-White-Matter
    8: "cerebellar_gm",         # Left-Cerebellum-Cortex
    10: "deep_gm",              # Left-Thalamus
    11: "deep_gm",              # Left-Caudate
    12: "deep_gm",              # Left-Putamen
    13: "deep_gm",              # Left-Pallidum
    14: "csf_ventricles",       # 3rd-Ventricle
    15: "csf_ventricles",       # 4th-Ventricle
    16: "brainstem",            # Brain-Stem
    17: "cerebral_gm",          # Left-Hippocampus
    18: "cerebral_gm",          # Left-Amygdala
    # 24: "csf_ventricles",       # CSF
    24: "background",       # CSF
    26: "deep_gm",              # Left-Accumbens-area
    28: "deep_gm",              # Left-VentralDC
    41: "cerebral_wm",          # Right-Cerebral-White-Matter
    42: "cerebral_gm",          # Right-Cerebral-Cortex
    43: "csf_ventricles",       # Right-Lateral-Ventricle
    44: "csf_ventricles",       # Right-Inf-Lat-Vent
    46: "cerebellar_wm",        # Right-Cerebellum-White-Matter
    47: "cerebellar_gm",        # Right-Cerebellum-Cortex
    49: "deep_gm",              # Right-Thalamus
    50: "deep_gm",              # Right-Caudate
    51: "deep_gm",              # Right-Putamen
    52: "deep_gm",              # Right-Pallidum
    53: "cerebral_gm",          # Right-Hippocampus
    54: "cerebral_gm",          # Right-Amygdala
    58: "deep_gm",              # Right-Accumbens-area
    60: "deep_gm",              # Right-VentralDC
}
# cortical parcellation (SynthSeg --parc)
for _base in (1000, 2000):
    for _i in range(36):
        SYNTHSEG_TO_L1[_base + _i] = "cerebral_gm"


def merge_array(seg, mapping=SYNTHSEG_TO_L1, labels=L1_LABELS):
    """Return a uint8 array with SynthSeg ids replaced by L1 ids."""
    seg = np.rint(np.asarray(seg)).astype(np.int32)
    unknown = sorted(set(np.unique(seg).tolist()) - set(mapping))
    if unknown:
        raise ValueError(f"SynthSeg labels without an L1 mapping: {unknown}")
    lut = np.zeros(max(mapping) + 1, dtype=np.uint8)
    for src, name in mapping.items():
        lut[src] = labels[name]
    return lut[seg]


def merge_file(input_path, output_path):
    img = nib.load(input_path)
    merged = merge_array(np.asanyarray(img.dataobj))
    header = img.header.copy()
    header.set_data_dtype(np.uint8)
    header.set_slope_inter(1, 0)
    out = nib.Nifti1Image(merged, img.affine, header)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    nib.save(out, output_path)
    return merged


def _output_name(input_path):
    name = os.path.basename(input_path)
    for ext in (".nii.gz", ".nii"):
        if name.endswith(ext):
            return name[: -len(ext)] + "_L1" + ext
    raise ValueError(f"Not a NIfTI file: {input_path}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-i", "--input", required=True, help="SynthSeg .nii(.gz) file or folder")
    p.add_argument("-o", "--output", required=True, help="Output folder (or output file if input is a file)")
    args = p.parse_args(argv)

    if os.path.isdir(args.input):
        inputs = sorted(
            os.path.join(args.input, f) for f in os.listdir(args.input)
            if f.endswith((".nii", ".nii.gz"))
        )
    else:
        inputs = [args.input]

    for path in inputs:
        if args.output.endswith((".nii", ".nii.gz")) and len(inputs) == 1:
            out_path = args.output
        else:
            out_path = os.path.join(args.output, _output_name(path))
        merged = merge_file(path, out_path)
        vals, counts = np.unique(merged, return_counts=True)
        inv = {v: k for k, v in L1_LABELS.items()}
        summary = ", ".join(f"{inv[v]}={c}" for v, c in zip(vals, counts))
        print(f"{path} -> {out_path}\n    voxels: {summary}")


if __name__ == "__main__":
    sys.exit(main())
