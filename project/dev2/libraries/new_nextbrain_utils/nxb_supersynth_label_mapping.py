"""Map a simplified NextBrain segmentation to SuperSynth labels.

Thin wrapper around nextbrain_utils.to_supersynth (libraries/nextbrain-utils), as used by
simplified_trees_and_labels/nxbsimplified_to_nxbsupersynth.py: the library is only pointed to
mapping/nextbrain_to_nxbsupersynth.yaml (ts.PATH_MAPPING, the library is not modified).
    * every class gets the SuperSynth id of one side (--side, default R) for ALL the voxels:
      the images are not split by hemisphere, only the label ids are homogenised
      (e.g. the whole cortex = 42 with R, 3 with L);
    * without --claustrum the Claustrum is merged into Cerebral-White-Matter.

The input should be a simplified NextBrain segmentation (nxb_label_mapping.py fills the labels
of mapping/labels_to_remove.yaml before calling this script with --mapping nxbsupersynth).

Usage:
    python nxb_supersynth_label_mapping.py -i seg.nii.gz -o out.nii.gz [--side L|R] [--claustrum]
                                           [--unknown 0|same|error]

--unknown : what to do with labels in the segmentation that are not in the mapping:
            a number (set them to it, default 0), "same" (keep their value, what the library
            does) or "error" (stop).
"""

import argparse
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "nextbrain-utils"))
from nextbrain_utils import to_supersynth as ts  # noqa: E402

MAPPING = HERE / "mapping" / "nextbrain_to_nxbsupersynth.yaml"


def map_segmentation(seg, side="R", claustrum=False, unknown=0):
    """Replace each NextBrain label in `seg` by its SuperSynth id (nextbrain_utils.to_supersynth).

    side: "L" or "R", side whose ids are used for all the voxels.
    claustrum: keep the claustrum as a class (default: merged into Cerebral-White-Matter).
    unknown: int, "same" or "error" -- handling of labels not in the mapping.
    """
    ts.PATH_MAPPING = str(MAPPING)
    seg = np.rint(np.asarray(seg)).astype(np.int64)

    with open(MAPPING) as f:
        known = {0} | {int(l) for labs in yaml.safe_load(f).values() for l in labs or []}
    missing = sorted(set(np.unique(seg).tolist()) - known)
    if missing and unknown == "error":
        raise ValueError(f"Labels in segmentation but not in the SuperSynth mapping: {missing}")

    # same as ts.to_supersynth(seg, side, claustrum), which fails with arrays (Nifti1Image without
    # affine). The library LUT keeps the unknown labels; extended for labels above its size.
    lut = ts.get_nextbrain2supersynth_map(side, claustrum).astype(np.int32)
    if seg.max() >= len(lut):
        lut = np.concatenate([lut, np.arange(len(lut), seg.max() + 1, dtype=np.int32)])
    if missing and unknown != "same":
        lut[missing] = int(unknown)
    return lut[seg]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-i", required=True, help="Simplified NextBrain segmentation")
    p.add_argument("-o", required=True, help="Output SuperSynth segmentation")
    p.add_argument("--side", choices=["L", "R"], default="R",
                   help="side whose label ids are used for all the voxels (default R)")
    p.add_argument("--claustrum", action="store_true",
                   help="keep the claustrum as a class (default: merged into the cerebral WM)")
    p.add_argument("--unknown", default="0", help='Number, "same" or "error" (default 0)')
    args = p.parse_args()

    unknown = args.unknown if args.unknown in ("same", "error") else int(args.unknown)
    img = nib.load(args.i)
    out = map_segmentation(np.asanyarray(img.dataobj), args.side, args.claustrum, unknown)
    hdr = img.header.copy()
    hdr.set_data_dtype(np.int16)
    hdr["scl_slope"], hdr["scl_inter"] = 1, 0
    nib.save(nib.Nifti1Image(out.astype(np.int16), img.affine, hdr), args.o)


if __name__ == "__main__":
    main()
