#!/usr/bin/env python
"""
Step 2 — convert the real test cases to the training label space (L1).

Reads every NIfTI of paths.test_raw_dir (full NextBrain label set, 0.2 mm) and
writes the merged version, same grid and affine, to paths.test_l1_dir.
The merge uses merge_labels.py as a library (same lookup table as its CLI).

The test cases contain a few labels that are not in the ontology; they are
mapped with labels.test_extra_label_map (see the config), any other unknown
label becomes background, and all of them are printed.

The raw test data is only read.

  python 02_prepare_test_cases.py --config configs/default.yaml
"""
import glob
import os

import nibabel as nib
import numpy as np

from lib.config import config_from_args, parser
from lib.labels import LabelMerger, assert_writable


def main():
    p = parser(__doc__)
    p.add_argument("--overwrite", action="store_true", help="redo cases that already exist")
    args = p.parse_args()
    cfg = config_from_args(args)
    src_dir, dst_dir = cfg["paths"]["test_raw_dir"], cfg["paths"]["test_l1_dir"]
    if os.path.realpath(src_dir) == os.path.realpath(dst_dir):
        raise ValueError("paths.test_l1_dir must differ from paths.test_raw_dir")
    os.makedirs(assert_writable(dst_dir), exist_ok=True)

    merger = LabelMerger(cfg)
    extra = cfg["labels"].get("test_extra_label_map") or {}
    for path in sorted(glob.glob(os.path.join(src_dir, "*.nii*"))):
        out = os.path.join(dst_dir, os.path.basename(path))
        if os.path.exists(out) and not args.overwrite:
            print(f"[skip] {out} exists")
            continue
        img = nib.load(path)
        merged, unknown = merger(np.asarray(img.dataobj), extra=extra)
        header = img.header.copy()
        header.set_data_dtype(merged.dtype)
        header.set_slope_inter(1, 0)
        nib.save(nib.Nifti1Image(merged, img.affine, header), out)
        print(f"[ok] {os.path.basename(path)}: labels {np.flatnonzero(np.bincount(merged.ravel())).tolist()}"
              f" | extra map used: {extra} | unknown labels set to 0: {unknown}", flush=True)


if __name__ == "__main__":
    main()
