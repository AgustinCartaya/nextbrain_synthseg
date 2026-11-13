#!/usr/bin/env python
"""
Naive baseline: fill every background voxel inside the closed brain outline with the label
of the nearest labelled voxel.

  brain = fill_holes(binary_closing(seg > 0, closing_iter))
  every voxel with seg == 0 inside brain -> nearest label (3D distance transform)

It fills block gaps and holes, but it also closes most sulci (see benchmark/compare_naive.py).
It never looks at label classes, so it works the same on the full label set and on L1 maps
(no --input-is-l1 needed).

  python baseline_naive.py input.nii.gz output.nii.gz
  python baseline_naive.py input.nii.gz output.nii.gz --closing-iter 6 --save-mask
"""
import argparse
import os
import sys
import time

import nibabel as nib
import numpy as np
from scipy import ndimage as ndi

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# from restore import fill_nearest   # noqa: E402


def naive_fill_mask(seg, closing_iter=3, fill_holes=True):
    """Background voxels that the naive baseline fills."""
    brain = seg > 0
    if closing_iter > 0:
        brain = ndi.binary_closing(brain, structure=ndi.generate_binary_structure(3, 1),
                                   iterations=closing_iter)
    if fill_holes:
        brain = ndi.binary_fill_holes(brain)
    return brain & (seg == 0)


def fill_nearest(seg, unknown, max_dist_vox=None, margin=None):
    """Assign to every `unknown` voxel the label of the nearest known voxel (3D EDT),
    processed per connected component bounding box."""
    out = seg.copy()
    lab, n = ndi.label(unknown, structure=np.ones((3, 3, 3)))
    if n == 0:
        return out
    objs = ndi.find_objects(lab)
    for i, sl in enumerate(objs, 1):
        ext = max(s.stop - s.start for s in sl)
        mg = margin if margin is not None else int(min(max(4, ext // 2 + 2), 60))
        box = tuple(slice(max(0, s.start - mg), min(d, s.stop + mg)) for s, d in zip(sl, seg.shape))
        unk = unknown[box]
        _, ind = ndi.distance_transform_edt(unk, return_indices=True)
        comp = lab[box] == i
        src = tuple(ind[a][comp] for a in range(3))
        out[box][comp] = seg[box][src]
    return out


def naive(seg, closing_iter=3, fill_holes=True):
    """Returns (filled label array, mask of filled voxels)."""
    m = naive_fill_mask(seg, closing_iter, fill_holes)
    return fill_nearest(seg, m), m


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input", help="label map (.nii/.nii.gz), full label set or L1")
    p.add_argument("output", help="filled label map")
    p.add_argument("--closing-iter", type=int, default=3,
                   help="iterations of binary closing of the brain mask (default 3); more closes wider gaps "
                        "and more sulci")
    p.add_argument("--no-fill-holes", action="store_true",
                   help="do not fill enclosed cavities of the closed mask (only what the closing covers)")
    p.add_argument("--save-mask", action="store_true", help="also write <output>_mask.nii.gz with the filled voxels")
    a = p.parse_args()

    t = time.time()
    img = nib.load(a.input)
    seg = np.asarray(img.dataobj)
    if np.issubdtype(seg.dtype, np.floating):
        seg = np.rint(seg).astype(np.int32)
    out, m = naive(seg, a.closing_iter, not a.no_fill_holes)

    hdr = img.header.copy()
    out_img = nib.Nifti1Image(out.astype(img.get_data_dtype()), img.affine, hdr)
    nib.save(out_img, a.output)
    print(f"{int(m.sum()):,} background voxels filled ({time.time() - t:.1f} s) -> {a.output}")
    if a.save_mask:
        stem = a.output[:-7] if a.output.endswith(".nii.gz") else os.path.splitext(a.output)[0]
        nib.save(nib.Nifti1Image(m.astype(np.uint8), img.affine), stem + "_mask.nii.gz")


if __name__ == "__main__":
    main()
