#!/usr/bin/env python3
"""Combine NextBrain hemispheres with nextbrain_utils.combine_hemis, robustly.

NextBrain writes seg.left / seg.right on grids whose voxel sizes differ slightly
(e.g. 0.4005 vs 0.3989 mm, because the output shape is rounded). combine_hemis
assumes both grids share the same voxel size and differ only by an integer
shift, so it crashes with a broadcast error. Here the right segmentation is
first resampled (nearest neighbour) onto the left voxel grid, extended to cover
the whole right FOV, and then passed to combine_hemis.

Usage: python combine_hemis_safe.py <left> <right> <out_combined> <out_sides>
"""
import sys
from itertools import product

import nibabel as nb
import numpy as np
from scipy.ndimage import map_coordinates

from nextbrain_utils.combine_hemis import combine_hemis


def resample_to_left_grid(left, right):
    """Nearest-neighbour resample of `right` onto a grid with `left`'s voxel
    axes, aligned to integer offsets of the left grid, covering right's FOV."""
    right2left = np.linalg.inv(left.affine) @ right.affine
    corners = np.array(list(product([0, 1], [0, 1], [0, 1]))) * (np.array(right.shape[:3]) - 1)
    corners = corners @ right2left[:3, :3].T + right2left[:3, 3]
    lo = np.floor(corners.min(0)).astype(int)
    hi = np.ceil(corners.max(0)).astype(int)
    shape = tuple(hi - lo + 1)

    affine = left.affine.copy()
    affine[:3, 3] = left.affine[:3, :3] @ lo + left.affine[:3, 3]

    vox2vox = np.linalg.inv(right.affine) @ affine
    grid = np.indices(shape, dtype=np.float32).reshape(3, -1)
    coords = vox2vox[:3, :3] @ grid + vox2vox[:3, 3:4]
    data = np.asarray(right.dataobj)
    out = map_coordinates(data, coords, order=0, mode="constant", cval=0)
    out = out.reshape(shape).astype(data.dtype)
    return nb.Nifti1Image(out, affine, left.header.copy())


def main():
    if len(sys.argv) != 5:
        sys.exit(__doc__)
    left_f, right_f, out_f, sides_f = sys.argv[1:]
    left = nb.load(left_f)
    right = nb.load(right_f)
    right_rs = resample_to_left_grid(left, right)
    print(f"left {left.shape} {left.header.get_zooms()}, right {right.shape} "
          f"{right.header.get_zooms()} -> resampled {right_rs.shape}")
    combine_hemis(left, right_rs, out_f, sides_f)
    print(f"saved {out_f} and {sides_f}")


if __name__ == "__main__":
    main()
