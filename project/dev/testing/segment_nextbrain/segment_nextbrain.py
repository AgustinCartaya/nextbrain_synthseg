#!/usr/bin/env python3
"""Segment one image with NextBrain (both hemispheres) and bring the result
back to the voxel grid of the input image.

Steps
  1. mri_histo_atlas_segment_fireants --side left   -> seg.left.nii.gz
  2. mri_histo_atlas_segment_fireants --side right  -> seg.right.nii.gz
     (the second run reuses the SuperSynth outputs of the first)
  3. combine both hemispheres                        -> seg.combined.nii.gz, seg.sides.nii.gz
     (right is first resampled onto the left voxel grid, because NextBrain
      writes both sides on grids with slightly different voxel sizes)
  4. mri_vol2vol --regheader --interp nearest        -> seg_registered.nii.gz
     (header-only resampling onto the input image grid, no registration)

Finished steps are skipped on re-runs (use --force to redo everything).

Requirements: FreeSurfer (NextBrain build) sourced, i.e. mri_histo_atlas_segment_fireants
and mri_vol2vol in PATH; python with numpy, scipy and nibabel.

Usage
  python segment_nextbrain.py <image.nii.gz> <output_dir> [--device cuda:0]
         [--mode invivo] [--threads N] [--force]
"""
import argparse
import shutil
import subprocess
import sys
import time
from itertools import product
from pathlib import Path

import nibabel as nb
import numpy as np
from scipy.ndimage import map_coordinates


# --------------------------------------------------------------------------- #
# Hemisphere combination
# --------------------------------------------------------------------------- #
def _corners(shape, vox2vox):
    """Corners of a volume of `shape`, mapped with `vox2vox` (4x4)."""
    c = np.array(list(product([0, 1], [0, 1], [0, 1]))) * (np.array(shape[:3]) - 1)
    return c @ vox2vox[:3, :3].T + vox2vox[:3, 3]


def resample_to_left_grid(left, right):
    """Nearest-neighbour resample of `right` onto a grid with `left`'s voxel
    axes, aligned to integer offsets of the left grid, covering right's FOV."""
    corners = _corners(right.shape, np.linalg.inv(left.affine) @ right.affine)
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
    return nb.Nifti1Image(out.reshape(shape).astype(data.dtype), affine, left.header.copy())


def combine_hemis(left, right):
    """Paste left and right (same voxel size, integer shift) into one volume.
    Left has priority where both overlap. Returns (combined, sides) images,
    sides: 1 = left, 2 = right. Adapted from nextbrain_utils.combine_hemis."""
    l_c = _corners(left.shape, np.eye(4))
    r_c = _corners(right.shape, np.linalg.inv(left.affine) @ right.affine)
    l_min, l_max = np.round(l_c.min(0)).astype(int), np.round(l_c.max(0)).astype(int)
    r_min, r_max = np.round(r_c.min(0)).astype(int), np.round(r_c.max(0)).astype(int)

    bmin = np.minimum(l_min, r_min)
    shape = tuple(np.maximum(l_max, r_max) - bmin + 1)
    l_box = tuple(slice(l_min[i] - bmin[i], l_max[i] - bmin[i] + 1) for i in range(3))
    r_box = tuple(slice(r_min[i] - bmin[i], r_max[i] - bmin[i] + 1) for i in range(3))

    comb2left = np.eye(4)
    comb2left[:3, 3] = bmin - l_min
    affine = left.affine @ comb2left

    ldata, rdata = np.asarray(left.dataobj), np.asarray(right.dataobj)

    seg = np.zeros(shape, dtype=ldata.dtype)
    seg[l_box] = ldata
    seg[r_box] += rdata * (seg[r_box] == 0)

    sides = np.zeros(shape, dtype=np.uint8)
    sides[l_box] = (ldata > 0).astype(np.uint8)
    free = sides[r_box] == 0
    sides[r_box] += (rdata > 0).astype(np.uint8) * 2 * free

    seg_img = nb.Nifti1Image(seg, affine, left.header.copy())
    sides_hdr = left.header.copy()
    sides_hdr.set_data_dtype(np.uint8)
    sides_img = nb.Nifti1Image(sides, affine, sides_hdr)
    return seg_img, sides_img


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def run(cmd, log_file):
    """Run `cmd`, writing stdout/stderr to `log_file`; exit on failure."""
    print("  $ " + " ".join(map(str, cmd)))
    t0 = time.time()
    with open(log_file, "w") as log:
        ret = subprocess.run(list(map(str, cmd)), stdout=log, stderr=subprocess.STDOUT)
    if ret.returncode != 0:
        sys.exit(f"  FAILED (exit {ret.returncode}), see {log_file}")
    print(f"  ok ({time.time() - t0:.0f}s)")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("image", type=Path, help="input image (.nii / .nii.gz / .mgz)")
    p.add_argument("out_dir", type=Path, help="output folder")
    p.add_argument("--device", default="cuda:0", help="device for NextBrain (default cuda:0)")
    p.add_argument("--mode", default="invivo", help="NextBrain mode (default invivo)")
    p.add_argument("--threads", type=int, help="CPU threads passed to NextBrain")
    p.add_argument("--force", action="store_true", help="redo steps already done")
    args = p.parse_args()

    image = args.image.resolve()
    out = args.out_dir.resolve()
    if not image.is_file():
        sys.exit(f"Input image not found: {image}")
    for tool in ("mri_histo_atlas_segment_fireants", "mri_vol2vol"):
        if shutil.which(tool) is None:
            sys.exit(f"{tool} not found in PATH (source FreeSurfer first)")
    out.mkdir(parents=True, exist_ok=True)

    seg_left = out / "seg.left.nii.gz"
    seg_right = out / "seg.right.nii.gz"
    seg_comb = out / "seg.combined.nii.gz"
    seg_sides = out / "seg.sides.nii.gz"
    seg_final = out / "seg_registered.nii.gz"

    print(f"Image : {image}\nOutput: {out}")

    # 1-2. NextBrain, one hemisphere at a time
    for side, seg in (("left", seg_left), ("right", seg_right)):
        print(f"[NextBrain {side}]")
        if seg.exists() and not args.force:
            print("  skip (exists)")
            continue
        cmd = ["mri_histo_atlas_segment_fireants", "--i", image, "--o", out,
               "--device", args.device, "--side", side, "--mode", args.mode]
        if args.threads:
            cmd += ["--threads", args.threads]
        run(cmd, out / f"segment.{side}.log")

    # 3. combine hemispheres
    print("[combine hemispheres]")
    if seg_comb.exists() and seg_sides.exists() and not args.force:
        print("  skip (exists)")
    else:
        left, right = nb.load(seg_left), nb.load(seg_right)
        right_rs = resample_to_left_grid(left, right)
        print(f"  left {left.shape} {left.header.get_zooms()[:3]}, "
              f"right {right.shape} {right.header.get_zooms()[:3]} -> {right_rs.shape}")
        comb, sides = combine_hemis(left, right_rs)
        nb.save(comb, seg_comb)
        nb.save(sides, seg_sides)
        print(f"  ok -> {seg_comb.name} {comb.shape}, {seg_sides.name}")

    # 4. resample onto the input image grid (header only, nearest neighbour)
    print("[resample to input grid]")
    if seg_final.exists() and not args.force:
        print("  skip (exists)")
    else:
        run(["mri_vol2vol", "--mov", seg_comb, "--targ", image, "--regheader",
             "--interp", "nearest", "--o", seg_final], out / "resample.log")

    print(f"Done: {seg_final}")


if __name__ == "__main__":
    main()
