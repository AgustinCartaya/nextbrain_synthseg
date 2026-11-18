#!/usr/bin/env python3
"""Assemble a whole-hemisphere 3D label volume for one NextBrain case.

Each case repo (github.com/UCL/NextBrain-Case-N) ships one registered 3D label
volume per histology block (labels_blocks/<N>/LABELS.nii.gz, 0.1 mm in-plane,
~0.33-0.8 mm between sections) whose header affine maps into the space of
mri_rotated/mri.nii.gz. This script mosaics all blocks onto one isotropic,
RAS-aligned grid covering the MRI, by nearest-neighbour sampling.

The logic follows scripts/Reconstruction/3d-recon-all-labels.py of
github.com/acasamitjana/ERC_reconstruction, with two changes:
  * each block is resampled only inside its own bounding box, so fine grids fit
    in memory;
  * where blocks overlap, the label from the block whose histology section is
    closest (through-plane) wins, instead of the last block processed.

Usage:
    python build_nn.py \
        --case-dir /home/agustin/phd/synthseg_nextbrain/project/nextbrain_atlas_files/registered_images/NextBrain-Case-1/P57-16 \
        --out-dir /home/agustin/phd/synthseg_nextbrain/project/dev2/atlas_subjects/data/outputs/originals/nn_test \
        --res 0.2
"""
import argparse
import itertools
import os
import time

import nibabel as nib
import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--case-dir', required=True,
                   help='Case directory containing labels_blocks/ and mri_rotated/ eg. ../NextBrain-Case-1/P57-16')
    p.add_argument('--out-dir', required=True,  help='Output folder')
    p.add_argument('--res', type=float, default=0.2, help='Isotropic output voxel size in mm (default 0.2)')
    p.add_argument('--chunk', type=int, default=64, help='Output slabs processed at once (memory/speed trade-off)')
    return p.parse_args()


def target_grid(mri_path, res):
    """Isotropic RAS-aligned grid covering the MRI field of view."""
    mri = nib.load(mri_path)
    n = np.array(mri.shape[:3])
    # outer corners of the MRI volume (voxel edges, not centres)
    corners = np.array(list(itertools.product(*[(-0.5, s - 0.5) for s in n])))
    ras = corners @ mri.affine[:3, :3].T + mri.affine[:3, 3]
    lo, hi = ras.min(0), ras.max(0)
    shape = tuple(int(s) for s in np.ceil((hi - lo) / res))
    affine = np.diag([res, res, res, 1.0])
    affine[:3, 3] = lo + res / 2  # centre of voxel (0,0,0)
    return shape, affine


def main():
    args = parse_args()
    case_dir = os.path.abspath(args.case_dir)
    case = os.path.basename(case_dir)
    blocks_dir = os.path.join(case_dir, 'labels_blocks')
    blocks = sorted((b for b in os.listdir(blocks_dir)
                     if os.path.exists(os.path.join(blocks_dir, b, 'LABELS.nii.gz'))), key=int)

    shape, affine = target_grid(os.path.join(case_dir, 'mri_rotated', 'mri.nii.gz'), args.res)
    print(f'{case}: {len(blocks)} blocks -> grid {shape} at {args.res} mm')

    seg = np.zeros(shape, dtype=np.uint16)
    # through-plane distance (mm) from each output voxel to the section it was taken from
    best = np.full(shape, np.inf, dtype=np.float32)
    n_overlap = 0

    t0 = time.time()
    for b in blocks:
        img = nib.load(os.path.join(blocks_dir, b, 'LABELS.nii.gz'))
        lab = np.asanyarray(img.dataobj)
        if lab.ndim == 2:
            lab = lab[..., None]
        nb = np.array(lab.shape[:3])
        dz = float(np.linalg.norm(img.affine[:3, 2]))  # section spacing in mm

        # bounding box of the block (outer voxel edges) in output voxel coordinates
        corners = np.array(list(itertools.product(*[(-0.5, s - 0.5) for s in nb])))
        ras = corners @ img.affine[:3, :3].T + img.affine[:3, 3]
        vox = (ras - affine[:3, 3]) / args.res
        lo = np.clip(np.floor(vox.min(0)).astype(int), 0, shape)
        hi = np.clip(np.ceil(vox.max(0)).astype(int) + 1, 0, shape)
        if np.any(hi <= lo):
            print(f'  block {b}: outside grid, skipped')
            continue

        # output voxel -> block voxel
        M = np.linalg.inv(img.affine) @ affine
        jj = np.arange(lo[1], hi[1], dtype=np.float32)[None, :, None]
        kk = np.arange(lo[2], hi[2], dtype=np.float32)[None, None, :]
        filled = 0
        for i0 in range(lo[0], hi[0], args.chunk):
            i1 = min(i0 + args.chunk, hi[0])
            ii = np.arange(i0, i1, dtype=np.float32)[:, None, None]
            c = [M[r, 0] * ii + M[r, 1] * jj + M[r, 2] * kk + M[r, 3] for r in range(3)]
            idx = [np.rint(x).astype(np.int32) for x in c]
            inside = np.ones(idx[0].shape, dtype=bool)
            for a in range(3):
                inside &= (idx[a] >= 0) & (idx[a] < nb[a])
            if not inside.any():
                continue
            dist = (np.abs(c[2] - idx[2]) * dz).astype(np.float32)
            vals = np.zeros(inside.shape, dtype=np.uint16)
            vals[inside] = lab[idx[0][inside], idx[1][inside], idx[2][inside]]

            sl = (slice(i0, i1), slice(lo[1], hi[1]), slice(lo[2], hi[2]))
            cur_seg, cur_best = seg[sl], best[sl]
            take = (vals > 0) & (dist < cur_best)
            n_overlap += int(np.count_nonzero((vals > 0) & (cur_seg > 0)))
            cur_seg[take] = vals[take]
            cur_best[take] = dist[take]
            filled += int(np.count_nonzero(take))
        print(f'  block {b:>2}: shape {lab.shape},section spacing {dz:.2f} mm, '
              f'{len(np.unique(lab)) - 1} labels, {filled} voxels written')
    del best

    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir, f'{case}_seg_{args.res:g}mm.nii.gz')
    hdr = nib.Nifti1Header()
    hdr.set_data_dtype(np.uint16)
    hdr.set_xyzt_units('mm')
    nib.save(nib.Nifti1Image(seg, affine, hdr), out)

    labels = np.unique(seg)
    n_lab = int(np.count_nonzero(seg))
    print(f'\nSaved {out}')
    print(f'  {len(labels) - 1} labels (max id {labels.max()}), {n_lab} labelled voxels '
          f'= {n_lab * args.res ** 3 / 1000:.1f} cm^3')
    print(f'  {n_overlap} voxels were claimed by more than one block '
          f'({100 * n_overlap / max(n_lab, 1):.2f}% of labelled voxels)')
    print(f'  done in {time.time() - t0:.0f} s')


if __name__ == '__main__':
    main()
