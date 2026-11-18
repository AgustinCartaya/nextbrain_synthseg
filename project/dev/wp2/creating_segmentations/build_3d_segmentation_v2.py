#!/usr/bin/env python3
"""Assemble a whole-hemisphere 3D label volume for one NextBrain case (version 2).

Same inputs and output grid as build_3d_segmentation.py, but it avoids most of the
artefacts of the plain nearest-section mosaic by using what is known at
reconstruction time (the geometry of every block and the co-registered MRI):

  1. Shape-based interpolation between sections. Inside each block, every label
     gets a 2D signed distance map per section; the maps of the two bracketing
     sections are interpolated linearly along the section axis and the label
     with the largest value wins. This replaces the through-plane staircase.
  2. Missing sections are bridged. Sections that are empty or mostly unlabelled
     (typically the first/last ones of a block, which is what produces the
     "block gaps") are ignored; the voxels they cover are interpolated between,
     or copied from, the nearest usable sections of the block.
  3. Blocks are mosaicked; where they overlap, the block with the closest usable
     section wins.
  4. Labels where the MRI has no tissue are removed (leaks into sulci/background).
     Labels that mostly lie where the MRI is zero (ventricles, and structures that
     are not in the stripped MRI such as the optic nerve or the pineal body) are
     detected automatically and never removed.
  5. Optional (--mri-fill, off by default): voxels inside the MRI brain mask that
     are still unlabelled get the nearest label in 3D, up to --max-fill-mm.
     WARNING: this closes the sulci. The MRI (mri_rotated/mri.nii.gz) is only
     brain-masked: sulci are bright in it, not zero, so "MRI > 0" does not
     separate tissue from sulcal space. Only use it with a real tissue mask.

Not handled: misalignment between blocks ("block steps"); that is residual
registration error and needs a post-hoc correction (e.g. restore.py shift step).

Outputs (in --out-dir):
  <case>_seg_<res>mm_v2.nii.gz        labels (NextBrain / AllenAtlasLUT ids)
  <case>_seg_<res>mm_v2_steps.nii.gz  how each voxel was obtained:
        1 = interpolated between two consecutive sections
        2 = bridged over missing sections / copied from the nearest usable section
        3 = MRI-guided fill (only with --mri-fill)
        4 = label removed because the MRI has no tissue there

Usage:
    python build_3d_segmentation_v2.py --case-dir ../NextBrain-Case-1/P57-16 --res 0.2
"""
import argparse
import itertools
import os
import time

import nibabel as nib
import numpy as np
from scipy import ndimage as ndi

HERE = os.path.dirname(os.path.abspath(__file__))
STEP_INTERP, STEP_BRIDGED, STEP_FILL, STEP_TRIM = 1, 2, 3, 4


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--case-dir', default=os.path.join(HERE, '..', 'NextBrain-Case-1', 'P57-16'),
                   help='Case directory containing labels_blocks/ and mri_rotated/')
    p.add_argument('--res', type=float, default=0.2, help='Isotropic output voxel size in mm (default 0.2)')
    p.add_argument('--like', default=None,
                   help='Optional NIfTI whose grid (shape + affine) is used instead of --res')
    p.add_argument('--out-dir', default=os.path.join(HERE, 'outputs'))
    p.add_argument('--min-section-frac', type=float, default=0.3,
                   help='A section is "missing" if its labelled area is below this fraction of the '
                        'median labelled area of its block (default 0.3)')
    p.add_argument('--sdt-cap-mm', type=float, default=1.0,
                   help='Signed distances are clipped to +/- this value (default 1 mm). Where a structure '
                        'moves more than this between two sections the result tends to nearest-section')
    p.add_argument('--inplane-stride', type=int, default=1,
                   help='Use every n-th in-plane pixel of the sections (default 1 = full 0.1 mm detail)')
    p.add_argument('--keep-outside-frac', type=float, default=0.25,
                   help='Labels with more than this fraction of their volume where the MRI is zero are '
                        'treated as not visible in the MRI: never trimmed, never used to fill (default 0.25)')
    p.add_argument('--max-fill-mm', type=float, default=2.0,
                   help='MRI-guided fill: maximum distance to the nearest label (default 2 mm)')
    p.add_argument('--no-trim', action='store_true', help='Keep labels where the MRI has no tissue')
    p.add_argument('--mri-fill', action='store_true',
                   help='Fill unlabelled voxels inside the MRI brain mask (closes the sulci, see above)')
    return p.parse_args()


def log(msg):
    print(msg, flush=True)


def corners_ras(shape, affine):
    c = np.array(list(itertools.product(*[(-0.5, s - 0.5) for s in shape])))
    return c @ affine[:3, :3].T + affine[:3, 3]


def target_grid(mri, res):
    """Isotropic RAS-aligned grid covering the MRI field of view (same as version 1)."""
    ras = corners_ras(mri.shape[:3], mri.affine)
    lo, hi = ras.min(0), ras.max(0)
    shape = tuple(int(s) for s in np.ceil((hi - lo) / res))
    affine = np.diag([res, res, res, 1.0])
    affine[:3, 3] = lo + res / 2
    return shape, affine


def section_sdt(sec, pix_mm, cap):
    """Signed distance map (mm, >0 inside) of every label of a 2D section, background included."""
    out = {}
    for lab in np.unique(sec):
        m = sec == lab
        if m.all():
            out[int(lab)] = np.full(sec.shape, cap, np.float32)
            continue
        sd = ndi.distance_transform_edt(m) - ndi.distance_transform_edt(~m)
        out[int(lab)] = np.clip(sd * pix_mm, -cap, cap).astype(np.float32)
    return out


def reconstruct_block(lab, blk_affine, grid_shape, grid_affine, res, min_frac, cap, stride):
    """Resample one block onto the output grid.

    Returns (slices of the block bounding box in the grid, labels, distance in mm to the nearest
    usable section, bridged flag) or None. Arrays cover the bounding box; labels are 0 outside
    the block."""
    if lab.ndim == 2:
        lab = lab[..., None]
    nb = np.array(lab.shape[:3])
    pix = float(np.linalg.norm(blk_affine[:3, 0]))  # in-plane pixel size (mm)
    dz = float(np.linalg.norm(blk_affine[:3, 2]))   # section spacing (mm)
    sub = np.ascontiguousarray(lab[::stride, ::stride, :])

    area = (sub > 0).sum(axis=(0, 1))
    if not area.any():
        return None
    valid = np.flatnonzero(area >= min_frac * np.median(area[area > 0]))

    vox = (corners_ras(nb, blk_affine) - grid_affine[:3, 3]) / res
    lo = np.clip(np.floor(vox.min(0)).astype(int), 0, grid_shape)
    hi = np.clip(np.ceil(vox.max(0)).astype(int) + 1, 0, grid_shape)
    if np.any(hi <= lo):
        return None
    box = tuple(slice(a, b) for a, b in zip(lo, hi))
    bshape = tuple(int(b - a) for a, b in zip(lo, hi))

    # output voxel -> block voxel
    M = np.linalg.inv(blk_affine) @ grid_affine
    ax = [np.arange(a, b, dtype=np.float32) for a, b in zip(lo, hi)]
    ii, jj, kk = ax[0][:, None, None], ax[1][None, :, None], ax[2][None, None, :]
    cx, cy, cz = (M[r, 0] * ii + M[r, 1] * jj + M[r, 2] * kk + M[r, 3] for r in range(3))
    inside = ((cx >= -0.5) & (cx <= nb[0] - 0.5) & (cy >= -0.5) & (cy <= nb[1] - 0.5)
              & (cz >= -0.5) & (cz <= nb[2] - 0.5))
    flat = np.flatnonzero(inside)
    if flat.size == 0:
        return None
    xi = np.clip(np.rint(cx[inside] / stride).astype(np.int32), 0, sub.shape[0] - 1)
    yi = np.clip(np.rint(cy[inside] / stride).astype(np.int32), 0, sub.shape[1] - 1)
    zc = cz[inside]
    del cx, cy, cz, inside

    # usable sections bracketing every voxel
    j = np.searchsorted(valid, zc)
    s_lo = valid[np.clip(j - 1, 0, len(valid) - 1)]
    s_hi = valid[np.clip(j, 0, len(valid) - 1)]
    s_lo = np.where(zc < valid[0], valid[0], s_lo)
    s_hi = np.where(zc > valid[-1], valid[-1], s_hi)
    span = s_hi - s_lo
    t = np.where(span > 0, (zc - s_lo) / np.maximum(span, 1), 0).astype(np.float32)
    dist = (np.minimum(np.abs(zc - s_lo), np.abs(zc - s_hi)) * dz).astype(np.float32)
    bridged = span != 1  # outside the usable range, or across skipped sections

    out = np.zeros(flat.size, np.uint16)
    pmin = np.float32(pix * stride)
    key = s_lo.astype(np.int64) * (nb[2] + 1) + s_hi
    order = np.argsort(key, kind='stable')
    bounds = np.flatnonzero(np.diff(key[order])) + 1
    cache = {}
    for grp in np.split(order, bounds):
        a, b = int(s_lo[grp[0]]), int(s_hi[grp[0]])
        gx, gy = xi[grp], yi[grp]
        if a == b:  # copy the nearest usable section
            out[grp] = sub[gx, gy, a]
            continue
        for s in (a, b):
            if s not in cache:
                cache[s] = section_sdt(sub[:, :, s], pix * stride, cap)
        for s in [s for s in cache if s < a]:
            del cache[s]
        sa, sb = cache[a], cache[b]
        tg = t[grp]
        best_val = np.full(grp.size, -np.inf, np.float32)
        best_lab = np.zeros(grp.size, np.uint16)
        for l in sorted(set(sa) | set(sb)):
            # a label absent from one section fades out halfway (as nearest-section would do),
            # instead of being pushed out by a large negative constant: keeps small structures
            if l in sa and l in sb:
                va, vb = sa[l][gx, gy], sb[l][gx, gy]
            elif l in sa:
                va = sa[l][gx, gy]; vb = -np.maximum(np.abs(va), pmin)
            else:
                vb = sb[l][gx, gy]; va = -np.maximum(np.abs(vb), pmin)
            val = (1 - tg) * va + tg * vb
            better = val > best_val
            best_val[better] = val[better]
            best_lab[better] = l
        out[grp] = best_lab

    n = int(np.prod(bshape))
    labels = np.zeros(n, np.uint16); labels[flat] = out
    d = np.full(n, np.inf, np.float32); d[flat] = dist
    br = np.zeros(n, bool); br[flat] = bridged
    info = dict(n_sections=int(nb[2]), n_missing=int(nb[2] - len(valid)), dz=dz,
                n_labels=int(len(np.unique(lab)) - 1))
    return box, labels.reshape(bshape), d.reshape(bshape), br.reshape(bshape), info


def main():
    args = parse_args()
    t0 = time.time()
    case_dir = os.path.abspath(args.case_dir)
    case = os.path.basename(case_dir)
    blocks_dir = os.path.join(case_dir, 'labels_blocks')
    blocks = sorted((b for b in os.listdir(blocks_dir)
                     if os.path.exists(os.path.join(blocks_dir, b, 'LABELS.nii.gz'))), key=int)
    mri = nib.load(os.path.join(case_dir, 'mri_rotated', 'mri.nii.gz'))

    if args.like:
        ref = nib.load(args.like)
        shape, affine = tuple(int(s) for s in ref.shape[:3]), ref.affine.astype(float)
        res = float(np.mean(ref.header.get_zooms()[:3]))
        tag = 'like-' + os.path.basename(args.like).split('.')[0]
    else:
        res = args.res
        shape, affine = target_grid(mri, res)
        tag = f'{res:g}mm'
    log(f'{case}: {len(blocks)} blocks -> grid {shape} at {res:g} mm')

    # ---- steps 1-3: interpolate inside every block and mosaic -------------------------------
    seg = np.zeros(shape, np.uint16)
    steps = np.zeros(shape, np.uint8)
    best = np.full(shape, np.inf, np.float32)
    tot_missing = tot_sections = 0
    for b in blocks:
        img = nib.load(os.path.join(blocks_dir, b, 'LABELS.nii.gz'))
        r = reconstruct_block(np.asanyarray(img.dataobj), img.affine, shape, affine, res,
                              args.min_section_frac, args.sdt_cap_mm, args.inplane_stride)
        if r is None:
            log(f'  block {b:>2}: nothing to write, skipped')
            continue
        box, lab, dist, bridged, info = r
        take = (lab > 0) & (dist < best[box])
        seg[box][take] = lab[take]
        best[box][take] = dist[take]
        steps[box][take] = np.where(bridged[take], STEP_BRIDGED, STEP_INTERP)
        tot_missing += info['n_missing']; tot_sections += info['n_sections']
        log(f'  block {b:>2}: {info["n_sections"]} sections ({info["n_missing"]} missing), '
            f'spacing {info["dz"]:.2f} mm, {info["n_labels"]} labels, {int(take.sum())} voxels written')
    del best
    log(f'  {tot_missing} of {tot_sections} sections treated as missing')
    n_mosaic = int(np.count_nonzero(seg))

    # ---- MRI tissue mask on the output grid ------------------------------------------------
    M = np.linalg.inv(mri.affine) @ affine
    tissue = ndi.affine_transform((np.asanyarray(mri.dataobj) > 0).astype(np.float32), M[:3, :3], M[:3, 3],
                                  output_shape=shape, order=1, mode='constant', cval=0) >= 0.5
    n_tissue = int(tissue.sum())
    cov_mosaic = np.count_nonzero(seg[tissue]) / n_tissue

    # ---- step 4: remove labels where the MRI has no tissue ----------------------------------
    # labels that mostly lie where the MRI is zero are not visible in the MRI (ventricles, optic
    # nerve, midline structures...): the MRI says nothing about them
    tot = np.bincount(seg.ravel(), minlength=int(seg.max()) + 1)
    outside = np.bincount(seg[~tissue], minlength=len(tot))
    not_in_mri = np.zeros(len(tot), bool)
    not_in_mri[1:] = outside[1:] > args.keep_outside_frac * np.maximum(tot[1:], 1)
    log(f'  labels not visible in the MRI (kept as they are): {np.flatnonzero(not_in_mri).tolist()}')

    n_trim = 0
    if not args.no_trim:
        trim = (seg > 0) & ~tissue & ~not_in_mri[seg]
        n_trim = int(trim.sum())
        trimmed_labels, trimmed_counts = np.unique(seg[trim], return_counts=True)
        seg[trim] = 0
        steps[trim] = STEP_TRIM

    # ---- step 5: MRI-guided fill ------------------------------------------------------------
    n_fill = 0
    if args.mri_fill:
        todo = tissue & (seg == 0)
        nz = np.nonzero(todo.any(axis=(1, 2)))[0], np.nonzero(todo.any(axis=(0, 2)))[0], np.nonzero(todo.any(axis=(0, 1)))[0]
        pad = int(np.ceil(args.max_fill_mm / res)) + 1
        crop = tuple(slice(max(int(a[0]) - pad, 0), min(int(a[-1]) + pad + 1, s)) for a, s in zip(nz, shape))
        sub = seg[crop]
        d, ind = ndi.distance_transform_edt((sub == 0) | not_in_mri[sub], sampling=(res,) * 3,
                                            return_indices=True)
        fill = todo[crop] & (d <= args.max_fill_mm)
        sub[fill] = sub[tuple(i[fill] for i in ind)]
        steps[crop][fill] = STEP_FILL
        n_fill = int(fill.sum())
        del d, ind

    # ---- save and report --------------------------------------------------------------------
    os.makedirs(args.out_dir, exist_ok=True)
    base = os.path.join(args.out_dir, f'{case}_seg_{tag}_v2')
    for arr, suffix, dt in ((seg, '.nii.gz', np.uint16), (steps, '_steps.nii.gz', np.uint8)):
        hdr = nib.Nifti1Header(); hdr.set_data_dtype(dt); hdr.set_xyzt_units('mm')
        nib.save(nib.Nifti1Image(arr, affine, hdr), base + suffix)

    n_lab = int(np.count_nonzero(seg))
    vox_cm3 = res ** 3 / 1000
    log(f'\nSaved {base}.nii.gz and {base}_steps.nii.gz')
    log(f'  {len(np.unique(seg)) - 1} labels, {n_lab} labelled voxels = {n_lab * vox_cm3:.1f} cm^3')
    log(f'  MRI tissue: {n_tissue * vox_cm3:.1f} cm^3; labelled fraction {cov_mosaic:.3f} after the mosaic, '
        f'{np.count_nonzero(seg[tissue]) / n_tissue:.3f} at the end')
    log(f'  interpolated/bridged: {n_mosaic} voxels ({int((steps == STEP_BRIDGED).sum())} bridged in the final volume)')
    log(f'  removed outside MRI tissue: {n_trim} voxels ({n_trim * vox_cm3:.1f} cm^3)')
    if n_trim:
        top = np.argsort(trimmed_counts)[::-1][:8]
        log('    most affected labels (id: fraction of that label removed): ' + ', '.join(
            f'{int(trimmed_labels[i])}: {trimmed_counts[i] / (trimmed_counts[i] + np.count_nonzero(seg == trimmed_labels[i])):.2f}'
            for i in top))
    log(f'  MRI-guided fill: {n_fill} voxels ({n_fill * vox_cm3:.1f} cm^3)')
    log(f'  done in {time.time() - t0:.0f} s')


if __name__ == '__main__':
    main()
