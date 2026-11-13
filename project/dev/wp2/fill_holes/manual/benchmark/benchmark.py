#!/usr/bin/env python
"""
benchmark.py — does restore.py work? Quantitative test with synthetic, known corruptions.

The real cases have no ground truth, so a real case is used as the reference ("GT") and
synthetic NextBrain-like artefacts are added on top of it:
  * block gaps   planes every 9-13 mm (tilted, near-coronal), 0.6-2 mm wide, partial
  * block shifts in-plane translation of each block (N(0, shift_sd) mm)
  * cracks/holes thin holes + blob holes from fill_holes/image_corruption/corrupt_nifti.py

Then ../restore.py is run on the corrupted volume and compared with GT, **only where the
synthetic corruption changed something** (pre-existing artefacts of the case are not scored):
  * error voxels before / after (label != GT), per corruption type, L1 and full labels
  * block steps: estimated vs true relative shift between neighbouring blocks
  * collateral damage: restore.py run on the uncorrupted GT; fraction of GT sulcal
    background (background touching cortex) that gets filled

  python benchmark.py --source 4 <FOMO seg.left.nii.gz> --downsample 2 --seed 0
"""
import argparse
import json
import os
import sys
import time

import nibabel as nib
import numpy as np
from scipy import ndimage as ndi

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                       # manual/
FILL_HOLES = os.path.dirname(ROOT)                 # fill_holes/
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(FILL_HOLES, "image_corruption"))
import restore as R          # noqa: E402
import qc                    # noqa: E402
from corrupt_nifti import corrupt_array   # noqa: E402

TEST = os.path.join(FILL_HOLES, "data/inputs/recon_holes_0.2mm")


def synthetic_blocks(seg, zooms, rng, axis=1, spacing_mm=(9, 13), width_mm=(0.6, 2.0),
                     pitch=(-4, 4), yaw=(0, 3), shift_sd_mm=0.6, partial=(0.5, 1.0)):
    """Block gaps + block shifts. Returns (corrupted, gap_mask, block_index, shifts_vox, planes)."""
    o = [a for a in range(3) if a != axis]
    n = R.normal_from_angles(rng.normal(*pitch), rng.normal(*yaw), axis)
    g = np.ogrid[tuple(slice(0, s) for s in seg.shape)]
    t = sum(g[a].astype(np.float32) * np.float32(zooms[a] * n[a]) for a in range(3))
    tmin, tmax = float(t.min()), float(t.max())
    pos, p = [], tmin + rng.uniform(*spacing_mm)
    while p < tmax:
        pos.append(p)
        p += rng.uniform(*spacing_mm)
    pos = np.array(pos)
    bidx = np.searchsorted(pos, t).astype(np.int16)              # block of every voxel
    nb = len(pos) + 1
    shifts = np.rint(rng.normal(0, shift_sd_mm, (nb, 2)) / np.array([zooms[o[0]], zooms[o[1]]])).astype(int)
    out = np.zeros_like(seg)
    for b in range(nb):
        m = bidx == b
        if not m.any():
            continue
        moved = np.where(m, seg, 0)
        moved = np.roll(moved, tuple(shifts[b]), axis=tuple(o))
        mm = np.roll(m, tuple(shifts[b]), axis=tuple(o))
        out[mm] = moved[mm]
    # gaps (partial: only on one side of a random in-plane line)
    gap = np.zeros(seg.shape, bool)
    widths = rng.uniform(*width_mm, len(pos))
    for k, (p, w) in enumerate(zip(pos, widths)):
        slab = np.abs(t - p) < w / 2
        frac = rng.uniform(*partial)
        if frac < 1:
            ang = rng.uniform(0, 2 * np.pi)
            d = np.cos(ang) * g[o[0]] * zooms[o[0]] + np.sin(ang) * g[o[1]] * zooms[o[1]]
            d = np.broadcast_to(d, seg.shape)
            dvals = d[slab & (seg > 0)] if slab.any() else d.ravel()
            if dvals.size:
                thr = np.quantile(dvals, 1 - frac)
                slab = slab & (d >= thr)
        gap |= slab
    gap &= out > 0
    out[gap] = 0
    planes = [dict(t_mm=float(p), width_mm=float(w)) for p, w in zip(pos, widths)]
    return out, gap, shifts, planes, bidx


def score(gt, x, region, lut):
    """Error counts (full labels and L1) inside `region`."""
    return dict(voxels=int(region.sum()),
                err_full=int((x != gt)[region].sum()),
                err_l1=int((lut[x] != lut[gt])[region].sum()))


def sulcal_background(gt, lut):
    """GT background voxels that touch cortex (sulci) inside the closed brain."""
    hull = ndi.binary_fill_holes(ndi.binary_closing(gt > 0, iterations=3))
    ctx = np.isin(lut[gt], R.CORTEX_L1)
    return (gt == 0) & hull & ndi.binary_dilation(ctx)


def load_reference(source, downsample):
    """source: a test case number (real NextBrain recon, has its own artefacts) or a path to
    a clean label map (e.g. a FOMO seg.left.nii.gz). Returns (gt, zooms, affine, name)."""
    if str(source).isdigit():
        path = os.path.join(TEST, f"case-{source}.nii.gz")
        name = f"case{source}"
    else:
        path = source
        parts = os.path.normpath(path).split(os.sep)
        name = "fomo_" + "_".join(parts[-5:-2]) + "_" + os.path.basename(path).split(".")[1]
    img = nib.load(path)
    gt = np.asarray(img.dataobj)
    if np.issubdtype(gt.dtype, np.floating):
        gt = np.rint(gt).astype(np.int32)
    ds = downsample
    gt = np.ascontiguousarray(gt[::ds, ::ds, ::ds])
    zooms = tuple(float(z) * ds for z in img.header.get_zooms()[:3])
    affine = img.affine.copy()
    affine[:3, :3] *= ds
    return gt, zooms, affine, name


def residual_steps(rep, shifts, bidx, gt, axis=1):
    """Net in-plane displacement of every synthetic block after restoration
    (applied shift + estimated correction); perfect restoration -> equal for all blocks.
    Returns (rms of true steps, rms of residual steps) in voxels."""
    T = np.array(rep.get("block_translations_vox", [[0, 0]]))
    bpl = rep.get("boundary_planes", [])
    o = [a for a in range(3) if a != axis]
    r = []
    for b in range(len(shifts)):
        m = (bidx == b) & (gt > 0)
        if m.sum() < 100:
            continue
        c = np.median(np.argwhere(m), axis=0)
        k = sum(1 for p in bpl if p["coef"][0] * c[o[0]] + p["coef"][1] * c[o[1]] + p["coef"][2] < c[axis])
        r.append(shifts[b] + (T[k] if len(T) > k else 0))
    r = np.array(r, float)
    true = np.array([shifts[b] for b in range(len(shifts)) if ((bidx == b) & (gt > 0)).sum() >= 100], float)
    rms = lambda v: float(np.sqrt((np.diff(v, axis=0) ** 2).sum(1).mean())) if len(v) > 1 else 0.0
    return rms(true), rms(r)


def run(source, downsample, seed, out_dir, shift_sd_mm=0.6, restore_kw=None, verbose=True):
    restore_kw = restore_kw or {}
    os.makedirs(out_dir, exist_ok=True)
    gt, zooms, affine, name = load_reference(source, downsample)
    ds = downsample
    case = name
    lut = R.load_l1_lut(int(gt.max()))
    rng = np.random.default_rng(seed)
    t0 = time.time()

    # --- synthetic corruption --------------------------------------------------------
    x, gap, shifts, planes, bidx = synthetic_blocks(gt, zooms, rng, shift_sd_mm=shift_sd_mm)
    shifted = (x != gt) & ~gap
    x2, params, _ = corrupt_array(x, affine, (-6, 6), "coronal", rng, thickness=0.4,
                                  p_section=0, p_block=0, n_blobs=(15, 25), radius_mm=(1.0, 4.0),
                                  thin_prob=0.6, thin_sections=(1, 3), p_slab=0, irregularity=0.5)
    holes = (x2 != x)
    x = x2
    corrupted = x != gt
    R.log(f"{case} @ {zooms[0]:.1f} mm: {len(planes)} synthetic gaps, "
          f"{gap.sum():,} gap vox, {shifted.sum():,} shifted vox, {holes.sum():,} hole vox", verbose)

    # --- restore -----------------------------------------------------------------------
    y, steps, rep = R.restore_array(x, zooms, lut, affine, verbose=verbose, **restore_kw)

    regions = {"all_corrupted": corrupted, "gaps": gap & corrupted, "shifted": shifted & corrupted,
               "holes_cracks": holes & corrupted}
    res = dict(case=case, voxel_mm=zooms[0], seed=seed, n_planes_true=len(planes), restore=rep,
               shifts_true_vox=shifts.tolist())
    res["scores"] = {k: dict(before=score(gt, x, m, lut), after=score(gt, y, m, lut))
                     for k, m in regions.items()}
    # changes outside the corrupted region (fixes of real artefacts, or damage)
    outside = ~ndi.binary_dilation(corrupted, iterations=2)
    res["changed_outside_corruption"] = int(((y != x) & outside).sum())

    # plane detection: matched planes (true plane position at centre vs detected)
    res["n_planes_detected"] = len(rep.get("planes", []))

    # block steps: residual relative shift between consecutive blocks after restoration
    res["step_rms_vox_before"], res["step_rms_vox_after"] = residual_steps(rep, shifts, bidx, gt)

    # --- collateral damage on the uncorrupted case ----------------------------------------
    y0, steps0, rep0 = R.restore_array(gt, zooms, lut, affine, verbose=False, **restore_kw)
    sulc = sulcal_background(gt, lut)
    res["gt_run"] = dict(changed=int((y0 != gt).sum()), sulcal_bg=int(sulc.sum()),
                         sulcal_filled=int(((y0 != 0) & sulc).sum()),
                         n_planes=len(rep0.get("planes", [])))
    res["seconds"] = round(time.time() - t0, 1)

    # --- QC figures --------------------------------------------------------------------------
    stem = os.path.join(out_dir, f"{case}_ds{ds}_seed{seed}")
    fg = gt > 0
    at = [int(np.mean(np.nonzero(fg.any(axis=tuple(a for a in range(3) if a != ax))))) for ax in range(3)]
    qc.qc_figure(lut[x], lut[y], steps, stem + "_restored.png", at=at,
                 title=f"{case}: synthetic corruption -> restored")
    err = np.zeros(gt.shape, np.uint8)
    err[(x != gt)] = 1                     # error before
    err[(y != gt) & corrupted] = 3         # remaining error inside corrupted region
    err[(y != gt) & ~corrupted] = 4        # new difference outside
    qc.qc_figure(lut[gt], lut[y], err, stem + "_errors.png", at=at,
                 title="reference vs restored | 3rd column: red = corrupted (before), "
                       "green = still wrong, yellow = changed outside the corruption",
                 labels=("reference", "restored", "errors"))
    with open(stem + ".json", "w") as f:
        json.dump(res, f, indent=1)
    return res


def summary(res):
    lines = [f"{res['case']} @ {res['voxel_mm']:.1f} mm  seed {res['seed']}  "
             f"(planes: {res['n_planes_true']} synthetic, {res['n_planes_detected']} detected)"]
    for k, s in res["scores"].items():
        b, a = s["before"], s["after"]
        if b["voxels"] == 0:
            continue
        lines.append(f"  {k:14s} {b['voxels']:>10,} vox | L1 errors {b['err_l1']:>10,} -> {a['err_l1']:>10,} "
                     f"({100 * (1 - a['err_l1'] / max(b['err_l1'], 1)):5.1f}% fixed) | "
                     f"full-label errors {b['err_full']:>10,} -> {a['err_full']:>10,} "
                     f"({100 * (1 - a['err_full'] / max(b['err_full'], 1)):5.1f}% fixed)")
    g = res["gt_run"]
    lines.append(f"  on uncorrupted case: {g['changed']:,} vox changed, sulcal background filled "
                 f"{g['sulcal_filled']:,}/{g['sulcal_bg']:,} ({100 * g['sulcal_filled'] / max(g['sulcal_bg'], 1):.2f}%)")
    lines.append(f"  changed outside the synthetic corruption: {res['changed_outside_corruption']:,} vox")
    lines.append(f"  block steps (rms, vox): {res['step_rms_vox_before']:.2f} -> {res['step_rms_vox_after']:.2f}")
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", nargs="+", default=["4"],
                   help="test case numbers (1-5) and/or paths to clean label maps (FOMO seg.*.nii.gz)")
    p.add_argument("--downsample", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--shift-sd-mm", type=float, default=0.6)
    p.add_argument("--no-shifts", action="store_true", help="disable the shift correction step")
    p.add_argument("--out", default=os.path.join(HERE, "results/benchmark"))
    a = p.parse_args()
    kw = dict(do_shifts=not a.no_shifts)
    for c in a.source:
        res = run(c, a.downsample, a.seed, a.out, a.shift_sd_mm, kw)
        print(summary(res), flush=True)


if __name__ == "__main__":
    main()
