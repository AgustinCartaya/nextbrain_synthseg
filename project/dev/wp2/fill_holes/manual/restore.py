#!/usr/bin/env python
"""
restore.py — classical (no deep learning) restoration of NextBrain-style label maps.

NextBrain hemispheres are reconstructed from ~1 cm tissue blocks cut in near-coronal slabs,
each sectioned for histology. The 3D label maps therefore show:
  * GAPS      thin background slabs at block boundaries (planes tilted a few degrees
              from coronal, often crossing only part of the brain)
  * STEPS     in-plane misalignment of a block relative to its neighbours
  * CRACKS    thin background cracks inside blocks (missing / torn sections)
  * HOLES     enclosed background cavities inside tissue

Pipeline (each step can be switched off):
  1. detect_planes      bounded short background runs along the slicing axis; their
                        midpoints concentrate on the gap planes -> estimate the global
                        normal, find peaks, robustly fit one plane per gap
  2. correct_shifts     per gap plane: in-plane shift between the label slices on both
                        sides, minus the natural anatomical drift measured inside each
                        block; blocks are translated with a regularised least-squares
                        solution (integer voxels, labels are never interpolated)
  3. fill_gaps          voxels of each gap slab -> nearest known label (3D). Background of
                        sulci next to the gap stays "known", so sulci crossing a gap stay open
  4. fill_cracks        short background runs (any axis) bounded at both ends by the same
                        NON-cortical tissue class (WM, deep GM, brainstem, ...). Sulci are
                        bounded by cortex, so they are not touched
  5. fill_holes         background components not connected to the outside and whose
                        border is mostly non-cortical tissue

Works on the full label set (any label ids). Tissue classes (to know which labels are
cortex) come from the NextBrain ontology + L1 mapping via wp1 merge_labels.py
(or use --input-is-l1 for L1 inputs).

CLI:
  python restore.py input.nii.gz output.nii.gz
"""
import argparse
import json
import os
import time

import nibabel as nib
import numpy as np
from scipy import ndimage as ndi

# L1 classes (mapping_l1.json)
L1 = {"Background": 0, "Brainstem": 1, "CSF": 2, "CerebellarGM": 3, "CerebellarWM": 4,
      "CerebralGM": 5, "CerebralWM": 6, "DeepGM": 7, "Other": 8}
CORTEX_L1 = (3, 5)          # labels that line sulci (background next to them may be real)

STEP_CODES = {"gap": 1, "crack": 2, "hole": 3, "shift": 4}


def log(msg, verbose=True):
    if verbose:
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# =============================================================================
# Label classes
# =============================================================================
MERGE_DIR = "/home/agustin/phd/synthseg_nextbrain/project/dev/wp1/create_merged_segmentations"
ONTOLOGY = "/home/agustin/phd/synthseg_nextbrain/data/basics/nextbrain_ontology_v1.csv"
MAPPING_L1 = os.path.join(MERGE_DIR, "data/inputs/mapping_l1.json")


# raw recon labels that are not in the ontology, with the L1 class they get in
# deep_learning/data/test/changed_recon_holes_0.2mm_L1 (3 = cerebral cortex)
EXTRA_L1 = {1: 1, 2: 2, 3: 5}


def load_l1_lut(max_label, ontology=ONTOLOGY, mapping=MAPPING_L1, input_is_l1=False, extra=None):
    """
    LUT original_id -> L1 class, built with wp1/create_merged_segmentations/merge_labels.py
    (same code that produced the *_L1 files). Labels missing from the ontology -> 8 (Other,
    i.e. treated as non-cortical tissue).
    """
    if input_is_l1:
        lut = np.full(max_label + 1, 8, np.uint8)
        ids = np.arange(min(max_label, 8) + 1)
        lut[ids] = ids
        return lut
    import sys
    sys.path.insert(0, MERGE_DIR)
    import merge_labels as ml
    onto = ml.load_ontology(ontology)
    with open(mapping) as f:
        mp = json.load(f)
    label_map = ml.build_label_map(onto, "L1", mp)
    lut = ml.build_lut(onto, label_map, max_label, unmapped_value=0, unknown_value=8)
    for k, v in (EXTRA_L1 if extra is None else extra).items():
        if k < len(lut):
            lut[k] = v
    return lut[: max(max_label, 1) + 1].astype(np.uint8) if len(lut) > max_label else lut.astype(np.uint8)


# =============================================================================
# Runs of background along one axis
# =============================================================================
def bg_runs(seg, axis=1):
    """
    For every voxel along `axis`: index of the previous / next tissue voxel.
    Returns (prev, nxt) int32 arrays (prev=-1 / nxt=L when there is none).
    A background voxel belongs to a run [prev+1, nxt-1] of length nxt-prev-1.
    """
    x = np.moveaxis(seg, axis, -1)
    L = x.shape[-1]
    tissue = x > 0
    idx = np.arange(L, dtype=np.int32)
    prev = np.where(tissue, idx, np.int32(-1))
    np.maximum.accumulate(prev, axis=-1, out=prev)
    nxt = np.where(tissue, idx, np.int32(L))[..., ::-1]
    nxt = np.minimum.accumulate(nxt, axis=-1)[..., ::-1]
    return np.moveaxis(prev, -1, axis), np.ascontiguousarray(np.moveaxis(nxt, -1, axis))


def short_run_points(seg, axis, max_len):
    """Midpoints, lengths and end labels of bounded background runs of length <= max_len."""
    prev, nxt = bg_runs(seg, axis)
    L = seg.shape[axis]
    j = np.arange(L).reshape([-1 if a == axis else 1 for a in range(3)])
    start = (seg == 0) & (prev == j - 1) & (prev >= 0) & (nxt < L) & (nxt - prev - 1 <= max_len)
    pts = np.argwhere(start)
    p_prev = prev[start]
    p_next = nxt[start]
    mid = pts.astype(np.float32)
    mid[:, axis] = (p_prev + p_next) / 2.0
    return mid, (p_next - p_prev - 1).astype(np.int32), p_prev, p_next


# =============================================================================
# 1. Gap planes
# =============================================================================
def normal_from_angles(pitch, yaw, axis=1):
    """Unit normal (voxel axes) near `axis`, tilted by pitch (towards axis 2) and yaw (axis 0)."""
    p, y = np.deg2rad(pitch), np.deg2rad(yaw)
    n = np.zeros(3)
    o = [a for a in range(3) if a != axis]                      # o[0] ~ x, o[1] ~ z
    n[axis] = np.cos(y) * np.cos(p)
    n[o[1]] = np.sin(p)
    n[o[0]] = -np.sin(y) * np.cos(p)
    return n


def spikiness(h, win):
    base = ndi.median_filter(h, size=win, mode="nearest")
    r = np.clip(h - base, 0, None)
    return float(np.sqrt(np.mean(r ** 2)) / (np.mean(h) + 1e-9))


def estimate_normal(mid, axis, zooms, angle_max=25, step=1.0, bin_mm=0.4, win_mm=6.0, rng=None):
    """Grid search of the normal maximising the spikiness of the run-midpoint histogram."""
    rng = rng or np.random.default_rng(0)
    if len(mid) > 400_000:
        mid = mid[rng.choice(len(mid), 400_000, replace=False)]
    P = mid * np.asarray(zooms, np.float32)
    win = max(3, int(round(win_mm / bin_mm)) | 1)

    def score(pitch, yaw):
        t = P @ normal_from_angles(pitch, yaw, axis).astype(np.float32)
        h = np.bincount(((t - t.min()) / bin_mm).astype(np.int64)).astype(float)
        return spikiness(h, win)

    best = max(((score(p, y), p, y) for p in np.arange(-angle_max, angle_max + 1e-6, 2 * step)
                for y in np.arange(-angle_max, angle_max + 1e-6, 2 * step)))
    _, p0, y0 = best
    best = max(((score(p, y), p, y) for p in np.arange(p0 - 2 * step, p0 + 2 * step + 1e-6, step / 2)
                for y in np.arange(y0 - 2 * step, y0 + 2 * step + 1e-6, step / 2)))
    return best[1], best[2], best[0]


def fit_plane_ransac(pts, axis, n0, max_dev_deg, tol, rng, iters=300):
    """Robust plane j = a*u + b*v + c (j along axis), restricted to <= max_dev_deg from n0."""
    o = [a for a in range(3) if a != axis]
    u, v, j = pts[:, o[0]], pts[:, o[1]], pts[:, axis]
    best_in, best = None, None
    cos_lim = np.cos(np.deg2rad(max_dev_deg))
    for _ in range(iters):
        s = pts[rng.choice(len(pts), 3, replace=False)]
        nrm = np.cross(s[1] - s[0], s[2] - s[0])
        if np.linalg.norm(nrm) < 1e-6:
            continue
        nrm /= np.linalg.norm(nrm)
        if abs(nrm @ n0) < cos_lim:
            continue
        d = np.abs((pts - s[0]) @ nrm)
        inl = d < tol
        if best_in is None or inl.sum() > best_in.sum():
            best_in = inl
    if best_in is None or best_in.sum() < 10:
        return None, None
    # least-squares refinement on inliers: j = a u + b v + c
    A = np.stack([u[best_in], v[best_in], np.ones(best_in.sum())], 1)
    coef, *_ = np.linalg.lstsq(A, j[best_in], rcond=None)
    for _ in range(2):
        res = np.abs(A @ coef - j[best_in])
        keep = res < tol
        coef, *_ = np.linalg.lstsq(A[keep], j[best_in][keep], rcond=None)
    d = np.abs(np.stack([u, v, np.ones(len(u))], 1) @ coef - j)
    return coef, d < tol


def detect_planes(seg, zooms, axis=1, max_gap_mm=2.4, angle_max=25, min_sep_mm=4.0,
                  peak_z=4.0, min_coverage=0.02, max_dev_deg=6.0, lut=None, min_noncortex=0.0,
                  verbose=True, seed=0):
    """
    Find gap planes. Returns (planes, info). Each plane: dict(coef=[a,b,c] with
    j = a*u + b*v + c in voxel indices (u, v = the two other axes in increasing order),
    n_runs, coverage, width_vox (median gap length), normal).
    """
    rng = np.random.default_rng(seed)
    vox = float(zooms[axis])
    max_len = max(1, int(round(max_gap_mm / vox)))
    mid, length, ip, inx = short_run_points(seg, axis, max_len)
    # runs bounded on both ends by non-cortical tissue (gaps cross WM, sulci are lined by cortex)
    if lut is not None:
        pts = np.round(mid).astype(np.int64)
        ca, cb = pts.copy(), pts.copy()
        ca[:, axis], cb[:, axis] = ip, inx
        la, lb = lut[seg[tuple(ca.T)]], lut[seg[tuple(cb.T)]]
        noncx = ~np.isin(la, CORTEX_L1) & ~np.isin(lb, CORTEX_L1)
    else:
        noncx = np.ones(len(mid), bool)
    log(f"  {len(mid):,} short background runs (<= {max_gap_mm} mm) along axis {axis}", verbose)
    pitch, yaw, sc = estimate_normal(mid, axis, zooms, angle_max=angle_max)
    n = normal_from_angles(pitch, yaw, axis)
    log(f"  global normal: pitch={pitch:.1f} yaw={yaw:.1f} deg (spikiness {sc:.2f})", verbose)

    # histogram along the normal, peaks above a robust baseline
    P = mid * np.asarray(zooms, np.float32)
    t = P @ n.astype(np.float32)
    bin_mm = vox
    t0 = t.min()
    b = ((t - t0) / bin_mm).astype(np.int64)
    h = np.bincount(b).astype(float)
    hs = ndi.uniform_filter1d(h, 3)
    base = ndi.median_filter(hs, size=int(round(8.0 / bin_mm)) | 1, mode="nearest")
    resid = hs - base
    mad = np.median(np.abs(resid - np.median(resid))) * 1.4826 + 1e-9
    cand = np.flatnonzero((resid > peak_z * mad) & (resid == ndi.maximum_filter1d(resid, int(round(min_sep_mm / bin_mm)) | 1)))
    cand = cand[np.argsort(-resid[cand])]
    # columns of brain crossing each position (for coverage)
    o = [a for a in range(3) if a != axis]
    proj_area = (seg > 0).any(axis=axis).sum()

    planes = []
    taken = np.zeros(len(mid), bool)
    for c in cand:
        sel = (np.abs(b - c) <= int(round(2.0 / bin_mm))) & ~taken
        if sel.sum() < 50:
            continue
        coef, inl = fit_plane_ransac(mid[sel], axis, n, max_dev_deg, tol=1.5, rng=rng)
        if coef is None:
            continue
        idx = np.flatnonzero(sel)[inl]
        cov = len(idx) / proj_area
        fnc = float(noncx[idx].mean())
        if cov < min_coverage or fnc < min_noncortex:
            if verbose:
                log(f"    rejected candidate plane: coverage {cov:.3f}, non-cortical runs {fnc:.2f}", verbose)
            continue
        taken[idx] = True
        nrm = np.zeros(3)
        nrm[axis] = 1.0
        nrm[o[0]], nrm[o[1]] = -coef[0], -coef[1]
        nrm /= np.linalg.norm(nrm)
        planes.append(dict(coef=[float(x) for x in coef], n_runs=int(len(idx)), coverage=float(cov),
                           width_vox=float(np.median(length[idx])),
                           width_p90_vox=float(np.percentile(length[idx], 90)),
                           normal=nrm.tolist(), peak_score=float(resid[c] / mad), noncortex_frac=fnc))
    # sort along the axis (position at the centre of the volume)
    cu, cv = seg.shape[o[0]] / 2, seg.shape[o[1]] / 2
    planes.sort(key=lambda p: p["coef"][0] * cu + p["coef"][1] * cv + p["coef"][2])
    info = dict(pitch_deg=pitch, yaw_deg=yaw, spikiness=sc, n_short_runs=int(len(mid)))
    log(f"  {len(planes)} gap planes (coverage / non-cortical runs): " +
        ", ".join(f"{p['coverage']:.2f}/{p['noncortex_frac']:.2f}" for p in planes), verbose)
    return planes, info


def plane_map(plane, shape, axis):
    """j position (float) of the plane for every (u, v) column."""
    o = [a for a in range(3) if a != axis]
    a, b, c = plane["coef"]
    u = np.arange(shape[o[0]], dtype=np.float32)[:, None]
    v = np.arange(shape[o[1]], dtype=np.float32)[None, :]
    return a * u + b * v + c


# =============================================================================
# 2. Block misalignment
# =============================================================================
def sample_along(seg, jmap, offset, axis):
    """2D label image seg[u, round(jmap + offset), v] (0 outside the volume)."""
    x = np.moveaxis(seg, axis, -1)                     # (U, V, J)
    j = np.rint(jmap + offset).astype(np.int64)
    ok = (j >= 0) & (j < x.shape[-1])
    out = np.zeros(jmap.shape, seg.dtype)
    uu, vv = np.nonzero(ok)
    out[uu, vv] = x[uu, vv, j[uu, vv]]
    return out


def best_shift(A, B, classes, max_shift):
    """
    Integer in-plane shift d (du, dv) maximising agreement of B(p + d) with A(p),
    over tissue classes, by FFT cross-correlation. Returns (d, score_gain).
    """
    U, V = A.shape
    fa = np.zeros((U + max_shift, V + max_shift))
    acc = None
    for c in classes:
        a = (A == c).astype(np.float32)
        bb = (B == c).astype(np.float32)
        if a.sum() < 50 or bb.sum() < 50:
            continue
        s = (U + 2 * max_shift, V + 2 * max_shift)
        cc = np.fft.irfft2(np.fft.rfft2(a, s) * np.conj(np.fft.rfft2(bb, s)), s)
        acc = cc if acc is None else acc + cc
    if acc is None:
        return np.zeros(2, int), 0.0
    shifts = np.arange(-max_shift, max_shift + 1)
    sub = acc[np.ix_(shifts % acc.shape[0], shifts % acc.shape[1])]
    k = np.unravel_index(np.argmax(sub), sub.shape)
    d = -np.array([shifts[k[0]], shifts[k[1]]])
    gain = float(sub[k] / max(sub[max_shift, max_shift], 1))
    return d, gain


def estimate_misalignments(seg_l1, planes, zooms, axis=1, max_shift_mm=3.0,
                           classes=(1, 2, 3, 4, 5, 6, 7), verbose=True):
    """
    For each gap plane: in-plane shift across the gap minus the natural drift of the anatomy,
    measured inside the two neighbouring blocks over the SAME distance as across the gap
    (so the drift estimate is not amplified).
    Returns list of dict(d_vox=[du, dv], across, drift_before, drift_after).
    """
    vox = float(zooms[axis])
    ms = max(1, int(round(max_shift_mm / vox)))
    out = []
    for p in planes:
        jm = plane_map(p, seg_l1.shape, axis)
        hw = int(np.ceil(p["width_p90_vox"] / 2)) + 1
        pr = 2 * hw
        a1 = sample_along(seg_l1, jm, -hw, axis)
        a2 = sample_along(seg_l1, jm, -hw - pr, axis)
        b1 = sample_along(seg_l1, jm, hw, axis)
        b2 = sample_along(seg_l1, jm, hw + pr, axis)
        d_x, g_x = best_shift(a1, b1, classes, ms)           # across the gap (2 hw apart)
        d_a, _ = best_shift(a2, a1, classes, ms)             # inside previous block (pr apart)
        d_b, _ = best_shift(b1, b2, classes, ms)             # inside next block
        drift = (d_a + d_b) / 2.0                             # expected natural shift across
        mis = d_x - drift
        out.append(dict(d_vox=[float(m) for m in mis], across=d_x.tolist(), gain=g_x,
                        drift_before=d_a.tolist(), drift_after=d_b.tolist()))
    log("  misalignment per plane (vox): " +
        " ".join(f"({m['d_vox'][0]:+.0f},{m['d_vox'][1]:+.0f})" for m in out), verbose)
    return out


def boundary_planes(planes, shape, axis, zooms, min_sep_mm=4.0):
    """Planes used as block boundaries: planes closer than min_sep_mm (at the volume centre)
    are the same gap fitted twice -> keep the one with the largest coverage."""
    o = [a for a in range(3) if a != axis]
    cu, cv = shape[o[0]] / 2, shape[o[1]] / 2
    pos = [p["coef"][0] * cu + p["coef"][1] * cv + p["coef"][2] for p in planes]
    sep = min_sep_mm / float(zooms[axis])
    groups = []
    for i in np.argsort(pos):
        if groups and pos[i] - pos[groups[-1][-1]] < sep:
            groups[-1].append(i)
        else:
            groups.append([i])
    return [planes[max(g, key=lambda k: planes[k]["coverage"])] for g in groups]


def solve_block_translations(mis, lam=0.3, min_shift_vox=1.0):
    """
    Blocks 0..K separated by K planes. Find translations T_k (2D) with
    T_{k+1} - T_k ~ -mis_k (undo the step) and a small penalty lam*|T_k|^2 (avoid drift).
    """
    K = len(mis)
    if K == 0:
        return np.zeros((1, 2))
    D = np.array([m["d_vox"] for m in mis], float)
    D[np.abs(D) < min_shift_vox] = 0
    n = K + 1
    A = np.zeros((K + n, n))
    for k in range(K):
        A[k, k], A[k, k + 1] = -1, 1
    A[K:, :] = np.sqrt(lam) * np.eye(n)
    T = np.zeros((n, 2))
    for c in range(2):
        rhs = np.concatenate([-D[:, c], np.zeros(n)])
        T[:, c] = np.linalg.lstsq(A, rhs, rcond=None)[0]
    return np.rint(T).astype(int)


def apply_block_translations(seg, planes, T, axis=1):
    """Translate each block (region between consecutive planes) in-plane by T[k] voxels."""
    if not np.any(T):
        return seg.copy(), np.zeros(seg.shape, bool)
    x = np.moveaxis(seg, axis, -1)                   # (U, V, J) view
    U, V, J = x.shape
    jj = np.arange(J, dtype=np.float32)
    bounds = [np.full((U, V), -np.inf, np.float32)] + [plane_map(p, seg.shape, axis) for p in planes] + \
             [np.full((U, V), np.inf, np.float32)]
    out = np.zeros_like(x)
    written = np.zeros(x.shape, bool)
    for k in range(len(bounds) - 1):
        du, dv = T[k]
        # voxels of block k: bounds[k] < j <= bounds[k+1]   (per column)
        lo = np.ceil(bounds[k] + 1e-3)
        hi = np.floor(bounds[k + 1])
        jl = int(np.clip(np.nanmin(np.where(np.isfinite(lo), lo, 0)), 0, J))
        jh = int(np.clip(np.nanmax(np.where(np.isfinite(hi), hi, J - 1)), -1, J - 1)) + 1
        if jh <= jl:
            continue
        sub = x[:, :, jl:jh]
        m = (jj[None, None, jl:jh] >= lo[..., None]) & (jj[None, None, jl:jh] <= hi[..., None])
        src_u = slice(max(0, -du), min(U, U - du))
        dst_u = slice(max(0, du), min(U, U + du))
        src_v = slice(max(0, -dv), min(V, V - dv))
        dst_v = slice(max(0, dv), min(V, V + dv))
        mm = m[src_u, src_v]
        o = out[dst_u, dst_v, jl:jh]
        w = written[dst_u, dst_v, jl:jh]
        o[mm] = sub[src_u, src_v][mm]
        w[mm] = True
    out = np.moveaxis(out, -1, axis)
    moved = np.moveaxis(written, -1, axis)
    return np.ascontiguousarray(out), moved


# =============================================================================
# Nearest-label filling
# =============================================================================
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


# =============================================================================
# 3. Gap filling
# =============================================================================
def gap_voxels(seg, planes, zooms, axis=1, max_gap_mm=2.4, extra_vox=1):
    """
    Background voxels of the gap slabs: bounded short runs along `axis` that cross the plane,
    restricted to |j - plane| <= half the gap width (p90) + extra_vox (so sulcus parts
    away from the plane stay 'known' background).
    """
    vox = float(zooms[axis])
    max_len = max(1, int(round(max_gap_mm / vox)))
    prev, nxt = bg_runs(seg, axis)
    L = seg.shape[axis]
    x_prev = np.moveaxis(prev, axis, -1)
    x_next = np.moveaxis(nxt, axis, -1)
    x_seg = np.moveaxis(seg, axis, -1)
    gap = np.zeros(x_seg.shape, bool)
    for p in planes:
        jm = plane_map(p, seg.shape, axis)
        hw = p["width_p90_vox"] / 2 + extra_vox
        jl = int(max(0, np.floor(jm.min() - hw)))
        jh = int(min(L, np.ceil(jm.max() + hw) + 1))
        if jh <= jl:
            continue
        jj = np.arange(jl, jh, dtype=np.float32)[None, None, :]
        pr, nx = x_prev[..., jl:jh], x_next[..., jl:jh]
        m = (x_seg[..., jl:jh] == 0) & (pr >= 0) & (nx < L) & (nx - pr - 1 <= max_len)
        m &= (pr < jm[..., None] + 1) & (nx > jm[..., None] - 1)         # run crosses the plane
        m &= np.abs(jj - jm[..., None]) <= hw
        gap[..., jl:jh] |= m
    return np.ascontiguousarray(np.moveaxis(gap, -1, axis))


# =============================================================================
# 4. Cracks
# =============================================================================
def crack_voxels(seg, lut, zooms, max_crack_mm=1.0, axes=(0, 1, 2), cortex=CORTEX_L1):
    """Background voxels in short runs bounded on both ends by the same non-cortical class."""
    crack = np.zeros(seg.shape, bool)
    for ax in axes:
        max_len = max(1, int(round(max_crack_mm / float(zooms[ax]))))
        prev, nxt = bg_runs(seg, ax)
        L = seg.shape[ax]
        m = (seg == 0) & (prev >= 0) & (nxt < L) & (nxt - prev - 1 <= max_len)
        if not m.any():
            continue
        pos = np.nonzero(m)
        ca = list(pos)
        cb = list(pos)
        ca[ax] = prev[m]
        cb[ax] = nxt[m]
        la = lut[seg[tuple(ca)]]
        lb = lut[seg[tuple(cb)]]
        ok = (la == lb) & ~np.isin(la, cortex) & (la > 0)
        crack[tuple(p[ok] for p in pos)] = True
    return crack


# =============================================================================
# 5. Enclosed holes
# =============================================================================
def hole_voxels(seg, lut, max_cortex_frac=0.3, max_csf_frac=0.02, min_vox=1, cortex=CORTEX_L1,
                csf=(2,)):
    """Background components not connected to the volume border whose border voxels are
    mostly non-cortical tissue. Components touching CSF (> max_csf_frac of the border) are
    kept: in NextBrain most of the lateral ventricle is unlabelled background."""
    bg = seg == 0
    lab, n = ndi.label(bg)
    if n == 0:
        return np.zeros(seg.shape, bool)
    border = np.unique(np.concatenate([lab[0].ravel(), lab[-1].ravel(), lab[:, 0].ravel(),
                                       lab[:, -1].ravel(), lab[:, :, 0].ravel(), lab[:, :, -1].ravel()]))
    inner = np.ones(n + 1, bool)
    inner[border] = False
    inner[0] = False
    # border composition: tissue voxels adjacent to each component
    tissue = ~bg
    ring = ndi.binary_dilation(bg, iterations=1) & tissue
    # assign each ring voxel to an adjacent component (max label among 6-neighbours)
    nb = ndi.maximum_filter(lab, footprint=ndi.generate_binary_structure(3, 1))
    comp_of_ring = nb[ring]
    cls = lut[seg[ring]]
    tot = np.bincount(comp_of_ring, minlength=n + 1)
    frac = np.bincount(comp_of_ring, weights=np.isin(cls, cortex), minlength=n + 1) / np.maximum(tot, 1)
    fcsf = np.bincount(comp_of_ring, weights=np.isin(cls, csf), minlength=n + 1) / np.maximum(tot, 1)
    size = np.bincount(lab.ravel(), minlength=n + 1)
    keep = inner & (frac <= max_cortex_frac) & (fcsf <= max_csf_frac) & (size >= min_vox) & (tot > 0)
    return keep[lab]


# =============================================================================
# Full pipeline
# =============================================================================
DEFAULTS = dict(axis=None, max_gap_mm=2.4, angle_max=25, min_coverage=0.02, max_dev_deg=10.0,
                min_noncortex=0.08,
                peak_z=4.0, do_shifts=True, max_shift_mm=3.0, shift_lambda=0.3, min_shift_mm=0.4,
                do_gaps=True, all_axes=False, do_cracks=True, max_crack_mm=1.0, crack_iters=2,
                do_holes=True, hole_max_cortex_frac=0.3, hole_max_csf_frac=0.02)


def slicing_axis(affine):
    """Array axis most aligned with world A-P (NextBrain blocks are coronal slabs)."""
    A = affine[:3, :3]
    return int(np.argmax(np.abs(A[1, :]) / np.linalg.norm(A, axis=0)))


def restore_array(seg, zooms, lut, affine=None, verbose=True, **kw):
    """
    Restore a label array. Returns (restored, steps, report):
      steps  uint8 volume, code of the step that changed each voxel (see STEP_CODES)
    """
    o = {**DEFAULTS, **kw}
    axis = o["axis"] if o["axis"] is not None else (slicing_axis(affine) if affine is not None else 1)
    rep = dict(axis=axis, params=o)
    x = seg.copy()
    steps = np.zeros(seg.shape, np.uint8)
    l1 = lambda a: lut[a]

    t = time.time()
    log("1. detecting gap planes", verbose)
    planes, info = detect_planes(x, zooms, axis, o["max_gap_mm"], o["angle_max"],
                                 min_coverage=o["min_coverage"], max_dev_deg=o["max_dev_deg"],
                                 peak_z=o["peak_z"], lut=lut, min_noncortex=o["min_noncortex"],
                                 verbose=verbose)
    rep.update(normal=info, planes=planes)

    if o["do_shifts"] and planes:
        log("2. block misalignment", verbose)
        bplanes = boundary_planes(planes, x.shape, axis, zooms)
        mis = estimate_misalignments(l1(x), bplanes, zooms, axis, max_shift_mm=o["max_shift_mm"],
                                     verbose=verbose)
        T = solve_block_translations(mis, o["shift_lambda"], o["min_shift_mm"] / float(zooms[axis]))
        rep.update(misalignment=mis, block_translations_vox=T.tolist(), boundary_planes=bplanes)
        log(f"  block translations (vox): {T.tolist()}", verbose)
        if np.any(T):
            y, _ = apply_block_translations(x, bplanes, T, axis)
            steps[(y != x)] = STEP_CODES["shift"]
            x = y

    if o["do_gaps"]:
        log("3. filling gaps", verbose)
        rep["gap_voxels"], rep["gap_filled"], rep["other_axes_planes"] = 0, 0, {}
        # main slicing axis first (planes already detected), then the other axes:
        # some blocks (e.g. around brainstem / cerebellum) are cut in other orientations
        gap_axes = [axis] + ([a for a in range(3) if a != axis] if o["all_axes"] else [])
        for ax in gap_axes:
            if ax == axis:
                pl = planes
            else:
                pl, _ = detect_planes(x, zooms, ax, o["max_gap_mm"], o["angle_max"],
                                      min_coverage=o["min_coverage"], max_dev_deg=o["max_dev_deg"],
                                      peak_z=o["peak_z"], lut=lut, min_noncortex=o["min_noncortex"],
                                      verbose=verbose)
                rep["other_axes_planes"][ax] = pl
            if not pl:
                continue
            g = gap_voxels(x, pl, zooms, ax, o["max_gap_mm"])
            y = fill_nearest(x, g)
            ch = y != x
            steps[ch] = STEP_CODES["gap"]
            rep["gap_voxels"] += int(g.sum())
            rep["gap_filled"] += int(ch.sum())
            log(f"  axis {ax}: {len(pl)} planes, {g.sum():,} gap voxels, {ch.sum():,} filled with tissue", verbose)
            x = y

    if o["do_cracks"]:
        log("4. filling cracks", verbose)
        tot = 0
        for it in range(o["crack_iters"]):
            c = crack_voxels(x, lut, zooms, o["max_crack_mm"])
            if not c.any():
                break
            y = fill_nearest(x, c, margin=3)
            ch = y != x
            steps[ch] = STEP_CODES["crack"]
            tot += int(ch.sum())
            x = y
        rep["crack_filled"] = tot
        log(f"  {tot:,} crack voxels filled", verbose)

    if o["do_holes"]:
        log("5. filling enclosed holes", verbose)
        h = hole_voxels(x, lut, o["hole_max_cortex_frac"], o["hole_max_csf_frac"])
        y = fill_nearest(x, h)
        ch = y != x
        steps[ch] = STEP_CODES["hole"]
        rep["hole_filled"] = int(ch.sum())
        log(f"  {ch.sum():,} hole voxels filled", verbose)
        x = y

    rep["changed_voxels"] = int((x != seg).sum())
    rep["seconds"] = round(time.time() - t, 1)
    log(f"done in {rep['seconds']} s, {rep['changed_voxels']:,} voxels changed", verbose)
    return x, steps, rep


def restore_nifti(path, mapping=None, verbose=True, **kw):
    """mapping: None (NextBrain labels, default ontology + L1 mapping) or a dict of
    load_l1_lut keyword arguments, e.g. {"input_is_l1": True}."""
    img = nib.load(path)
    seg = np.asarray(img.dataobj)
    if np.issubdtype(seg.dtype, np.floating):
        seg = np.rint(seg).astype(np.int32)
    lut = load_l1_lut(int(seg.max()), **(mapping or {}))
    x, steps, rep = restore_array(seg, img.header.get_zooms()[:3], lut, img.affine, verbose, **kw)
    out = nib.Nifti1Image(x.astype(seg.dtype), img.affine, img.header)
    st = nib.Nifti1Image(steps, img.affine)
    return out, st, rep


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input")
    p.add_argument("output")
    p.add_argument("--ontology", default=ONTOLOGY)
    p.add_argument("--l1-mapping", default=MAPPING_L1)
    p.add_argument("--input-is-l1", action="store_true", help="input labels are already L1 (0-8)")
    p.add_argument("--steps-out", default=None, help="save the per-voxel step codes (default: <output>_steps.nii.gz)")
    for k, v in DEFAULTS.items():
        if isinstance(v, bool):
            name = k[3:] if k.startswith("do_") else k
            p.add_argument(f"--no-{name.replace('_', '-')}" if v else f"--{name.replace('_', '-')}",
                           dest=k, action="store_false" if v else "store_true")
        else:
            p.add_argument(f"--{k.replace('_', '-')}", dest=k, type=type(v) if v is not None else int, default=v)
    a = p.parse_args()
    kw = {k: getattr(a, k) for k in DEFAULTS}
    mp = dict(ontology=a.ontology, mapping=a.l1_mapping, input_is_l1=a.input_is_l1)
    out, st, rep = restore_nifti(a.input, mp, **kw)
    nib.save(out, a.output)
    stem = a.output.replace(".nii.gz", "").replace(".nii", "")
    nib.save(st, a.steps_out or stem + "_steps.nii.gz")
    with open(stem + "_report.json", "w") as f:
        json.dump(rep, f, indent=1)


if __name__ == "__main__":
    main()
