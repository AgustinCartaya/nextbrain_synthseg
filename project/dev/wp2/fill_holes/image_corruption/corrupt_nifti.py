#!/usr/bin/env python
"""
corrupt_nifti.py — Random NextBrain-style corruption of a NIfTI label map.

Simulates the errors of 3D segmentations reconstructed from 2D histological
sections cut along an OBLIQUE plane:
  * shifted sections (single sections and blocks of sections, moved within their plane)
  * holes (irregular blobs, some confined to 1-3 sections)
  * fully missing sections

All geometry is computed in world coordinates (RAS, mm) using the NIfTI affine,
so voxel orientation, anisotropic spacing and oblique affines are handled
automatically. The image is never resampled: each voxel is assigned to the
section it belongs to, and only voxels of corrupted sections are modified.

Section geometry
----------------
  --axis        nominal cutting plane: coronal | axial | sagittal
                (section normal = A-P, S-I or L-R respectively)
  --tilt-about  world axis the plane is tilted around: LR | AP | SI
                (default: LR for coronal/axial, AP for sagittal).
                A coronal plane tilted around LR shows its tilt in SAGITTAL views.
  --angle-range min max   tilt in degrees, sampled uniformly per output image
  --random-sign also allow the opposite tilt direction

Example (planes 9-14 deg from coronal, tilt visible in sagittal):
  python corrupt_nifti.py seg.nii.gz --out-dir corrupted --n 20 \
         --axis coronal --angle-range 9 14 --thickness 1.0 --save-boundaries

Python API:
  from corrupt_nifti import corrupt_nifti
  for img, params in corrupt_nifti("seg.nii.gz", (9, 14), "coronal", n=5, seed=0):
      ...


      
"""
import argparse
import json
import os

import nibabel as nib
import numpy as np
from scipy import ndimage as ndi

WORLD = {"LR": np.array([1.0, 0, 0]), "AP": np.array([0, 1.0, 0]), "SI": np.array([0, 0, 1.0])}
PLANE_NORMAL = {"sagittal": "LR", "coronal": "AP", "axial": "SI"}
DEFAULT_TILT = {"coronal": "LR", "axial": "LR", "sagittal": "AP"}


# =============================================================================
# Geometry
# =============================================================================
def rodrigues(axis, angle_rad):
    k = np.asarray(axis, float) / np.linalg.norm(axis)
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(angle_rad) * K + (1 - np.cos(angle_rad)) * K @ K


def section_normal(axis, tilt_about, angle_deg):
    """World-space (RAS) unit normal of the tilted section planes."""
    n0 = WORLD[PLANE_NORMAL[axis]]
    t = WORLD[tilt_about]
    if abs(n0 @ t) > 0.99:
        raise ValueError(f"Cannot tilt a {axis} plane around its own normal ({tilt_about}).")
    n = rodrigues(t, np.deg2rad(angle_deg)) @ n0
    return n / np.linalg.norm(n)


def plane_basis(n):
    a = WORLD["LR"] if abs(n[0]) < 0.9 else WORLD["SI"]
    u = np.cross(n, a)
    u /= np.linalg.norm(u)
    return u, np.cross(n, u)


def world_projection(shape, affine, vec, origin_world=None, region=None):
    """
    (world(voxel) - origin) . vec  for every voxel, without building full coordinate
    arrays. `region` is an optional tuple of slices (sub-box of the volume).
    """
    A, b = affine[:3, :3], affine[:3, 3]
    coef = A.T @ vec                                 # contribution of each voxel index
    off = b @ vec - (0.0 if origin_world is None else np.asarray(origin_world) @ vec)
    if region is None:
        region = tuple(slice(0, s) for s in shape)
    g = np.ogrid[region]
    return sum(g[i].astype(np.float32) * np.float32(coef[i]) for i in range(3)) + np.float32(off)


def section_index(shape, affine, n, thickness, phase):
    t = world_projection(shape, affine, n)
    t -= np.float32(phase)                      # in place: same float32 arithmetic,
    t /= np.float32(thickness)                  # no extra full-size temporaries
    np.floor(t, out=t)
    return t.astype(np.int32)


def thickness_floor(affine, n):
    """Smallest sensible section thickness: the voxel extent along n."""
    return float(np.abs(affine[:3, :3].T @ n).max())


# =============================================================================
# Speed helpers (results identical to the plain full-volume versions)
# =============================================================================
def bbox(mask, pad=0):
    """Tuple of slices of the bounding box of mask (padded, clipped), or None if empty."""
    reg = []
    for ax in range(mask.ndim):
        idx = np.flatnonzero(mask.any(axis=tuple(a for a in range(mask.ndim) if a != ax)))
        if idx.size == 0:
            return None
        reg.append(slice(max(int(idx[0]) - pad, 0), min(int(idx[-1]) + 1 + pad, mask.shape[ax])))
    return tuple(reg)


def unique_ints(vals):
    """np.unique for an int array, via bincount (much faster than sorting)."""
    if vals.size == 0:
        return vals.astype(np.int32)
    mn = int(vals.min())
    return (np.flatnonzero(np.bincount(vals - mn)) + mn).astype(vals.dtype)


class SectionLookup:
    """
    Fast access to the voxels of a range of sections without scanning the volume.

    sidx is projected along the axis where it varies the least (min/max per
    column), which gives the few columns that can contain a section; only those
    columns are read. Exact: uses the stored sidx values, no float re-derivation.
    Voxel coordinates are returned in C order (same as np.nonzero on a full mask).
    """
    def __init__(self, sidx, affine, n):
        self.sidx = sidx
        self.shape = sidx.shape
        self.ax = int(np.argmin(np.abs(affine[:3, :3].T @ n)))   # least-varying axis
        self.mn = sidx.min(axis=self.ax)
        self.mx = sidx.max(axis=self.ax)

    def voxels(self, s_first, s_last, extra=None):
        """Coordinates (tuple of 3 int arrays) of voxels with s_first <= sidx <= s_last
        (and extra[voxel] True, if `extra` is given)."""
        c1, c2 = np.nonzero((self.mx >= s_first) & (self.mn <= s_last))
        sv = np.moveaxis(self.sidx, self.ax, 0)
        col = sv[:, c1, c2]                                    # (len_ax, n_columns)
        m = (col >= s_first) & (col <= s_last)
        if extra is not None:
            m &= np.moveaxis(extra, self.ax, 0)[:, c1, c2]
        a, c = np.nonzero(m)
        coords = [None] * 3
        others = [i for i in range(3) if i != self.ax]
        coords[self.ax], coords[others[0]], coords[others[1]] = a, c1[c], c2[c]
        lin = np.ravel_multi_index(coords, self.shape)
        lin.sort()
        return np.unravel_index(lin, self.shape)


# =============================================================================
# Corruptions
# =============================================================================
def brain_mask(seg, fill_holes=False, closing_iter=3):
    """
    Region where holes / missing sections may be created.

    fill_holes=False (default, for clean input labels): simply seg > 0.
    fill_holes=True : binary_closing + binary_fill_holes, computed on the padded bounding box of the
    labels (identical result: padding > closing_iter keeps the crop border empty,
    and at the true volume border nothing is padded). Holes are filled by
    labelling the background and keeping components that touch the border.
    """
    fg = seg > 0
    if not fill_holes:
        return fg
    out = np.zeros(seg.shape, bool)
    reg = bbox(fg, pad=closing_iter + 2)
    if reg is None:
        return out
    m = ndi.binary_closing(fg[reg], iterations=closing_iter)
    lab, nl = ndi.label(~m)
    faces = np.concatenate([lab[0].ravel(), lab[-1].ravel(), lab[:, 0].ravel(),
                            lab[:, -1].ravel(), lab[:, :, 0].ravel(), lab[:, :, -1].ravel()])
    outside = np.zeros(nl + 1, bool)
    outside[faces] = True
    outside[0] = False
    out[reg] = ~outside[lab]
    return out


def move_voxels(src, out, mask, affine, n, d_world, angle_deg, center_world, coords=None):
    """
    For voxels p in mask: out[p] = src[ source(p) ], an in-plane rigid motion in world mm.
    Instead of a full boolean `mask`, the voxel coordinates can be given as `coords`
    (tuple of index arrays in C order, as returned by np.nonzero).
    """
    if coords is None:
        coords = np.nonzero(mask)
    if len(coords[0]) == 0:
        return
    A, b = affine[:3, :3], affine[:3, 3]
    A_inv = np.linalg.inv(A)
    vox = np.stack(coords, axis=1).astype(np.float64)
    w = vox @ A.T + b
    R_inv = rodrigues(n, -np.deg2rad(angle_deg))
    w_src = (w - center_world) @ R_inv.T + center_world - d_world
    vox_src = (w_src - b) @ A_inv.T
    out[coords] = ndi.map_coordinates(src, vox_src.T, order=0, mode="constant", cval=0)


def corrupt_shifts(seg, affine, n, sidx, rng, p_section, p_block, shift_sd_mm,
                   rot_sd_deg, block_sections, lookup=None, fg_box=None):
    if fg_box is None:
        fg_box = bbox(seg > 0) or tuple(slice(0, 0) for _ in seg.shape)
    secs = unique_ints(sidx[fg_box][seg[fg_box] > 0])
    lookup = lookup or SectionLookup(sidx, affine, n)

    u, v = plane_basis(n)
    center = affine[:3, :3] @ (np.array(seg.shape) / 2.0) + affine[:3, 3]
    out = seg.copy()
    events = []

    def motion():
        a, c = rng.normal(0, shift_sd_mm, 2)
        return a * u + c * v, float(rng.normal(0, rot_sd_deg))

    for s in secs[rng.random(len(secs)) < p_section]:
        d, ang = motion()
        move_voxels(seg, out, None, affine, n, d, ang, center, coords=lookup.voxels(s, s))
        events.append(dict(type="section_shift", sections=[int(s)], shift_mm=d.tolist(), rot_deg=ang))
    if len(secs) and rng.random() < p_block:
        L = int(rng.integers(block_sections[0], block_sections[1] + 1))
        s0 = int(rng.choice(secs))
        d, ang = motion()
        move_voxels(seg, out, None, affine, n, d, ang, center,
                    coords=lookup.voxels(s0, s0 + L - 1))
        events.append(dict(type="block_shift", sections=[s0, s0 + L - 1], shift_mm=d.tolist(), rot_deg=ang))
    return out, events


def corrupt_holes(seg, brain, affine, n, sidx, rng, n_blobs, radius_mm, thin_prob,
                  thin_sections, p_slab, slab_sections, irregularity, lookup=None,
                  brain_box=None):
    out = seg.copy()
    events = []
    u, v = plane_basis(n)
    # brain voxels in C order (searched inside the brain box: same order as np.argwhere)
    bb = brain_box or bbox(brain) or tuple(slice(0, 0) for _ in seg.shape)
    inside_flat = np.flatnonzero(brain[bb])
    if len(inside_flat) == 0:
        return out, events
    bb_shape = brain[bb].shape
    bb_lo = np.array([r.start for r in bb])
    A, b = affine[:3, :3], affine[:3, 3]
    vox_size = np.sqrt((A ** 2).sum(0))

    for _ in range(int(rng.integers(n_blobs[0], n_blobs[1] + 1))):
        c_vox = np.array(np.unravel_index(inside_flat[rng.integers(len(inside_flat))],
                                          bb_shape)) + bb_lo
        c_w = A @ c_vox + b
        r = rng.uniform(*radius_mm, size=3)                       # radii along u, v, n
        half = np.ceil(1.6 * r.max() / vox_size).astype(int) + 2   # local box (voxels)
        lo = np.maximum(c_vox - half, 0)
        hi = np.minimum(c_vox + half + 1, seg.shape)
        reg = tuple(slice(l, h) for l, h in zip(lo, hi))
        d = sum((world_projection(seg.shape, affine, ax, c_w, reg) / ri) ** 2
                for ax, ri in zip((u, v, n), r))
        noise = ndi.gaussian_filter(rng.normal(size=d.shape).astype(np.float32), 2)
        noise /= noise.std() + 1e-6
        blob = d < 1.0 + irregularity * noise
        thin = bool(rng.random() < thin_prob)
        if thin:
            s_c = sidx[tuple(c_vox)]
            k = int(rng.integers(thin_sections[0], thin_sections[1] + 1))
            blob &= (sidx[reg] >= s_c) & (sidx[reg] < s_c + k)
        blob &= brain[reg]
        out[reg][blob] = 0
        events.append(dict(type="thin_hole" if thin else "blob_hole",
                           center_vox=c_vox.tolist(), radii_mm=r.tolist(), voxels=int(blob.sum())))

    if rng.random() < p_slab:
        secs = unique_ints(sidx[bb][brain[bb]])
        s0 = int(rng.choice(secs))
        k = int(rng.integers(slab_sections[0], slab_sections[1] + 1))
        idx = (lookup or SectionLookup(sidx, affine, n)).voxels(s0, s0 + k - 1, extra=brain)
        out[idx] = 0
        events.append(dict(type="missing_sections", sections=[s0, s0 + k - 1], voxels=len(idx[0])))
    return out, events


# =============================================================================
# Main API
# =============================================================================
DEFAULTS = dict(
    thickness=None,          # mm; None -> voxel extent along the normal
    tilt_about=None,         # None -> DEFAULT_TILT[axis]
    random_sign=False,
    p_section=0.05, p_block=0.3, shift_sd_mm=1.0, rot_sd_deg=1.5, block_sections=(3, 15),
    n_blobs=(0, 6), radius_mm=(1.0, 8.0), thin_prob=0.5, thin_sections=(1, 3),
    p_slab=0.2, slab_sections=(1, 3), irregularity=0.5,
    fill_brain_holes=False,  # True -> brain mask = closing + hole filling of seg > 0
)


def corrupt_array(seg, affine, angle_range, axis, rng, brain=None, fg_box=None, **kw):
    """
    Corrupt one label array. Returns (corrupted_array, params, section_index).
    `brain` (= brain_mask(seg)) and `fg_box` (= bbox(seg > 0)) can be passed to avoid
    recomputing them for every sample.
    """
    o = {**DEFAULTS, **kw}
    tilt_about = o["tilt_about"] or DEFAULT_TILT[axis]
    angle = float(rng.uniform(*angle_range))
    if o["random_sign"] and rng.random() < 0.5:
        angle = -angle
    n = section_normal(axis, tilt_about, angle)
    T = o["thickness"] or thickness_floor(affine, n)
    T = max(T, thickness_floor(affine, n))
    phase = float(rng.uniform(0, T))
    sidx = section_index(seg.shape, affine, n, T, phase)
    if brain is None:
        brain = brain_mask(seg, o["fill_brain_holes"])
    lookup = SectionLookup(sidx, affine, n)
    empty = tuple(slice(0, 0) for _ in seg.shape)
    brain_box = bbox(brain) or empty
    if fg_box is None:
        fg_box = bbox(seg > 0) or empty

    x, ev1 = corrupt_shifts(seg, affine, n, sidx, rng, o["p_section"], o["p_block"],
                            o["shift_sd_mm"], o["rot_sd_deg"], o["block_sections"],
                            lookup=lookup, fg_box=fg_box)
    x, ev2 = corrupt_holes(x, brain, affine, n, sidx, rng, o["n_blobs"], o["radius_mm"],
                           o["thin_prob"], o["thin_sections"], o["p_slab"],
                           o["slab_sections"], o["irregularity"], lookup=lookup,
                           brain_box=brain_box)
    bb = brain_box
    b_sub = brain[bb]
    n_brain = int(b_sub.sum())
    n_changed = int(((x[bb] != seg[bb]) & b_sub).sum())
    params = dict(axis=axis, tilt_about=tilt_about, angle_deg=angle, normal_ras=n.tolist(),
                  thickness_mm=T, phase_mm=phase,
                  changed_fraction=float(n_changed / n_brain) if n_brain else float("nan"),
                  events=ev1 + ev2)
    return x, params, sidx


def corrupt_nifti(path, angle_range, axis="coronal", n=1, seed=None, **kw):
    """Generator yielding (corrupted nibabel image, params dict)."""
    img = nib.load(path)
    seg = np.asarray(img.dataobj)
    if np.issubdtype(seg.dtype, np.floating):
        seg = np.rint(seg).astype(np.int32)
    affine = img.affine
    rng = np.random.default_rng(seed)
    brain = brain_mask(seg, kw.get("fill_brain_holes", DEFAULTS["fill_brain_holes"]))  # once
    fg_box = bbox(seg > 0) or tuple(slice(0, 0) for _ in seg.shape)
    for _ in range(n):
        x, params, _ = corrupt_array(seg, affine, angle_range, axis, rng, brain=brain,
                                     fg_box=fg_box, **kw)
        out = nib.Nifti1Image(x.astype(seg.dtype, copy=False), affine, img.header)
        out.set_data_dtype(seg.dtype)
        yield out, params


def boundaries_image(img, axis, tilt_about, angle_deg, thickness, phase=0.0):
    """Section boundaries as a NIfTI, to overlay in a viewer and check the angle."""
    n = section_normal(axis, tilt_about or DEFAULT_TILT[axis], angle_deg)
    s = section_index(img.shape[:3], img.affine, n, thickness, phase)
    b = np.zeros(s.shape, np.uint8)
    for ax in range(3):
        b |= (s != np.roll(s, 1, axis=ax)).astype(np.uint8)
    return nib.Nifti1Image(b, img.affine)


# =============================================================================
# CLI
# =============================================================================
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--n", type=int, default=10, help="number of corrupted images")
    p.add_argument("--angle-range", type=float, nargs=2, required=True, metavar=("MIN", "MAX"))
    p.add_argument("--axis", choices=list(PLANE_NORMAL), default="coronal")
    p.add_argument("--tilt-about", choices=list(WORLD), default=None)
    p.add_argument("--random-sign", action="store_true")
    p.add_argument("--thickness", type=float, default=None, help="section thickness (mm)")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--p-section", type=float, default=DEFAULTS["p_section"])
    p.add_argument("--p-block", type=float, default=DEFAULTS["p_block"])
    p.add_argument("--shift-sd-mm", type=float, default=DEFAULTS["shift_sd_mm"])
    p.add_argument("--rot-sd-deg", type=float, default=DEFAULTS["rot_sd_deg"])
    p.add_argument("--block-sections", type=int, nargs=2, default=DEFAULTS["block_sections"])
    p.add_argument("--n-blobs", type=int, nargs=2, default=DEFAULTS["n_blobs"])
    p.add_argument("--radius-mm", type=float, nargs=2, default=DEFAULTS["radius_mm"])
    p.add_argument("--thin-prob", type=float, default=DEFAULTS["thin_prob"])
    p.add_argument("--thin-sections", type=int, nargs=2, default=DEFAULTS["thin_sections"])
    p.add_argument("--p-slab", type=float, default=DEFAULTS["p_slab"])
    p.add_argument("--slab-sections", type=int, nargs=2, default=DEFAULTS["slab_sections"])
    p.add_argument("--irregularity", type=float, default=DEFAULTS["irregularity"])
    p.add_argument("--fill-brain-holes", action="store_true",
                   help="brain mask = closing + hole filling of the labels (for inputs that "
                        "already have holes); default: brain mask = labels > 0")
    p.add_argument("--save-boundaries", action="store_true",
                   help="also save the section boundaries of each sample (for QC)")
    a = p.parse_args()

    os.makedirs(a.out_dir, exist_ok=True)
    base = os.path.basename(a.input).replace(".nii.gz", "").replace(".nii", "")
    kw = {k: getattr(a, k) for k in DEFAULTS if hasattr(a, k)}
    kw["tilt_about"] = a.tilt_about
    kw = {k: (tuple(v) if isinstance(v, list) else v) for k, v in kw.items()}

    src = nib.load(a.input)
    for i, (img, params) in enumerate(corrupt_nifti(a.input, tuple(a.angle_range), a.axis,
                                                     n=a.n, seed=a.seed, **kw)):
        stem = os.path.join(a.out_dir, f"{base}_corrupt_{i:03d}")
        nib.save(img, stem + ".nii.gz")
        json.dump(params, open(stem + ".json", "w"), indent=2)
        if a.save_boundaries:
            nib.save(boundaries_image(src, a.axis, params["tilt_about"], params["angle_deg"],
                                      params["thickness_mm"], params["phase_mm"]),
                     stem + "_boundaries.nii.gz")
        print(f"[{i + 1}/{a.n}] angle={params['angle_deg']:.2f} deg  "
              f"thickness={params['thickness_mm']:.2f} mm  "
              f"changed={100 * params['changed_fraction']:.2f}%  events={len(params['events'])}")


if __name__ == "__main__":
    main()
