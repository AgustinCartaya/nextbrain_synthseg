"""
Step 2: spatial deformation (HR grid): random affine + elastic.

For every voxel v of the output HR grid, the label is read (nearest
neighbour) from the template at

    x   = (v - c) * h                      voxel-axis mm, centred
    p   = M x + t + e(v)                   deformed position (mm)
    src = p / h + c                        template voxel

with h the HR voxel size, c the centre of the template, M = R S Z the
linear part (`nxgen.core.geometry.affine_matrix`), t the translation, and
e the elastic displacement: a cubic B-spline (`nxgen.core.bspline`) of
n x n x n control vectors drawn in U(-dmax, dmax) mm, spread over the whole
template. Positions outside the template give background.

Only the requested region of the output grid is computed. `source_box`
gives the part of the template that region reads, so the pipeline can
crop the template before the morphology step. Both are exact: the affine
maps the region's box to a parallelepiped (bounded by its 8 corners) and
|e| <= max |control value| (the B-spline weights are a convex combination).

Differences with v2: the parameters are drawn here and stored in the plan
(matrix, translation, control points), instead of inside cornucopia; the
elastic field is evaluated directly at the HR voxels (no coarse grid +
upsampling), slab by slab.
"""

import numpy as np
import torch

from ..core import bspline
from ..core.geometry import affine_matrix
from ..core.rng import draw


def sample(cfg, rng):
    """
    Draw one deformation.

    Parameters
    ----------
    cfg : dict, config section 'spatial'
    rng : numpy Generator

    Returns
    -------
    dict with
        rotation_deg, zoom, shear, translation_mm : 3 values each (shear =
            xy, xz, yz)
        matrix : 3x3 list, M = R S Z
        elastic_dmax_mm : float
        elastic_ctrl_mm : (n, n, n, 3) nested list of control vectors (mm),
            or None if dmax = 0
    """
    rot = [float(draw(cfg['rotation_deg'], rng)) for _ in range(3)]
    zoom = [float(draw(cfg['zoom'], rng)) for _ in range(3)]
    shear = [float(draw(cfg['shear'], rng)) for _ in range(3)]
    trans = [float(draw(cfg['translation_mm'], rng)) for _ in range(3)]
    dmax = float(draw(cfg['elastic_dmax_mm'], rng))
    n = int(draw(cfg['elastic_ctrl'], rng))
    if n < 2:
        raise ValueError(f'spatial.elastic_ctrl must be >= 2, got {n}')
    ctrl = rng.uniform(-dmax, dmax, size=(n, n, n, 3))
    return dict(rotation_deg=rot, zoom=zoom, shear=shear,
                translation_mm=trans,
                matrix=affine_matrix(rot, zoom, shear).tolist(),
                elastic_dmax_mm=dmax,
                elastic_ctrl_mm=ctrl.tolist() if dmax > 0 else None)


def identity_params():
    """Parameters of the identity (no deformation)."""
    return dict(rotation_deg=[0.] * 3, zoom=[1.] * 3, shear=[0.] * 3,
                translation_mm=[0.] * 3, matrix=np.eye(3).tolist(),
                elastic_dmax_mm=0., elastic_ctrl_mm=None)


def _max_displacement_mm(params):
    """Largest |elastic displacement| per component (mm)."""
    if params['elastic_ctrl_mm'] is None:
        return np.zeros(3)
    return np.abs(np.asarray(params['elastic_ctrl_mm'])).reshape(-1, 3).max(0)


def source_box(params, start, size, shape, voxel_mm, halo=0):
    """
    Part of the template read by the output region [start, start + size).

    Parameters
    ----------
    params : dict from `sample` (or `identity_params`)
    start, size : 3 ints, region on the HR grid
    shape : 3 ints, template shape
    voxel_mm : float, HR voxel size
    halo : int, extra voxels added on every side (for the morphology)

    Returns
    -------
    lo, hi : (3,) int arrays, box [lo, hi) clipped to the template. The box
        is empty (lo >= hi on some axis) if the region reads only outside.
    """
    shape = np.asarray(shape)
    c = (shape - 1) / 2
    start = np.asarray(start)
    end = start + np.asarray(size) - 1
    corners = np.array(np.meshgrid(*zip(start, end), indexing='ij'))
    corners = corners.reshape(3, -1).T                       # (8, 3) voxels
    x = (corners - c) * voxel_mm
    p = x @ np.asarray(params['matrix']).T + params['translation_mm']
    src = p / voxel_mm + c
    reach = _max_displacement_mm(params) / voxel_mm
    lo = np.floor(src.min(0) - reach).astype(int) - 1 - halo
    hi = np.ceil(src.max(0) + reach).astype(int) + 2 + halo
    return np.clip(lo, 0, shape), np.clip(hi, 0, shape)


def apply(crop, crop_lo, params, start, size, shape, voxel_mm,
          slab_voxels=1 << 24):
    """
    Deformed labels of the output region (nearest neighbour).

    Parameters
    ----------
    crop : (a, b, c) int16 tensor, template[lo:hi] (after morphology)
    crop_lo : 3 ints, position of the crop in the template
    params : dict from `sample` / `identity_params`
    start, size : 3 ints, output region on the HR grid
    shape : 3 ints, full template shape
    voxel_mm : float, HR voxel size
    slab_voxels : int, output voxels computed at once (memory)

    Returns
    -------
    (size) int16 tensor of label indices (0 outside the template)
    """
    dev = crop.device
    out = torch.zeros(list(size), dtype=crop.dtype, device=dev)
    if crop.numel() == 0:
        return out
    h = float(voxel_mm)
    c = torch.tensor([(s - 1) / 2 for s in shape], device=dev)
    lo = torch.tensor(list(crop_lo), device=dev)
    cs = torch.tensor(list(crop.shape), device=dev)
    flat = crop.reshape(-1)
    M = torch.tensor(params['matrix'], dtype=torch.float32, device=dev)
    t = torch.tensor(params['translation_mm'], dtype=torch.float32,
                     device=dev)
    ctrl = params['elastic_ctrl_mm']
    if ctrl is not None:
        ctrl = torch.tensor(ctrl, dtype=torch.float32, device=dev)
        n = ctrl.shape[0]

    axes = [torch.arange(start[d], start[d] + size[d], device=dev)
            for d in range(3)]
    ymm = ((axes[1] - c[1]) * h)[None, :, None, None]
    zmm = ((axes[2] - c[2]) * h)[None, None, :, None]
    if ctrl is not None:
        wy = bspline.axis_weights(n, axes[1], shape[1], dev)
        wz = bspline.axis_weights(n, axes[2], shape[2], dev)

    X, Y, Z = size
    step = max(1, slab_voxels // (Y * Z))
    for i0 in range(0, X, step):
        i1 = min(i0 + step, X)
        xs = axes[0][i0:i1]
        xmm = ((xs - c[0]) * h)[:, None, None, None]
        p = M[:, 0] * xmm + M[:, 1] * ymm + M[:, 2] * zmm + t   # (s,Y,Z,3)
        if ctrl is not None:
            wx = bspline.axis_weights(n, xs, shape[0], dev)
            p += bspline.evaluate(ctrl, wx, wy, wz)
        idx = torch.round(p / h + c).long() - lo
        del p
        valid = ((idx >= 0) & (idx < cs)).all(-1)
        idx = torch.minimum(idx.clamp_min(0), cs - 1)
        lin = (idx[..., 0] * cs[1] + idx[..., 1]) * cs[2] + idx[..., 2]
        out[i0:i1] = torch.where(valid, flat[lin], 0)
        del idx, valid, lin
    return out
