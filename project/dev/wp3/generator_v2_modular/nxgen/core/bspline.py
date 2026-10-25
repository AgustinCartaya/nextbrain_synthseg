"""
Smooth random fields from a few control points (cubic B-splines).

Used by the elastic deformation (3 components) and the bias field
(1 component). A field is defined over the *whole* volume by an n^3 grid of
control values, and can be evaluated at any subset of voxels: evaluating it
on a patch gives exactly the crop of the full-volume field.

Evaluation is separable. With W_x, W_y, W_z the B-spline weights of each
axis (output positions x control points),

    field[x, y, z, :] = sum_abc W_x[x, a] W_y[y, b] W_z[z, c] ctrl[a, b, c, :]

The weights of each output position are normalised to sum to 1 (this also
handles the borders). They are non-negative, so the field is a convex
combination of control values: if every control value is in [lo, hi], the
field is too.
"""

import torch


def cubic_bspline(t):
    """Cubic B-spline basis function (support [-2, 2]) of a tensor."""
    t = t.abs()
    return torch.where(
        t < 1, 2 / 3 - t ** 2 + t ** 3 / 2,
        torch.where(t < 2, (2 - t) ** 3 / 6, torch.zeros_like(t)))


def axis_weights(n_ctrl, index, n_voxels, device=None):
    """
    B-spline weights of some voxels of one axis.

    The n_ctrl control points are spread evenly over the axis: control
    point j sits at voxel j * (n_voxels - 1) / (n_ctrl - 1).

    Parameters
    ----------
    n_ctrl : int (>= 2)
        Control points along the axis.
    index : 1D integer tensor
        Voxel indices (in the full volume) where the field is evaluated.
    n_voxels : int
        Size of the full volume along the axis.

    Returns
    -------
    (len(index), n_ctrl) float tensor, rows sum to 1
    """
    pos = index.to(device=device, dtype=torch.float32)
    u = pos * (n_ctrl - 1) / max(n_voxels - 1, 1)     # in control units
    j = torch.arange(n_ctrl, device=pos.device, dtype=torch.float32)
    w = cubic_bspline(u[:, None] - j[None, :])
    return w / w.sum(1, keepdim=True)


def evaluate(ctrl, wx, wy, wz):
    """
    Field at the voxels given by the weights of each axis.

    Parameters
    ----------
    ctrl : (a, b, c, C) float tensor, control values (C components)
    wx, wy, wz : (X, a), (Y, b), (Z, c) weights from `axis_weights`

    Returns
    -------
    (X, Y, Z, C) float tensor
    """
    t = torch.einsum('xa,abck->xbck', wx, ctrl)
    t = torch.einsum('yb,xbck->xyck', wy, t)
    return torch.einsum('zc,xyck->xyzk', wz, t)
