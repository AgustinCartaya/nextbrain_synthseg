"""
Step 5a: bias field (output grid).

Smooth multiplicative field: image * (1 + B(x)), with B a cubic B-spline
(`nxgen.core.bspline`) of ctrl^3 control values drawn in U(-vmax, vmax).
The B-spline weights are a convex combination, so the field stays in
[1 - vmax, 1 + vmax].

The field is defined over the whole output volume and evaluated only on the
region being generated, so a patch gets exactly the crop of the full-volume
field.

Difference with v2: v2 used cornucopia's RandomMulFieldTransform (same
idea, parameters not returned).
"""

import torch

from ..core import bspline


def sample(cfg, rng):
    """
    Draw a bias field.

    Parameters
    ----------
    cfg : dict, config section 'bias'
    rng : numpy Generator

    Returns
    -------
    None (no bias field, probability 1 - prob) or
    dict(ctrl_values=(n, n, n) nested list, vmax=float)
    """
    apply_it = rng.random() < cfg['prob']
    n = int(cfg['ctrl'])
    values = rng.uniform(-cfg['vmax'], cfg['vmax'], size=(n, n, n))
    if not apply_it:
        return None
    return dict(ctrl_values=values.tolist(), vmax=float(cfg['vmax']))


def field(params, start, size, shape, device=None):
    """
    The multiplicative field (1 + B) on the region [start, start + size).

    Parameters
    ----------
    params : dict from `sample`
    start, size : 3 ints, region on the output grid
    shape : 3 ints, full output volume

    Returns
    -------
    (size) float32 tensor
    """
    ctrl = torch.tensor(params['ctrl_values'], dtype=torch.float32,
                        device=device)[..., None]
    n = ctrl.shape[0]
    w = [bspline.axis_weights(
        n, torch.arange(start[d], start[d] + size[d], device=device),
        shape[d], device) for d in range(3)]
    return 1 + bspline.evaluate(ctrl, *w)[..., 0]


def apply(img, params, start, shape):
    """
    Multiply `img` (the region [start, start + img.shape) of a volume of
    size `shape`) by the bias field. `params` None -> unchanged.
    """
    if params is None:
        return img
    return img * field(params, start, img.shape, shape, img.device)
