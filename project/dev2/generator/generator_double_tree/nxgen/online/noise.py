"""
Step 5d: Gaussian noise.

    image + sigma * N(0, 1)

`sigma` is drawn per image from `noise.sigma` (on the image divided by
`tree.intensity_scale`, so values of order 1). The noise field is not
stored: the plan stores a seed and the field is rebuilt from it.

Where the noise is added depends on `noise.space`:
* 'lowres': on the low-resolution grid of step 5c, before upsampling (as in
  a real scan). If the acquisition is not degraded, on the output grid.
* 'highres': on the output grid, before step 5c (SynthSeg's order).
The pipeline makes that choice; this module only adds noise to what it is
given.
"""

import torch

from ..core.rng import draw, new_seed, torch_generator


def sample(cfg, rng):
    """
    Draw the noise level of one image.

    Returns
    -------
    dict(sigma=float, seed=int, space='lowres' | 'highres')
    """
    return dict(sigma=float(draw(cfg['sigma'], rng)), seed=new_seed(rng),
                space=cfg['space'])


def apply(img, params):
    """Add the noise of `params` to `img` (any shape). sigma 0 -> unchanged."""
    if params is None or params['sigma'] == 0:
        return img
    g = torch_generator(params['seed'], img.device)
    return img + params['sigma'] * torch.randn(
        img.shape, generator=g, device=img.device)
