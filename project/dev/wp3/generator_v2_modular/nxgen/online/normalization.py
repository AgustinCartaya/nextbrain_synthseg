"""
Step 6: intensity normalisation.

    'minmax'      (x - min) / (max - min)                         -> [0, 1]
    'percentile'  clip to the [p_lo, p_hi] percentiles, then rescale to
                  [0, 1]; robust to a few extreme voxels (noise, motion)
    'none'        unchanged

Nothing is random. The values used (lo, hi) are returned so they can be
recorded.

Note for patch mode: statistics of a patch differ from those of the full
image, so a patch is normalised differently from the same crop of a full
image. Training pipelines usually normalise again anyway (e.g. nnU-Net's
z-score); use 'none' to leave that to them.

Percentiles are computed on at most 2^24 evenly strided voxels (the limit
of torch.quantile).
"""

import torch


def apply(img, cfg):
    """
    Normalise `img`.

    Parameters
    ----------
    img : float tensor
    cfg : dict, config section 'normalization'

    Returns
    -------
    img : float tensor
    params : dict(method, lo, hi) (lo, hi None for 'none')
    """
    method = cfg['method']
    if method == 'none':
        return img, dict(method=method, lo=None, hi=None)
    if method == 'minmax':
        lo, hi = img.min(), img.max()
    else:
        flat = img.reshape(-1)
        flat = flat[::max(1, flat.numel() // (1 << 24) + 1)].float()
        q = torch.tensor([p / 100 for p in cfg['percentiles']],
                         device=img.device)
        lo, hi = torch.quantile(flat, q)
        img = img.clamp(lo, hi)
    out = (img - lo) / (hi - lo).clamp_min(1e-12)
    return out, dict(method=method, lo=float(lo), hi=float(hi))
