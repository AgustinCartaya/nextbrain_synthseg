"""
Step 5c: resolution (output grid -> low-res grid -> output grid).

Simulates the acquisition drawn at step 0 (SynthSeg Sec. 3.1.4):

    1. blur each degraded axis (spacing > output voxel) with a Gaussian
       slice profile (std = `acquisition.sigma_vox`, computed from the
       thickness and alpha; kernel truncated at 3 std, edges replicated);
    2. downsample each degraded axis to ceil(n / ratio) voxels, with
       ratio = spacing / output voxel (trilinear, align_corners=True);
    3. optionally add noise on this low-resolution grid (noise.space =
       'lowres', as in a real scan);
    4. upsample back to the output grid (trilinear, align_corners=True).

Same steps as v2 `LowResAcquisitionFinalTransform` (which used cornucopia's
`smoothnd` for step 1). The segmentation is never degraded.
`acquisition` None, or no degraded axis: the image is returned unchanged.

There is nothing random here: everything comes from the acquisition.
"""

import math

import torch
import torch.nn.functional as F

from . import acquisition as acquisition_mod


def gaussian_blur_axis(img, sigma, axis):
    """
    1D Gaussian blur of a (X, Y, Z) tensor along `axis`.

    Kernel truncated at 3 std and normalised; edges replicated.
    """
    if sigma <= 0:
        return img
    r = max(1, math.ceil(3 * sigma))
    d = torch.arange(-r, r + 1, dtype=torch.float32, device=img.device)
    k = torch.exp(-d ** 2 / (2 * sigma ** 2))
    k = k / k.sum()
    shape = [1, 1, 1, 1, 1]
    shape[2 + axis] = len(k)
    pad = [0] * 6                      # F.pad order: last axis first
    pad[2 * (2 - axis)] = pad[2 * (2 - axis) + 1] = r
    x = F.pad(img[None, None], pad, mode='replicate')
    return F.conv3d(x, k.view(shape))[0, 0]


def lowres_shape(shape, acquisition, voxel_mm):
    """
    Shape of the low-resolution grid of an acquisition.

    Axis d keeps its size if spacing_mm[d] <= voxel_mm, else it gets
    ceil(n / ratio) voxels (at least 2), ratio = spacing_mm[d] / voxel_mm.
    """
    out = []
    for n, spacing in zip(shape, acquisition['spacing_mm']):
        ratio = spacing / voxel_mm
        out.append(n if ratio <= 1 + 1e-6 else max(2, math.ceil(n / ratio)))
    return out


def apply(img, acquisition, voxel_mm, noise_fn=None):
    """
    Degrade `img` to the acquisition.

    Parameters
    ----------
    img : (X, Y, Z) float tensor
    acquisition : dict from `acquisition.sample`, or None (not degraded)
    voxel_mm : float, output voxel size
    noise_fn : callable or None
        Applied to the low-resolution image (step 3).

    Returns
    -------
    img : (X, Y, Z) float tensor, on the output grid
    lowres : (x, y, z) float tensor on the low-resolution grid, or None if
        the acquisition is not degraded (then `img` is returned unchanged
        and the noise is NOT added)
    """
    if not acquisition_mod.degraded_axes(acquisition, voxel_mm):
        return img, None
    shape = list(img.shape)
    y = img
    for d, sigma in enumerate(acquisition_mod.sigma_vox(acquisition,
                                                        voxel_mm)):
        y = gaussian_blur_axis(y, sigma, d)
    small = lowres_shape(shape, acquisition, voxel_mm)
    if small != shape:
        y = F.interpolate(y[None, None], size=small, mode='trilinear',
                          align_corners=True)[0, 0]
    if noise_fn is not None:
        y = noise_fn(y)
    out = y
    if small != shape:
        out = F.interpolate(y[None, None], size=shape, mode='trilinear',
                            align_corners=True)[0, 0]
    return out, y
