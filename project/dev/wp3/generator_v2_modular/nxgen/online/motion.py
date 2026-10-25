"""
Step 5b: motion artefacts (output grid).

Simulates a head that moves between k-space shots. The k-space lines along
one axis (the phase-encoding axis, chosen at random) are split into
`shots` consecutive blocks. Each block is acquired with the head in its own
rigid pose:

    for every shot s:
        moved_s = image after rotation R_s and translation t_s (about the
                  centre of the volume, trilinear, zeros outside)
        lines of block s  <-  1D FFT of moved_s along the axis
    image <- real part of the inverse 1D FFT of the assembled k-space

Only the axis of the blocks needs a Fourier transform: the other two axes
are fully sampled in every shot, so their transforms cancel out.

The shot that holds the centre of k-space sets the overall position of the
image; it is kept at the reference pose (no motion) so the image stays
aligned with the segmentation. The other shots give ghosting and blurring.
(Same choice as the v2 workaround on cornucopia's IntraScanMotionTransform.)

In patch mode the motion is simulated on the patch only. Real ghosts span
the whole field of view; on a patch they wrap inside the patch. This is an
approximation (see generator_structure.md, Q9).
"""

import math

import torch
import torch.nn.functional as F

from ..core.geometry import rotation_matrix
from ..core.rng import draw


def reference_shot(n, shots):
    """Shot holding the k-space centre (index n // 2 after fftshift)."""
    length = math.ceil(n / shots)
    return (n // 2) // length


def sample(cfg, rng, shape):
    """
    Draw motion parameters.

    Parameters
    ----------
    cfg : dict, config section 'motion'
    rng : numpy Generator
    shape : 3 ints, shape of the volume the motion is applied to

    Returns
    -------
    None (no motion, probability 1 - prob) or dict with
        axis : phase-encoding axis (0, 1 or 2)
        shots : int
        reference_shot : shot kept at the reference pose
        rotation_deg, translation_mm : per shot, 3 values each (zeros for
            the reference shot)
    """
    apply_it = rng.random() < cfg['prob']
    shots = int(cfg['shots'])
    axis = int(rng.integers(3))
    rot = [[float(draw(cfg['rotation_deg'], rng)) for _ in range(3)]
           for _ in range(shots)]
    trans = [[float(draw(cfg['translation_mm'], rng)) for _ in range(3)]
             for _ in range(shots)]
    if not apply_it or shots < 2:
        return None
    ref = reference_shot(shape[axis], shots)
    rot[ref] = [0.0] * 3
    trans[ref] = [0.0] * 3
    return dict(axis=axis, shots=shots, reference_shot=ref,
                rotation_deg=rot, translation_mm=trans)


def rigid(img, rotation_deg, translation_mm, voxel_mm):
    """
    Rigidly move the content of `img` about its centre.

    The object point at q (voxel-axis mm, centred) moves to R q + t, so the
    value at x is read at R^T (x - t). Trilinear, zeros outside.

    Parameters
    ----------
    img : (X, Y, Z) float tensor
    rotation_deg, translation_mm : 3 values
    voxel_mm : float

    Returns
    -------
    (X, Y, Z) float tensor
    """
    dev = img.device
    shape = img.shape
    R = torch.tensor(rotation_matrix(rotation_deg), dtype=torch.float32,
                     device=dev)
    t = torch.tensor(translation_mm, dtype=torch.float32, device=dev)
    c = torch.tensor([(s - 1) / 2 for s in shape], device=dev)
    axes = [(torch.arange(s, device=dev) - c[d]) * voxel_mm
            for d, s in enumerate(shape)]
    x = torch.stack(torch.meshgrid(*axes, indexing='ij'), -1)   # (X,Y,Z,3)
    src = ((x - t) @ R) / voxel_mm + c        # R^T (x - t), row vectors
    del x
    # grid_sample wants coordinates in [-1, 1], in (z, y, x) order
    grid = (2 * src / (torch.tensor(shape, device=dev) - 1) - 1).flip(-1)
    del src
    out = F.grid_sample(img[None, None], grid[None], mode='bilinear',
                        padding_mode='zeros', align_corners=True)
    return out[0, 0]


def apply(img, params, voxel_mm):
    """
    Motion artefacts on `img` ((X, Y, Z) float tensor). `params` None ->
    unchanged.
    """
    if params is None:
        return img
    axis, shots = params['axis'], params['shots']
    n = img.shape[axis]
    length = math.ceil(n / shots)
    spec = torch.zeros(img.shape, dtype=torch.complex64, device=img.device)
    for s in range(shots):
        a, b = s * length, min((s + 1) * length, n)
        if a >= b:
            continue
        moved = img if s == params['reference_shot'] else rigid(
            img, params['rotation_deg'][s], params['translation_mm'][s],
            voxel_mm)
        k = torch.fft.fftshift(torch.fft.fft(moved, dim=axis), dim=axis)
        spec.narrow(axis, a, b - a).copy_(k.narrow(axis, a, b - a))
        del moved, k
    out = torch.fft.ifft(torch.fft.ifftshift(spec, dim=axis), dim=axis)
    return out.real.contiguous()
