"""
SIAM-style synthetic data generator implemented with cornucopia.

>>> ANNOTATED VERSION <<<
This file is a line-by-line commented copy of `data_augmentation_pipeline.py`.
The code is identical (it runs and behaves the same); only comments, worked
examples and longer docstrings were added. Keep the two in sync if you edit.

Reproduces the generative model of:
    "SIAM: Head and Brain MRI Segmentation from Few High-Quality Templates via
    Synthetic Training" (arXiv:2605.02737), Sec. 3.2.

Big picture
-----------
A segmentation network is trained only on *synthetic* images. Each synthetic
image is created from a label map (a "template"): we randomly change the
shapes in the label map, then paint every tissue with a random intensity and
add scanner-like artefacts. Because contrasts are random, the network learns
to segment *any* MRI contrast (T1w, T2w, FLAIR, ...).

     label template (1 mm)
            |
            |  OFFLINE, once per template  ---------------------------------
            v
     [1] group labels           FreeSurfer 99 labels -> 14 tissue classes
     [2] super-resolution       1 mm -> 0.25 mm (smooth, sub-mm boundaries)
            |
            |  ONLINE, every sample  ---------------------------------------
            v
     [3] erosion / dilation     random thicker/thinner tissues  (0.25 mm)
     [4] affine + elastic       random pose / shape             (0.25 mm)
     [5] partial volume maps    0.25 mm one-hot -> 0.75 mm fractions
     [6] GMM intensities        random contrast, mixed by fractions
     [7] bias, motion, noise,   scanner artefacts                (0.75 mm)
         normalise [0, 1]
            v
     (image, label) pair at 0.75 mm  -> nnU-Net

Why 0.25 mm at all, if the network trains at 0.75 mm? Because doing the shape
changes at a finer grid and then *averaging* 3x3x3 small voxels into one big
voxel gives realistic partial volume: a 0.75 mm voxel on the GM/WM border
becomes e.g. 60% GM + 40% WM instead of 100% one or the other.

Parameters not given in the paper (deformation magnitudes, bias field,
motion, probabilities) are marked "ASSUMPTION" below. The deformation and
motion defaults follow TorchIO's defaults, since the paper uses TorchIO.
"""

import argparse
import json
import math
import os
import random
from dataclasses import dataclass, field

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F

import cornucopia as cc
# `Uniform(a, b)` is a cornucopia "sampler": every time a transform needs a
# random value it calls the sampler, e.g. Uniform(0, 1)() -> 0.37, and
# Uniform(0, 1)(3) -> [0.12, 0.88, 0.45] (one value per tissue).
from cornucopia.random import Uniform
# Low-level resampling helpers (wrap torch.nn.functional.grid_sample).
from cornucopia.utils import warps


# ===========================================================================
# 1. Label grouping
# ===========================================================================
# The input segmentation has FreeSurfer labels (your seg.nii.gz has 99 of
# them: 2 = left WM, 17 = left hippocampus, 1001..1035 = left cortical
# parcels, ...). The paper works with a smaller set of *tissue* classes,
# because what matters for image synthesis is "which tissues look different
# in MRI", not anatomical parcellation: all 68 cortical parcels are the same
# tissue (GM) and get the same intensity.

# The 12 brain tissues of the paper (+ background). `vessel` belongs to the
# paper's extra-cerebral group but is kept because the CSF<->vessel
# morphology rule needs it. Extra-cerebral head tissues (skin, fat, muscle,
# skull, ...) can be appended here when using full-head templates.
#
# The position in this list IS the label value in all outputs:
#   synthetic seg voxel == 2  ->  'wm'
TISSUES = [
    'background',    # 0
    'gm',            # 1  cerebral cortex
    'wm',            # 2
    'cerebellar_gm', # 3
    'csf',           # 4  extra-ventricular CSF
    'ventricles',    # 5
    'thalamus',      # 6
    'putamen',       # 7
    'pallidum',      # 8
    'caudate',       # 9
    'accumbens',     # 10
    'amygdala',      # 11
    'hippocampus',   # 12
    'vessel',        # 13
]
# Name -> index lookup, so the code can say T['wm'] instead of 2.
#   T = {'background': 0, 'gm': 1, 'wm': 2, ..., 'vessel': 13}
T = {name: i for i, name in enumerate(TISSUES)}

# FreeSurfer (aseg / aparc+aseg / SynthSeg) -> tissue groups.
# Structures the paper does not list separately are merged into the closest
# group: cerebellar WM, brainstem, ventral DC and corpus callosum -> WM,
# choroid plexus -> ventricles.
#
# `{**a, **b}` merges dictionaries; each `{lab: T[...] for lab in [...]}`
# builds entries like {3: 1, 42: 1}. The final dict looks like:
#   {3: 1, 42: 1, 1001: 1, ..., 2: 2, 41: 2, 7: 2, ..., 17: 12, 53: 12, ...}
#    ^FS label -> ^tissue index
FREESURFER_TO_TISSUE = {
    **{lab: T['gm'] for lab in [3, 42]},                 # cortex (aseg)
    **{lab: T['gm'] for lab in range(1000, 1036)},       # L cortex parcels
    **{lab: T['gm'] for lab in range(2000, 2036)},       # R cortex parcels
    **{lab: T['wm'] for lab in [2, 41, 7, 46, 16, 28, 60, 77, 78, 79, 85,
                                251, 252, 253, 254, 255]},
    **{lab: T['cerebellar_gm'] for lab in [8, 47]},
    **{lab: T['csf'] for lab in [24]},
    **{lab: T['ventricles'] for lab in [4, 43, 5, 44, 14, 15, 31, 63, 72]},
    **{lab: T['thalamus'] for lab in [9, 10, 48, 49]},
    **{lab: T['putamen'] for lab in [12, 51]},
    **{lab: T['pallidum'] for lab in [13, 52]},
    **{lab: T['caudate'] for lab in [11, 50]},
    **{lab: T['accumbens'] for lab in [26, 58]},
    **{lab: T['amygdala'] for lab in [18, 54]},
    **{lab: T['hippocampus'] for lab in [17, 53]},
    **{lab: T['vessel'] for lab in [30, 62]},
}

DEEP_NUCLEI = ['thalamus', 'putamen', 'pallidum', 'caudate', 'accumbens',
               'amygdala', 'hippocampus']

# Morphology rules (Sec. 3.2.2): "WM within GM, ventricles and deep nuclei",
# "CSF within GM and vasculature". Each pair (a, b) is randomly either
# a-grows-into-b (dilation of a) or b-grows-into-a (erosion of a).
#
# Resulting list (11 pairs):
#   [('wm','gm'), ('wm','ventricles'), ('wm','thalamus'), ('wm','putamen'),
#    ('wm','pallidum'), ('wm','caudate'), ('wm','accumbens'),
#    ('wm','amygdala'), ('wm','hippocampus'), ('csf','gm'), ('csf','vessel')]
#
# Why pairs and not "dilate everything"? Anatomically plausible variation:
# cortex thickness changes at the GM/WM border (WM<->GM), ventricles enlarge
# into WM with atrophy (WM<->ventricles), but e.g. CSF should never invade
# WM directly. BUT THIS IS NOT GURANTEED BY THE GENERATOR
MORPHO_PAIRS = (
    [('wm', 'gm'), ('wm', 'ventricles')]
    + [('wm', nucleus) for nucleus in DEEP_NUCLEI]
    + [('csf', 'gm'), ('csf', 'vessel')]
)


def group_labels(seg, mapping=FREESURFER_TO_TISSUE):
    """
    Map a raw label map to tissue-group indices (unmapped -> background).

    Uses a look-up table (LUT): an array where position `i` holds the new
    label of old label `i`. Then `lut[seg]` relabels the whole volume in one
    vectorised step (no Python loop over voxels).

    Example
    -------
    >>> seg = np.array([0, 2, 17, 1001, 41])
    >>> group_labels(seg)
    array([0, 2, 12, 1, 2], dtype=uint8)
    #      bg wm hip gm  wm
    """
    seg = np.asarray(seg).astype(np.int64)
    # LUT big enough to be indexed by the largest label (2035 here);
    # every entry defaults to 0 (= background).
    lut = np.zeros(seg.max() + 1, dtype=np.uint8)
    for src, dst in mapping.items():
        if src < len(lut):           # skip labels larger than seg.max()
            lut[src] = dst
    # Labels present in the image but not in the mapping would silently
    # become background -> warn so you can extend the mapping.
    unmapped = sorted(set(np.unique(seg).tolist()) - set(mapping) - {0})
    if unmapped:
        print(f'[group_labels] WARNING: labels {unmapped} unmapped '
              f'-> background')
    # Fancy indexing: out[v] = lut[seg[v]] for every voxel v.
    return lut[seg]


# ===========================================================================
# 2. Label super-resolution (offline)
# ===========================================================================
# Goal: turn a blocky 1 mm label map into a smooth 0.25 mm label map, where
# boundaries are no longer staircases of 1 mm cubes.
#
# Paper recipe (Sec. 3.2.1), per label k:
#     binary mask (lab == k)          -> "one-hot channel k"
#     Sinc interpolation to 0.25 mm   -> smooth, band-limited "soft mask"
#     Gaussian smoothing (0.5 mm)     -> removes Sinc ringing
# then, at each HR voxel, pick the label whose soft mask is largest (argmax).
#
# 1D intuition (one row of voxels, 1 mm, two labels):
#     label  :  A   A   A   B   B
#     mask A :  1   1   1   0   0
#     mask B :  0   0   0   1   1
#   after upsampling x4 and smoothing (values roughly):
#     mask A :  1 1 1 1 1 1 1 1 1 .9 .7 .5 .3 .1 0 0 ...
#     mask B :  0 0 0 0 0 0 0 0 0 .1 .3 .5 .7 .9 1 1 ...
#   argmax  :  A A A A A A A A A  A  A  ?  B  B B B ...
#   -> the A/B boundary is placed with 0.25 mm precision, and in 3D curved
#      boundaries become smooth instead of staircased.

def _sinc_upsample_smooth(x, factor, sigma_mm, out_voxel_mm):
    """
    Band-limited (Sinc) upsampling by zero-padding the centred spectrum,
    combined with a Gaussian smoothing applied in the Fourier domain.
    Output voxel `j` sits at input coordinate `j / factor`.

    Why the Fourier domain?
    -----------------------
    * Sinc interpolation == "keep all the frequencies the image has, add no
      new ones". In Fourier space that is literally: copy the spectrum into
      the centre of a bigger array full of zeros, then inverse FFT.
    * Gaussian smoothing == convolution with a Gaussian == multiplication of
      the spectrum by another Gaussian. So both steps cost one FFT + one
      inverse FFT, instead of a huge 3D convolution at 0.25 mm.

    Parameters
    ----------
    x : (X, Y, Z) float tensor     binary mask of one label (0./1.)
    factor : int                   upsampling factor (4: 1 mm -> 0.25 mm)
    sigma_mm : float               Gaussian sigma in mm (0 = no smoothing)
    out_voxel_mm : float           output voxel size in mm (0.25)

    Returns
    -------
    y : (X*factor, Y*factor, Z*factor) float tensor   soft mask
    """
    shape = x.shape                                  # e.g. (180, 222, 195)
    new_shape = [s * factor for s in shape]          # e.g. (720, 888, 780)

    # Forward FFT; fftshift moves frequency 0 ("DC", the mean) to the centre
    # of the array so the spectrum is symmetric around the middle.
    X = torch.fft.fftshift(torch.fft.fftn(x))

    # Bigger, empty (all-zero) spectrum. The zeros are the "new" high
    # frequencies -> they stay zero -> no new detail is invented.
    Xp = torch.zeros(new_shape, dtype=X.dtype, device=X.device)

    # keep the DC component (index s//2 after fftshift) aligned
    # Paste the small spectrum in the middle of the big one so that DC lands
    # on the DC position of the big array (n//2).
    #   1D example: s = 4, factor = 4 -> n = 16
    #     small DC index = 4//2 = 2 ; big DC index = 16//2 = 8
    #     slice(8 - 2, 8 - 2 + 4) = slice(6, 10)  -> small[2] lands at big[8]
    slicer = tuple(slice(n // 2 - s // 2, n // 2 - s // 2 + s)
                   for n, s in zip(new_shape, shape))
    Xp[slicer] = X
    del X                                            # free GPU memory

    if sigma_mm:
        # frequency grid of the HR volume in cycles/mm
        # The Fourier transform of a Gaussian with std `sigma` (mm) is
        #     G(f) = exp(-2 * pi^2 * sigma^2 * f^2)       (f in cycles/mm)
        # A 3D Gaussian is separable: G(fx, fy, fz) = G(fx) G(fy) G(fz),
        # so we multiply by a 1D profile along each axis in turn.
        #
        # Example with sigma = 0.5 mm:
        #   f = 0    (mean intensity)        -> G = 1.00 (kept)
        #   f = 0.25 cycles/mm (4 mm detail) -> G = 0.73
        #   f = 0.5  cycles/mm (1 mm Nyquist)-> G = 0.29 (ringing damped)
        for d, n in enumerate(new_shape):
            # fftfreq gives the frequency of every FFT bin; shift it so it
            # matches the shifted spectrum (0 in the middle).
            f = torch.fft.fftshift(torch.fft.fftfreq(
                n, d=out_voxel_mm, device=x.device))
            g = torch.exp(-2 * (math.pi * sigma_mm * f) ** 2)
            # Reshape the 1D profile so it broadcasts along axis d only:
            #   d=0 -> (n,1,1) ; d=1 -> (1,n,1) ; d=2 -> (1,1,n)
            view = [1] * len(new_shape)
            view[d] = n
            Xp *= g.view(view)

    # Undo the shift, inverse FFT, keep the real part (the imaginary part is
    # only numerical round-off for a real input).
    y = torch.fft.ifftn(torch.fft.ifftshift(Xp)).real

    # Amplitude correction: ifftn divides by the number of voxels, which is
    # factor^3 = 64 times larger than before. Without this, a mask of 1s
    # would come back as 1/64. (Not strictly needed for argmax, which only
    # compares channels, but keeps values interpretable: inside ~1, outside
    # ~0, boundary ~0.5.)
    return y * (factor ** len(shape))


def super_resolve_labels(lab, nb_labels, factor, voxel_mm, sigma_mm=0.5):
    """
    One-hot -> Sinc upsampling -> Gaussian smoothing -> argmax (Sec. 3.2.1).
    Channels are processed one at a time and a running argmax is kept, so
    the full HR one-hot volume is never held in memory.

    Memory reasoning
    ----------------
    HR volume here: 720 x 888 x 780 = 499 M voxels.
    Full one-hot, 14 channels x 4 bytes  -> 28 GB
    Running argmax: 1 float "best" + 1 uint8 "out" + 1 channel -> ~5 GB

    Running argmax, 1-voxel example (3 labels):
        k=0: y=0.1 -> best=0.1, out=0
        k=1: y=0.7 -> 0.7 > 0.1 -> best=0.7, out=1
        k=2: y=0.2 -> 0.2 < 0.7 -> unchanged
        final label = 1   (same result as argmax([0.1, 0.7, 0.2]))
    """
    out_voxel_mm = voxel_mm / factor                 # 1 / 4 = 0.25 mm
    best = None   # highest soft-mask value seen so far, per HR voxel
    out = None    # label that produced it, per HR voxel
    for k in range(nb_labels):
        mask = (lab == k)                            # one-hot channel k
        if not mask.any():                           # label absent (e.g. vessel)
            continue
        y = _sinc_upsample_smooth(mask.float(), factor, sigma_mm,
                                  out_voxel_mm)
        if best is None:
            # first present label: everything is initially assigned to it
            best = y
            out = torch.full_like(y, k, dtype=torch.uint8)
        else:
            better = y > best                        # where label k wins
            out.masked_fill_(better, k)              # out[better] = k
            torch.maximum(best, y, out=best)         # in-place, no new memory
        del y
    return out


def prepare_template(seg_path, out_path=None, target_voxel=0.75, factor=3,
                     margin_mm=16., sigma_mm=0.5, mapping=FREESURFER_TO_TISSUE,
                     device='cpu'):
    """
    Build a high-resolution tissue template from a label map.

    The label map is grouped, cropped around the head/brain (+margin), and
    super-resolved to `target_voxel / factor` mm. The HR shape is a multiple
    of `factor` so that average pooling lands exactly on the training grid.

    Worked example with data/inputs/seg.nii.gz
    ------------------------------------------
    input           : 197 x 233 x 189 @ 1 mm
    brain bbox      : voxels [24,24,0] .. [171,211,161]
    + 16 mm margin  : lo = [8, 8, -16], hi = [188, 228, 178]
    size            : [180, 220, 194]
    fix to x3 rule  : [180, 222, 195]   (size*4 must be divisible by 3)
    HR template     : 720 x 888 x 780 @ 0.25 mm
    training grid   : 240 x 296 x 260 @ 0.75 mm  (= HR / 3)

    Returns
    -------
    lab_hr : (X, Y, Z) uint8 tensor
    affine_hr : (4, 4) ndarray
    """
    nii = nib.load(seg_path)
    seg = group_labels(np.asarray(nii.dataobj), mapping)
    # The affine maps voxel indices (i, j, k) to world/scanner mm (x, y, z):
    #   [x, y, z, 1]^T = affine @ [i, j, k, 1]^T
    # Every time we crop or resample, we must update it so the outputs stay
    # aligned with the input in world space (e.g. when overlaid in a viewer).
    affine = nii.affine
    voxel_mm = float(np.mean(nii.header.get_zooms()[:3]))     # 1.0

    # FFT upsampling ratio from the input grid to the HR grid
    #   hr_voxel = 0.75 / 3 = 0.25 mm ;  up = 1.0 / 0.25 = 4
    hr_voxel = target_voxel / factor
    up = voxel_mm / hr_voxel
    # FFT zero-padding only supports integer factors. E.g. a 0.8 mm input
    # would give up = 3.2 -> error (resample the input first in that case).
    if abs(up - round(up)) > 1e-3:
        raise ValueError(f'input voxel {voxel_mm} mm must be an integer '
                         f'multiple of the HR voxel {hr_voxel} mm')
    up = int(round(up))

    # crop/pad around foreground with a margin (out-of-bounds -> background)
    # Why crop: the HR volume is up^3 = 64x bigger than the input, so every
    # background voxel we drop saves a lot of memory and time.
    # Why a margin: the random deformation later moves/zooms the brain; the
    # margin leaves room so it is not cut at the border of the volume.
    idx = np.argwhere(seg > 0)                       # coords of all brain voxels
    margin = int(math.ceil(margin_mm / voxel_mm))    # 16 voxels
    lo = idx.min(0) - margin                         # [8, 8, -16]  (can be < 0!)
    hi = idx.max(0) + 1 + margin                     # [188, 228, 178]
    # make HR shape a multiple of `factor`
    # Average pooling later merges blocks of 3x3x3 HR voxels; if the HR size
    # is not divisible by 3 the last partial block would be dropped and the
    # grids would not line up. Grow the crop by 1 voxel until it works:
    #   size=220 -> 880 % 3 = 1 -> 221 -> 884 % 3 = 2 -> 222 -> 888 % 3 = 0 ok
    size = hi - lo
    while np.any((size * up) % factor):
        size = np.where((size * up) % factor, size + 1, size)
    hi = lo + size
    # The crop may go outside the image (lo[2] = -16 here, because the
    # brainstem touches the bottom of the FOV). Pad with zeros (= background)
    # first, then shift the crop indices by the padding.
    pad_lo = np.maximum(-lo, 0)                      # [0, 0, 16]
    pad_hi = np.maximum(hi - np.array(seg.shape), 0) # [0, 0, 0]
    seg = np.pad(seg, list(zip(pad_lo, pad_hi)))
    lo = lo + pad_lo                                 # indices in padded array
    hi = hi + pad_lo
    seg = seg[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
    # Update the affine: new voxel (0,0,0) is old voxel (lo - pad_lo)
    # = (8, 8, -16). Composing with a translation matrix does exactly that:
    #   old affine translation  (-98, -134, -72)
    #   new affine translation  (-98+8, -134+8, -72-16) = (-90, -126, -88)
    crop_shift = np.eye(4)
    crop_shift[:3, 3] = lo - pad_lo
    affine = affine @ crop_shift

    lab = torch.as_tensor(seg, device=device)
    # no_grad: we never backpropagate here; avoids storing autograd buffers.
    with torch.no_grad():
        lab_hr = super_resolve_labels(lab, len(TISSUES), up, voxel_mm,
                                      sigma_mm)
    # HR voxel j sits at input voxel coordinate j / up
    # -> scale the voxel axes by 1/4 (voxel size 1 mm -> 0.25 mm), no shift:
    #   HR voxel 0 == input voxel 0 ; HR voxel 4 == input voxel 1 ; ...
    affine_hr = affine @ np.diag([1 / up, 1 / up, 1 / up, 1])

    if out_path:
        # uint8 + gzip: 499 M voxels -> ~14 MB on disk
        nib.save(nib.Nifti1Image(lab_hr.cpu().numpy(), affine_hr), out_path)
    return lab_hr, affine_hr


def load_template(path, device='cpu'):
    """Load a template saved by `prepare_template` (skips the slow step)."""
    nii = nib.load(path)
    lab = torch.as_tensor(np.asarray(nii.dataobj, dtype=np.uint8),
                          device=device)
    return lab, nii.affine


# ===========================================================================
# 3. Online generator
# ===========================================================================

def _bbox(mask, margin=0):
    """
    Slicer of the bounding box of a 3D mask (None if empty).

    Used to run the morphology only around the two tissues involved: e.g.
    WM<->pallidum only needs a small box around the pallidum instead of the
    whole 499 M-voxel volume.

    Example (a 3 x 4 x 1 volume, i.e. a single z slice)
    ---------------------------------------------------
        mask[:, :, 0] = [[0 0 0 0]
                         [0 1 1 0]
                         [0 0 1 0]]
        x with any 1 -> 1..2 ; y with any 1 -> 1..2 ; z -> 0
        _bbox(mask)           -> (slice(1, 3), slice(1, 3), slice(0, 1))
        _bbox(mask, margin=1) -> (slice(0, 3), slice(0, 4), slice(0, 1))
                                 (margin clipped at the volume borders)
    """
    box = []
    for d in range(mask.ndim):
        # Collapse the two *other* axes with `any` -> 1D profile along d
        # telling which slices contain at least one True voxel.
        other = [i for i in range(mask.ndim) if i != d]
        nz = mask.any(dim=other[1]).any(dim=other[0]).nonzero()
        if len(nz) == 0:
            return None
        lo = max(int(nz[0]) - margin, 0)              # first slice - margin
        hi = min(int(nz[-1]) + 1 + margin, mask.shape[d])  # last + margin
        box.append(slice(lo, hi))
    return tuple(box)


@dataclass
class SIAMConfig:
    """
    All the knobs of the generator in one place.

    Usage:
        cfg = SIAMConfig()                         # defaults
        cfg = SIAMConfig(motion_prob=0.0)          # e.g. disable motion
        cfg = SIAMConfig(rotations=20, zooms=0.2)  # stronger deformation
    """
    target_voxel: float = 0.75   # training resolution (mm)
    factor: int = 3              # HR = target_voxel / factor = 0.25 mm

    # -- shape augmentation (paper: 1-4 iterations of 0.25 mm) ------------
    # 1 iteration moves a boundary by 1 HR voxel = 0.25 mm, so the maximum
    # change is 4 x 0.25 = 1 mm (e.g. cortex 1 mm thicker or thinner).
    morpho_iter: tuple = (1, 4)
    morpho_prob: float = 0.5     # ASSUMPTION: per-pair probability
    morpho_pairs: tuple = tuple(MORPHO_PAIRS)

    # -- spatial deformation (ASSUMPTION: TorchIO defaults) ----------------
    rotations: float = 10.       # deg, U(-10, 10)
    zooms: float = 0.1           # U(-0.1, 0.1) about 1  -> scale 0.9..1.1
    shears: float = 0.
    translations_mm: float = 0.
    elastic_dmax_mm: float = 7.5 # max non-linear displacement
    elastic_ctrl: int = 7        # control-point grid size (up to 7x7x7)

    # -- intensities (paper) -----------------------------------------------
    gmm_mu: tuple = (0., 1.)          # tissue mean ~ U(0, 1)
    gmm_sigma: tuple = (0.001, 0.01)  # tissue std  ~ U(0.001, 0.01)
    zero_background: bool = False   # True: background stays at 0

    # -- artefacts ----------------------------------------------------------
    bias_prob: float = 1.0       # ASSUMPTION
    bias_vmax: float = 0.5       # ASSUMPTION: field in (1-v, 1+v)
    bias_ctrl: int = 5           # ASSUMPTION
    motion_prob: float = 0.3     # ASSUMPTION
    motion_shots: int = 3        # ASSUMPTION (TorchIO: num_transforms=2)
    motion_rotations: float = 10.   # deg (TorchIO default)
    motion_translations_mm: float = 10.  # (TorchIO default)
    noise_sigma: tuple = (0.01, 0.1)  # paper

    nb_labels: int = len(TISSUES)
    names: tuple = tuple(TISSUES)


class SIAMGenerator:
    """
    Draw (image, label) pairs from one or several HR templates.

    Parameters
    ----------
    templates : list of (lab_hr, affine_hr)
        Outputs of `prepare_template` / `load_template`.
    config : SIAMConfig
    device : torch.device

    Example
    -------
    >>> lab_hr, aff_hr = load_template('data/outputs/template_hr0.25mm.nii.gz')
    >>> gen = SIAMGenerator([(lab_hr, aff_hr)], device='cuda')
    >>> img, seg, pv, affine = gen()      # every call -> a new random sample
    >>> img.shape, seg.shape, pv.shape
    ((1, 240, 296, 260), (240, 296, 260), (14, 240, 296, 260))
    """

    def __init__(self, templates, config=None, device='cpu'):
        self.cfg = config or SIAMConfig()
        self.device = torch.device(device)
        # Templates are moved to the GPU once and reused for every sample.
        self.templates = [(lab.to(self.device), aff)
                          for lab, aff in templates]
        cfg = self.cfg

        # --- cornucopia transforms -----------------------------------------
        # A cornucopia "Random...Transform" stores *distributions*. Each call
        # draws new parameters and applies them:
        #     t = cc.RandomGaussianNoiseTransform(sigma=Uniform(0.01, 0.1))
        #     a = t(img)   # noise with, say, sigma = 0.034
        #     b = t(img)   # new draw,   say, sigma = 0.081
        # So we build them once here and simply call them per sample.

        # Affine + elastic sampling field. It is drawn by cornucopia on the
        # training (0.75 mm) grid, then linearly upsampled slab-by-slab and
        # used to resample the 0.25 mm labels with nearest neighbour (see
        # `_deform`). The field is smooth (affine + 7^3 control points), so
        # this is equivalent to drawing it at 0.25 mm, at a fraction of the
        # memory (~110 GB -> a few GB for a 0.5 G-voxel template).
        #
        # unit='vox' -> dmax and translations are in voxels of the grid the
        # transform is applied to (0.75 mm), hence the division:
        #     7.5 mm / 0.75 mm = 10 voxels
        # returns='flow' -> instead of a deformed image, return the sampling
        # coordinates (see `_deform`).
        self.deform = cc.RandomAffineElasticTransform(
            dmax=cfg.elastic_dmax_mm / cfg.target_voxel,
            shape=cfg.elastic_ctrl,
            rotations=cfg.rotations,
            zooms=cfg.zooms,
            shears=cfg.shears,
            translations=cfg.translations_mm / cfg.target_voxel,
            unit='vox',
            returns='flow',
        )

        # GMM with partial volume: input is the (K, X, Y, Z) PV maps, so
        # cornucopia computes  sum_k pv_k * N(mu_k, sigma_k)
        #
        # When cornucopia receives a *float* multi-channel tensor it treats
        # channels as soft class memberships (instead of an integer label
        # map) and does, for each class k:
        #     img += pv[k] * (mu_k + sigma_k * randn(X, Y, Z))
        # Each call draws 14 new means and 14 new stds -> new contrast.
        # fwhm=0: noise inside a tissue is white (not spatially smoothed).
        self.gmm = cc.RandomGaussianMixtureTransform(
            mu=Uniform(*cfg.gmm_mu),
            sigma=Uniform(*cfg.gmm_sigma),
            fwhm=0,
            background=0 if cfg.zero_background else None,
        )

        # Multiplicative bias field (receive-coil inhomogeneity):
        #     img = img * field
        # The field is a smooth spline through a random grid of up to 5x5x5
        # control values; with symmetric=True the values lie in (1-v, 1+v),
        # v ~ U(0, 0.5). E.g. v = 0.3 -> one side of the brain up to 30%
        # brighter, the other up to 30% darker.
        # MaybeTransform(t, p): apply t with probability p (here always).
        self.bias = cc.MaybeTransform(
            cc.RandomMulFieldTransform(
                shape=cfg.bias_ctrl, vmax=cfg.bias_vmax, symmetric=True),
            cfg.bias_prob)

        # Additive Gaussian noise, std ~ U(0.01, 0.1) (paper):
        #     img = img + sigma * randn(X, Y, Z)
        self.noise = cc.RandomGaussianNoiseTransform(
            sigma=Uniform(*cfg.noise_sigma))
        # Min-max rescale to [0, 1]:  img = (img - min) / (max - min)
        #     e.g. values in [-0.2, 1.3] -> [0, 1]
        self.normalize = cc.MinMaxTransform()

    # -- steps ---------------------------------------------------------------

    def _morphology(self, lab):
        """
        Constrained random dilation/erosion between neighbouring pairs.

        For each tissue pair (a, b), with probability `morpho_prob`:
          * pick a direction: a grows into b, or b grows into a
            (b growing into a == eroding a);
          * pick n_iter in 1..4 (= 0.25..1 mm);
          * dilate the growing tissue by n_iter voxels, but only accept the
            new voxels that previously belonged to the *paired* tissue.

        1D example (WM<->GM, WM grows, n_iter = 2)
        ------------------------------------------
            before   : WM WM WM GM GM GM CSF
            dilated  : WM WM WM WM WM GM CSF   (WM pushed 2 voxels)
            result   : WM WM WM WM WM GM CSF   (GM voxels taken -> accepted)

        Constraint example (WM grows, but next to CSF)
        ----------------------------------------------
            before   : WM WM CSF CSF
            dilated  : WM WM WM  WM
            result   : WM WM CSF CSF   (CSF is not the paired tissue -> kept)
        """
        cfg = self.cfg
        lo, hi = cfg.morpho_iter
        for a, b in cfg.morpho_pairs:
            # skip this pair with probability 1 - morpho_prob
            if random.random() > cfg.morpho_prob:
                continue
            a, b = T[a], T[b]                        # names -> indices
            # coin flip for the direction
            grow, shrink = (a, b) if random.random() < 0.5 else (b, a)
            n_iter = random.randint(lo, hi)          # inclusive: 1, 2, 3 or 4
            # work in the bounding box of the pair (+ margin)
            # (margin so the dilation is not clipped at the box border)
            box = _bbox((lab == grow) | (lab == shrink), margin=n_iter + 1)
            if box is None:                          # tissue absent (vessel)
                continue
            sub = lab[box]                           # a *view*: writing to it
                                                     # later writes into `lab`
            # cornucopia's 'conv' dilation by `radius` = radius-1 iterations
            # (checked empirically: radius=1 changes nothing, radius=2 grows
            # by 1 voxel in 6-connectivity, ...), hence n_iter + 1.
            # `sub[None]` adds the channel dimension cornucopia expects:
            # (X, Y, Z) -> (1, X, Y, Z); `[0]` removes it again.
            dilated = cc.DilateLabelTransform(
                labels=[grow], radius=n_iter + 1)(sub[None])[0]
            # only take over voxels of the paired neighbour
            #   new voxel = grow    if dilation reached it AND it was `shrink`
            #             = old     otherwise
            lab[box] = torch.where((dilated == grow) & (sub == shrink),
                                   grow, sub)
            del dilated, sub
        return lab

    def _deform(self, lab, slab=16):
        """
        Random affine+elastic deformation of the HR labels (nearest).

        How a deformation is applied ("pull" / backward warping)
        --------------------------------------------------------
        For every OUTPUT voxel p we compute where to read from in the INPUT:
            out[p] = in[ phi(p) ]
        `phi` is the "flow": a (X, Y, Z, 3) array of input coordinates.
        Identity: phi(p) = p. A shift of +2 voxels in x: phi(p) = p + (2,0,0).
        (We verified that cornucopia returns phi in this absolute form.)

        Trick to save memory
        --------------------
        Building phi directly at 0.25 mm costs 499 M x 3 floats (+ spline
        intermediates) -> ~113 GB. But phi is smooth, so we:
          1. let cornucopia draw phi on the 0.75 mm grid (240x296x260);
          2. convert its coordinates from LR voxels to HR voxels;
          3. upsample it x3 with trilinear interpolation, 16 LR slices
             (48 HR slices) at a time;
          4. read the HR labels at those coordinates (nearest neighbour).
        Linear interpolation of an affine map is exact, and of a 7-control-
        point spline is accurate to a tiny fraction of a voxel.
        """
        f = self.cfg.factor                          # 3
        lr_shape = [s // f for s in lab.shape]       # [240, 296, 260]
        # (1, x, y, z, 3) absolute sampling coordinates on the LR grid
        # We feed a dummy all-zero LR image: only its shape matters, since
        # we asked cornucopia to return the flow, not the warped image.
        flow = self.deform(lab.new_zeros([1, *lr_shape], dtype=torch.float))
        # LR voxel i is centred on HR voxel f*i + (f-1)/2
        # Each 0.75 mm voxel covers 3 HR voxels, its centre is the middle one:
        #   LR voxel 0 covers HR 0,1,2   -> centre HR 1   = 3*0  + 1
        #   LR voxel 10 covers HR 30..32 -> centre HR 31  = 3*10 + 1
        #   LR coord 10.5                -> HR coord 32.5
        # .movedim(-1, 1): (1, x, y, z, 3) -> (1, 3, x, y, z), the layout
        # F.interpolate expects (batch, channels, spatial...).
        flow = (flow * f + (f - 1) / 2).movedim(-1, 1)   # (1, 3, x, y, z)
        # grid_sample needs (batch, channel, X, Y, Z) floats. Labels <= 13
        # are exact in float32, and nearest mode never mixes values.
        src = lab[None, None].float()
        out = torch.empty_like(lab)
        # Process the volume in slabs along x to bound memory.
        for i0 in range(0, lr_shape[0], slab):
            i1 = min(i0 + slab, lr_shape[0])         # LR slices [i0, i1)
            # Add 1 LR slice on each side (halo) so trilinear interpolation
            # at the slab edges sees the true neighbours, not a clamped edge.
            a, b = max(i0 - 1, 0), min(i1 + 1, lr_shape[0])  # 1-voxel halo
            # align_corners=False maps HR voxel j to LR coord (j - (f-1)/2)/f
            # -> the same LR/HR correspondence as above (HR 31 <-> LR 10).
            up = F.interpolate(flow[:, :, a:b], scale_factor=f,
                               mode='trilinear', align_corners=False)
            # drop the halo again: keep HR slices of LR [i0, i1)
            up = up[:, :, (i0 - a) * f:(i1 - a) * f].movedim(1, -1)
            # Pull the labels: out[p] = src[nearest(up[p])].
            # padding_mode='zeros': reading outside the volume -> background.
            # has_identity=True: `up` holds absolute coordinates, not offsets.
            out[i0 * f:i1 * f] = warps.apply_flow(
                src, up, has_identity=True, mode='nearest',
                padding_mode='zeros')[0, 0].to(lab.dtype)
            del up
        return out

    def _partial_volume(self, lab):
        """
        Average-pool the one-hot HR labels to the training grid.

        Each 0.75 mm voxel contains 3x3x3 = 27 HR voxels. The fraction of
        them belonging to tissue k is its partial-volume (PV) value.

        Example: a voxel on the cortex boundary with 18 GM and 9 WM HR voxels
            pv[gm] = 18/27 = 0.667 ; pv[wm] = 9/27 = 0.333 ; others = 0
        Fractions always sum to 1 at every voxel.

        Returns (K, X/3, Y/3, Z/3) = (14, 240, 296, 260).
        """
        f = self.cfg.factor
        pv = []
        # One tissue at a time: a full HR one-hot (14 x 499 M floats) would
        # need 28 GB; one channel needs 2 GB.
        for k in range(self.cfg.nb_labels):
            m = (lab == k)[None, None].float()       # (1, 1, X, Y, Z) 0./1.
            # avg_pool3d(m, 3): mean over non-overlapping 3x3x3 blocks
            pv.append(F.avg_pool3d(m, f)[0, 0])
            del m
        return torch.stack(pv)

    def _motion(self, img):
        """
        Random rigid head motion during the acquisition (k-space model).

        An MRI image is acquired in k-space (Fourier space), line by line.
        If the head moves halfway through, the first part of k-space comes
        from position A and the rest from position B. The reconstructed
        image then shows ghosting / ringing. cornucopia simulates this:
          1. split k-space along `axis` into `shots` contiguous chunks;
          2. for each shot, rigidly move the image (random rotation and
             translation), FFT, and keep only that shot's chunk;
          3. assemble the chunks, inverse FFT, take the magnitude.

        Pattern example: n = 10 k-space lines, 3 shots -> length = 4
            shot 0: lines 0-3   [1 1 1 1 0 0 0 0 0 0]
            shot 1: lines 4-7   [0 0 0 0 1 1 1 1 0 0]
            shot 2: lines 8-9   [0 0 0 0 0 0 0 0 1 1]
        """
        cfg = self.cfg
        if random.random() > cfg.motion_prob:        # 70% of the time: no-op
            return img
        # cornucopia 0.5.1 workarounds for non-cubic volumes:
        # - the motion is sampled for the input shape but applied after
        #   moving `axis` to dim 1 -> we move the axis ourselves, use axis=1;
        # - the (shots, n) pattern broadcasts against the *last* dim -> we
        #   pass a (shots, n, 1, 1) pattern.
        #
        # Random phase-encode axis (1, 2 or 3 = x, y or z; dim 0 is channel).
        # E.g. axis = 2: (1, 240, 296, 260) -> (1, 296, 240, 260)
        axis = random.choice([1, 2, 3])
        img = img.movedim(axis, 1)
        n = img.shape[1]                             # number of k-space lines
        # cornucopia expects translations as a fraction of the FOV:
        #   10 mm / (240 * 0.75 mm = 180 mm) = 0.056
        fov_mm = min(img.shape[1:]) * cfg.target_voxel
        # sequential k-space pattern
        length = math.ceil(n / cfg.motion_shots)
        pattern = torch.zeros([cfg.motion_shots, n, 1, 1], dtype=torch.bool,
                              device=img.device)
        for shot in range(cfg.motion_shots):
            pattern[shot, shot * length:(shot + 1) * length] = True
        motion = cc.IntraScanMotionTransform(
            shots=cfg.motion_shots,
            axis=1,
            translations=cfg.motion_translations_mm / fov_mm,
            rotations=cfg.motion_rotations,
            pattern=pattern,
        )
        # apply, then put the axis back where it was
        return motion(img).movedim(1, axis)

    # -- main ----------------------------------------------------------------

    @torch.no_grad()
    def __call__(self, template_index=None):
        """
        Generate one random (image, label) pair.

        Returns
        -------
        img : (1, X, Y, Z) float tensor in [0, 1]
        lab : (X, Y, Z) uint8 tensor (tissue indices)
        pv : (K, X, Y, Z) float tensor, partial-volume maps
        affine : (4, 4) ndarray, voxel-to-world of the 0.75 mm grid
        """
        # With several templates (the paper uses 6), pick one at random.
        if template_index is None:
            template_index = random.randrange(len(self.templates))
        lab, affine_hr = self.templates[template_index]

        # ---- shape, at 0.25 mm --------------------------------------------
        # clone(): _morphology edits in place; the stored template must stay
        # untouched for the next sample.
        lab = self._morphology(lab.clone())          # [3] erosion/dilation
        lab = self._deform(lab)                      # [4] affine + elastic
        pv = self._partial_volume(lab)               # [5] 0.25 -> 0.75 mm (pv are the soft one hot maps of the 14 tissues)
        del lab                                      # free ~0.5 GB

        # ---- appearance, at 0.75 mm ---------------------------------------
        # Shapes: (14, X, Y, Z) PV maps -> (1, X, Y, Z) image from here on.
        img = self.gmm(pv)                           # [6] random contrast
        img = self.bias(img)                         # [7] bias field
        img = self._motion(img)                      #     motion (p = 0.3)
        img = self.noise(img)                        #     Gaussian noise
        img = self.normalize(img)                    #     -> [0, 1]

        # Training label at 0.75 mm = the tissue with the largest fraction
        # (majority vote of the 27 HR voxels). E.g. pv = [.., gm .667,
        # wm .333, ..] -> label = gm.
        seg = pv.argmax(0).to(torch.uint8)

        # pooled voxel i is centred on HR voxel f*i + (f-1)/2
        # Same correspondence as in `_deform`, written as a matrix:
        #   pool = [[3, 0, 0, 1],
        #           [0, 3, 0, 1],
        #           [0, 0, 3, 1],
        #           [0, 0, 0, 1]]
        # With affine_hr = diag(0.25) + translation (-90, -126, -88):
        #   affine = diag(0.75) + translation (-89.75, -125.75, -87.75)
        f = self.cfg.factor
        pool = np.diag([f, f, f, 1.]).astype(float)
        pool[:3, 3] = (f - 1) / 2
        affine = affine_hr @ pool
        return img, seg, pv, affine


# ===========================================================================
# I/O helpers and CLI (nnU-Net raw dataset layout)
# ===========================================================================

def save_sample(img, seg, affine, img_path, seg_path):
    """Write image (float32) and labels (uint8) as NIfTI with the same affine."""
    # img[0]: drop the channel dimension, (1, X, Y, Z) -> (X, Y, Z)
    nib.save(nib.Nifti1Image(img[0].cpu().numpy().astype(np.float32),
                             affine), img_path)
    nib.save(nib.Nifti1Image(seg.cpu().numpy().astype(np.uint8),
                             affine), seg_path)


def write_nnunet_dataset_json(out_dir, n, names=TISSUES):
    """
    nnU-Net v2 needs a dataset.json describing channels and labels, e.g.
        {"channel_names": {"0": "synthetic"},
         "labels": {"background": 0, "gm": 1, ..., "vessel": 13},
         "numTraining": 1000, "file_ending": ".nii.gz"}
    """
    meta = {
        'channel_names': {'0': 'synthetic'},
        'labels': {name: i for i, name in enumerate(names)},
        'numTraining': n,
        'file_ending': '.nii.gz',
    }
    with open(os.path.join(out_dir, 'dataset.json'), 'w') as f:
        json.dump(meta, f, indent=2)


def seed_everything(seed):
    """
    Make runs reproducible. cornucopia draws from Python's `random` and
    torch's RNG, so both must be seeded (numpy too, for safety).
    Same seed + same GPU -> same samples.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main():
    """
    Command-line entry point. Example:

        python data_augmentation_pipeline_comments.py \\
            --seg data/inputs/seg.nii.gz --out /path/Dataset001_SIAM -n 1000

    produces the nnU-Net raw layout:
        Dataset001_SIAM/
            templates/seg_hr0.25mm.nii.gz      (cached HR template)
            imagesTr/synth_0000_0000.nii.gz    (_0000 = channel 0)
            labelsTr/synth_0000.nii.gz
            dataset.json
    """
    p = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    p.add_argument('--seg', nargs='+', required=True,
                   help='label map(s) used as templates')
    p.add_argument('--out', required=True, help='output directory')
    p.add_argument('-n', type=int, default=10, help='number of samples')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available()
                   else 'cpu')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--factor', type=int, default=3,
                   help='HR factor: HR voxel = 0.75 / factor mm')
    p.add_argument('--prefix', default='synth')
    args = p.parse_args()

    seed_everything(args.seed)
    cfg = SIAMConfig(factor=args.factor)
    os.makedirs(os.path.join(args.out, 'templates'), exist_ok=True)
    os.makedirs(os.path.join(args.out, 'imagesTr'), exist_ok=True)
    os.makedirs(os.path.join(args.out, 'labelsTr'), exist_ok=True)

    # Build (or load from cache) one HR template per input label map.
    templates = []
    for path in args.seg:
        name = os.path.basename(path).split('.')[0]  # 'seg.nii.gz' -> 'seg'
        hr = os.path.join(args.out, 'templates',
                          f'{name}_hr{cfg.target_voxel / cfg.factor:g}mm.nii.gz')
        if os.path.exists(hr):
            templates.append(load_template(hr))
        else:
            print(f'Super-resolving {path} -> {hr}')
            lab, aff = prepare_template(path, hr, cfg.target_voxel,
                                        cfg.factor, device=args.device)
            # keep a CPU copy; the generator moves it to `device` itself
            templates.append((lab.cpu(), aff))

    gen = SIAMGenerator(templates, cfg, device=args.device)
    for i in range(args.n):
        img, seg, _, aff = gen()
        case = f'{args.prefix}_{i:04d}'              # synth_0000, synth_0001...
        save_sample(img, seg, aff,
                    os.path.join(args.out, 'imagesTr', f'{case}_0000.nii.gz'),
                    os.path.join(args.out, 'labelsTr', f'{case}.nii.gz'))
        print(f'[{i + 1}/{args.n}] {case} {tuple(seg.shape)}')
    write_nnunet_dataset_json(args.out, args.n, cfg.names)


if __name__ == '__main__':
    main()
