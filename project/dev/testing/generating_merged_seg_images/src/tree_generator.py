"""
Tree-based synthetic data generator, v1.

Same pipeline as `../generator_v0/tree_generator.py`, with two changes:

* Label maps are written as **uint16** (int32 if a value does not fit), never
  uint8, so label sets with hundreds of labels (NextBrain, ~333) are safe.
  Internally the label indices are int16 (up to 32767 labels).
* **Resolution augmentation** as in SynthSeg (Billot et al. 2023, Sec. 3.1.4,
  eqs. 20-26), built on cornucopia's low-resolution transforms
  (`cornucopia/psf.py`). See `../siam_generator/understanding/
  resolution_variability.ipynb`. The sampled acquisition resolution also
  decides which tree nodes are visible, so the intensity model and the image
  resolution stay consistent.

Pipeline
--------
Offline, once per template (`prepare_template`):
    1. Keep the raw label values (no regrouping). Labels missing from the
       tree can be sent to background.
    2. Label super-resolution to `target_voxel / factor` (0.25 mm):
       one-hot -> Sinc upsampling -> Gaussian smoothing (0.5 mm) -> argmax.
       Each label is processed in its own bounding box, and background is
       obtained as 1 - sum(other labels), so the cost scales with the size
       of the structures, not with the number of labels.

Online, per sample (`TreeGenerator.__call__`):
    0. Sample the acquisition (SynthSeg priors): none, isotropic or one thick
       slice axis; spacing r_spac, thickness r_thick, alpha.
   At high resolution (0.25 mm):
    3. Random erosion/dilation between pairs of label groups (tree nodes or
       label values).
    4. Random affine + elastic deformation.
    5. Partial volume: fraction of each label inside every training voxel
       (0.75 mm), computed slab by slab with a histogram.
   At training resolution (0.75 mm):
    6. Tree intensities: principal colors C ~ U, V ~ U, then
       `TreeSampling.compute_stats` gives one (mu, sigma) per label, with
       node visibility decided by the acquisition resolution of step 0.
       Labels whose node collapses share one Gaussian (generation class).
    7. Bias field, motion artefacts.
    8. Resolution: Gaussian slice profile -> downsample to r_spac ->
       (noise, in low-res space) -> upsample back to 0.75 mm.
    9. Gaussian noise (if not added in step 8), min-max to [0, 1].

The label map is never degraded: the segmentation target stays at 0.75 mm
with sharp boundaries (SynthSeg Sec. 3.1.5).

The output segmentation is in index space (0 = background, i = labels[i]);
`TreeGenerator.labels` maps indices back to raw label values. Every sample
also returns the label groups that share a Gaussian (for merging labels in
the loss at that resolution).
"""

import argparse
import copy
import json
import math
import os
import random
import sys
from dataclasses import dataclass, field, fields
from math import inf

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F

try:   # local cornucopia fork (same as generator_v0), else the installed one
    sys.path.append('/home/agustin/phd/local_preprocessing/augmentation')
    from local_cornucopia import cornucopia as cc
    from local_cornucopia.cornucopia.base import FinalTransform, \
        NonFinalTransform
    from local_cornucopia.cornucopia.baseutils import prepare_output
    from local_cornucopia.cornucopia.random import Uniform
    from local_cornucopia.cornucopia.utils import warps
    from local_cornucopia.cornucopia.utils.conv import smoothnd
except ImportError:
    import cornucopia as cc
    from cornucopia.base import FinalTransform, NonFinalTransform
    from cornucopia.baseutils import prepare_output
    from cornucopia.random import Uniform
    from cornucopia.utils import warps
    from cornucopia.utils.conv import smoothnd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tree_sampling import LabelTree, TreeSampling  # noqa: E402


# ---------------------------------------------------------------------------
# 1. Label maps
# ---------------------------------------------------------------------------

def label_dtype(min_value, max_value):
    """
    Numpy dtype used to store a label map on disk.

    uint16 whenever the values fit (0..65535), so the files are small and
    hold any realistic label set (NextBrain has ~333 labels with values
    above 255). Otherwise int32. uint8 is never used on purpose: it silently
    wraps values > 255.

    Parameters
    ----------
    min_value, max_value : int
        Smallest and largest label value of the map.

    Returns
    -------
    numpy dtype
    """
    if min_value >= 0 and max_value <= np.iinfo(np.uint16).max:
        return np.uint16
    if np.iinfo(np.int32).min <= min_value and \
            max_value <= np.iinfo(np.int32).max:
        return np.int32
    return np.int64


def save_labels(arr, affine, path):
    """
    Save an integer label map as NIfTI with the dtype of `label_dtype`.

    Parameters
    ----------
    arr : array-like of int
        Label map.
    affine : (4, 4) ndarray
        Voxel-to-world matrix.
    path : str
        Output file (.nii or .nii.gz).
    """
    arr = np.asarray(arr).astype(np.int64)
    dt = label_dtype(int(arr.min()), int(arr.max()))
    nii = nib.Nifti1Image(arr.astype(dt), affine)
    nii.set_data_dtype(dt)
    nii.header.set_slope_inter(1, 0)   # integers stored as-is, no scaling
    nib.save(nii, path)


def restrict_labels(seg, labels, background=0):
    """
    Send every label not in `labels` to `background` (with a warning).

    Parameters
    ----------
    seg : array-like of int
        Label map with raw label values.
    labels : iterable of int
        Allowed label values (e.g. the labels of the tree).
    background : int
        Value given to the other labels.

    Returns
    -------
    seg : int64 ndarray
    """
    seg = np.asarray(seg).astype(np.int64)
    present = set(np.unique(seg).tolist())
    unknown = sorted(present - set(int(l) for l in labels) - {background})
    if unknown:
        print(f'[restrict_labels] WARNING: labels {unknown} not in the tree '
              f'-> background ({background})')
        seg[np.isin(seg, unknown)] = background
    return seg


# ---------------------------------------------------------------------------
# 2. Label super-resolution (offline)
# ---------------------------------------------------------------------------

def _bbox(mask, margin=0):
    """
    Bounding box of a 3D boolean mask.

    Parameters
    ----------
    mask : (X, Y, Z) bool tensor
    margin : int
        Voxels added on each side (clipped to the volume).

    Returns
    -------
    tuple of 3 slices, or None if the mask is empty.
    """
    box = []
    for d in range(mask.ndim):
        other = [i for i in range(mask.ndim) if i != d]
        nz = mask.any(dim=other[1]).any(dim=other[0]).nonzero()
        if len(nz) == 0:
            return None
        lo = max(int(nz[0]) - margin, 0)
        hi = min(int(nz[-1]) + 1 + margin, mask.shape[d])
        box.append(slice(lo, hi))
    return tuple(box)


def _sinc_upsample_smooth(x, factor, sigma_mm, out_voxel_mm):
    """
    Band-limited (Sinc) upsampling with Gaussian smoothing, in Fourier.

    The centred spectrum is zero-padded to `factor` times the size (Sinc
    interpolation) and multiplied by the transfer function of a Gaussian of
    std `sigma_mm`. Output voxel `j` sits at input coordinate `j / factor`.

    Parameters
    ----------
    x : (X, Y, Z) float tensor
    factor : int
        Integer upsampling factor.
    sigma_mm : float
        Std of the Gaussian smoothing, in mm (0 = no smoothing).
    out_voxel_mm : float
        Voxel size of the output, in mm.

    Returns
    -------
    (factor*X, factor*Y, factor*Z) float tensor
    """
    shape = x.shape
    new_shape = [s * factor for s in shape]
    X = torch.fft.fftshift(torch.fft.fftn(x))
    Xp = torch.zeros(new_shape, dtype=X.dtype, device=X.device)
    # keep the DC component (index s//2 after fftshift) aligned
    slicer = tuple(slice(n // 2 - s // 2, n // 2 - s // 2 + s)
                   for n, s in zip(new_shape, shape))
    Xp[slicer] = X
    del X
    if sigma_mm:
        # frequency grid of the HR volume in cycles/mm
        for d, n in enumerate(new_shape):
            f = torch.fft.fftshift(torch.fft.fftfreq(
                n, d=out_voxel_mm, device=x.device))
            g = torch.exp(-2 * (math.pi * sigma_mm * f) ** 2)
            view = [1] * len(new_shape)
            view[d] = n
            Xp *= g.view(view)
    y = torch.fft.ifftn(torch.fft.ifftshift(Xp)).real
    return y * (factor ** len(shape))


def super_resolve_labels(seg, factor, voxel_mm, sigma_mm=0.5, background=0,
                         margin=None):
    """
    One-hot -> Sinc upsampling -> Gaussian smoothing -> argmax.

    Every foreground label is upsampled inside its bounding box (+ `margin`
    input voxels; outside it the smoothed indicator is ~0). The background
    channel is never computed: upsampling and smoothing are linear and the
    one-hot channels sum to 1, so background = 1 - sum(foreground).

    Parameters
    ----------
    seg : (X, Y, Z) int tensor
        Label map with raw label values.
    factor : int
        Upsampling factor.
    voxel_mm : float
        Voxel size of `seg`, in mm.
    sigma_mm : float
        Std of the Gaussian smoothing, in mm.
    background : int
        Raw value of the background.
    margin : int, optional
        Bounding-box margin in input voxels (default: past the kernel
        support).

    Returns
    -------
    (factor*X, factor*Y, factor*Z) int16/int32 tensor of raw label values.
    """
    out_voxel_mm = voxel_mm / factor
    if margin is None:
        # well past the support of the smoothing kernel and of sinc ringing
        margin = max(4, int(math.ceil(4 * sigma_mm / voxel_mm)) + 2)
    hr_shape = [s * factor for s in seg.shape]
    best = torch.zeros(hr_shape, dtype=torch.float32, device=seg.device)
    total = torch.zeros_like(best)
    dtype = torch.int32 if int(seg.max()) > 32767 else torch.int16
    out = torch.full(hr_shape, background, dtype=dtype, device=seg.device)
    values = [int(v) for v in torch.unique(seg).tolist() if v != background]
    for v in values:
        mask = seg == v
        box = _bbox(mask, margin)
        y = _sinc_upsample_smooth(mask[box].float(), factor, sigma_mm,
                                  out_voxel_mm)
        del mask
        hr_box = tuple(slice(b.start * factor, b.stop * factor) for b in box)
        better = y > best[hr_box]
        out[hr_box].masked_fill_(better, v)
        best[hr_box] = torch.maximum(best[hr_box], y)
        total[hr_box] += y
        del y, better
    out.masked_fill_(1 - total > best, background)
    return out


def prepare_template(seg_path, out_path=None, target_voxel=0.75, factor=3,
                     margin_mm=16., sigma_mm=0.5, labels=None, background=0,
                     device='cpu'):
    """
    Build a high-resolution template from a label map, keeping its labels.

    The label map is cropped around the foreground (+margin) and
    super-resolved to `target_voxel / factor` mm. The HR shape is a multiple
    of `factor` so that the partial-volume pooling lands exactly on the
    training grid.

    Parameters
    ----------
    seg_path : str
        Input label map (NIfTI, isotropic, voxel size an integer multiple
        of the HR voxel).
    out_path : str, optional
        Where to save the HR template (uint16).
    target_voxel : float
        Training voxel size in mm.
    factor : int
        HR voxel = target_voxel / factor.
    margin_mm : float
        Background margin kept around the foreground.
    sigma_mm : float
        Smoothing of the one-hot channels before argmax.
    labels : iterable of int, optional
        Allowed labels (e.g. those of the tree); others go to background.
    background : int
        Raw background value.
    device : str or torch.device

    Returns
    -------
    lab_hr : (X, Y, Z) int tensor of raw label values
    affine_hr : (4, 4) ndarray
    """
    nii = nib.load(seg_path)
    seg = np.asarray(nii.dataobj).round().astype(np.int64)
    if labels is not None:
        seg = restrict_labels(seg, labels, background)
    affine = nii.affine
    voxel_mm = float(np.mean(nii.header.get_zooms()[:3]))

    # FFT upsampling ratio from the input grid to the HR grid
    hr_voxel = target_voxel / factor
    up = voxel_mm / hr_voxel
    if abs(up - round(up)) > 1e-3:
        raise ValueError(f'input voxel {voxel_mm} mm must be an integer '
                         f'multiple of the HR voxel {hr_voxel} mm')
    up = int(round(up))

    # crop/pad around foreground with a margin (out-of-bounds -> background)
    idx = np.argwhere(seg != background)
    margin = int(math.ceil(margin_mm / voxel_mm))
    lo = idx.min(0) - margin
    hi = idx.max(0) + 1 + margin
    # make HR shape a multiple of `factor`
    size = hi - lo
    while np.any((size * up) % factor):
        size = np.where((size * up) % factor, size + 1, size)
    hi = lo + size
    pad_lo = np.maximum(-lo, 0)
    pad_hi = np.maximum(hi - np.array(seg.shape), 0)
    seg = np.pad(seg, list(zip(pad_lo, pad_hi)), constant_values=background)
    lo = lo + pad_lo
    hi = hi + pad_lo
    seg = seg[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
    crop_shift = np.eye(4)
    crop_shift[:3, 3] = lo - pad_lo
    affine = affine @ crop_shift

    lab = torch.as_tensor(seg, device=device)
    with torch.no_grad():
        lab_hr = super_resolve_labels(lab, up, voxel_mm, sigma_mm, background)
    # HR voxel j sits at input voxel coordinate j / up
    affine_hr = affine @ np.diag([1 / up, 1 / up, 1 / up, 1])

    if out_path:
        save_labels(lab_hr.cpu().numpy(), affine_hr, out_path)
    return lab_hr, affine_hr


def load_template(path, device='cpu'):
    """
    Load a cached HR template.

    Returns
    -------
    lab : (X, Y, Z) int32 tensor of raw label values
    affine : (4, 4) ndarray
    """
    nii = nib.load(path)
    lab = torch.as_tensor(np.asarray(nii.dataobj).astype(np.int32),
                          device=device)
    return lab, nii.affine


# ---------------------------------------------------------------------------
# 3. Resolution augmentation (SynthSeg Sec. 3.1.4, on top of cornucopia)
# ---------------------------------------------------------------------------

FWHM_PER_SIGMA = 2 * math.sqrt(2 * math.log(2))


def sigma_thick_vox(r_thick, alpha, r_hr):
    """
    Std of the Gaussian slice profile, in voxels of the input grid (eq. 23).

    sigma = 2 alpha log(10) / (2 pi) * r_thick / r_hr: the power of the HR
    signal is divided by 10 at the cut-off frequency of the thick slice.

    Parameters
    ----------
    r_thick : float
        Slice thickness in mm.
    alpha : float
        Random deviation of the nominal thickness (eq. 22).
    r_hr : float
        Voxel size of the image being degraded, in mm.
    """
    return 2 * alpha * math.log(10) / (2 * math.pi) * r_thick / r_hr


class LowResAcquisitionFinalTransform(FinalTransform):
    """
    Deterministic thick-slice acquisition along any subset of axes.

    Per-axis generalisation of `cc.LowResSliceFinalTransform` (same steps,
    same cornucopia primitives), so that isotropic low-resolution images can
    also have a slice thickness different from the spacing:

        1. smooth with a Gaussian of FWHM `fwhm[d]` voxels along each axis
           (`cornucopia.utils.conv.smoothnd`);
        2. downsample to ceil(n_d / spacing[d]) voxels (trilinear,
           align_corners=True);
        3. optionally add noise in low-resolution space;
        4. upsample back to the input grid.

    With a single degraded axis it is identical to
    `cc.LowResSliceFinalTransform(axis, spacing, fwhm / spacing, noise)`.

    Parameters
    ----------
    spacing : sequence of 3 floats
        Slice spacing in input voxels (1 = axis untouched).
    fwhm : sequence of 3 floats
        FWHM of the slice profile in input voxels (0 = no smoothing).
    noise : cornucopia Transform, optional
        Applied to the low-resolution image.
    returns : see cornucopia ('input', 'lowres', 'output').
    """

    def __init__(self, spacing, fwhm, noise=None, **kwargs):
        super().__init__(**kwargs)
        self.spacing = [float(s) for s in spacing]
        self.fwhm = [float(f) for f in fwhm]
        self.noise = noise

    @property
    def is_final(self):
        return self.noise.is_final if self.noise else True

    def _get_smallshape(self, x):
        """Spatial shape of the low-resolution grid."""
        return [n if s <= 1 else max(2, math.ceil(n / s))
                for n, s in zip(x.shape[1:], self.spacing)]

    def _unroll(self, x, max_depth=inf):
        """Fix the random noise parameters on the low-resolution shape."""
        if max_depth == 0:
            return self
        noise = None
        if self.noise:
            fake = x.new_zeros([]).expand([len(x), *self._get_smallshape(x)])
            noise = self.noise.unroll(fake, max_depth - 1)
        return type(self)(self.spacing, self.fwhm, noise, **self.get_prm())

    def _xform(self, x):
        """Apply steps 1-4 to a (C, X, Y, Z) tensor."""
        def interpol(v, shape):
            return F.interpolate(v[None], size=shape, align_corners=True,
                                 mode='trilinear')[0]

        ishape = list(x.shape[1:])
        oshape = self._get_smallshape(x)
        y = smoothnd(x, fwhm=self.fwhm) if any(self.fwhm) else x
        if oshape != ishape:
            y = interpol(y, oshape)
        if self.noise:
            y = self.noise(y)
        z = interpol(y, ishape) if oshape != ishape else y
        return prepare_output(dict(input=x, lowres=y, output=z), self.returns)


class SynthSegLowResTransform(NonFinalTransform):
    """
    Random acquisition resolution with SynthSeg's priors.

    Per image, with probability `1 - prob` the image is left at its native
    resolution. Otherwise (SynthSeg `SampleResolution` + eqs. 20-23):

    * isotropic (probability `iso_prob`): r_spac ~ U(r_hr, max_iso) on the
      three axes;
    * anisotropic: one random axis, r_spac ~ U(r_hr, max_aniso), the other
      two stay at r_hr;
    * per degraded axis: r_thick ~ U(r_hr, r_spac) (or r_thick = r_spac if
      `random_thickness` is False), alpha ~ U(*alpha) and
      sigma = 2 alpha log(10) / (2 pi) r_thick / r_hr.

    The parameters are then passed to `LowResAcquisitionFinalTransform`.

    Sampling (`sample`) and building the deterministic transform
    (`make_final`) are separate methods, so the generator can draw the
    acquisition first and use its resolution for the tree intensities.

    Parameters
    ----------
    r_hr : float
        Voxel size of the image to degrade, in mm.
    prob : float
        Probability of degrading the resolution.
    iso_prob : float
        Probability of an isotropic acquisition (when degraded).
    max_iso : float
        Largest isotropic voxel size in mm.
    max_aniso : float
        Largest slice spacing in mm of an anisotropic acquisition.
    alpha : (float, float)
        Range of alpha (eq. 22).
    random_thickness : bool
        Sample r_thick < r_spac (slice gap) or use r_thick = r_spac.
    noise : cornucopia Transform, optional
        Noise added in low-resolution space.
    """

    def __init__(self, r_hr=0.75, prob=0.95, iso_prob=0.1, max_iso=4.,
                 max_aniso=8., alpha=(1 / 1.03, 1.03), random_thickness=True,
                 noise=None, **kwargs):
        super().__init__(**kwargs)
        self.r_hr = float(r_hr)
        self.prob = prob
        self.iso_prob = iso_prob
        self.max_iso = max_iso
        self.max_aniso = max_aniso
        self.alpha = alpha
        self.random_thickness = random_thickness
        self.noise = noise

    def sample(self):
        """
        Draw one acquisition.

        Returns
        -------
        dict with
            mode : 'none', 'iso' or 'aniso'
            axis : degraded axis (aniso) or None
            spacing_mm, thickness_mm : 3 floats each
            alpha : 3 floats
            sigma_vox : 3 floats, std of the slice profile (input voxels)
        """
        r = self.r_hr
        prm = dict(mode='none', axis=None, spacing_mm=[r] * 3,
                   thickness_mm=[r] * 3, alpha=[1.] * 3, sigma_vox=[0.] * 3)
        if random.random() >= self.prob:
            return prm
        if random.random() < self.iso_prob:
            if self.max_iso <= r:
                return prm
            prm['mode'] = 'iso'
            axes = [0, 1, 2]
            prm['spacing_mm'] = [random.uniform(r, self.max_iso)] * 3
        else:
            if self.max_aniso <= r:
                return prm
            prm['mode'] = 'aniso'
            prm['axis'] = random.randrange(3)
            axes = [prm['axis']]
            prm['spacing_mm'][prm['axis']] = random.uniform(r, self.max_aniso)
        for d in axes:
            spac = prm['spacing_mm'][d]                                # eq. 20
            thick = random.uniform(r, spac) if self.random_thickness \
                else spac                                              # eq. 21
            alpha = random.uniform(*self.alpha)                        # eq. 22
            prm['thickness_mm'][d] = thick
            prm['alpha'][d] = alpha
            prm['sigma_vox'][d] = sigma_thick_vox(thick, alpha, r)     # eq. 23
        return prm

    def make_final(self, prm, noise=True):
        """
        Deterministic transform for the acquisition `prm` (from `sample`).

        Parameters
        ----------
        prm : dict
        noise : bool
            Attach `self.noise` (added in low-resolution space).

        Returns
        -------
        LowResAcquisitionFinalTransform, or None if `prm['mode'] == 'none'`.
        """
        if prm['mode'] == 'none':
            return None
        spacing = [s / self.r_hr for s in prm['spacing_mm']]
        fwhm = [FWHM_PER_SIGMA * s for s in prm['sigma_vox']]
        return LowResAcquisitionFinalTransform(
            spacing, fwhm, self.noise if noise else None, **self.get_prm())

    def _unroll(self, x, max_depth=inf):
        """cornucopia API: sample, then return the deterministic transform."""
        if max_depth == 0:
            return self
        prm = self.sample()
        final = self.make_final(prm)
        if final is None:
            return cc.IdentityTransform() if self.noise is None \
                else self.noise.unroll(x, max_depth - 1)
        final = final.unroll(x, max_depth - 1)
        final.synthseg_prm = prm
        return final


# ---------------------------------------------------------------------------
# 4. Configuration
# ---------------------------------------------------------------------------

def _f(default, help):
    """Dataclass field with a help string (used by `run_example.py`)."""
    return field(default=default, metadata={'help': help})


@dataclass
class TreeGeneratorConfig:
    """
    All the parameters of the generator. Ranges are (lo, hi) tuples sampled
    uniformly per image. Load with `from_json` (lists -> tuples); every field
    is also exposed as a command-line flag in `run_example.py`.
    """
    # -- grid ----------------------------------------------------------------
    target_voxel: float = _f(0.75, 'training voxel size (mm)')
    factor: int = _f(3, 'HR template voxel = target_voxel / factor')
    background: int = _f(0, 'raw label value of the background')
    template_sigma_mm: float = _f(
        0.5, 'Gaussian smoothing (mm) of the labels when building the HR '
             'template (0 = none)')

    # -- label space ---------------------------------------------------------
    output_labels: str = _f(
        'templates', "'templates': labels found in the templates; "
                     "'tree': every label of the tree")
    unassigned: str = _f(
        'error', "template labels missing from the tree: 'error' or "
                 "'background'")

    # -- shape augmentation --------------------------------------------------
    morpho_pairs: tuple = _f(
        (), 'pairs (a, b) of label groups (tree node name, label value or '
            'list of those); each pair randomly a grows into b or b into a')
    morpho_iter: tuple = _f((1, 4), 'range of HR voxel layers grown')
    morpho_prob: float = _f(0.5, 'probability of applying each pair')

    # -- spatial deformation -------------------------------------------------
    rotations: float = _f(10., 'max rotation (deg), U(-r, r)')
    zooms: float = _f(0.1, 'max zoom, 1 + U(-z, z)')
    shears: float = _f(0., 'max shear')
    translations_mm: float = _f(0., 'max translation (mm)')
    elastic_dmax_mm: float = _f(7.5, 'max elastic displacement (mm)')
    elastic_ctrl: int = _f(7, 'elastic control points per axis')

    # -- tree intensities ----------------------------------------------------
    colors_mu: tuple = _f(
        (0., 255.), 'range of the principal color means (tree units)')
    colors_sigma: tuple = _f(
        (0.255, 2.55), 'range of the principal color stds (tree units)')
    min_sigma: float = _f(0., 'lower bound of every label std')
    intensity_scale: float = _f(
        255., 'image / scale before the artefacts')
    tree_resolution: str = _f(
        'acquisition', "resolution used for node visibility: 'acquisition' "
                       "(coarsest sampled spacing, target_voxel if not "
                       "degraded) or 'config' (field `resolution`)")
    resolution: object = _f(
        None, "with tree_resolution='config': None (= target_voxel), a "
              "number or a (lo, hi) range in mm")
    contrasts: object = _f(
        None, 'contrasts for the r_max lookup; one picked per image '
              "(None = r_max 'default')")

    # -- artefacts -----------------------------------------------------------
    bias_prob: float = _f(1.0, 'probability of a bias field')
    bias_vmax: float = _f(0.5, 'bias field in (1 - v, 1 + v)')
    bias_ctrl: int = _f(5, 'bias field control points per axis')
    motion_prob: float = _f(0.3, 'probability of motion artefacts')
    motion_shots: int = _f(3, 'number of k-space shots')
    motion_rotations: float = _f(10., 'max motion rotation (deg)')
    motion_translations_mm: float = _f(10., 'max motion translation (mm)')
    noise_sigma: tuple = _f((0.01, 0.1), 'range of the Gaussian noise std')

    # -- resolution augmentation (SynthSeg Sec. 3.1.4) ---------------------
    lowres_prob: float = _f(
        0.95, 'probability of simulating a lower acquisition resolution')
    lowres_iso_prob: float = _f(
        0.1, 'probability that the acquisition is isotropic')
    lowres_max_iso: float = _f(4., 'max isotropic voxel size (mm)')
    lowres_max_aniso: float = _f(
        8., 'max slice spacing of anisotropic acquisitions (mm)')
    lowres_alpha: tuple = _f(
        (1 / 1.03, 1.03), 'range of alpha (deviation of the slice profile)')
    lowres_random_thickness: bool = _f(
        True, 'thickness ~ U(target_voxel, spacing) (else = spacing)')
    noise_space: str = _f(
        'lowres', "'lowres': noise added in low-res space, before "
                  "upsampling (as in a real scan); 'highres': on the "
                  "training grid, before the resolution step (SynthSeg)")

    @classmethod
    def from_json(cls, path, **overrides):
        """
        Load a config from JSON; unknown keys raise. `overrides` win over
        the file. JSON lists become tuples (except `contrasts`).
        """
        with open(path) as f:
            d = json.load(f)
        d.update(overrides)
        return cls.from_dict(d)

    @classmethod
    def from_dict(cls, d):
        """Build from a dict (same rules as `from_json`)."""
        known = {f.name for f in fields(cls)}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f'Unknown config fields: {sorted(unknown)}')
        d = dict(d)
        for key, value in d.items():   # JSON lists -> tuples
            if isinstance(value, list) and key != 'contrasts':
                d[key] = tuple(tuple(v) if isinstance(v, list) else v
                               for v in value)
        return cls(**d)

    def to_dict(self):
        """Plain dict of all the fields (JSON serialisable)."""
        return {f.name: getattr(self, f.name) for f in fields(self)}


# ---------------------------------------------------------------------------
# 5. Online generator
# ---------------------------------------------------------------------------

def _uniform(r):
    """A number, or a (lo, hi) range sampled uniformly."""
    if isinstance(r, (int, float)):
        return float(r)
    return random.uniform(*r)


class TreeGenerator:
    """
    Draw (image, label) pairs from one or several HR templates, with
    intensities from a label tree.

    Parameters
    ----------
    templates : list of (lab_hr, affine_hr)
        Outputs of `prepare_template` / `load_template` (raw label values).
    tree : LabelTree, dict or path to the tree JSON
    config : TreeGeneratorConfig
    device : str or torch.device

    Attributes
    ----------
    labels : (K,) int ndarray
        Raw label value of each output index (index 0 = background).
    index : dict
        Raw label value -> output index.
    """

    def __init__(self, templates, tree, config=None, device='cpu'):
        self.cfg = config or TreeGeneratorConfig()
        self.device = torch.device(device)
        self.tree = LabelTree.load(tree)
        cfg = self.cfg
        bg = cfg.background

        # -- label space: index 0 = background, i = self.labels[i] ----------
        tree_labels = set(self.tree.label_to_node)
        found = set()
        for lab, _ in templates:
            found |= set(torch.unique(lab).tolist())
        missing = sorted(found - tree_labels - {bg})
        if missing and cfg.unassigned == 'error':
            raise ValueError(f'Template labels {missing} are not in the tree. '
                             f'Add them to the tree or use '
                             f"unassigned='background'.")
        if cfg.output_labels == 'tree':
            labels = tree_labels
        elif cfg.output_labels == 'templates':
            labels = found & tree_labels
        else:
            raise ValueError(f'output_labels must be tree or templates, '
                             f'got {cfg.output_labels!r}')
        self.labels = np.array([bg] + sorted(set(labels) - {bg}), dtype=int)
        self.nb_labels = len(self.labels)
        if self.nb_labels > np.iinfo(np.int16).max:
            raise ValueError(f'{self.nb_labels} labels: too many for int16')
        self.index = {int(l): i for i, l in enumerate(self.labels)}

        # raw values -> indices; labels not kept go to background (index 0).
        # int16 indices: no uint8 overflow with > 255 labels.
        lut = torch.zeros(int(max(max(found), self.labels.max())) + 1,
                          dtype=torch.long)
        for lab, i in self.index.items():
            lut[lab] = i
        self.templates = []
        for lab, aff in templates:
            lab = lut[lab.long().cpu()].to(torch.int16)
            self.templates.append((lab.to(self.device), aff))

        self._build_steps()

    # fields that change the label space or the template grid: they cannot
    # be changed by `with_config` (a new TreeGenerator is needed)
    _TEMPLATE_FIELDS = ('target_voxel', 'factor', 'background',
                        'template_sigma_mm', 'output_labels', 'unassigned')

    def _build_steps(self):
        """
        Build everything that depends on the config but not on the
        templates: intensity model, morphology rules and the cornucopia
        transforms. Called by `__init__` and `with_config`.
        """
        cfg = self.cfg
        if cfg.noise_space not in ('lowres', 'highres'):
            raise ValueError(f"noise_space must be 'lowres' or 'highres', "
                             f'got {cfg.noise_space!r}')
        if cfg.tree_resolution not in ('acquisition', 'config'):
            raise ValueError(f"tree_resolution must be 'acquisition' or "
                             f"'config', got {cfg.tree_resolution!r}")

        # -- intensity model ----------------------------------------------
        # std_noise=0: the acquisition noise is added later
        self.scheme = TreeSampling(
            self.labels, self.tree, std_noise=0., min_sigma=cfg.min_sigma,
            unassigned='zero', seed=int(np.random.randint(2 ** 31)),
            verbose=False)

        # -- morphology rules, resolved to index sets ---------------------
        self.morpho_pairs = []
        for a, b in cfg.morpho_pairs:
            ia, ib = self._resolve(a), self._resolve(b)
            if ia and ib:
                self.morpho_pairs.append((a, b, ia, ib))

        # Affine + elastic sampling field. It is drawn by cornucopia on the
        # training (0.75 mm) grid, then linearly upsampled slab-by-slab and
        # used to resample the 0.25 mm labels with nearest neighbour (see
        # `_deform`). The field is smooth, so this is equivalent to drawing
        # it at 0.25 mm, at a fraction of the memory.
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
        self.bias = cc.MaybeTransform(
            cc.RandomMulFieldTransform(
                shape=cfg.bias_ctrl, vmax=cfg.bias_vmax, symmetric=True),
            cfg.bias_prob)
        self.noise = cc.RandomGaussianNoiseTransform(
            sigma=Uniform(*cfg.noise_sigma))
        self.lowres = SynthSegLowResTransform(
            r_hr=cfg.target_voxel,
            prob=cfg.lowres_prob,
            iso_prob=cfg.lowres_iso_prob,
            max_iso=cfg.lowres_max_iso,
            max_aniso=cfg.lowres_max_aniso,
            alpha=tuple(cfg.lowres_alpha),
            random_thickness=cfg.lowres_random_thickness,
            noise=self.noise if cfg.noise_space == 'lowres' else None,
        )
        self.normalize = cc.MinMaxTransform()

    def with_config(self, **overrides):
        """
        New generator with some config fields changed, sharing the
        (already indexed) templates of this one: cheap, no extra memory.

        Useful to compare settings, e.g.
        `gen.with_config(lowres_prob=1., noise_space='highres')`.

        Parameters
        ----------
        **overrides : TreeGeneratorConfig fields. The fields that change
            the label space or the grid (`_TEMPLATE_FIELDS`) are not
            allowed: build a new TreeGenerator for those.

        Returns
        -------
        TreeGenerator
        """
        bad = sorted(set(overrides) & set(self._TEMPLATE_FIELDS))
        if bad:
            raise ValueError(f'{bad} change the label space / grid: build a '
                             f'new TreeGenerator instead')
        d = self.cfg.to_dict()
        d.update(overrides)
        new = copy.copy(self)    # shares templates, labels, index, tree
        new.cfg = TreeGeneratorConfig.from_dict(d)
        new._build_steps()
        return new

    # -- label helpers ---------------------------------------------------------

    def _resolve(self, spec):
        """
        Label group -> sorted output indices.

        Parameters
        ----------
        spec : str, int or list
            Tree node name (all the labels of its subtree), raw label value,
            or a list of those.
        """
        if isinstance(spec, (list, tuple)):
            out = set()
            for s in spec:
                out |= set(self._resolve(s))
            return sorted(out)
        if isinstance(spec, str):
            if spec not in self.tree.nodes:
                raise ValueError(f'morpho_pairs: unknown tree node {spec!r}')
            labs = []
            for name in [spec] + self.tree.descendants(spec):
                labs += self.tree.nodes[name].labels
        else:
            labs = [int(spec)]
        return sorted(self.index[l] for l in labs if l in self.index)

    def label_names(self):
        """Unique name per output index: tree node (+ value if shared)."""
        l2n = self.tree.label_to_node
        names = []
        for i, lab in enumerate(self.labels):
            node = l2n.get(int(lab), 'background' if i == 0 else 'unknown')
            shared = node in self.tree.nodes and \
                len(self.tree.nodes[node].labels) > 1
            names.append(f'{node}_{lab}' if shared else node)
        return names

    def output_affine(self, template_index=0):
        """Voxel-to-world matrix of the training grid of a template."""
        f = self.cfg.factor
        pool = np.diag([f, f, f, 1.]).astype(float)
        pool[:3, 3] = (f - 1) / 2   # pooled voxel i <-> HR voxel f*i+(f-1)/2
        return self.templates[template_index][1] @ pool

    # -- steps ---------------------------------------------------------------

    def _isin(self, lab, idx):
        """Boolean mask of the voxels of `lab` whose index is in `idx`."""
        return torch.isin(lab, torch.as_tensor(idx, device=lab.device,
                                               dtype=lab.dtype))

    def _morphology(self, lab):
        """
        Constrained random dilation/erosion between pairs of label groups.

        For each pair, with probability `morpho_prob`, one group (chosen at
        random) grows into the other by `morpho_iter` voxel layers
        (26-neighbourhood). With several labels in the growing group each new
        voxel copies a neighbouring label. Works in the pair's bounding box.

        Parameters
        ----------
        lab : (X, Y, Z) int16 tensor of indices (modified in place)

        Returns
        -------
        lab
        """
        cfg = self.cfg
        lo, hi = cfg.morpho_iter
        for _, _, a, b in self.morpho_pairs:
            if random.random() > cfg.morpho_prob:
                continue
            grow, shrink = (a, b) if random.random() < 0.5 else (b, a)
            n_iter = random.randint(lo, hi)
            box = _bbox(self._isin(lab, grow) | self._isin(lab, shrink),
                        margin=n_iter + 1)
            if box is None:
                continue
            sub = lab[box]
            for _ in range(n_iter):
                # index + 1 of the growing labels, 0 elsewhere; the max over
                # the neighbourhood picks one neighbouring growing label
                val = torch.where(self._isin(sub, grow), sub.float() + 1, 0.)
                nbr = F.max_pool3d(val[None, None], 3, stride=1,
                                   padding=1)[0, 0]
                take = self._isin(sub, shrink) & (nbr > 0)
                sub = torch.where(take, (nbr - 1).to(sub.dtype), sub)
                del val, nbr, take
            lab[box] = sub
            del sub
        return lab

    def _deform(self, lab, slab=16):
        """
        Random affine + elastic deformation of the HR labels (nearest).

        Parameters
        ----------
        lab : (X, Y, Z) int tensor, shape multiple of `factor`
        slab : int
            Training-grid slices upsampled at once (memory).

        Returns
        -------
        (X, Y, Z) tensor, same dtype
        """
        f = self.cfg.factor
        lr_shape = [s // f for s in lab.shape]
        # (1, x, y, z, 3) absolute sampling coordinates on the LR grid
        flow = self.deform(lab.new_zeros([1, *lr_shape], dtype=torch.float))
        # LR voxel i is centred on HR voxel f*i + (f-1)/2
        flow = (flow * f + (f - 1) / 2).movedim(-1, 1)   # (1, 3, x, y, z)
        src = lab[None, None].float()   # exact for indices < 2**24
        out = torch.empty_like(lab)
        for i0 in range(0, lr_shape[0], slab):
            i1 = min(i0 + slab, lr_shape[0])
            a, b = max(i0 - 1, 0), min(i1 + 1, lr_shape[0])  # 1-voxel halo
            # align_corners=False maps HR voxel j to LR coord (j - (f-1)/2)/f
            up = F.interpolate(flow[:, :, a:b], scale_factor=f,
                               mode='trilinear', align_corners=False)
            up = up[:, :, (i0 - a) * f:(i1 - a) * f].movedim(1, -1)
            out[i0 * f:i1 * f] = warps.apply_flow(
                src, up, has_identity=True, mode='nearest',
                padding_mode='zeros')[0, 0].to(lab.dtype)
            del up
        return out

    def _sample_contrast(self, acquisition):
        """
        Principal colors, tree resolution and contrast of one image.

        Parameters
        ----------
        acquisition : dict
            Output of `SynthSegLowResTransform.sample`.

        Returns
        -------
        C, V : (n_colors,) ndarrays
        res : float, resolution (mm) used for node visibility
        contrast : str or None
        """
        cfg = self.cfg
        n = self.tree.n_colors
        C = np.array([random.uniform(*cfg.colors_mu) for _ in range(n)])
        V = np.array([random.uniform(*cfg.colors_sigma) for _ in range(n)])
        if cfg.tree_resolution == 'acquisition':
            # coarsest axis: a structure must be resolved along every axis
            res = float(max(acquisition['spacing_mm']))
        else:
            res = cfg.target_voxel if cfg.resolution is None \
                else _uniform(cfg.resolution)
        contrast = random.choice(list(cfg.contrasts)) if cfg.contrasts \
            else None
        return C, V, res, contrast

    def _synthesize(self, lab, mus, sigmas, group_of, return_pv=False,
                    slab=None):
        """
        Partial volume + tree intensities, slab by slab on the LR grid.

        For each LR voxel the f^3 HR labels are counted (histogram), which
        gives the partial-volume fraction of every label without one-hot
        volumes. The segmentation is the label with the largest fraction.
        Labels of one generation class share one Gaussian, and the image is
        sum_g pv_g * (mu_g + sigma_g * eps_g), with one noise draw per class.

        Parameters
        ----------
        lab : (X*f, Y*f, Z*f) int tensor of indices
        mus, sigmas : (K,) arrays, Gaussian of each label index
        group_of : (K,) int array, generation class of each label (-1 =
            unassigned)
        return_pv : bool
            Also return the partial volume of every class.
        slab : int, optional
            LR slices processed at once (default ~256 MB of counts).

        Returns
        -------
        img : (1, X, Y, Z) float tensor (tree units)
        seg : (X, Y, Z) int tensor of indices
        pv : (G, X, Y, Z) float tensor or None
        """
        f = self.cfg.factor
        K = self.nb_labels
        X, Y, Z = (s // f for s in lab.shape)
        dev = lab.device

        # generation classes (unassigned labels -> their own zero class)
        group_of = np.where(group_of < 0, group_of.max() + 1, group_of)
        _, group_of = np.unique(group_of, return_inverse=True)
        G = int(group_of.max()) + 1
        onehot = torch.zeros(K, G, device=dev)
        onehot[torch.arange(K), torch.as_tensor(group_of)] = 1
        mu_g = torch.zeros(G, device=dev)
        sd_g = torch.zeros(G, device=dev)
        mu_g[torch.as_tensor(group_of)] = torch.as_tensor(
            mus, dtype=torch.float, device=dev)
        sd_g[torch.as_tensor(group_of)] = torch.as_tensor(
            sigmas, dtype=torch.float, device=dev)

        img = torch.empty([X, Y, Z], device=dev)
        seg = torch.empty([X, Y, Z], dtype=lab.dtype, device=dev)
        pv = torch.empty([G, X, Y, Z], device=dev) if return_pv else None
        if slab is None:   # ~256 MB of label counts per slab
            slab = max(1, (1 << 26) // (Y * Z * K))
        for i0 in range(0, X, slab):
            i1 = min(i0 + slab, X)
            s = i1 - i0
            blk = lab[i0 * f:i1 * f].long()
            blk = blk.view(s, f, Y, f, Z, f).permute(0, 2, 4, 1, 3, 5)
            blk = blk.reshape(-1, f ** 3)
            p = torch.zeros(blk.shape[0], K, device=dev)
            p.scatter_add_(1, blk, torch.ones_like(blk, dtype=p.dtype))
            p /= f ** 3
            seg[i0:i1] = p.argmax(1).view(s, Y, Z).to(seg.dtype)
            pg = p @ onehot
            del p, blk
            eps = torch.randn_like(pg)
            img[i0:i1] = (pg * (mu_g + sd_g * eps)).sum(1).view(s, Y, Z)
            if return_pv:
                pv[:, i0:i1] = pg.T.reshape(G, s, Y, Z)
            del pg, eps
        return img[None], seg, pv

    def _motion(self, img, force=None):
        """
        Intra-scan motion artefacts (cornucopia), with probability
        `motion_prob` (or always/never with `force=True/False`).

        Parameters
        ----------
        img : (1, X, Y, Z) tensor
        force : bool, optional

        Returns
        -------
        (1, X, Y, Z) tensor
        """
        cfg = self.cfg
        apply = (random.random() < cfg.motion_prob) if force is None \
            else force
        if not apply:
            return img
        # cornucopia 0.5.1 workarounds for non-cubic volumes:
        # - the motion is sampled for the input shape but applied after
        #   moving `axis` to dim 1 -> we move the axis ourselves, use axis=1;
        # - the (shots, n) pattern broadcasts against the *last* dim -> we
        #   pass a (shots, n, 1, 1) pattern.
        axis = random.choice([1, 2, 3])
        img = img.movedim(axis, 1)
        n = img.shape[1]
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
        ).final(img)
        # cornucopia moves the object in every shot, and the shot holding
        # the k-space centre sets the global pose of the image, which would
        # then be misaligned with `seg`. Keep that shot at the reference
        # pose: the other shots still move and produce ghosting/blurring.
        centre = (n // 2) // length   # DC sits at index n//2 (fftshift)
        motion.motion[centre] = cc.AffineTransform(
            bound='reflection').final(img)
        return motion(img).movedim(1, axis)

    def _resolution(self, img, acquisition):
        """
        Resolution + noise steps (8-9 of the pipeline).

        With `noise_space='lowres'` the noise is added on the low-resolution
        grid (inside the cornucopia transform), otherwise before it. Without
        degradation the noise is simply added on the training grid.

        Parameters
        ----------
        img : (1, X, Y, Z) tensor
        acquisition : dict from `SynthSegLowResTransform.sample`

        Returns
        -------
        img : (1, X, Y, Z) tensor
        lowres : (1, x, y, z) tensor of the acquired image, or None
        """
        if self.cfg.noise_space == 'highres':
            img = self.noise(img)
        final = self.lowres.make_final(acquisition)
        if final is None:
            if self.cfg.noise_space == 'lowres':
                img = self.noise(img)
            return img, None
        final.returns = dict(lowres='lowres', output='output')
        out = final(img)
        return out['output'], out['lowres']

    # -- main ----------------------------------------------------------------

    @torch.no_grad()
    def __call__(self, template_index=None, return_pv=False,
                 return_steps=False, acquisition=None, motion=None):
        """
        Generate one (image, segmentation) pair.

        Parameters
        ----------
        template_index : int, optional
            Template to use (random by default).
        return_pv : bool
            Return the partial volume per generation class in `info['pv']`.
        return_steps : bool
            Return the intermediate volumes in `info['steps']` (for figures).
        acquisition : dict, optional
            Fixed acquisition instead of a random one, e.g.
            `dict(mode='aniso', axis=2, spacing_mm=[.75, .75, 5],
            thickness_mm=[.75, .75, 4], alpha=[1, 1, 1],
            sigma_vox=[0, 0, sigma_thick_vox(4, 1, .75)])`; easiest is to
            take one from `gen.lowres.sample()` and edit it.
        motion : bool, optional
            Force (True) or disable (False) the motion artefacts.

        Returns
        -------
        img : (1, X, Y, Z) float tensor in [0, 1]
        seg : (X, Y, Z) int16 tensor, indices into `self.labels`
        affine : (4, 4) ndarray, voxel-to-world of the training grid
        info : dict
            colors_mu, colors_sigma, resolution, contrast : what was drawn
            acquisition : simulated acquisition (mode, axis, spacing_mm,
                thickness_mm, alpha, sigma_vox)
            mus, sigmas : per-label Gaussian (index order, tree units)
            group_of : generation class of each label index; labels with
                the same class are indistinguishable in this image
            class_names : tree node of each generation class
            pv : (G, X, Y, Z) partial volume per class (if return_pv)
            steps : dict of intermediate volumes (if return_steps)
        """
        if template_index is None:
            template_index = random.randrange(len(self.templates))
        lab = self.templates[template_index][0]
        steps = {}

        # 0. acquisition, then the tree intensities that go with it
        if acquisition is None:
            acquisition = self.lowres.sample()
        C, V, res, contrast = self._sample_contrast(acquisition)
        mus, sigmas = self.scheme.compute_stats(C, V, res, contrast)
        group_of, class_names = self.scheme.label_set_lut(res, contrast)

        # 3-5. shape at 0.25 mm, then partial volume on the 0.75 mm grid
        lab = self._morphology(lab.clone())
        if return_steps:
            steps['morphology'] = lab.cpu()
        lab = self._deform(lab)
        if return_steps:
            steps['deformed'] = lab.cpu()
        img, seg, pv = self._synthesize(lab, mus, sigmas, group_of,
                                        return_pv or return_steps)
        del lab
        if return_steps:
            steps['pv'] = pv.cpu()
            pv = pv if return_pv else None

        # 6-9. intensities and artefacts on the 0.75 mm grid
        img = img / self.cfg.intensity_scale
        if return_steps:
            steps['tree'] = img.cpu()
        img = self.bias(img)
        if return_steps:
            steps['bias'] = img.cpu()
        img = self._motion(img, motion)
        if return_steps:
            steps['motion'] = img.cpu()
        img, lowres = self._resolution(img, acquisition)
        if return_steps:
            steps['lowres'] = None if lowres is None else lowres.cpu()
            steps['resolution'] = img.cpu()
        img = self.normalize(img)

        info = dict(colors_mu=C, colors_sigma=V, resolution=res,
                    contrast=contrast, acquisition=acquisition, mus=mus,
                    sigmas=sigmas, group_of=group_of,
                    class_names=class_names, pv=pv)
        if return_steps:
            info['steps'] = steps
        return img, seg, self.output_affine(template_index), info


# ---------------------------------------------------------------------------
# 6. I/O helpers and CLI (nnU-Net raw dataset layout)
# ---------------------------------------------------------------------------

def save_sample(img, seg, affine, img_path, seg_path, labels=None):
    """
    Save one sample.

    Parameters
    ----------
    img : (1, X, Y, Z) tensor, saved as float32
    seg : (X, Y, Z) int tensor of indices, saved as uint16 (see
        `label_dtype`)
    affine : (4, 4) ndarray
    img_path, seg_path : str
    labels : array-like, optional
        `gen.labels`: if given, indices are converted to raw label values.
    """
    seg = seg.cpu().numpy().astype(np.int64)
    if labels is not None:
        seg = np.asarray(labels)[seg]
    nib.save(nib.Nifti1Image(img[0].cpu().numpy().astype(np.float32),
                             affine), img_path)
    save_labels(seg, affine, seg_path)


def write_nnunet_dataset_json(out_dir, n, names, labels):
    """Write nnU-Net v2's dataset.json (indices) + the raw label values."""
    meta = {
        'channel_names': {'0': 'synthetic'},
        'labels': {name: i for i, name in enumerate(names)},
        'numTraining': n,
        'file_ending': '.nii.gz',
        'original_label_values': [int(l) for l in labels],
    }
    with open(os.path.join(out_dir, 'dataset.json'), 'w') as f:
        json.dump(meta, f, indent=2)


def seed_everything(seed):
    """Seed python, numpy and torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_templates(seg_paths, tree, cfg, cache_dir, device='cpu'):
    """
    Super-resolve each label map once and cache it in `cache_dir`.

    Parameters
    ----------
    seg_paths : list of str
    tree : LabelTree, dict or path
    cfg : TreeGeneratorConfig
    cache_dir : str
        Cached HR templates `<name>_hr<voxel>mm.nii.gz` are reused.
    device : str

    Returns
    -------
    list of (lab_hr, affine_hr)
    """
    os.makedirs(cache_dir, exist_ok=True)
    labels = None
    if cfg.unassigned == 'background':
        labels = list(LabelTree.load(tree).label_to_node)
    templates = []
    for path in seg_paths:
        name = os.path.basename(path).split('.')[0]
        hr = os.path.join(cache_dir, f'{name}_hr'
                          f'{cfg.target_voxel / cfg.factor:g}mm.nii.gz')
        if os.path.exists(hr):
            templates.append(load_template(hr))
        else:
            print(f'Super-resolving {path} -> {hr}')
            lab, aff = prepare_template(path, hr, cfg.target_voxel,
                                        cfg.factor,
                                        sigma_mm=cfg.template_sigma_mm,
                                        labels=labels,
                                        background=cfg.background,
                                        device=device)
            templates.append((lab.cpu(), aff))
    return templates


def main():
    """CLI: write an nnU-Net raw dataset (imagesTr, labelsTr, dataset.json)."""
    p = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    p.add_argument('--seg', nargs='+', required=True,
                   help='label map(s) used as templates')
    p.add_argument('--tree', required=True, help='label tree JSON')
    p.add_argument('--config', help='TreeGeneratorConfig JSON (optional)')
    p.add_argument('--out', required=True, help='output directory')
    p.add_argument('-n', type=int, default=10, help='number of samples')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available()
                   else 'cpu')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--prefix', default='synth')
    p.add_argument('--raw-labels', action='store_true',
                   help='save raw label values instead of indices')
    args = p.parse_args()

    seed_everything(args.seed)
    cfg = TreeGeneratorConfig.from_json(args.config) if args.config \
        else TreeGeneratorConfig()
    os.makedirs(os.path.join(args.out, 'imagesTr'), exist_ok=True)
    os.makedirs(os.path.join(args.out, 'labelsTr'), exist_ok=True)

    templates = build_templates(args.seg, args.tree, cfg,
                                os.path.join(args.out, 'templates'),
                                args.device)
    gen = TreeGenerator(templates, args.tree, cfg, device=args.device)
    for i in range(args.n):
        img, seg, aff, info = gen()
        case = f'{args.prefix}_{i:04d}'
        save_sample(img, seg, aff,
                    os.path.join(args.out, 'imagesTr', f'{case}_0000.nii.gz'),
                    os.path.join(args.out, 'labelsTr', f'{case}.nii.gz'),
                    gen.labels if args.raw_labels else None)
        acq = info['acquisition']
        print(f'[{i + 1}/{args.n}] {case} {tuple(seg.shape)} '
              f'{len(info["class_names"])} generation classes, '
              f'acquisition {acq["mode"]} '
              f'{[round(s, 2) for s in acq["spacing_mm"]]} mm')
    write_nnunet_dataset_json(args.out, args.n, gen.label_names(), gen.labels)


if __name__ == '__main__':
    main()
