"""
Step 0: acquisition (scan resolution and contrast).

Drawn first because two later steps use it:
* `tree_intensities` (step 4): the coarsest spacing decides which tree
  nodes are visible (labels the scan cannot resolve share one Gaussian);
* `resolution` (step 5c): the image is degraded to this acquisition.

Priors (SynthSeg, Billot et al. 2023, Sec. 3.1.4, eqs. 20-23), with r the
output voxel size (`grid.target_voxel_mm`):

* with probability 1 - lowres_prob: no degradation (mode 'none');
* else, with probability iso_prob: isotropic, spacing ~ U(r, max_iso_mm)
  on the three axes (mode 'iso');
* else: one random axis, spacing ~ U(r, max_aniso_mm) (mode 'aniso');
* per degraded axis: thickness ~ U(r, spacing) (or = spacing if
  random_thickness is false), alpha ~ U(alpha); the std of the Gaussian
  slice profile, in output voxels, is computed from them at render time
      sigma = 2 alpha log(10) / (2 pi) * thickness / r     (eq. 23).

This is v2's `SynthSegLowResTransform.sample`, drawing from an explicit
generator.

The plan stores only physical values (spacing, thickness, alpha) and the
contrast. They are the single source of truth: the mode (`describe`) and the
slice-profile std (`sigma_vox`) are derived from them when needed, so editing
a plan by hand cannot leave two descriptions that disagree. An axis is
degraded when its spacing is larger than the output voxel; on the other axes
thickness and alpha are ignored. `plan['acquisition'] = None` removes the
whole step (same as all spacings = output voxel).
"""

import math


def slice_sigma_vox(thickness_mm, alpha, voxel_mm):
    """
    Std (in voxels of size `voxel_mm`) of the Gaussian slice profile.

    sigma = 2 alpha log(10) / (2 pi) * thickness / voxel: the power of the
    signal is divided by 10 at the cut-off frequency of the slice.
    """
    return 2 * alpha * math.log(10) / (2 * math.pi) * thickness_mm / voxel_mm


def none_params(voxel_mm, contrast=None):
    """Parameters of an acquisition at the output resolution (no change)."""
    r = float(voxel_mm)
    return dict(spacing_mm=[r] * 3, thickness_mm=[r] * 3, alpha=[1.0] * 3,
                contrast=contrast)


def degraded_axes(acq, voxel_mm):
    """Axes whose spacing is larger than the output voxel (None -> [])."""
    if acq is None:
        return []
    r = float(voxel_mm)
    return [d for d in range(3) if acq['spacing_mm'][d] > r * (1 + 1e-6)]


def sigma_vox(acq, voxel_mm):
    """
    Std of the Gaussian slice profile of each axis, in output voxels
    (`slice_sigma_vox`, eq. 23; 0 on the axes that are not degraded).
    """
    axes = degraded_axes(acq, voxel_mm)
    return [slice_sigma_vox(acq['thickness_mm'][d], acq['alpha'][d], voxel_mm)
            if d in axes else 0.0 for d in range(3)]


def describe(acq, voxel_mm):
    """
    Mode of an acquisition, derived from its spacings.

    Returns
    -------
    mode : 'none' (None or no degraded axis), 'iso' (3 equal degraded
        spacings), 'aniso' (one degraded axis) or 'custom' (anything else,
        e.g. a hand-edited plan with two degraded axes)
    axes : list of the degraded axes
    """
    axes = degraded_axes(acq, voxel_mm)
    if not axes:
        return 'none', axes
    if len(axes) == 1:
        return 'aniso', axes
    sp = acq['spacing_mm']
    if len(axes) == 3 and max(sp) - min(sp) <= 1e-6 * max(sp):
        return 'iso', axes
    return 'custom', axes


def sample(cfg, rng, voxel_mm):
    """
    Draw one acquisition.

    Parameters
    ----------
    cfg : dict, config section 'acquisition'
    rng : numpy Generator
    voxel_mm : float, output voxel size

    Returns
    -------
    dict with
        spacing_mm, thickness_mm, alpha : 3 values each (an axis is degraded
            when its spacing > voxel_mm; see `describe`, `sigma_vox`)
        contrast : str or None
    """
    r = float(voxel_mm)
    # the draws below always happen in the same order, so the contrast does
    # not depend on whether the image was degraded
    contrasts = cfg['contrasts']
    contrast = None
    if contrasts:
        contrast = contrasts[int(rng.integers(len(contrasts)))]
    prm = none_params(r, contrast)
    if rng.random() >= cfg['lowres_prob']:
        return prm
    if rng.random() < cfg['iso_prob']:
        if cfg['max_iso_mm'] <= r:
            return prm
        axes = [0, 1, 2]
        prm['spacing_mm'] = [float(rng.uniform(r, cfg['max_iso_mm']))] * 3
    else:
        if cfg['max_aniso_mm'] <= r:
            return prm
        axis = int(rng.integers(3))
        axes = [axis]
        prm['spacing_mm'][axis] = float(rng.uniform(r, cfg['max_aniso_mm']))
    for d in axes:
        spacing = prm['spacing_mm'][d]                               # eq. 20
        thick = float(rng.uniform(r, spacing)) \
            if cfg['random_thickness'] else spacing                  # eq. 21
        prm['thickness_mm'][d] = thick
        prm['alpha'][d] = float(rng.uniform(*cfg['alpha']))          # eq. 22
    return prm
