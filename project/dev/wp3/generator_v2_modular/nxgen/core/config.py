"""
Configuration: defaults, loading, validation.

One JSON file holds every setting, grouped in sections (one per module).
The offline launcher (template building) and the online generator read the
same file; each one only uses its own sections.

Rules
-----
* A file only needs the values that differ from `DEFAULTS`. Missing
  sections or keys take the default.
* Unknown sections or keys raise an error (typos never pass silently).
* Values with a fixed set of options (e.g. `noise.space`) are checked.
* Random parameters are written as *distribution specs* (see
  `nxgen.core.rng.draw`):

      5.0                      fixed value
      [lo, hi]                 uniform in [lo, hi]
      {"uniform": [lo, hi]}    same, explicit
      {"randint": [lo, hi]}    integer in [lo, hi], both included
      {"choice": [a, b, ...]}  one of the values, equally likely

  A fixed value pins the parameter, which is how a config gives
  predictable results.
* Relative paths (`labels.tree`) are relative to the config file.

Difference with v2: ranges are explicit. In v2, `"rotations": 10` meant
U(-10, 10); here `"rotation_deg": 10` means exactly 10 degrees and the v2
behaviour is `"rotation_deg": [-10, 10]`.
"""

import copy
import hashlib
import json
import os

from .rng import check_spec

# Every setting, with its default. The comment above each key is its
# documentation (also in docs/config.md).
DEFAULTS = {
    # -- grids ---------------------------------------------------------------
    'grid': {
        # Voxel size (mm) of the output images and segmentations. The HR
        # voxel is read from the templates; target_voxel_mm / hr_voxel must
        # be an integer (the partial-volume factor).
        'target_voxel_mm': 0.6,
    },

    # -- label space ----------------------------------------------------------
    'labels': {
        # Label tree JSON (path relative to the config file).
        'tree': None,
        # Raw label value of the background.
        'background': 0,
        # Template labels that are not in the tree: 'error' (raise) or
        # 'background' (mapped to the background).
        'unassigned': 'error',
        # Output labels: 'templates' (labels present in the templates) or
        # 'tree' (every label of the tree, some may never appear).
        'output_labels': 'templates',
    },

    # -- offline: HR template building (nxgen.offline.template) -------------
    'template': {
        # Voxel size (mm) of the HR templates. The input label maps must
        # have a voxel size that is an integer multiple of it.
        'hr_voxel_mm': 0.2,
        # Std (mm) of the Gaussian smoothing of the one-hot labels before
        # the argmax (0 = none).
        'sigma_mm': 0.0,
        # Background kept around the foreground when cropping (mm): one
        # number for all axes or [x, y, z].
        'margin_mm': 16.0,
    },

    # -- online: training patch -----------------------------------------------
    'patch': {
        # Size [x, y, z] of the generated patch, in output voxels.
        # null = full volume. Only the part of the template the patch needs
        # is processed, so patches are much faster than full volumes.
        'size': None,
    },

    # -- step 0: acquisition (nxgen.online.acquisition) ---------------------
    'acquisition': {
        # Probability of simulating a lower resolution than target_voxel_mm.
        'lowres_prob': 0.95,
        # Probability that a degraded acquisition is isotropic (else one
        # random axis gets thick slices).
        'iso_prob': 0.1,
        # Largest isotropic voxel (mm).
        'max_iso_mm': 4.0,
        # Largest slice spacing (mm) of an anisotropic acquisition.
        'max_aniso_mm': 8.0,
        # Deviation of the slice profile (SynthSeg eq. 22).
        'alpha': [0.970873786407767, 1.03],
        # true: slice thickness ~ U(target_voxel, spacing) (slice gap);
        # false: thickness = spacing.
        'random_thickness': True,
        # Contrasts for the per-contrast r_max of the tree; one is chosen
        # per image. null = r_max "default".
        'contrasts': None,
    },

    # -- step 1: morphology (nxgen.online.morphology) ------------------------
    'morphology': {
        # Pairs [a, b] of label groups. A group is a tree node name (all
        # labels of its subtree), a raw label value, or a list of those.
        # For each pair, one group (chosen at random) grows into the other.
        'pairs': [],
        # Number of HR voxel layers grown.
        'n_iter': {'randint': [1, 4]},
        # Probability of applying each pair.
        'prob': 0.5,
    },

    # -- step 2: spatial deformation (nxgen.online.spatial) -----------------
    'spatial': {
        # Rotation (degrees) about each axis, drawn per axis.
        'rotation_deg': [-10.0, 10.0],
        # Zoom factor per axis (1 = none), drawn per axis.
        'zoom': [0.9, 1.1],
        # Shear per pair of axes (xy, xz, yz), drawn per pair.
        'shear': 0.0,
        # Translation (mm) per axis, drawn per axis.
        'translation_mm': 0.0,
        # Largest elastic displacement (mm); control points are drawn in
        # U(-dmax, dmax) per component.
        'elastic_dmax_mm': 7.5,
        # Number of elastic control points per axis (>= 2; fewer =
        # smoother).
        'elastic_ctrl': {'randint': [2, 7]},
    },

    # -- step 3: partial volume (nxgen.online.partial_volume) ---------------
    'partial_volume': {
        # 'downsample': fraction of each label among the factor^3 HR voxels
        # of an output voxel (the HR grid must be finer than the output).
        # 'smooth': Gaussian-weighted fraction in a small window around
        # each voxel (for factor = 1, same grid).
        'method': 'downsample',
        # Labels kept per voxel (the largest fractions). The others are
        # dropped and the kept fractions renormalised.
        'top_k': 8,
        # 'smooth' only: std of the window, in voxels. The window is
        # truncated at 2 std.
        'smooth_sigma_vox': 0.5,
    },

    # -- step 4: tree intensities (nxgen.online.tree_intensities) -----------
    'tree': {
        # Range of the principal color means C (one per tree color).
        'colors_mu': [0.0, 255.0],
        # Range of the principal color stds V.
        'colors_sigma': [0.255, 2.55],
        # Lower bound of every label std.
        'min_sigma': 0.0,
        # The image is divided by this after the synthesis (so bias and
        # noise act on values of order 1).
        'intensity_scale': 255.0,
        # Resolution (mm) deciding node visibility: 'acquisition' (coarsest
        # spacing of step 0) or a distribution spec (e.g. [0.75, 3]).
        'resolution': 'acquisition',
    },

    # -- step 5a: bias field (nxgen.online.bias) -----------------------------
    'bias': {
        # Probability of a bias field.
        'prob': 1.0,
        # Field values in (1 - vmax, 1 + vmax).
        'vmax': 0.5,
        # Control points per axis (smoothness).
        'ctrl': 5,
    },

    # -- step 5b: motion (nxgen.online.motion) ------------------------------
    'motion': {
        # Probability of motion artefacts.
        'prob': 0.3,
        # Number of k-space shots (the head moves between shots).
        'shots': 3,
        # Rotation (degrees) of each shot, per axis.
        'rotation_deg': [-10.0, 10.0],
        # Translation (mm) of each shot, per axis.
        'translation_mm': [-10.0, 10.0],
    },

    # -- step 5d: noise (nxgen.online.noise) ---------------------------------
    'noise': {
        # Std of the Gaussian noise (image divided by intensity_scale).
        'sigma': [0.01, 0.1],
        # 'lowres': added on the low-resolution grid of step 5c (as in a
        # real scan); 'highres': added on the output grid before step 5c
        # (SynthSeg).
        'space': 'lowres',
    },

    # -- step 6: normalisation (nxgen.online.normalization) ------------------
    'normalization': {
        # 'minmax' (to [0, 1]), 'percentile' (clip to the percentiles, then
        # [0, 1]) or 'none'.
        'method': 'minmax',
        # 'percentile' only: [low, high] percentiles.
        'percentiles': [0.5, 99.5],
    },
}

# Allowed values of the keys that have a fixed set of options.
CHOICES = {
    ('labels', 'unassigned'): ('error', 'background'),
    ('labels', 'output_labels'): ('templates', 'tree'),
    ('partial_volume', 'method'): ('downsample', 'smooth'),
    ('noise', 'space'): ('lowres', 'highres'),
    ('normalization', 'method'): ('minmax', 'percentile', 'none'),
}

# Keys that hold distribution specs (checked with `check_spec`).
SPECS = {
    'morphology': ('n_iter',),
    'spatial': ('rotation_deg', 'zoom', 'shear', 'translation_mm',
                'elastic_dmax_mm', 'elastic_ctrl'),
    'tree': ('colors_mu', 'colors_sigma'),
    'motion': ('rotation_deg', 'translation_mm'),
    'noise': ('sigma',),
}


def load_config(path=None, **overrides):
    """
    Load a config: `DEFAULTS` < JSON file < `overrides`.

    Parameters
    ----------
    path : str, optional
        JSON file. Without it, only the defaults (and overrides) are used.
    **overrides : dict per section
        E.g. `load_config('c.json', noise={'sigma': 0.0})`. Only the given
        keys of a section are replaced.

    Returns
    -------
    cfg : dict
        Complete config (every section, every key), validated. Relative
        paths are made absolute.
    """
    cfg = copy.deepcopy(DEFAULTS)
    base = os.getcwd()
    if path is not None:
        with open(path) as f:
            _merge(cfg, json.load(f))
        base = os.path.dirname(os.path.abspath(path))
    _merge(cfg, overrides)
    tree = cfg['labels']['tree']
    if tree is not None and not os.path.isabs(tree):
        cfg['labels']['tree'] = os.path.normpath(os.path.join(base, tree))
    validate(cfg)
    return cfg


def _merge(cfg, new):
    """Copy the sections/keys of `new` into `cfg` (unknown names raise)."""
    for section, values in new.items():
        if section not in DEFAULTS:
            raise ValueError(f'Unknown config section {section!r}; known: '
                             f'{sorted(DEFAULTS)}')
        if not isinstance(values, dict):
            raise ValueError(f'Config section {section!r} must be a dict')
        for key, value in values.items():
            if key not in DEFAULTS[section]:
                raise ValueError(f'Unknown config key {section}.{key}; '
                                 f'known: {sorted(DEFAULTS[section])}')
            cfg[section][key] = copy.deepcopy(value)


def validate(cfg):
    """Check option values and distribution specs. Raises ValueError."""
    for (section, key), allowed in CHOICES.items():
        if cfg[section][key] not in allowed:
            raise ValueError(f'{section}.{key} must be one of {allowed}, '
                             f'got {cfg[section][key]!r}')
    for section, keys in SPECS.items():
        for key in keys:
            check_spec(cfg[section][key], f'{section}.{key}')
    res = cfg['tree']['resolution']
    if res != 'acquisition':
        check_spec(res, 'tree.resolution')
    size = cfg['patch']['size']
    if size is not None and (len(size) != 3 or min(size) < 1):
        raise ValueError(f'patch.size must be null or [x, y, z], got {size}')
    for key in ('lowres_prob', 'iso_prob'):
        if not 0 <= cfg['acquisition'][key] <= 1:
            raise ValueError(f'acquisition.{key} must be in [0, 1]')
    for section in ('morphology', 'bias', 'motion'):
        if not 0 <= cfg[section]['prob'] <= 1:
            raise ValueError(f'{section}.prob must be in [0, 1]')
    if cfg['motion']['shots'] < 1:
        raise ValueError('motion.shots must be >= 1')
    if cfg['bias']['ctrl'] < 2:
        raise ValueError('bias.ctrl must be >= 2')
    for pair in cfg['morphology']['pairs']:
        if len(pair) != 2:
            raise ValueError(f'morphology.pairs: each pair must be [a, b], '
                             f'got {pair!r}')


def config_hash(cfg):
    """Short hash (10 hex digits) of a config, to name saved copies."""
    text = json.dumps(cfg, sort_keys=True)
    return hashlib.sha1(text.encode()).hexdigest()[:10]
