"""
The online generator: sample a plan, then render it.

    gen = Generator(['templates/case-1_hr0.2mm.nii.gz'], 'config.json',
                    device='cuda')
    sample = gen(seed=42)                 # = gen.render(gen.sample_plan(42))
    sample.image, sample.seg, sample.affine, sample.plan

Two phases
----------
1. `sample_plan(seed)` draws every random parameter of one image (template,
   patch position, acquisition, morphology, deformation, tree Gaussians,
   bias, motion, noise) into a JSON-serialisable dict, the *plan*. No voxel
   is touched; it takes milliseconds.
2. `render(plan)` produces the image. It is deterministic: the same plan
   gives the same segmentation, and the same image up to GPU floating-point
   non-determinism.

A plan can be saved (`nxgen.core.io.save_json`), edited and rendered again:

    plan = gen.sample_plan(42)
    plan['motion'] = None                 # same image, without motion
    plan['noise'] = None                  # ... without noise
    plan['acquisition'] = None            # ... and without low resolution
    sample = gen.render(plan, Recorder(steps='all', out_dir='debug/'))

Every augmentation can be removed this way: `morphology`, `spatial`,
`acquisition`, `bias`, `motion` and `noise` accept None (= step skipped).
`tree` (the intensities) is the image itself and cannot be None.

Each module draws from its own random stream (`nxgen.core.rng.module_rng`),
so changing or disabling one module does not change the others.

Steps of `render`
-----------------
    crop          part of the HR template the patch needs (+ morphology halo)
    morphology    step 1, on the crop
    spatial       step 2, deformed HR labels of the patch
    partial_volume step 3, -> segmentation + sparse partial volumes
    synthesis     step 4, image from the tree Gaussians
    bias          step 5a
    motion        step 5b
    resolution    step 5c (+ noise on the low-res grid, step 5d)
    normalization step 6

Grids: the template is on the HR grid (voxel h). The output grid is the HR
grid pooled by `factor` = target_voxel_mm / h. A patch is a box of the
output grid; `patch.size: null` = the whole output grid.
"""

from dataclasses import dataclass, field

import numpy as np
import torch

from .core.config import config_hash, load_config
from .core.geometry import pooled_affine, shifted_affine
from .core.io import load_labels
from .core.label_space import LabelSpace
from .core.recorder import Recorder
from .core.rng import module_rng
from .core.tree import LabelTree
from .online import (acquisition, bias, morphology, motion, noise,
                     normalization, partial_volume, resolution, spatial,
                     synthesis, tree_intensities)

# 2: acquisition stores only spacing_mm, thickness_mm, alpha, contrast (mode,
#    axis and sigma_vox are derived, see `acquisition.describe/sigma_vox`).
#    Version 1 plans still render the same (the extra keys are ignored).
PLAN_VERSION = 2

# plan sections that may be None (step not applied)
OPTIONAL = ('morphology', 'spatial', 'acquisition', 'bias', 'motion', 'noise')


@dataclass
class Sample:
    """
    One generated sample.

    image : (X, Y, Z) float32 tensor (normalised)
    seg : (X, Y, Z) int16 tensor of label indices (see `Generator.labels`)
    affine : (4, 4) ndarray, voxel -> world of the output (patch) grid
    plan : dict, every parameter used (see `Generator.sample_plan`)
    normalization : dict, values used by the normalisation
    recorder : Recorder (intermediate steps, timings)
    """
    image: torch.Tensor
    seg: torch.Tensor
    affine: np.ndarray
    plan: dict
    normalization: dict
    recorder: Recorder = field(repr=False, default=None)

    @property
    def class_of(self):
        """Generation class of each label index (labels sharing a Gaussian)."""
        return self.plan['tree']['class_of']


class Generator:
    """
    Online generator.

    Parameters
    ----------
    templates : list of str
        HR templates (NIfTI label maps with raw values, all with the same
        isotropic voxel size), e.g. from `nxgen.offline.template`.
    config : str or dict
        Config file, or a dict from `nxgen.core.config.load_config`.
    label_space : LabelSpace or str, optional
        Fixed label space (or its JSON). Default: built from the tree and
        the templates (`labels.output_labels`, `labels.unassigned`).
    device : str

    Attributes
    ----------
    labels : LabelSpace (index <-> raw value and name)
    factor : int, HR voxels per output voxel along each axis
    hr_voxel_mm, voxel_mm : HR and output voxel sizes
    templates : list of dicts (path, labels (int16 indices on `device`),
        affine, hr_shape, out_shape, fg_lo, fg_hi (foreground box on the
        output grid))
    """

    def __init__(self, templates, config, label_space=None, device='cpu'):
        self.cfg = config if isinstance(config, dict) else load_config(config)
        self.config_hash = config_hash(self.cfg)
        self.device = torch.device(device)
        if self.cfg['labels']['tree'] is None:
            raise ValueError('labels.tree is not set in the config')
        self.tree = LabelTree.from_json(self.cfg['labels']['tree'])
        bg = self.cfg['labels']['background']

        raw = []
        for path in templates:
            seg, affine, voxel = load_labels(path)
            if np.ptp(voxel) > 1e-4:
                raise ValueError(f'{path}: voxels must be isotropic, got '
                                 f'{voxel}')
            raw.append((path, seg, affine, float(voxel[0])))
        hr = {round(v, 6) for _, _, _, v in raw}
        if len(hr) != 1:
            raise ValueError(f'templates have different voxel sizes: {hr}')
        # rounded: NIfTI stores voxel sizes as float32 (0.2 -> 0.20000000298)
        self.hr_voxel_mm = round(raw[0][3], 6)
        self.voxel_mm = float(self.cfg['grid']['target_voxel_mm'])
        f = self.voxel_mm / self.hr_voxel_mm
        if abs(f - round(f)) > 1e-3 or round(f) < 1:
            raise ValueError(f'grid.target_voxel_mm {self.voxel_mm} is not an '
                             f'integer multiple of the HR voxel '
                             f'{self.hr_voxel_mm}')
        self.factor = int(round(f))
        if self.cfg['partial_volume']['method'] == 'smooth' and \
                self.factor != 1:
            raise ValueError("partial_volume.method 'smooth' needs "
                             'target_voxel_mm == HR voxel')

        values = set()
        for _, seg, _, _ in raw:
            values |= set(np.unique(seg).tolist())
        if label_space is None:
            label_space = LabelSpace.build(
                self.tree, values, bg, self.cfg['labels']['output_labels'],
                self.cfg['labels']['unassigned'])
        elif isinstance(label_space, str):
            label_space = LabelSpace.load(label_space)
        unknown = sorted(values - set(label_space.values))
        if unknown and self.cfg['labels']['unassigned'] == 'error':
            raise ValueError(f'Template labels {unknown} are not in the label '
                             f"space (labels.unassigned = 'error')")
        self.labels = label_space

        self.templates = [self._load(path, seg, affine)
                          for path, seg, affine, _ in raw]

    def _load(self, path, seg, affine):
        """Raw values -> int16 indices on the device, padded to factor."""
        f = self.factor
        lut = self.labels.lut(int(seg.max()))
        idx = lut[torch.as_tensor(seg)]
        shape = [-(-s // f) * f for s in idx.shape]     # multiple of factor
        lab = torch.zeros(shape, dtype=torch.int16)
        lab[:idx.shape[0], :idx.shape[1], :idx.shape[2]] = idx
        lab = lab.to(self.device)
        fg = [torch.nonzero(lab.ne(0).any(dim=[a for a in range(3) if a != d]))
              .flatten() for d in range(3)]
        return dict(path=path, labels=lab, affine=np.asarray(affine, float),
                    hr_shape=shape, out_shape=[s // f for s in shape],
                    fg_lo=[int(x[0]) // f for x in fg],
                    fg_hi=[int(x[-1]) // f + 1 for x in fg])

    # -- phase 1 ---------------------------------------------------------------

    def sample_plan(self, seed, template=None, start=None):
        """
        Draw every random parameter of one image.

        Parameters
        ----------
        seed : int, image seed
        template : int, optional, template index (random by default)
        start : 3 ints, optional, patch position on the output grid
            (random by default: patch centre uniform in the foreground box)

        Returns
        -------
        plan : dict with
            version, seed, config_hash : identification
            template, template_path : template used
            roi : dict(start, size), patch on the output grid
            acquisition, morphology, spatial, tree, bias, motion, noise :
                parameters of each module (see its `sample`). None = module
                not applied; every section but `tree` can also be set to
                None by hand to skip that step in `render`
        """
        cfg = self.cfg

        def rng(name):
            return module_rng(seed, name)

        if template is None:
            template = int(rng('template').integers(len(self.templates)))
        t = self.templates[template]
        shape = t['out_shape']
        size = cfg['patch']['size'] or shape
        size = [min(int(s), n) for s, n in zip(size, shape)]
        if start is None:
            r = rng('patch')
            centre = [int(r.integers(lo, hi))
                      for lo, hi in zip(t['fg_lo'], t['fg_hi'])]
            start = [int(np.clip(c - s // 2, 0, n - s))
                     for c, s, n in zip(centre, size, shape)]

        acq = acquisition.sample(cfg['acquisition'], rng('acquisition'),
                                 self.voxel_mm)
        return dict(
            version=PLAN_VERSION, seed=int(seed),
            config_hash=self.config_hash,
            template=template, template_path=t['path'],
            roi=dict(start=[int(s) for s in start], size=size),
            acquisition=acq,
            morphology=morphology.sample(cfg['morphology'],
                                         rng('morphology'), self.labels,
                                         self.tree),
            spatial=spatial.sample(cfg['spatial'], rng('spatial')),
            tree=tree_intensities.sample(cfg['tree'], rng('tree'), self.tree,
                                         self.labels, acq),
            bias=bias.sample(cfg['bias'], rng('bias')),
            motion=motion.sample(cfg['motion'], rng('motion'), size),
            noise=noise.sample(cfg['noise'], rng('noise')),
        )

    # -- phase 2 ---------------------------------------------------------------

    @torch.no_grad()
    def render(self, plan, recorder=None):
        """
        Produce the image of a plan.

        Parameters
        ----------
        plan : dict from `sample_plan` (possibly edited; the sections in
            `OPTIONAL` may be None or missing, = step skipped)
        recorder : Recorder, optional (intermediate steps, timings)

        Returns
        -------
        Sample
        """
        if plan.get('tree') is None:
            raise ValueError("plan['tree'] (the label Gaussians) is required; "
                             'it cannot be None')
        plan = dict(plan)
        for key in OPTIONAL:
            plan.setdefault(key, None)
        rec = recorder or Recorder()
        dev = self.device
        cfg = self.cfg
        f, h = self.factor, self.hr_voxel_mm
        t = self.templates[plan['template']]
        start, size = plan['roi']['start'], plan['roi']['size']
        start_hr = [s * f for s in start]
        size_hr = [s * f for s in size]
        aff_out = shifted_affine(pooled_affine(t['affine'], f), start)
        aff_hr = shifted_affine(t['affine'], start_hr)
        sp = plan['spatial'] or spatial.identity_params()

        with rec.step('crop', dev):
            lo, hi = spatial.source_box(
                sp, start_hr, size_hr, t['hr_shape'], h,
                halo=morphology.halo(plan['morphology']))
            crop = t['labels'][lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
        rec.record('template_crop', crop, shifted_affine(t['affine'], lo))

        with rec.step('morphology', dev):
            crop = morphology.apply(crop, plan['morphology'], len(self.labels))
        rec.record('morphology', crop, shifted_affine(t['affine'], lo))

        with rec.step('spatial', dev):
            lab = spatial.apply(crop, lo, sp, start_hr, size_hr,
                                t['hr_shape'], h)
        del crop
        rec.record('spatial', lab, aff_hr)

        pvc = cfg['partial_volume']
        with rec.step('partial_volume', dev):
            pv = partial_volume.apply(lab, pvc['method'], f, pvc['top_k'],
                                      pvc['smooth_sigma_vox'])
        del lab
        seg = pv['seg']
        rec.record('pv_seg', seg, aff_out)
        rec.record('pv_max', pv['weight'][0], aff_out)
        rec.record('pv_count', (pv['weight'] > 0).sum(0), aff_out)

        with rec.step('synthesis', dev):
            img = synthesis.apply(pv, plan['tree'])
        del pv
        rec.record('synthesis', img, aff_out)

        with rec.step('bias', dev):
            img = bias.apply(img, plan['bias'], start, t['out_shape'])
        rec.record('bias', img, aff_out)

        with rec.step('motion', dev):
            img = motion.apply(img, plan['motion'], self.voxel_mm)
        rec.record('motion', img, aff_out)

        acq, nz = plan['acquisition'], plan['noise']
        with rec.step('resolution', dev):
            if nz is not None and nz['space'] == 'highres':
                img = noise.apply(img, nz)
                rec.record('noise', img, aff_out)
                img, low = resolution.apply(img, acq, self.voxel_mm)
            else:
                img, low = resolution.apply(
                    img, acq, self.voxel_mm,
                    noise_fn=lambda x: noise.apply(x, nz))
                if low is None:          # not degraded: noise on this grid
                    img = noise.apply(img, nz)
        if low is not None:
            scale = [(n - 1) / max(m - 1, 1)
                     for n, m in zip(img.shape, low.shape)]
            rec.record('lowres', low, aff_out @ np.diag(scale + [1.0]))
        rec.record('resolution', img, aff_out)

        with rec.step('normalization', dev):
            img, norm = normalization.apply(img, cfg['normalization'])
        rec.record('image', img, aff_out)

        return Sample(image=img, seg=seg, affine=aff_out, plan=plan,
                      normalization=norm, recorder=rec)

    def __call__(self, seed, template=None, start=None, recorder=None):
        """`render(sample_plan(seed, template, start), recorder)`."""
        return self.render(self.sample_plan(seed, template, start), recorder)
