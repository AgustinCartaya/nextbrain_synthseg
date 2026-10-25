"""
Tree-based synthetic data generator.

Same generative pipeline as the SIAM generator in
`../augmentation/pipeline/data_augmentation_pipeline.py`, but the intensity
model is the label ontology tree of `../simple_generator/tree_sampling.py`
instead of an independent Gaussian per tissue, and nothing is tied to a fixed
set of tissues: any label map works as long as its labels are in the tree.

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

Online, per sample (`TreeGenerator`), at high resolution (0.25 mm):
    3. Random erosion/dilation between pairs of label groups. A group is a
       tree node (all the labels of its subtree) or a list of label values,
       so the rules are written in terms of the tree, not of fixed tissues.
    4. Random affine + elastic deformation.
    5. Partial volume: fraction of each label inside every training voxel
       (0.75 mm), computed slab-by-slab with a histogram, so memory does not
       grow with the number of labels.
   At training resolution (0.75 mm):
    6. Tree intensities: principal colors C ~ U, V ~ U, then
       `TreeSampling.compute_stats` gives one (mu, sigma) per label. Labels
       whose node collapses at the sampled resolution/contrast share one
       Gaussian (one generation class). Image = sum_g pv_g * N(mu_g, sigma_g).
    7. Bias field, motion artefacts, Gaussian noise, min-max to [0, 1].

The output segmentation is in index space (0 = background, i = labels[i]);
`TreeGenerator.labels` maps indices back to raw label values. Every sample
also returns the label groups that share a Gaussian (for merging labels in
the loss at that resolution).
"""

import argparse
import json
import math
import os
import random
import sys
from dataclasses import dataclass, fields

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F

# import cornucopia as cc
# from cornucopia.random import Uniform
# from cornucopia.utils import warps
import sys
import sys
sys.path.append('/home/agustin/phd/local_preprocessing/augmentation')

from local_cornucopia import cornucopia as cc
from local_cornucopia.cornucopia.random import Uniform
from local_cornucopia.cornucopia.utils import warps

from tree_sampling import LabelTree, TreeSampling 


# ---------------------------------------------------------------------------
# 1. Label maps
# ---------------------------------------------------------------------------

def _label_dtype(max_value):
    for dt in (np.uint8, np.int16, np.int32):
        if max_value <= np.iinfo(dt).max:
            return dt
    return np.int64


def restrict_labels(seg, labels, background=0):
    """Send every label not in `labels` to `background` (with a warning)."""
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

def _sinc_upsample_smooth(x, factor, sigma_mm, out_voxel_mm):
    """
    Band-limited (Sinc) upsampling by zero-padding the centred spectrum,
    combined with a Gaussian smoothing applied in the Fourier domain.
    Output voxel `j` sits at input coordinate `j / factor`.
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
    One-hot -> Sinc upsampling -> Gaussian smoothing -> argmax, for any set
    of label values.

    Every foreground label is upsampled inside its bounding box (+ `margin`
    input voxels; outside it the smoothed indicator is ~0). The background
    channel is never computed: upsampling and smoothing are linear and the
    one-hot channels sum to 1, so background = 1 - sum(foreground).

    Returns the HR label map with the raw label values.
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
    training grid. If `labels` is given (e.g. the labels of the tree), any
    other label goes to background.

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
        arr = lab_hr.cpu().numpy()
        arr = arr.astype(_label_dtype(int(arr.max())))
        nib.save(nib.Nifti1Image(arr, affine_hr), out_path)
    return lab_hr, affine_hr


def load_template(path, device='cpu'):
    nii = nib.load(path)
    lab = torch.as_tensor(np.asarray(nii.dataobj).astype(np.int32),
                          device=device)
    return lab, nii.affine


# ---------------------------------------------------------------------------
# 3. Online generator
# ---------------------------------------------------------------------------

def _bbox(mask, margin=0):
    """Slicer of the bounding box of a 3D mask (None if empty)."""
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


def _uniform(r):
    """A number, or a (lo, hi) range sampled uniformly."""
    if isinstance(r, (int, float)):
        return float(r)
    return random.uniform(*r)


@dataclass
class TreeGeneratorConfig:
    target_voxel: float = 0.75   # training resolution (mm)
    factor: int = 3              # HR = target_voxel / factor = 0.25 mm
    background: int = 0          # raw label value of the background

    # -- label space ---------------------------------------------------------
    # 'templates': labels present in the templates; 'tree': every tree label
    output_labels: str = 'templates'
    # template labels missing from the tree: 'error' or 'background'
    unassigned: str = 'error'

    # -- shape augmentation (paper: 1-4 iterations of 0.25 mm) ------------
    # Pairs (a, b) of label groups. A group is a tree node name (all labels
    # of its subtree), a label value, or a list of those. Each pair is
    # randomly either a-grows-into-b or b-grows-into-a. Empty = no-op.
    morpho_pairs: tuple = ()
    morpho_iter: tuple = (1, 4)
    morpho_prob: float = 0.5     # ASSUMPTION: per-pair probability

    # -- spatial deformation (ASSUMPTION: TorchIO defaults) ----------------
    rotations: float = 10.       # deg, U(-10, 10)
    zooms: float = 0.1           # U(-0.1, 0.1) about 1
    shears: float = 0.
    translations_mm: float = 0.
    elastic_dmax_mm: float = 7.5
    elastic_ctrl: int = 7

    # -- tree intensities ----------------------------------------------------
    # Principal colors, in the units of the tree's wr_mu / wr_sigma (0-255).
    # Defaults are the paper's mu ~ U(0, 1), sigma ~ U(0.001, 0.01) x 255.
    colors_mu: tuple = (0., 255.)
    colors_sigma: tuple = (0.255, 2.55)
    min_sigma: float = 0.
    intensity_scale: float = 255.  # image / scale before the artefacts
    # Resolution (mm) used for node visibility: None = target_voxel, a
    # number, or a (lo, hi) range sampled per image.
    resolution: object = None
    # Contrast used for r_max lookup: None (r_max 'default') or a list,
    # one is picked at random per image.
    contrasts: object = None

    # -- artefacts ----------------------------------------------------------
    bias_prob: float = 1.0       # ASSUMPTION
    bias_vmax: float = 0.5       # ASSUMPTION: field in (1-v, 1+v)
    bias_ctrl: int = 5           # ASSUMPTION
    motion_prob: float = 0.3     # ASSUMPTION
    motion_shots: int = 3        # ASSUMPTION (TorchIO: num_transforms=2)
    motion_rotations: float = 10.   # deg (TorchIO default)
    motion_translations_mm: float = 10.  # (TorchIO default)
    noise_sigma: tuple = (0.01, 0.1)  # paper

    @classmethod
    def from_json(cls, path, **overrides):
        with open(path) as f:
            d = json.load(f)
        d.update(overrides)
        known = {f.name for f in fields(cls)}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f'Unknown config fields: {sorted(unknown)}')
        for key, value in d.items():   # JSON lists -> tuples
            if isinstance(value, list) and key != 'contrasts':
                d[key] = tuple(value)
        return cls(**d)

    # class to convert to dict
    def to_dict(self):
        return {f.name: getattr(self, f.name) for f in fields(self)}

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
    device : torch.device
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
        self.index = {int(l): i for i, l in enumerate(self.labels)}

        lut = torch.zeros(int(max(max(found), self.labels.max())) + 1,
                          dtype=torch.long)
        for lab, i in self.index.items():
            lut[lab] = i
        idx_dtype = torch.uint8 if self.nb_labels <= 256 else torch.int16
        self.templates = []
        for lab, aff in templates:
            lab = lut[lab.long().cpu()].to(idx_dtype)
            self.templates.append((lab.to(self.device), aff))

        # -- intensity model ----------------------------------------------
        # std_noise=0: the acquisition noise is added later (step 7)
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
        self.normalize = cc.MinMaxTransform()

    # -- label helpers ---------------------------------------------------------

    def _resolve(self, spec):
        """Tree node / label value / list of those -> sorted label indices."""
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

    # -- steps ---------------------------------------------------------------

    def _isin(self, lab, idx):
        return torch.isin(lab, torch.as_tensor(idx, device=lab.device,
                                               dtype=lab.dtype))

    def _morphology(self, lab):
        """
        Constrained random dilation/erosion between pairs of label groups.
        The growing group takes over voxels of the shrinking group one
        voxel layer (26-neighbourhood) per iteration; with several labels in
        the growing group each new voxel copies a neighbouring label.
        """
        cfg = self.cfg
        lo, hi = cfg.morpho_iter
        for _, _, a, b in self.morpho_pairs:
            if random.random() > cfg.morpho_prob:
                continue
            grow, shrink = (a, b) if random.random() < 0.5 else (b, a)
            n_iter = random.randint(lo, hi)
            # work in the bounding box of the pair (+ margin)
            box = _bbox(self._isin(lab, grow) | self._isin(lab, shrink),
                        margin=n_iter + 1)
            if box is None:
                continue
            sub = lab[box]
            for _ in range(n_iter):
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
        """Random affine+elastic deformation of the HR labels (nearest)."""
        f = self.cfg.factor
        lr_shape = [s // f for s in lab.shape]
        # (1, x, y, z, 3) absolute sampling coordinates on the LR grid
        flow = self.deform(lab.new_zeros([1, *lr_shape], dtype=torch.float))
        # LR voxel i is centred on HR voxel f*i + (f-1)/2
        flow = (flow * f + (f - 1) / 2).movedim(-1, 1)   # (1, 3, x, y, z)
        src = lab[None, None].float()
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

    def _sample_contrast(self):
        """Principal colors, resolution and contrast of one image."""
        cfg = self.cfg
        n = self.tree.n_colors
        C = np.array([random.uniform(*cfg.colors_mu) for _ in range(n)])
        V = np.array([random.uniform(*cfg.colors_sigma) for _ in range(n)])
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

    def _motion(self, img):
        cfg = self.cfg
        if random.random() > cfg.motion_prob:
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

    # -- main ----------------------------------------------------------------

    @torch.no_grad()
    def __call__(self, template_index=None, return_pv=False):
        """
        Returns
        -------
        img : (1, X, Y, Z) float tensor in [0, 1]
        seg : (X, Y, Z) int tensor, indices into `self.labels`
        affine : (4, 4) ndarray, voxel-to-world of the training grid
        info : dict
            colors_mu, colors_sigma, resolution, contrast : what was drawn
            mus, sigmas : per-label Gaussian (index order, tree units)
            group_of : generation class of each label index; labels with
                the same class are indistinguishable in this image
            class_names : tree node of each generation class
            pv : (G, X, Y, Z) partial volume per class (if return_pv)
        """
        if template_index is None:
            template_index = random.randrange(len(self.templates))
        lab, affine_hr = self.templates[template_index]

        C, V, res, contrast = self._sample_contrast()
        mus, sigmas = self.scheme.compute_stats(C, V, res, contrast)
        group_of, class_names = self.scheme.label_set_lut(res, contrast)

        lab = self._morphology(lab.clone())
        lab = self._deform(lab)
        img, seg, pv = self._synthesize(lab, mus, sigmas, group_of,
                                        return_pv)
        del lab

        img = img / self.cfg.intensity_scale
        img = self.bias(img)
        img = self._motion(img)
        img = self.noise(img)
        img = self.normalize(img)

        # pooled voxel i is centred on HR voxel f*i + (f-1)/2
        f = self.cfg.factor
        pool = np.diag([f, f, f, 1.]).astype(float)
        pool[:3, 3] = (f - 1) / 2
        affine = affine_hr @ pool
        info = dict(colors_mu=C, colors_sigma=V, resolution=res,
                    contrast=contrast, mus=mus, sigmas=sigmas,
                    group_of=group_of, class_names=class_names, pv=pv)
        return img, seg, affine, info


# ---------------------------------------------------------------------------
# I/O helpers and CLI (nnU-Net raw dataset layout)
# ---------------------------------------------------------------------------

def save_sample(img, seg, affine, img_path, seg_path, labels=None):
    """Save image and labels. With `labels`, indices -> raw label values."""
    seg = seg.cpu().numpy().astype(np.int64)
    if labels is not None:
        seg = np.asarray(labels)[seg]
    nib.save(nib.Nifti1Image(img[0].cpu().numpy().astype(np.float32),
                             affine), img_path)
    nib.save(nib.Nifti1Image(seg.astype(_label_dtype(int(seg.max()))),
                             affine), seg_path)


def write_nnunet_dataset_json(out_dir, n, names, labels):
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
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_templates(seg_paths, tree, cfg, cache_dir, device='cpu'):
    """Super-resolve each label map once and cache it in `cache_dir`."""
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
                                        cfg.factor, labels=labels,
                                        background=cfg.background,
                                        device=device)
            templates.append((lab.cpu(), aff))
    return templates


def main():
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
        print(f'[{i + 1}/{args.n}] {case} {tuple(seg.shape)} '
              f'{len(info["class_names"])} generation classes '
              f'@ {info["resolution"]:g} mm')
    write_nnunet_dataset_json(args.out, args.n, gen.label_names(), gen.labels)


if __name__ == '__main__':
    main()
