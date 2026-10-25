"""
SIAM-style synthetic data generator implemented with cornucopia.

Reproduces the generative model of:
    "SIAM: Head and Brain MRI Segmentation from Few High-Quality Templates via
    Synthetic Training" (arXiv:2605.02737), Sec. 3.2.

Pipeline
--------
Offline, once per template (`prepare_template`):
    1. Regroup labels into tissue classes (Sec. 3.1.1).
    2. Label super-resolution to `target_voxel / factor` (0.25 mm):
       one-hot -> Sinc upsampling per channel -> Gaussian smoothing
       (0.5 mm) per channel -> argmax (Sec. 3.2.1).

Online, per sample (`SIAMGenerator`), at high resolution (0.25 mm):
    3. Random erosion/dilation between neighbouring tissue pairs,
       1-4 iterations of 0.25 mm (Sec. 3.2.2).
    4. Random affine + elastic deformation (Sec. 3.2.2).
    5. Partial-volume maps: average pooling of the one-hot HR labels to the
       training resolution (0.75 mm) (Sec. 3.2.2).
   At training resolution (0.75 mm):
    6. GMM intensities, mu ~ U(0, 1), sigma ~ U(0.001, 0.01), mixed with the
       partial-volume maps as weights (Sec. 3.2.3).
    7. Bias field, motion artefacts, Gaussian noise with sd ~ U(0.01, 0.1),
       min-max normalisation to [0, 1] (Sec. 3.2.3).

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
from cornucopia.random import Uniform
from cornucopia.utils import warps


# ---------------------------------------------------------------------------
# 1. Label grouping
# ---------------------------------------------------------------------------

# The 12 brain tissues of the paper (+ background). `vessel` belongs to the
# paper's extra-cerebral group but is kept because the CSF<->vessel
# morphology rule needs it. Extra-cerebral head tissues (skin, fat, muscle,
# skull, ...) can be appended here when using full-head templates.
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
T = {name: i for i, name in enumerate(TISSUES)}

# FreeSurfer (aseg / aparc+aseg / SynthSeg) -> tissue groups.
# Structures the paper does not list separately are merged into the closest
# group: cerebellar WM, brainstem, ventral DC and corpus callosum -> WM,
# choroid plexus -> ventricles.
FREESURFER_TO_TISSUE = {
    **{lab: T['gm'] for lab in [3, 42]},
    **{lab: T['gm'] for lab in range(1000, 1036)},
    **{lab: T['gm'] for lab in range(2000, 2036)},
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
MORPHO_PAIRS = (
    [('wm', 'gm'), ('wm', 'ventricles')]
    + [('wm', nucleus) for nucleus in DEEP_NUCLEI]
    + [('csf', 'gm'), ('csf', 'vessel')]
)


def group_labels(seg, mapping=FREESURFER_TO_TISSUE):
    """Map a raw label map to tissue-group indices (unmapped -> background)."""
    seg = np.asarray(seg).astype(np.int64)
    lut = np.zeros(seg.max() + 1, dtype=np.uint8)
    for src, dst in mapping.items():
        if src < len(lut):
            lut[src] = dst
    unmapped = sorted(set(np.unique(seg).tolist()) - set(mapping) - {0})
    if unmapped:
        print(f'[group_labels] WARNING: labels {unmapped} unmapped '
              f'-> background')
    return lut[seg]


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


def super_resolve_labels(lab, nb_labels, factor, voxel_mm, sigma_mm=0.5):
    """
    One-hot -> Sinc upsampling -> Gaussian smoothing -> argmax (Sec. 3.2.1).
    Channels are processed one at a time and a running argmax is kept, so
    the full HR one-hot volume is never held in memory.
    """
    out_voxel_mm = voxel_mm / factor
    best = None
    out = None
    for k in range(nb_labels):
        mask = (lab == k)
        if not mask.any():
            continue
        y = _sinc_upsample_smooth(mask.float(), factor, sigma_mm,
                                  out_voxel_mm)
        if best is None:
            best = y
            out = torch.full_like(y, k, dtype=torch.uint8)
        else:
            better = y > best
            out.masked_fill_(better, k)
            torch.maximum(best, y, out=best)
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

    Returns
    -------
    lab_hr : (X, Y, Z) uint8 tensor
    affine_hr : (4, 4) ndarray
    """
    nii = nib.load(seg_path)
    seg = group_labels(np.asarray(nii.dataobj), mapping)
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
    idx = np.argwhere(seg > 0)
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
    seg = np.pad(seg, list(zip(pad_lo, pad_hi)))
    lo = lo + pad_lo
    hi = hi + pad_lo
    seg = seg[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
    crop_shift = np.eye(4)
    crop_shift[:3, 3] = lo - pad_lo
    affine = affine @ crop_shift

    lab = torch.as_tensor(seg, device=device)
    with torch.no_grad():
        lab_hr = super_resolve_labels(lab, len(TISSUES), up, voxel_mm,
                                      sigma_mm)
    # HR voxel j sits at input voxel coordinate j / up
    affine_hr = affine @ np.diag([1 / up, 1 / up, 1 / up, 1])

    if out_path:
        nib.save(nib.Nifti1Image(lab_hr.cpu().numpy(), affine_hr), out_path)
    return lab_hr, affine_hr


def load_template(path, device='cpu'):
    nii = nib.load(path)
    lab = torch.as_tensor(np.asarray(nii.dataobj, dtype=np.uint8),
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


@dataclass
class SIAMConfig:
    target_voxel: float = 0.75   # training resolution (mm)
    factor: int = 3              # HR = target_voxel / factor = 0.25 mm

    # -- shape augmentation (paper: 1-4 iterations of 0.25 mm) ------------
    morpho_iter: tuple = (1, 4)
    morpho_prob: float = 0.5     # ASSUMPTION: per-pair probability
    morpho_pairs: tuple = tuple(MORPHO_PAIRS)

    # -- spatial deformation (ASSUMPTION: TorchIO defaults) ----------------
    rotations: float = 10.       # deg, U(-10, 10)
    zooms: float = 0.1           # U(-0.1, 0.1) about 1
    shears: float = 0.
    translations_mm: float = 0.
    elastic_dmax_mm: float = 7.5
    elastic_ctrl: int = 7

    # -- intensities (paper) -----------------------------------------------
    gmm_mu: tuple = (0., 1.)
    gmm_sigma: tuple = (0.001, 0.01)
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
    """

    def __init__(self, templates, config=None, device='cpu'):
        self.cfg = config or SIAMConfig()
        self.device = torch.device(device)
        self.templates = [(lab.to(self.device), aff)
                          for lab, aff in templates]
        cfg = self.cfg

        # Affine + elastic sampling field. It is drawn by cornucopia on the
        # training (0.75 mm) grid, then linearly upsampled slab-by-slab and
        # used to resample the 0.25 mm labels with nearest neighbour (see
        # `_deform`). The field is smooth (affine + 7^3 control points), so
        # this is equivalent to drawing it at 0.25 mm, at a fraction of the
        # memory (~110 GB -> a few GB for a 0.5 G-voxel template).
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
        self.gmm = cc.RandomGaussianMixtureTransform(
            mu=Uniform(*cfg.gmm_mu),
            sigma=Uniform(*cfg.gmm_sigma),
            fwhm=0,
            background=0 if cfg.zero_background else None,
        )

        self.bias = cc.MaybeTransform(
            cc.RandomMulFieldTransform(
                shape=cfg.bias_ctrl, vmax=cfg.bias_vmax, symmetric=True),
            cfg.bias_prob)

        self.noise = cc.RandomGaussianNoiseTransform(
            sigma=Uniform(*cfg.noise_sigma))
        self.normalize = cc.MinMaxTransform()

    # -- steps ---------------------------------------------------------------

    def _morphology(self, lab):
        """Constrained random dilation/erosion between neighbouring pairs."""
        cfg = self.cfg
        lo, hi = cfg.morpho_iter
        for a, b in cfg.morpho_pairs:
            if random.random() > cfg.morpho_prob:
                continue
            a, b = T[a], T[b]
            grow, shrink = (a, b) if random.random() < 0.5 else (b, a)
            n_iter = random.randint(lo, hi)
            # work in the bounding box of the pair (+ margin)
            box = _bbox((lab == grow) | (lab == shrink), margin=n_iter + 1)
            if box is None:
                continue
            sub = lab[box]
            # cornucopia's 'conv' dilation by `radius` = radius-1 iterations
            dilated = cc.DilateLabelTransform(
                labels=[grow], radius=n_iter + 1)(sub[None])[0]
            # only take over voxels of the paired neighbour
            lab[box] = torch.where((dilated == grow) & (sub == shrink),
                                   grow, sub)
            del dilated, sub
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

    def _partial_volume(self, lab):
        """Average-pool the one-hot HR labels to the training grid."""
        f = self.cfg.factor
        pv = []
        for k in range(self.cfg.nb_labels):
            m = (lab == k)[None, None].float()
            pv.append(F.avg_pool3d(m, f)[0, 0])
            del m
        return torch.stack(pv)

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
        )
        return motion(img).movedim(1, axis)

    # -- main ----------------------------------------------------------------

    @torch.no_grad()
    def __call__(self, template_index=None):
        """
        Returns
        -------
        img : (1, X, Y, Z) float tensor in [0, 1]
        lab : (X, Y, Z) uint8 tensor (tissue indices)
        pv : (K, X, Y, Z) float tensor, partial-volume maps
        affine : (4, 4) ndarray, voxel-to-world of the 0.75 mm grid
        """
        if template_index is None:
            template_index = random.randrange(len(self.templates))
        lab, affine_hr = self.templates[template_index]

        lab = self._morphology(lab.clone())
        lab = self._deform(lab)
        pv = self._partial_volume(lab)
        del lab

        img = self.gmm(pv)
        img = self.bias(img)
        img = self._motion(img)
        img = self.noise(img)
        img = self.normalize(img)

        seg = pv.argmax(0).to(torch.uint8)

        # pooled voxel i is centred on HR voxel f*i + (f-1)/2
        f = self.cfg.factor
        pool = np.diag([f, f, f, 1.]).astype(float)
        pool[:3, 3] = (f - 1) / 2
        affine = affine_hr @ pool
        return img, seg, pv, affine


# ---------------------------------------------------------------------------
# I/O helpers and CLI (nnU-Net raw dataset layout)
# ---------------------------------------------------------------------------

def save_sample(img, seg, affine, img_path, seg_path):
    nib.save(nib.Nifti1Image(img[0].cpu().numpy().astype(np.float32),
                             affine), img_path)
    nib.save(nib.Nifti1Image(seg.cpu().numpy().astype(np.uint8),
                             affine), seg_path)


def write_nnunet_dataset_json(out_dir, n, names=TISSUES):
    meta = {
        'channel_names': {'0': 'synthetic'},
        'labels': {name: i for i, name in enumerate(names)},
        'numTraining': n,
        'file_ending': '.nii.gz',
    }
    with open(os.path.join(out_dir, 'dataset.json'), 'w') as f:
        json.dump(meta, f, indent=2)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main():
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

    templates = []
    for path in args.seg:
        name = os.path.basename(path).split('.')[0]
        hr = os.path.join(args.out, 'templates',
                          f'{name}_hr{cfg.target_voxel / cfg.factor:g}mm.nii.gz')
        if os.path.exists(hr):
            templates.append(load_template(hr))
        else:
            print(f'Super-resolving {path} -> {hr}')
            lab, aff = prepare_template(path, hr, cfg.target_voxel,
                                        cfg.factor, device=args.device)
            templates.append((lab.cpu(), aff))

    gen = SIAMGenerator(templates, cfg, device=args.device)
    for i in range(args.n):
        img, seg, _, aff = gen()
        case = f'{args.prefix}_{i:04d}'
        save_sample(img, seg, aff,
                    os.path.join(args.out, 'imagesTr', f'{case}_0000.nii.gz'),
                    os.path.join(args.out, 'labelsTr', f'{case}.nii.gz'))
        print(f'[{i + 1}/{args.n}] {case} {tuple(seg.shape)}')
    write_nnunet_dataset_json(args.out, args.n, cfg.names)


if __name__ == '__main__':
    main()
