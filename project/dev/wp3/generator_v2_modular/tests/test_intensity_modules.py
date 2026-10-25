"""Online steps 4-6: tree intensities, synthesis, artefacts, normalisation."""

import os
import sys

import numpy as np
import torch

from phantoms import BG, CSF, GM, GM_PART, WM, Skip, sphere_labels, tree
from nxgen.checks import gaussian_fit_pure
from nxgen.core.config import load_config
from nxgen.core.label_space import LabelSpace
from nxgen.online import (acquisition, bias, motion, noise, normalization,
                          partial_volume, resolution, synthesis,
                          tree_intensities)

V2 = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(
    __file__))), '..', 'generator_v2')


def _tree_params(res, contrast=None, seed=0, **tree_cfg):
    t = tree()
    ls = LabelSpace.build(t, {BG, CSF, GM, GM_PART, WM})
    cfg = load_config(tree=dict(resolution=res, **tree_cfg))['tree']
    acq = acquisition.none_params(1.0, contrast)
    return tree_intensities.sample(cfg, np.random.default_rng(seed), t, ls,
                                   acq), ls


# -- step 4a -------------------------------------------------------------------

def test_collapsed_node_shares_parent_gaussian():
    p, ls = _tree_params(res=2.0)                      # gm_part r_max = 1
    gm, part = ls.index[GM], ls.index[GM_PART]
    assert (p['mu'][part], p['sigma'][part]) == (p['mu'][gm], p['sigma'][gm])
    assert p['class_of'][part] == p['class_of'][gm]
    q, _ = _tree_params(res=0.8)                       # visible
    assert q['mu'][part] != q['mu'][gm]
    assert q['class_of'][part] != q['class_of'][gm]
    t2, _ = _tree_params(res=2.0, contrast='T2')       # r_max(T2) = 2
    assert t2['class_of'][part] != t2['class_of'][gm]


def test_visible_nodes_same_values_at_any_resolution():
    fine, ls = _tree_params(res=0.5, seed=4)
    coarse, _ = _tree_params(res=5.0, seed=4)
    for v in (CSF, GM, WM):                            # always visible
        i = ls.index[v]
        assert fine['mu'][i] == coarse['mu'][i]


def test_null_nodes_and_min_sigma():
    p, ls = _tree_params(res=1.0, min_sigma=3.0)
    assert p['mu'][ls.index[BG]] == 0 and p['sigma'][ls.index[BG]] == 0
    assert min(p['sigma'][1:]) >= 3.0


# -- step 4b -------------------------------------------------------------------

def test_synthesis_pure_voxels_follow_gaussians(device='cpu'):
    p, ls = _tree_params(res=1.0, colors_sigma=[5.0, 10.0])
    seg = sphere_labels((60, 60, 60), scale=2)
    lab = ls.lut(int(seg.max()))[torch.as_tensor(seg)].to(device)
    pv = partial_volume.apply(lab, 'downsample', factor=2)
    img = synthesis.apply(pv, p)
    fit = gaussian_fit_pure(img, pv, p)
    assert set(fit) >= {ls.index[CSF], ls.index[GM], ls.index[WM]}
    for i, r in fit.items():
        if r['sigma'] > 0:
            assert abs(r['mean'] - r['mu']) < 4 * r['sigma'] / np.sqrt(r['n'])
            assert abs(r['std'] / r['sigma'] - 1) < 0.1
            assert r['ks'] is None or r['ks'] < 0.05


def test_synthesis_same_class_shares_noise(device='cpu'):
    p, ls = _tree_params(res=2.0)                      # gm + gm_part merged
    gm, part = ls.index[GM], ls.index[GM_PART]
    shape = (1, 6, 6, 6)
    pv = dict(index=torch.stack([torch.full(shape[1:], gm),
                                 torch.full(shape[1:], part)]).short(),
              weight=torch.full((2, 6, 6, 6), 0.5))
    img = synthesis.apply(pv, p) * p['intensity_scale']
    # same eps for both entries -> exactly one Gaussian draw per voxel
    assert torch.allclose((img - p['mu'][gm]) / p['sigma'][gm],
                          torch.randn((2, 6, 6, 6), generator=torch.Generator()
                                      .manual_seed(p['noise_seed']))[0],
                          atol=1e-4)


# -- step 5 --------------------------------------------------------------------

def test_bias_range_and_patch_consistency(device='cpu'):
    cfg = load_config(bias=dict(vmax=0.3, ctrl=4))['bias']
    p = bias.sample(cfg, np.random.default_rng(1))
    full = bias.field(p, [0, 0, 0], [30, 20, 25], [30, 20, 25], device)
    assert full.min() >= 0.7 - 1e-5 and full.max() <= 1.3 + 1e-5
    part = bias.field(p, [5, 2, 7], [10, 10, 10], [30, 20, 25], device)
    assert torch.allclose(part, full[5:15, 2:12, 7:17], atol=1e-6)
    assert bias.sample(load_config(bias=dict(prob=0))['bias'],
                       np.random.default_rng(1)) is None


def test_motion_identity_and_effect(device='cpu'):
    img = torch.rand(24, 20, 16, device=device)
    p = dict(axis=1, shots=3, reference_shot=1,
             rotation_deg=[[0] * 3] * 3, translation_mm=[[0] * 3] * 3)
    assert torch.allclose(motion.apply(img, p, 1.0), img, atol=1e-5)
    p['translation_mm'] = [[0, 3, 0], [0, 0, 0], [2, 0, 0]]
    out = motion.apply(img, p, 1.0)
    assert not torch.allclose(out, img, atol=1e-2)
    assert motion.reference_shot(20, 3) == 1


def test_motion_rigid_translation(device='cpu'):
    img = torch.zeros(10, 10, 10, device=device)
    img[4, 5, 5] = 1
    out = motion.rigid(img, [0, 0, 0], [2.0, 0, 0], voxel_mm=1.0)
    assert out[6, 5, 5] == 1 and out.sum() == 1


def test_resolution_none_and_shapes(device='cpu'):
    img = torch.rand(30, 30, 30, device=device)
    none = acquisition.none_params(1.0)
    for acq in (none, None):
        out, low = resolution.apply(img, acq, 1.0)
        assert out is img and low is None
    acq = dict(none, spacing_mm=[1, 1, 4.0], thickness_mm=[1, 1, 3.0],
               alpha=[1, 1, 1])
    calls = []
    out, low = resolution.apply(img, acq, 1.0,
                                noise_fn=lambda x: calls.append(x.shape) or x)
    assert list(low.shape) == [30, 30, 8] and calls == [low.shape]
    assert out.shape == img.shape
    # smoothing along z only: variance drops along z, not along x
    assert out.diff(dim=2).abs().mean() < 0.5 * img.diff(dim=2).abs().mean()


def test_resolution_close_to_v2(device='cpu'):
    sys.path.insert(0, V2)
    try:
        import tree_generator as v2
    except Exception as e:
        raise Skip(f'generator_v2 not importable: {e}')
    torch.manual_seed(0)
    img = torch.rand(32, 32, 32)
    acq = dict(acquisition.none_params(1.0), spacing_mm=[3.0, 1, 1],
               thickness_mm=[2.5, 1, 1], alpha=[1, 1, 1])
    new, _ = resolution.apply(img, acq, 1.0)
    # v2 parameters: the mode and sigma_vox are stored
    old = v2.SynthSegLowResTransform(r_hr=1.0).make_final(
        dict(acq, mode='aniso', axis=0,
             sigma_vox=acquisition.sigma_vox(acq, 1.0)),
        noise=False)(img[None])[0]
    # same steps; only the kernel truncation / edges differ
    inner = (slice(6, -6),) * 3
    assert (new[inner] - old[inner]).abs().max() < 0.02


def test_noise_level_and_seed(device='cpu'):
    img = torch.zeros(50, 50, 50, device=device)
    p = dict(sigma=0.2, seed=5, space='lowres')
    a, b = noise.apply(img, p), noise.apply(img, p)
    assert torch.equal(a, b) and abs(a.std().item() - 0.2) < 0.005
    assert noise.apply(img, dict(p, sigma=0)) is img


# -- step 6 --------------------------------------------------------------------

def test_normalization(device='cpu'):
    img = torch.randn(20, 20, 20, device=device) * 5 + 3
    out, p = normalization.apply(img, dict(method='minmax'))
    assert out.min() == 0 and out.max() == 1 and p['lo'] < p['hi']
    out, p = normalization.apply(img, dict(method='percentile',
                                           percentiles=[1, 99]))
    assert out.min() == 0 and out.max() == 1
    out, _ = normalization.apply(img, dict(method='none'))
    assert out is img
