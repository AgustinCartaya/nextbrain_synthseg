"""Online steps 0-3: acquisition, morphology, spatial, partial volume."""

import numpy as np
import torch

from phantoms import BG, CSF, GM, WM, sphere_labels, tree
from nxgen.core.config import load_config
from nxgen.core.label_space import LabelSpace
from nxgen.online import acquisition, morphology, partial_volume, spatial


def _index_map(device='cpu', shape=(40, 40, 40)):
    """Phantom as int16 indices + its label space."""
    seg = sphere_labels(shape)
    ls = LabelSpace.build(tree(), set(np.unique(seg).tolist()))
    lab = ls.lut(int(seg.max()))[torch.as_tensor(seg)].to(device)
    return lab, ls


# -- step 0 --------------------------------------------------------------------

def test_acquisition_priors():
    cfg = load_config(acquisition=dict(lowres_prob=0.8, iso_prob=0.25,
                                       max_iso_mm=3.0, max_aniso_mm=5.0,
                                       contrasts=['T1', 'T2']))['acquisition']
    rng = np.random.default_rng(0)
    draws = [acquisition.sample(cfg, rng, 1.0) for _ in range(4000)]
    modes = [acquisition.describe(d, 1.0)[0] for d in draws]
    assert set(modes) == {'none', 'iso', 'aniso'}       # never 'custom'
    assert abs(modes.count('none') / 4000 - 0.2) < 0.03
    assert abs(modes.count('iso') / 4000 - 0.8 * 0.25) < 0.03
    for d, mode in zip(draws, modes):
        assert set(d) == {'spacing_mm', 'thickness_mm', 'alpha', 'contrast'}
        for s, t in zip(d['spacing_mm'], d['thickness_mm']):
            assert 1.0 <= t <= s + 1e-9
        if mode == 'aniso':
            assert sum(s > 1 for s in d['spacing_mm']) == 1
            assert max(d['spacing_mm']) <= 5.0
        if mode == 'iso':
            assert len(set(d['spacing_mm'])) == 1 and d['spacing_mm'][0] <= 3
    assert {d['contrast'] for d in draws} == {'T1', 'T2'}


def test_acquisition_derived_values():
    r = 0.4
    none = acquisition.none_params(r)
    assert acquisition.describe(None, r) == ('none', [])
    assert acquisition.describe(none, r) == ('none', [])
    assert acquisition.sigma_vox(none, r) == [0.0] * 3   # thickness r ignored
    aniso = dict(none, spacing_mm=[r, r, 4.0], thickness_mm=[3.0, r, 3.0],
                 alpha=[1, 1, 0.5])
    assert acquisition.describe(aniso, r) == ('aniso', [2])
    # thickness of a non-degraded axis is ignored
    assert acquisition.sigma_vox(aniso, r) == [
        0.0, 0.0, acquisition.slice_sigma_vox(3.0, 0.5, r)]
    iso = dict(none, spacing_mm=[2.0] * 3, thickness_mm=[2.0] * 3)
    assert acquisition.describe(iso, r) == ('iso', [0, 1, 2])
    two = dict(none, spacing_mm=[2.0, r, 3.0])
    assert acquisition.describe(two, r) == ('custom', [0, 2])


# -- step 1 --------------------------------------------------------------------

def _op(ls, grow, shrink, n):
    t = tree()
    return dict(ops=[dict(grow=grow, shrink=shrink, n_iter=n,
                          grow_idx=ls.resolve(grow, t),
                          shrink_idx=ls.resolve(shrink, t))])


def test_morphology_grows_only_into_the_other_group(device='cpu'):
    lab, ls = _index_map(device)
    out = morphology.apply(lab, _op(ls, 'wm', 'gm', 2), len(ls))
    wm, gm, csf = ls.index[WM], ls.index[GM], ls.index[CSF]
    assert (out == wm).sum() > (lab == wm).sum()            # wm grew
    changed = out != lab
    assert set(lab[changed].tolist()) <= {gm, ls.index[42]}  # taken from gm
    assert set(out[changed].tolist()) == {wm}
    assert torch.equal(out == csf, lab == csf)              # others untouched
    assert morphology.apply(lab, None, len(ls)) is lab


def test_morphology_on_crop_equals_full(device='cpu'):
    lab, ls = _index_map(device)
    params = _op(ls, 'csf', 'gm', 2)
    params['ops'] += _op(ls, 'wm', 'gm', 1)['ops']
    full = morphology.apply(lab, params, len(ls))
    h = morphology.halo(params)
    assert h == 3
    lo, hi = np.array([5, 12, 20]), np.array([25, 30, 36])
    crop = lab[lo[0] - h:hi[0] + h, lo[1] - h:hi[1] + h, lo[2] - h:hi[2] + h]
    part = morphology.apply(crop, params, len(ls))[h:-h, h:-h, h:-h]
    assert torch.equal(part, full[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]])


# -- step 2 --------------------------------------------------------------------

def test_spatial_identity(device='cpu'):
    lab, _ = _index_map(device)
    p = spatial.identity_params()
    shape = list(lab.shape)
    lo, hi = spatial.source_box(p, [0, 0, 0], shape, shape, 0.5)
    out = spatial.apply(lab[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]], lo, p,
                        [0, 0, 0], shape, shape, 0.5)
    assert torch.equal(out, lab)


def test_spatial_patch_equals_crop_of_full(device='cpu'):
    lab, _ = _index_map(device)
    cfg = load_config(spatial=dict(rotation_deg=[-20, 20], shear=[-.1, .1],
                                   translation_mm=[-2, 2],
                                   elastic_dmax_mm=2.0))['spatial']
    p = spatial.sample(cfg, np.random.default_rng(3))
    shape = list(lab.shape)
    full = spatial.apply(lab, [0, 0, 0], p, [0, 0, 0], shape, shape, 0.5)
    start, size = [6, 10, 3], [16, 12, 20]
    lo, hi = spatial.source_box(p, start, size, shape, 0.5)
    crop = lab[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
    part = spatial.apply(crop, lo, p, start, size, shape, 0.5,
                         slab_voxels=1000)                   # several slabs
    assert torch.equal(part, full[6:22, 10:22, 3:23])


def test_spatial_rotation_moves_labels(device='cpu'):
    lab, _ = _index_map(device)
    p = spatial.identity_params()
    from nxgen.core.geometry import affine_matrix
    p['matrix'] = affine_matrix([0, 0, 180], [1, 1, 1], [0, 0, 0]).tolist()
    shape = list(lab.shape)
    out = spatial.apply(lab, [0, 0, 0], p, [0, 0, 0], shape, shape, 0.5)
    # 180 degrees about z flips x and y (the gm/gm_part halves swap)
    assert torch.equal(out, lab.flip(0).flip(1))
    assert set(torch.unique(out).tolist()) <= set(torch.unique(lab).tolist())


# -- step 3 --------------------------------------------------------------------

def test_pv_factor_one_is_hard_labels(device='cpu'):
    lab, _ = _index_map(device)
    pv = partial_volume.apply(lab, 'downsample', factor=1, top_k=8)
    assert pv['index'].shape[0] == 1
    assert torch.equal(pv['seg'], lab) and torch.all(pv['weight'] == 1)


def test_pv_downsample_mass_and_majority(device='cpu'):
    lab, ls = _index_map(device)
    f = 2
    pv = partial_volume.apply(lab, 'downsample', factor=f, top_k=8,
                              budget_bytes=1 << 16)          # many slabs
    assert (pv['weight'].sum(0) - 1).abs().max() < 1e-5
    assert pv['lost'].abs().max() < 1e-6                     # k = 8 exact
    for i in range(len(ls)):
        mass = (pv['weight'] * (pv['index'] == i)).sum().item()
        assert abs(mass - (lab == i).sum().item() / f ** 3) < 1e-2
    # majority label, ties -> smallest index (brute force)
    blk = lab.long().view(20, 2, 20, 2, 20, 2).permute(0, 2, 4, 1, 3, 5)
    counts = torch.nn.functional.one_hot(blk.reshape(20, 20, 20, 8),
                                         len(ls)).sum(3)
    assert torch.equal(pv['seg'].long(), counts.argmax(-1))


def test_pv_topk_truncation_reports_lost_mass(device='cpu'):
    lab, _ = _index_map(device)
    pv = partial_volume.apply(lab, 'downsample', factor=2, top_k=1)
    assert torch.equal(pv['index'][0].long(), pv['seg'].long())
    assert pv['lost'].max() > 0 and torch.all(pv['weight'] == 1)


def test_pv_smooth_keeps_seg_and_mixes_boundaries(device='cpu'):
    lab, _ = _index_map(device)
    pv = partial_volume.apply(lab, 'smooth', factor=1, top_k=4, sigma_vox=0.7)
    assert torch.equal(pv['seg'], lab)
    pure = pv['weight'][0] > 1 - 1e-6
    # interior voxels stay pure, boundary voxels get mixed
    assert pure[20, 20, 20] and not pure.all()
    assert (pv['weight'].sum(0) - 1).abs().max() < 1e-5
