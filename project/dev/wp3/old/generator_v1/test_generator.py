"""
Checks of generator_v1.

    python test_generator.py [--device cuda]

1. `LowResAcquisitionFinalTransform` on one axis == cornucopia's
   `LowResSliceFinalTransform` (same parameters).
2. `SynthSegLowResTransform` follows SynthSeg's priors (modes, ranges).
3. Label I/O: uint16 on disk, values > 255 survive the round trip.
4. Many labels: a template with > 255 labels (the WM split into 300
   parcels, values 5000-5299, added to the tree) goes through the whole
   generator without index overflow.
5. Full generator on the example template: shapes, range, no NaN, both
   noise spaces, fixed acquisitions, and the tree resolution follows the
   acquisition (fewer generation classes at 6 mm than at 0.75 mm).
"""

import argparse
import copy
import json
import math
import os
import tempfile

import nibabel as nib
import numpy as np
import torch

import tree_generator as tg

HERE = os.path.dirname(os.path.abspath(__file__))
SEG = os.path.join(HERE, 'data', 'inputs', 'seg.nii.gz')
TREE = os.path.join(HERE, 'data', 'inputs', 'synthseg_ctx_tree.json')
CONFIG = os.path.join(HERE, 'configs', 'freesurfer.json')


def check(cond, msg):
    """Print and assert."""
    print(('  OK   ' if cond else '  FAIL ') + msg)
    assert cond, msg


def test_cornucopia_equivalence():
    print('1. single-axis low-res == cc.LowResSliceFinalTransform')
    x = torch.rand(1, 40, 48, 36)
    for axis, spac, fwhm in [(0, 4.0, 3.1), (1, 2.5, 2.5), (2, 7.3, 9.0)]:
        spacing, fw = [1.] * 3, [0.] * 3
        spacing[axis], fw[axis] = spac, fwhm
        ours = tg.LowResAcquisitionFinalTransform(spacing, fw)(x)
        ref = tg.cc.LowResSliceFinalTransform(axis, spac, fwhm / spac)(x)
        err = (ours - ref).abs().max().item()
        check(err < 1e-5, f'axis {axis} spacing {spac}: max |diff| {err:.2e}')


def test_priors(n=4000):
    print('2. SynthSeg priors')
    tg.seed_everything(0)
    t = tg.SynthSegLowResTransform(r_hr=0.75, prob=0.9, iso_prob=0.2,
                                   max_iso=4., max_aniso=8.)
    draws = [t.sample() for _ in range(n)]
    modes = [d['mode'] for d in draws]
    p_none = modes.count('none') / n
    p_iso = modes.count('iso') / max(1, n - modes.count('none'))
    check(abs(p_none - 0.1) < 0.02, f'P(none) = {p_none:.3f} (0.1)')
    check(abs(p_iso - 0.2) < 0.03, f'P(iso | degraded) = {p_iso:.3f} (0.2)')
    ok = True
    for d in draws:
        sp, th = np.array(d['spacing_mm']), np.array(d['thickness_mm'])
        ok &= bool(np.all(th <= sp + 1e-9) and np.all(sp >= 0.75))
        if d['mode'] == 'iso':
            ok &= bool(sp.max() <= 4 and np.ptp(sp) == 0)
        if d['mode'] == 'aniso':
            ok &= bool(sp.max() <= 8 and np.sum(sp > 0.75) <= 1)
            a = d['axis']
            ref = tg.sigma_thick_vox(th[a], d['alpha'][a], 0.75)
            ok &= math.isclose(d['sigma_vox'][a], ref)
    check(ok, 'spacing / thickness / sigma within the priors')
    # cornucopia API: random call keeps the shape
    x = torch.rand(1, 30, 30, 30)
    check(t(x).shape == x.shape, 'random call returns the input shape')


def test_label_io(tmp):
    print('3. label I/O')
    check(tg.label_dtype(0, 255) == np.uint16, 'uint8 never used')
    check(tg.label_dtype(0, 65535) == np.uint16, 'uint16 up to 65535')
    check(tg.label_dtype(0, 70000) == np.int32, 'int32 above 65535')
    arr = np.random.default_rng(0).integers(0, 900, (10, 11, 12))
    path = os.path.join(tmp, 'lab.nii.gz')
    tg.save_labels(arr, np.eye(4), path)
    nii = nib.load(path)
    back = np.asarray(nii.dataobj)
    check(nii.get_data_dtype() == np.uint16,
          f'saved dtype {nii.get_data_dtype()}')
    check(np.array_equal(back, arr), 'values > 255 survive the round trip')


def many_label_inputs(tmp, n_parcels=300, first=5000):
    """Split the WM (2, 41) into parcels and add them to the tree."""
    nii = nib.load(SEG)
    seg = np.asarray(nii.dataobj).astype(np.int64)
    wm = np.isin(seg, [2, 41])
    idx = np.argwhere(wm)
    rng = np.random.default_rng(0)
    centres = idx[rng.choice(len(idx), n_parcels, replace=False)]
    # nearest centre, in chunks to bound memory
    near = np.concatenate([
        np.argmin(((c[:, None] - centres[None]) ** 2).sum(-1), 1)
        for c in np.array_split(idx, 50)])
    seg[tuple(idx.T)] = first + near
    seg_path = os.path.join(tmp, 'seg_many.nii.gz')
    tg.save_labels(seg, nii.affine, seg_path)

    tree = json.load(open(TREE))
    parent = tree['nodes']['cerebral_wm']
    for k in range(n_parcels):
        name = f'wm_parcel_{k}'
        parent['children'].append(name)
        tree['nodes'][name] = dict(parent='cerebral_wm', children=[], level=2,
                                   labels=[first + k], wr_mu=10.,
                                   wr_sigma=0.5, r_max=1.5)
    tree_path = os.path.join(tmp, 'tree_many.json')
    json.dump(tree, open(tree_path, 'w'))
    return seg_path, tree_path


def test_many_labels(tmp, device):
    print('4. > 255 labels through the whole generator')
    seg_path, tree_path = many_label_inputs(tmp)
    cfg = tg.TreeGeneratorConfig.from_json(CONFIG)
    templates = tg.build_templates([seg_path], tree_path, cfg,
                                   os.path.join(tmp, 'templates'), device)
    hr = nib.load(os.path.join(tmp, 'templates', 'seg_many_hr0.25mm.nii.gz'))
    check(hr.get_data_dtype() == np.uint16, 'HR template saved as uint16')
    gen = tg.TreeGenerator(templates, tree_path, cfg, device=device)
    check(gen.nb_labels > 256, f'{gen.nb_labels} output labels')
    img, seg, aff, info = gen(acquisition=dict(
        mode='none', axis=None, spacing_mm=[.75] * 3, thickness_mm=[.75] * 3,
        alpha=[1.] * 3, sigma_vox=[0.] * 3))
    present = torch.unique(seg).cpu().numpy()
    check(seg.dtype == torch.int16, f'index dtype {seg.dtype}')
    check(present.max() > 255, f'max index {present.max()} > 255')
    check(len(present) > 0.9 * gen.nb_labels,
          f'{len(present)}/{gen.nb_labels} labels present after deformation')
    out = os.path.join(tmp, 'many_seg.nii.gz')
    tg.save_sample(img, seg, aff, os.path.join(tmp, 'many_img.nii.gz'), out,
                   labels=gen.labels)
    back = np.asarray(nib.load(out).dataobj)
    ref = gen.labels[seg.cpu().numpy().astype(int)]
    check(np.array_equal(back, ref), 'raw label values (5000+) saved exactly')
    # 300 visible WM parcels at 0.75 mm, collapsed into the WM at 6 mm
    g_hr = len(info['class_names'])
    _, _, _, info6 = gen(acquisition=fixed_aniso(gen, 6.))
    check(len(info6['class_names']) < g_hr - 250,
          f'generation classes {g_hr} @ 0.75 mm -> '
          f'{len(info6["class_names"])} @ 6 mm')


def fixed_aniso(gen, spacing, axis=2):
    """Acquisition with one thick-slice axis (thickness = spacing)."""
    r = gen.cfg.target_voxel
    sp, th, sg = [r] * 3, [r] * 3, [0.] * 3
    sp[axis] = th[axis] = spacing
    sg[axis] = tg.sigma_thick_vox(spacing, 1., r)
    return dict(mode='aniso', axis=axis, spacing_mm=sp, thickness_mm=th,
                alpha=[1.] * 3, sigma_vox=sg)


def test_generator(tmp, device):
    print('5. full generator on the example template')
    cfg = tg.TreeGeneratorConfig.from_json(CONFIG)
    templates = tg.build_templates([SEG], TREE, cfg,
                                   os.path.join(tmp, 'templates'), device)
    for space in ('lowres', 'highres'):
        c = copy.deepcopy(cfg)
        c.noise_space = space
        gen = tg.TreeGenerator(templates, TREE, c, device=device)
        for i in range(4):
            img, seg, aff, info = gen(motion=i % 2 == 0)
            ok = (img.shape[1:] == seg.shape and torch.isfinite(img).all()
                  and img.min() >= 0 and img.max() <= 1)
            acq = info['acquisition']
            check(bool(ok), f'noise {space}: {acq["mode"]:5s} '
                  f'{[round(s, 2) for s in acq["spacing_mm"]]} mm, tree res '
                  f'{info["resolution"]:.2f}, '
                  f'{len(info["class_names"])} classes')
            check(math.isclose(info['resolution'], max(acq['spacing_mm'])),
                  'tree resolution = coarsest acquired spacing')
    # with_config: shares the templates, changes only the given fields
    gen = tg.TreeGenerator(templates, TREE, cfg, device=device)
    g2 = gen.with_config(lowres_prob=0., noise_space='highres')
    check(g2.templates[0][0] is gen.templates[0][0]
          and g2.cfg.lowres_prob == 0. and gen.cfg.lowres_prob == cfg.lowres_prob,
          'with_config shares templates and leaves the original untouched')
    _, _, _, inf = g2()
    check(inf['acquisition']['mode'] == 'none', 'with_config(lowres_prob=0)')
    try:
        gen.with_config(factor=2)
        check(False, 'with_config(factor=...) must raise')
    except ValueError:
        check(True, 'with_config refuses label-space / grid fields')
    # the low-res step only changes the image: same seed -> same labels
    gen = tg.TreeGenerator(templates, TREE, cfg, device=device)
    tg.seed_everything(1)
    _, s1, _, i1 = gen(template_index=0, acquisition=fixed_aniso(gen, 6.),
                       motion=False, return_steps=True)
    tg.seed_everything(1)
    _, s2, _, _ = gen(template_index=0, acquisition=fixed_aniso(gen, 0.75),
                      motion=False)
    check(torch.equal(s1, s2), 'segmentation independent of the acquisition')
    lr = i1['steps']['lowres']
    exp = math.ceil(s1.shape[2] / (6. / 0.75))
    check(lr is not None and lr.shape[3] == exp,
          f'low-res grid {tuple(lr.shape[1:])} ({exp} slices of 6 mm)')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--device', default='cuda' if torch.cuda.is_available()
                   else 'cpu')
    args = p.parse_args()
    tg.seed_everything(0)
    with tempfile.TemporaryDirectory() as tmp:
        test_cornucopia_equivalence()
        test_priors()
        test_label_io(tmp)
        test_many_labels(tmp, args.device)
        test_generator(tmp, args.device)
    print('all checks passed')


if __name__ == '__main__':
    main()
