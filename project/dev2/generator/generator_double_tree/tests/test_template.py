"""Offline module: HR template building."""

import os
import sys

import numpy as np
import torch

from phantoms import BG, CSF, GM, Skip, sphere_labels, workdir
from nxgen.core.io import load_labels, save_labels
from nxgen.offline.template import (build_template, crop_to_foreground,
                                    restrict_labels, super_resolve)

V2 = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(
    __file__))), '..', 'generator_v2')


def test_identity_when_same_voxel_and_no_smoothing():
    seg = torch.as_tensor(sphere_labels((20, 20, 20)))
    out = super_resolve(seg, factor=1, voxel_mm=0.5, sigma_mm=0.0)
    assert torch.equal(out.long(), seg)


def test_no_new_labels_and_volumes_kept():
    seg = torch.as_tensor(sphere_labels((24, 24, 24)))
    out = super_resolve(seg, factor=2, voxel_mm=1.0, sigma_mm=0.5)
    assert set(torch.unique(out).tolist()) <= set(torch.unique(seg).tolist())
    for v in torch.unique(seg).tolist():
        before = (seg == v).sum().item() * 8          # in HR voxels
        after = (out == v).sum().item()
        assert abs(after - before) / before < 0.15, (v, before, after)


def test_crop_margin_and_padding():
    seg = np.zeros((20, 20, 20), int)
    seg[2:5, 8:12, 15:19] = GM
    out, affine, box = crop_to_foreground(seg, np.eye(4), 1.0, [3, 1, 2])
    assert out.shape == (3 + 6, 4 + 2, 4 + 4)
    assert box['lo'] == [-1, 7, 13] and box['hi'] == [8, 13, 21]
    assert np.allclose(affine[:3, 3], [-1, 7, 13])    # world of voxel 0
    assert (out == GM).sum() == (seg == GM).sum()


def test_restrict_labels():
    seg = np.array([BG, GM, CSF, 99])
    out, removed = restrict_labels(seg, [GM, CSF])
    assert removed == [99] and out.tolist() == [BG, GM, CSF, BG]


def test_build_template_record_and_cache():
    folder = workdir()
    src = os.path.join(folder, 'map.nii.gz')
    save_labels(sphere_labels((30, 30, 30)), np.diag([0.4] * 3 + [1]), src)
    path, rec = build_template(src, folder, hr_voxel_mm=0.2, sigma_mm=0.1,
                               margin_mm=1.0)
    lab, affine, voxel = load_labels(path)
    assert np.allclose(voxel, 0.2)
    assert rec['vanished_labels'] == [] and rec['upsampling_factor'] == 2
    assert set(np.unique(lab)) == set(np.unique(sphere_labels((30, 30, 30))))
    _, rec2 = build_template(src, folder, 0.2, 0.1, 1.0)    # cached
    assert rec2['source_sha256'] == rec['source_sha256']
    try:
        build_template(src, folder, hr_voxel_mm=0.3)
        raise AssertionError('accepted a non-integer upsampling factor')
    except ValueError:
        pass


def test_equivalent_to_v2_super_resolution():
    sys.path.insert(0, V2)
    try:
        import tree_generator as v2
    except Exception as e:                       # cornucopia missing, ...
        raise Skip(f'generator_v2 not importable: {e}')
    seg = torch.as_tensor(sphere_labels((24, 24, 24)))
    new = super_resolve(seg, 2, 1.0, 0.5)
    old = v2.super_resolve_labels(seg, 2, 1.0, 0.5)
    assert torch.equal(new.long(), old.long())
