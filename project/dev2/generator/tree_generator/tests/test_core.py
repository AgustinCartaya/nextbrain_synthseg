"""Core: config, distribution specs, seeds, label space, B-splines."""

import numpy as np
import torch

from phantoms import BG, CSF, GM, GM_PART, WM, tree, write_inputs, workdir
from nxgen.core import bspline
from nxgen.core.config import load_config
from nxgen.core.label_space import LabelSpace
from nxgen.core.rng import check_spec, draw, image_seed, module_rng


def test_config_defaults_and_overrides():
    cfg = load_config(noise={'sigma': 0.0})
    assert cfg['noise']['sigma'] == 0.0
    assert cfg['noise']['space'] == 'lowres'           # untouched default


def test_config_rejects_unknown_and_invalid():
    for bad in [dict(nosie={'sigma': 0}), dict(noise={'sgma': 0}),
                dict(noise={'space': 'middle'}),
                dict(labels={'unassigned': 'background_l1'}),
                dict(spatial={'zoom': [1.1, 0.9]})]:
        try:
            load_config(**bad)
        except ValueError:
            continue
        raise AssertionError(f'accepted {bad}')


def test_config_relative_tree_path():
    folder = workdir()
    _, tree_path = write_inputs(folder)
    import json, os
    path = os.path.join(folder, 'c.json')
    with open(path, 'w') as f:
        json.dump({'labels': {'tree': 'tree.json'}}, f)
    assert load_config(path)['labels']['tree'] == tree_path


def test_draw_specs():
    rng = np.random.default_rng(0)
    assert draw(5, rng) == 5 and isinstance(draw(5, rng), int)
    v = [draw([1.0, 2.0], rng) for _ in range(200)]
    assert 1 <= min(v) and max(v) <= 2
    v = {draw({'randint': [1, 3]}, rng) for _ in range(200)}
    assert v == {1, 2, 3}
    v = {draw({'choice': ['a', 'b']}, rng) for _ in range(100)}
    assert v == {'a', 'b'}
    for bad in ([2, 1], {'randint': [3]}, {'choice': []}, True, 'x'):
        try:
            check_spec(bad)
        except ValueError:
            continue
        raise AssertionError(f'accepted {bad}')


def test_seeds_independent_and_reproducible():
    assert image_seed(0, 1) == image_seed(0, 1) != image_seed(0, 2)
    a = module_rng(7, 'spatial').random(3)
    b = module_rng(7, 'spatial').random(3)
    c = module_rng(7, 'bias').random(3)
    assert np.array_equal(a, b) and not np.array_equal(a, c)


def test_label_space():
    t = tree()
    ls = LabelSpace.build(t, {BG, CSF, GM, WM}, output_labels='templates')
    assert ls.values == [BG, WM, GM, CSF]              # sorted after bg
    ls_tree = LabelSpace.build(t, {BG, GM}, output_labels='tree')
    assert set(ls_tree.values) == {BG, CSF, GM, GM_PART, WM}
    assert ls.resolve('gm', t) == [ls.index[GM]]       # GM_PART not present
    assert ls_tree.resolve('gm', t) == sorted([ls_tree.index[GM],
                                              ls_tree.index[GM_PART]])
    lut = ls.lut(100)
    assert lut[GM] == ls.index[GM] and lut[99] == 0    # unknown -> bg
    try:
        LabelSpace.build(t, {BG, 77}, unassigned='error')
        raise AssertionError('accepted a label missing from the tree')
    except ValueError:
        pass
    assert LabelSpace.from_dict(ls.to_dict()).values == ls.values


def test_bspline_convex_and_patch_consistent():
    torch.manual_seed(0)
    ctrl = torch.rand(5, 5, 5, 1) * 2 - 1              # in [-1, 1]
    shape = (30, 20, 25)
    full = bspline.evaluate(ctrl, *[bspline.axis_weights(
        5, torch.arange(n), n) for n in shape])
    assert full.min() >= -1 - 1e-5 and full.max() <= 1 + 1e-5
    part = bspline.evaluate(ctrl, *[bspline.axis_weights(
        5, torch.arange(a, a + 7), n) for a, n in zip((3, 4, 5), shape)])
    assert torch.allclose(part, full[3:10, 4:11, 5:12], atol=1e-6)
    const = bspline.evaluate(torch.full((4, 4, 4, 1), 0.3), *[
        bspline.axis_weights(4, torch.arange(n), n) for n in shape])
    assert torch.allclose(const, torch.full_like(const, 0.3), atol=1e-6)
