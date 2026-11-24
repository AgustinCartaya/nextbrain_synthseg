"""Label levels: online merge of the augmented labels (nxgen.core.merging)."""

import json
import os

import torch

from phantoms import BG, CSF, GM, GM_PART, WM, config, workdir, write_inputs
from nxgen.core.recorder import Recorder
from nxgen.online import partial_volume

# coarse level: gm_part merged into gm; one colour per class
COARSE_TREE = {
    'version': 'coarse', 'description': 'phantom coarse tree',
    'colors': ['CSF', 'GM', 'WM'], 'contrasts': ['T1', 'T2'], 'root': 'root',
    'nodes': {
        'root': {'parent': None, 'children': ['background', 'csf', 'gm', 'wm'],
                 'level': 0, 'labels': []},
        'background': {'parent': 'root', 'children': [], 'level': 1,
                       'labels': [0], 'null': True},
        'csf': {'parent': 'root', 'children': [], 'level': 1, 'labels': [1],
                'w_mu': [1, 0, 0]},
        'gm': {'parent': 'root', 'children': [], 'level': 1, 'labels': [2],
               'w_mu': [0, 1, 0]},
        'wm': {'parent': 'root', 'children': [], 'level': 1, 'labels': [3],
               'w_mu': [0, 0, 1]},
    },
}
COARSE_MAP = {BG: 0, CSF: 1, GM: 2, GM_PART: 2, WM: 3}

SCRIPT = '''
MAPPING = {mapping}

def label_mapping(drop=None):
    return {{k: v for k, v in MAPPING.items() if k != drop}}
'''


def level_inputs(folder):
    """Phantom template + MT, and the coarse merging script + tree."""
    tpl, mt = write_inputs(folder)
    script = os.path.join(folder, 'to_coarse.py')
    with open(script, 'w') as f:
        f.write(SCRIPT.format(mapping=COARSE_MAP))
    tree = os.path.join(folder, 'coarse_tree.json')
    with open(tree, 'w') as f:
        json.dump(COARSE_TREE, f)
    return tpl, mt, script, tree


def double_generator(device='cpu', options=None, **sections):
    from nxgen.pipeline import Generator
    tpl, mt, script, tree = level_inputs(workdir())
    levels = {'all': None,
              'coarse': {'script': script, 'tree': tree,
                         'options': options or {}}}
    cfg = config(mt, merging={'levels': levels, 'principal': 'coarse'},
                 **sections)
    return Generator([tpl], cfg, device=device)


def test_levels_and_intensity_tree(device='cpu'):
    gen = double_generator(device)
    assert list(gen.levels) == ['all', 'coarse']           # principal last
    assert gen.principal == 'coarse'
    assert gen.itree.version == 'coarse' and gen.tree.version == 'test'
    coarse = gen.levels['coarse']
    assert coarse.labels.values == [0, 1, 2, 3]
    # fine index -> coarse index follows the mapping
    for i, v in enumerate(gen.labels.values):
        assert coarse.labels.values[int(coarse.lut[i])] == COARSE_MAP[v]
    # one Gaussian per coarse label (the image uses IT)
    plan = gen.sample_plan(0)
    assert len(plan['tree']['mu']) == len(coarse.labels)


def test_one_segmentation_per_level(device='cpu'):
    gen = double_generator(device)
    for seed in range(3):
        s = gen(seed)
        assert set(s.segs) == {'all', 'coarse'}
        assert s.seg is s.segs['coarse']
        for name, seg in s.segs.items():
            assert seg.shape == s.image.shape and seg.dtype == torch.int16
            assert int(seg.max()) < len(gen.levels[name].labels)
        assert int(s.segs['all'].max()) > 3        # fine level is not merged


def test_merge_before_partial_volume(device='cpu'):
    """Level seg = PV of the merged HR labels, ties -> class of the fine seg."""
    gen = double_generator(device)
    s = gen(4, recorder=Recorder(steps=['spatial', 'merge_coarse']))
    hr = s.recorder.steps['spatial'].to(device)
    lut = gen.levels['coarse'].lut
    merged = lut[hr.long()]
    assert torch.equal(s.recorder.steps['merge_coarse'].to(device), merged)
    pv_all = partial_volume.apply(hr, 'downsample', gen.factor, 1)
    assert torch.equal(pv_all['seg'], s.segs['all'])
    pv = partial_volume.apply(merged, 'downsample', gen.factor, 1,
                              prefer=lut[pv_all['seg'].long()])
    assert torch.equal(pv['seg'], s.segs['coarse'])


def test_levels_agree_except_where_a_class_covers_more(device='cpu'):
    """coarse seg == lut[fine seg], unless the coarse class has a strictly
    larger fraction than the class of the fine winner."""
    gen = double_generator(device)
    lut = gen.levels['coarse'].lut
    for seed in range(4):
        s = gen(seed, recorder=Recorder(steps=['spatial']))
        hr = s.recorder.steps['spatial'].to(device)
        relabel = lut[s.segs['all'].long()]
        differ = relabel != s.segs['coarse']
        full = partial_volume.apply(lut[hr.long()], 'downsample', gen.factor,
                                    gen.factor ** 3)
        frac = lambda c: ((full['index'] == c[None]) * full['weight']).sum(0)
        assert (frac(s.segs['coarse'])[differ] >
                frac(relabel)[differ] + 1e-6).all()


def test_prefer_breaks_ties_only():
    # 4 + 4 voxels: a tie; prefer decides. 5 + 3: the majority wins anyway
    tie = torch.tensor([1] * 4 + [2] * 4, dtype=torch.int16).view(2, 2, 2)
    maj = torch.tensor([1] * 5 + [2] * 3, dtype=torch.int16).view(2, 2, 2)
    two = torch.full((1, 1, 1), 2)
    assert int(partial_volume.apply(tie, 'downsample', 2, 1)['seg']) == 1
    assert int(partial_volume.apply(tie, 'downsample', 2, 1,
                                    prefer=two)['seg']) == 2
    assert int(partial_volume.apply(maj, 'downsample', 2, 1,
                                    prefer=two)['seg']) == 1


def test_same_plan_same_segmentations(device='cpu'):
    gen = double_generator(device)
    plan = gen.sample_plan(9)
    a, b = gen.render(plan), gen.render(json.loads(json.dumps(plan)))
    for name in gen.levels:
        assert torch.equal(a.segs[name], b.segs[name])
    assert torch.allclose(a.image, b.image, atol=1e-5)


def test_label_missing_from_mapping_raises():
    try:
        double_generator(options={'drop': GM_PART})
    except ValueError as e:
        assert str(GM_PART) in str(e)
    else:
        raise AssertionError('a label missing from the mapping must raise')


def test_unknown_principal_raises():
    from nxgen.core.config import load_config
    try:
        load_config(merging={'levels': {'a': None}, 'principal': 'b'})
    except ValueError as e:
        assert 'principal' in str(e)
    else:
        raise AssertionError('an unknown principal level must raise')


def test_no_levels_is_the_single_tree_generator(device='cpu'):
    from phantoms import generator
    gen = generator(device)
    assert list(gen.levels) == ['all'] and gen.itree is gen.tree
    s = gen(1)
    assert torch.equal(s.seg, s.segs['all'])


def test_training_stream_levels(device='cpu'):
    from nxgen.training import SyntheticStream
    gen = double_generator(device, patch={'size': [8, 8, 8]})
    loader = torch.utils.data.DataLoader(
        SyntheticStream(gen, return_levels=True), batch_size=2)
    image, segs = next(iter(loader))
    assert set(segs) == {'all', 'coarse'}
    for g in segs.values():
        assert g.shape == image.shape and g.dtype == torch.int64
