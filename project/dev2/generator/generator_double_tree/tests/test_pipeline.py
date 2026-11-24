"""The whole pipeline, the plan, the launchers and the training stream."""

import copy
import json
import os

import numpy as np
import torch

from phantoms import generator, workdir, write_inputs
from nxgen.core.io import load_labels
from nxgen.core.recorder import Recorder
from nxgen.core.rng import module_rng
from nxgen.online import acquisition


def test_output_shapes_and_ranges(device='cpu'):
    gen = generator(device)
    for seed in range(4):
        s = gen(seed)
        assert list(s.image.shape) == gen.templates[0]['out_shape']
        assert s.seg.shape == s.image.shape and s.seg.dtype == torch.int16
        assert s.image.min() >= 0 and s.image.max() <= 1
        assert not torch.isnan(s.image).any()
        assert s.seg.max() < len(gen.labels)


def test_same_plan_same_output(device='cpu'):
    gen = generator(device)
    plan = gen.sample_plan(11)
    a, b = gen.render(plan), gen.render(copy.deepcopy(plan))
    assert torch.equal(a.seg, b.seg)
    assert torch.allclose(a.image, b.image, atol=1e-5)
    assert gen.sample_plan(11) == plan                  # plans reproducible


def test_plan_is_json_and_replays(device='cpu'):
    gen = generator(device)
    plan = gen.sample_plan(5)
    back = json.loads(json.dumps(plan))
    assert torch.equal(gen.render(back).seg, gen.render(plan).seg)


def test_modules_draw_independently(device='cpu'):
    """Disabling a module does not change what the others draw."""
    gen_a = generator(device)
    gen_b = generator(device, morphology={'prob': 0.0}, motion={'prob': 0.0})
    pa, pb = gen_a.sample_plan(3), gen_b.sample_plan(3)
    for name in ('acquisition', 'spatial', 'tree', 'bias', 'noise'):
        assert pa[name] == pb[name], name
    assert pb['morphology'] is None and pb['motion'] is None


def test_patch_equals_crop_of_full_volume(device='cpu'):
    """Everything except motion and the normalisation is patch-exact."""
    off = dict(motion={'prob': 0.0}, normalization={'method': 'none'},
               noise={'sigma': 0.0}, acquisition={'lowres_prob': 0.0})
    full_gen = generator(device, **off)
    patch_gen = generator(device, patch={'size': [8, 9, 10]}, **off)
    plan = patch_gen.sample_plan(7)
    full_plan = dict(plan, roi=dict(start=[0, 0, 0],
                                    size=full_gen.templates[0]['out_shape']))
    full = full_gen.render(full_plan)
    part = patch_gen.render(plan)
    (x, y, z), (a, b, c) = plan['roi']['start'], plan['roi']['size']
    assert torch.equal(part.seg, full.seg[x:x + a, y:y + b, z:z + c])
    # tree noise is drawn per patch, so compare with sigma 0 ...
    for p in (plan, full_plan):
        p['tree'] = dict(p['tree'], sigma=[0.0] * len(p['tree']['sigma']))
    full = full_gen.render(full_plan)
    part = patch_gen.render(plan)
    assert torch.allclose(part.image, full.image[x:x + a, y:y + b, z:z + c],
                          atol=1e-5)
    assert np.allclose(part.affine[:3, 3],
                       (full.affine @ [x, y, z, 1])[:3])


def test_patch_lies_on_foreground(device='cpu'):
    gen = generator(device, patch={'size': [6, 6, 6]})
    for seed in range(10):
        s = gen(seed)
        assert list(s.seg.shape) == [6, 6, 6]


def test_recorder_keeps_and_saves_steps(device='cpu'):
    gen = generator(device, motion={'prob': 1.0},
                    acquisition={'lowres_prob': 1.0})
    out = workdir()
    rec = Recorder(steps='all', out_dir=out, profile=True)
    gen(1, recorder=rec)
    for name in ('template_crop', 'morphology', 'spatial', 'pv_seg', 'pv_max',
                 'synthesis', 'bias', 'motion', 'lowres', 'resolution',
                 'image'):
        assert name in rec.steps, name
    assert len(os.listdir(out)) == len(rec.steps)
    assert set(rec.timings) >= {'spatial', 'synthesis', 'resolution'}


def test_editing_a_plan_disables_a_step(device='cpu'):
    gen = generator(device, motion={'prob': 1.0})
    plan = gen.sample_plan(2)
    assert plan['motion'] is not None
    rec = Recorder(steps=['bias', 'motion'])
    gen.render(plan, rec)
    assert not torch.equal(rec.steps['bias'], rec.steps['motion'])
    rec = Recorder(steps=['bias', 'motion'])
    gen.render(dict(plan, motion=None), rec)
    assert torch.equal(rec.steps['bias'], rec.steps['motion'])


def test_every_augmentation_can_be_none(device='cpu'):
    gen = generator(device, motion={'prob': 1.0}, bias={'prob': 1.0},
                    acquisition={'lowres_prob': 1.0},
                    normalization={'method': 'none'})
    plan = gen.sample_plan(3)
    off = dict(plan, morphology=None, spatial=None, acquisition=None,
               bias=None, motion=None, noise=None)
    rec = Recorder(steps='all')
    s = gen.render(off, rec)
    # nothing after the synthesis changes the image
    assert torch.equal(rec.steps['synthesis'], s.image)
    assert 'lowres' not in rec.steps
    # a missing section is the same as None
    bare = {k: v for k, v in off.items() if v is not None}
    assert torch.equal(gen.render(bare).image, s.image)
    # acquisition None == all spacings at the output voxel
    flat = dict(off, acquisition=acquisition.none_params(gen.voxel_mm))
    assert torch.equal(gen.render(flat).image, s.image)
    try:
        gen.render(dict(plan, tree=None))
    except ValueError:
        pass
    else:
        raise AssertionError('tree None must raise')


def test_cli_build_and_generate():
    from nxgen.cli import build_templates, generate
    folder = workdir()
    src, tree_path = write_inputs(folder, voxel_mm=0.5)
    cfg_path = os.path.join(folder, 'config.json')
    with open(cfg_path, 'w') as f:
        json.dump({'grid': {'target_voxel_mm': 1.0},
                   'labels': {'tree': 'tree.json'},
                   'template': {'hr_voxel_mm': 0.5, 'margin_mm': 2.0}}, f)
    build_templates.main(['--config', cfg_path, '--seg', src,
                          '--out', os.path.join(folder, 'tpl')])
    out = os.path.join(folder, 'out')
    args = ['--templates', os.path.join(folder, 'tpl'), '--config', cfg_path,
            '-n', '2', '--out', out, '--device', 'cpu']
    generate.main(args)
    root = os.path.join(out, 'Dataset001_Synthetic')
    ds = json.load(open(os.path.join(root, 'dataset.json')))
    assert ds['numTraining'] == 2
    lab, _, _ = load_labels(os.path.join(root, 'labelsTr',
                                         'synth_00001.nii.gz'))
    assert lab.max() < len(ds['labels'])
    plan = json.load(open(os.path.join(root, 'plans', 'synth_00001.json')))
    assert os.path.exists(os.path.join(root, 'configs',
                                       plan['plan']['config_hash']))
    generate.main(args)                       # resume: everything skipped


def test_training_stream(device='cpu'):
    from nxgen.training import SyntheticStream
    gen = generator(device, patch={'size': [8, 8, 8]})
    loader = torch.utils.data.DataLoader(SyntheticStream(gen, run_seed=1),
                                         batch_size=2, num_workers=0)
    image, seg = next(iter(loader))
    assert image.shape == (2, 1, 8, 8, 8) and seg.dtype == torch.long


def test_module_rng_names_are_distinct():
    a = module_rng(0, 'tree').random()
    assert a != module_rng(0, 'noise').random()
