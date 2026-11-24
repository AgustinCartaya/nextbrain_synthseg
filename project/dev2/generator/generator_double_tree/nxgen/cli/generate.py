"""
Online launcher: write synthetic images as an nnU-Net v2 raw dataset.

    python -m nxgen.cli.generate --templates templates/ --config config.json \
        -n 100 --out data/outputs --dataset-id 1 --dataset-name NextBrain \
        [--seed 0] [--start 0] [--device cuda:0] [--overwrite] \
        [--save-steps all | --save-steps pv_seg synthesis ...] [--profile]

Output (<out>/Dataset<ID>_<NAME>/):
    imagesTr/<prefix>_<i>_0000.nii.gz   image, float32 (5-digit index i)
    labelsTr/<prefix>_<i>.nii.gz        segmentation of the principal level,
                                        label indices (uint16)
    levels/<level>/<prefix>_<i>.nii.gz  segmentation of every level (merging
                                        section), label indices (uint16)
    plans/<prefix>_<i>.json             every parameter of the image (the
                                        plan), the normalisation values and
                                        the config hash
    steps/<prefix>_<i>/                 intermediate steps (--save-steps)
    label_space.json                    index <-> raw value and name of the
                                        fine labels (MT). Created by the
                                        first run, then reused, so indices
                                        never change in a dataset
    label_spaces/<level>.json           same for every level (a later run
                                        whose levels differ raises an error)
    configs/<hash>/config.json, tree.json, tree_<level>.json
                                        every config used in the dataset
                                        (plans point to theirs by hash)
    dataset.json                        nnU-Net v2 description (principal
                                        level)

Image i uses seed image_seed(--seed, i). An interrupted run is resumed by
running the same command (existing cases are skipped). Image i is
regenerated with `--start i -n 1 --overwrite`. To replay an edited plan, use
the Python API (`Generator.render(plan)`).
"""

import argparse
import glob
import json
import os
import shutil

import torch

from ..core.config import load_config
from ..core.io import save_image, save_json, save_labels
from ..core.label_space import LabelSpace
from ..core.recorder import Recorder
from ..core.rng import image_seed
from ..online.acquisition import describe
from ..pipeline import Generator


def build_parser():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[1],
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('--templates', nargs='+', required=True,
                   help='HR template files, or folders of *.nii.gz')
    p.add_argument('--config', required=True, help='config JSON')
    p.add_argument('-n', type=int, required=True, help='number of images')
    p.add_argument('--out', required=True, help='output root folder')
    p.add_argument('--dataset-id', type=int, default=1)
    p.add_argument('--dataset-name', default='Synthetic')
    p.add_argument('--prefix', default='synth', help='case name prefix')
    p.add_argument('--seed', type=int, default=0, help='run seed')
    p.add_argument('--start', type=int, default=0, help='first image index')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available()
                   else 'cpu')
    p.add_argument('--overwrite', action='store_true',
                   help='regenerate cases that already exist')
    p.add_argument('--save-steps', nargs='+', default=None,
                   help="intermediate steps to save ('all' or names, see "
                        'nxgen.core.recorder)')
    p.add_argument('--profile', action='store_true',
                   help='print time and peak GPU memory of every step')
    return p


def template_paths(items):
    """Files and folders -> sorted list of NIfTI files."""
    paths = []
    for item in items:
        if os.path.isdir(item):
            paths += sorted(glob.glob(os.path.join(item, '*.nii*')))
        else:
            paths.append(item)
    return paths


def main(argv=None):
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config)
    root = os.path.join(args.out,
                        f'Dataset{args.dataset_id:03d}_{args.dataset_name}')
    for sub in ('imagesTr', 'labelsTr', 'plans'):
        os.makedirs(os.path.join(root, sub), exist_ok=True)

    # frozen label space: built by the first run, reused afterwards
    ls_path = os.path.join(root, 'label_space.json')
    label_space = LabelSpace.load(ls_path) if os.path.exists(ls_path) \
        else None
    gen = Generator(template_paths(args.templates), cfg, label_space,
                    args.device)
    if label_space is None:
        gen.labels.save(ls_path)
    os.makedirs(os.path.join(root, 'label_spaces'), exist_ok=True)
    for name, level in gen.levels.items():
        os.makedirs(os.path.join(root, 'levels', name), exist_ok=True)
        path = os.path.join(root, 'label_spaces', f'{name}.json')
        if not os.path.exists(path):
            level.labels.save(path)
        elif LabelSpace.load(path).to_dict() != level.labels.to_dict():
            raise ValueError(f'Level {name!r}: its labels differ from '
                             f'{path} (the dataset was made with another '
                             f'mapping or tree)')

    # every config used in the dataset, by hash
    cdir = os.path.join(root, 'configs', gen.config_hash)
    os.makedirs(cdir, exist_ok=True)
    save_json(cfg, os.path.join(cdir, 'config.json'))
    shutil.copy(cfg['labels']['tree'], os.path.join(cdir, 'tree.json'))
    for name, spec in cfg['merging']['levels'].items():
        if spec is not None:
            shutil.copy(spec['tree'], os.path.join(cdir, f'tree_{name}.json'))

    steps = None
    if args.save_steps:
        steps = 'all' if args.save_steps == ['all'] else args.save_steps
    for i in range(args.start, args.start + args.n):
        case = f'{args.prefix}_{i:05d}'
        files = [os.path.join(root, 'imagesTr', f'{case}_0000.nii.gz'),
                 os.path.join(root, 'labelsTr', f'{case}.nii.gz'),
                 os.path.join(root, 'plans', f'{case}.json')]
        level_files = {name: os.path.join(root, 'levels', name,
                                          f'{case}.nii.gz')
                       for name in gen.levels}
        files += list(level_files.values())
        if not args.overwrite and all(os.path.exists(f) for f in files):
            print(f'{case}: exists, skipped')
            continue
        rec = Recorder(steps, os.path.join(root, 'steps', case)
                       if steps else None, args.profile)
        s = gen(image_seed(args.seed, i), recorder=rec)
        save_image(s.image, s.affine, files[0])
        save_labels(s.seg, s.affine, files[1])
        for name, path in level_files.items():
            save_labels(s.segs[name], s.affine, path)
        save_json(dict(case=case, run_seed=args.seed, index=i,
                       plan=s.plan, normalization=s.normalization,
                       timings=rec.timings or None), files[2])
        acq = s.plan['acquisition']
        spacing = [round(x, 2) for x in acq['spacing_mm']] if acq else '-'
        print(f'{case}: {list(s.seg.shape)} voxels, acquisition '
              f'{describe(acq, gen.voxel_mm)[0]} {spacing} mm, '
              f'{len(s.plan["tree"]["class_names"])} generation classes')
        if args.profile:
            for name, t in rec.timings.items():
                peak = '' if t['peak_gb'] is None else \
                    f', peak {t["peak_gb"]:.2f} GB'
                print(f'    {name:15s} {t["time_s"]:7.3f} s{peak}')

    main = gen.levels[gen.principal].labels
    n_cases = len(glob.glob(os.path.join(root, 'labelsTr', '*.nii.gz')))
    with open(os.path.join(root, 'dataset.json'), 'w') as f:
        json.dump(dict(channel_names={'0': 'synthetic'},
                       labels={n: i for i, n in enumerate(main.names)},
                       numTraining=n_cases, file_ending='.nii.gz',
                       original_label_values=main.values,
                       level=gen.principal), f, indent=2)


if __name__ == '__main__':
    main()
