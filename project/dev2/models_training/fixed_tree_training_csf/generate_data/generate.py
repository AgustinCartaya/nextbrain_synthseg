"""
1000 synthetic images -> two nnU-Net v2 raw datasets, one per label level.

    conda activate nxb
    python generate.py                           # 1000 images on every GPU
    python generate.py --gpus 0 2                # only GPUs 0 and 2
    python generate.py -n 8 --out /tmp/try/generation --models /tmp/try/models   # quick try
    ./launch.sh                                  # same as the first, in the
                                                 # background, log in logs/

The generator (generator_double_tree/nxgen) augments one fine label map per
image (morphology with MT, then spatial), merges the augmented labels into
the two levels of config.json (nxb_all = identity, nxb18), gives each level
its own partial volume (0.4 mm GT) and synthesises the image from the
principal level nxb18 with its tree (IT). So every image has two ground
truths that agree voxel by voxel, and the same image is used by the two
nnU-Nets.

The label maps L are the nxbsimplified maps with ventricles (50) and CSF (24)
added (simplified_trees_and_labels/data/outputs/nxbsimplified_csf), and MT is
the nxbsimplified tree with the ventricles_of_forebrain_l1 branch (ventricles,
csf) added back.

Steps
-----
1. HR templates: the label maps of --labels (L, 0.2 mm) re-cropped with a
   background margin of template.margin_mm (12 mm). The maps of L touch
   their borders; the margin keeps the anatomy inside the field of view
   after rotation / zoom / elastic. Built once, cached in <out>/templates.
2. Images: one worker process per GPU. Image i uses seed
   image_seed(--seed, i) and template i % n_templates, so the result does
   not depend on the number of GPUs, and a run is resumed by running the
   same command (finished cases are skipped).
3. dataset.json of every dataset (written at the end, also by --finalize).

Output
------
--out (default ../data_shared/generation), shared by the two nets:
    templates/                          HR templates (+ build records)
    images/synth_<i>.nii.gz             image in [0, 1] (--dtype, uint8)
    labels/<net>/synth_<i>.nii.gz       GT of each level: label INDICES
                                        0..N-1 (what nnU-Net needs)
    plans/synth_<i>.json                every random parameter of the image
    config.json, trees/                 config and trees used
--models (default ../models), one nnU-Net raw dataset per level (--ids, --names):
    <net>/nnUNet_raw/Dataset<ID>_<net>/
        imagesTr/synth_<i>_0000.nii.gz  hard link to images/ (no extra disk)
        labelsTr -> labels/<net>/       symlink to the shared labels
        label_space.json                index <-> raw value and name (to map
                                        predictions back to raw labels)
        dataset.json                    nnU-Net description
The level nxb_all of the config is the net nxbsimplified_csf.

Train: see ../README.md (source ../env.sh <net>, then 01/02/03_*.sh).
"""

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent          # fixed_tree_training_csf/generate_data
TRAIN_ROOT = HERE.parent                        # fixed_tree_training_csf
DEV2 = HERE.parents[2]
NXGEN = DEV2 / 'generator' / 'generator_double_tree'
sys.path.insert(0, str(NXGEN))

LABELS = DEV2 / 'simplified_trees_and_labels' / 'data' / 'outputs' / 'nxbsimplified_csf' / 'labels'
LEVELS = ['nxb_all', 'nxb18']                  # generator levels (config.json)
NETS = ['nxbsimplified_csf', 'nxb18']          # net of each level (models/<net>)


def build_parser():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[1],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config', default=str(HERE / 'config.json'))
    p.add_argument('--labels', default=str(LABELS), help='folder of label maps L')
    p.add_argument('--out', default=str(TRAIN_ROOT / 'data_shared' / 'generation'),
                   help='shared data: images, labels, plans, templates, trees')
    p.add_argument('--models', default=str(TRAIN_ROOT / 'models'),
                   help='one folder per net, the raw datasets go to <net>/nnUNet_raw')
    p.add_argument('-n', type=int, default=5, help='number of images')
    p.add_argument('--start', type=int, default=0, help='first image index')
    p.add_argument('--seed', type=int, default=0, help='run seed')
    p.add_argument('--gpus', nargs='+', default=None,
                   help='GPU indices, one worker each (default: every GPU)')
    p.add_argument('--dtype', default='uint8', choices=['uint8', 'uint16', 'float32'],
                   help='image storage; uint8 steps (1/255) are below the noise')
    p.add_argument('--ids', nargs=len(LEVELS), type=int, default=[101, 102],
                   help=f'nnU-Net dataset ids of {LEVELS}')
    p.add_argument('--names', nargs=len(LEVELS), default=NETS,
                   help=f'nnU-Net dataset names of {LEVELS}')
    p.add_argument('--overwrite', action='store_true', help='regenerate existing cases')
    p.add_argument('--finalize', action='store_true',
                   help='only (re)write the dataset.json files')
    # internal: one worker
    p.add_argument('--worker', type=int, default=None, help=argparse.SUPPRESS)
    p.add_argument('--n-workers', type=int, default=1, help=argparse.SUPPRESS)
    p.add_argument('--device', default=None, help=argparse.SUPPRESS)
    return p


def dataset_dirs(args):
    """level -> models/<net>/nnUNet_raw/Dataset<ID>_<name>"""
    return {lvl: Path(args.models) / net / 'nnUNet_raw' / f'Dataset{i:03d}_{n}'
            for lvl, net, i, n in zip(LEVELS, NETS, args.ids, args.names)}


def label_dirs(args):
    """level -> <out>/labels/<net> (what labelsTr links to)"""
    return {lvl: Path(args.out) / 'labels' / net for lvl, net in zip(LEVELS, NETS)}


def case_name(i):
    return f'synth_{i:05d}'


# ----------------------------------------------------------------- templates
def build_templates(args, cfg):
    from nxgen.core.tree import load_tree
    from nxgen.offline.template import build_template
    paths = sorted(glob.glob(os.path.join(args.labels, '*.nii*')))
    if not paths:
        sys.exit(f'No label maps in {args.labels}')
    tree = load_tree(cfg['labels']['tree'])
    allowed = list(tree.label_to_node) if cfg['labels']['unassigned'] == 'background' else None
    t = cfg['template']
    out = []
    for p in paths:
        path, rec = build_template(p, os.path.join(args.out, 'templates'), t['hr_voxel_mm'],
                                   t['sigma_mm'], t['margin_mm'], allowed,
                                   cfg['labels']['background'], 'cpu')
        print(f'    {os.path.basename(path)}: {rec["shape"]} voxels'
              + (f', vanished {rec["vanished_labels"]}' if rec['vanished_labels'] else ''))
        out.append(path)
    return out


# -------------------------------------------------------------------- worker
def worker(args):
    import torch
    from nxgen import Generator, load_config
    from nxgen.core.io import save_image, save_json, save_labels
    from nxgen.core.rng import image_seed

    cfg = load_config(args.config)
    templates = sorted(glob.glob(os.path.join(args.out, 'templates', '*.nii.gz')))
    gen = Generator(templates, cfg, device=args.device)
    if list(gen.levels) != [l for l in LEVELS if l != gen.principal] + [gen.principal]:
        sys.exit(f'config levels {list(gen.levels)} != {LEVELS}')
    out, dirs, ldirs = Path(args.out), dataset_dirs(args), label_dirs(args)
    tag = f'[worker {args.worker} {args.device}]'
    from nxgen.core.label_space import LabelSpace
    for l, d in dirs.items():        # indices must be those of label_space.json
        if LabelSpace.load(str(d / 'label_space.json')).to_dict() != gen.levels[l].labels.to_dict():
            sys.exit(f'{tag} level {l}: labels differ from {d / "label_space.json"}')

    todo = [i for i in range(args.start, args.start + args.n) if i % args.n_workers == args.worker]
    t_start, done = time.time(), 0
    for i in todo:
        case = case_name(i)
        image = out / 'images' / f'{case}.nii.gz'
        plan = out / 'plans' / f'{case}.json'
        links = {l: d / 'imagesTr' / f'{case}_0000.nii.gz' for l, d in dirs.items()}
        segs = {l: d / f'{case}.nii.gz' for l, d in ldirs.items()}
        files = [image, plan, *links.values(), *segs.values()]
        if not args.overwrite and all(f.exists() for f in files):
            continue
        t0 = time.time()
        s = gen(image_seed(args.seed, i), template=i % len(gen.templates))
        # labels first, plan last: the plan marks a finished case
        for l, path in segs.items():
            save_labels(s.segs[l], s.affine, str(path))
        save_image(s.image, s.affine, str(image), args.dtype)
        for path in links.values():
            path.unlink(missing_ok=True)
            try:
                os.link(image, path)
            except OSError:
                shutil.copy(image, path)
        save_json(dict(case=case, run_seed=args.seed, index=i,
                       template=templates[i % len(templates)], principal=gen.principal,
                       config_hash=gen.config_hash, plan=s.plan,
                       normalization=s.normalization), str(plan))
        done += 1
        acq = s.plan['acquisition']
        spacing = [round(x, 2) for x in acq['spacing_mm']] if acq else [gen.voxel_mm] * 3
        eta = (time.time() - t_start) / done * (len(todo) - todo.index(i) - 1)
        print(f'{tag} {case}: {list(s.seg.shape)}, {time.time() - t0:.1f} s, '
              f'{acq.get("contrast") if acq else "-"}, spacing {spacing} mm, '
              f'ETA {eta / 60:.0f} min', flush=True)
        del s
        torch.cuda.empty_cache()
    print(f'{tag} finished: {done} images in {(time.time() - t_start) / 60:.1f} min', flush=True)


# ------------------------------------------------------------------ datasets
def prepare(args, cfg):
    """Folders, label spaces, config copy. Errors if an existing dataset
    was made with other labels."""
    import numpy as np
    from nxgen.core.label_space import LabelSpace
    from nxgen.core.tree import load_tree
    from nxgen.core.merging import build_levels
    from nxgen.core.io import load_labels
    out = Path(args.out)
    for sub in ['images', 'plans', 'trees']:
        (out / sub).mkdir(parents=True, exist_ok=True)
    # label spaces exactly as the Generator builds them (templates' labels)
    tree = load_tree(cfg['labels']['tree'])
    values = set()
    for t in sorted(glob.glob(str(out / 'templates' / '*.nii.gz'))):
        values |= set(np.unique(load_labels(t)[0]).tolist())
    fine = LabelSpace.build(tree, values, cfg['labels']['background'],
                            cfg['labels']['output_labels'], cfg['labels']['unassigned'])
    levels, principal = build_levels(cfg, tree, fine)
    ldirs = label_dirs(args)
    for lvl, d in dataset_dirs(args).items():
        (d / 'imagesTr').mkdir(parents=True, exist_ok=True)
        ldirs[lvl].mkdir(parents=True, exist_ok=True)
        link = d / 'labelsTr'
        if not link.is_symlink():
            if link.exists():
                sys.exit(f'{link} exists and is not a link to {ldirs[lvl]}')
            link.symlink_to(os.path.relpath(ldirs[lvl], d))
        elif link.resolve() != ldirs[lvl].resolve():
            sys.exit(f'{link} points to {link.resolve()}, expected {ldirs[lvl]}')
        path = d / 'label_space.json'
        space = levels[lvl].labels
        if not path.exists():
            space.save(str(path))
        elif LabelSpace.load(str(path)).to_dict() != space.to_dict():
            sys.exit(f'{path}: the labels of level {lvl} changed (other mapping, tree or '
                     f'templates). Use another --out or --ids.')
    save_json_plain(cfg, out / 'config.json')
    shutil.copy(cfg['labels']['tree'], out / 'trees' / 'tree_mt.json')
    for lvl, spec in cfg['merging']['levels'].items():
        if spec:
            shutil.copy(spec['tree'], out / 'trees' / f'tree_{lvl}.json')
    return levels, principal


def save_json_plain(obj, path):
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2)


def finalize(args, cfg):
    from nxgen.core.label_space import LabelSpace
    dirs = dataset_dirs(args)
    principal = cfg['merging']['principal']
    for lvl, d in dirs.items():
        space = LabelSpace.load(str(d / 'label_space.json'))
        names, seen = [], set()
        for i, (n, v) in enumerate(zip(space.names, space.values)):
            n = 'background' if i == 0 else (n if n not in seen else f'{n}_{v}')
            seen.add(n)
            names.append(n)
        cases = sorted(p.name[:-len('.nii.gz')] for p in (d / 'labelsTr').glob('*.nii.gz'))
        cases = [c for c in cases if (Path(args.out) / 'plans' / f'{c}.json').exists()
                 and (d / 'imagesTr' / f'{c}_0000.nii.gz').exists()]
        save_json_plain(dict(
            channel_names={'0': 'synthetic'},
            labels={n: i for i, n in enumerate(names)},
            numTraining=len(cases),
            file_ending='.nii.gz',
            name=d.name,
            description=f'Synthetic images (generator_double_tree), level {lvl}; image '
                        f'synthesised from level {principal}. Labels are indices; '
                        f'label_space.json maps them to raw values.',
            level=lvl,
            original_label_values=space.values,
        ), d / 'dataset.json')
        print(f'    {d.name}: {len(cases)} cases, {len(names)} labels (level {lvl})')


# ---------------------------------------------------------------------- main
def main():
    args = build_parser().parse_args()
    if args.worker is not None:
        return worker(args)

    from nxgen import load_config
    cfg = load_config(args.config)
    assert list(cfg['merging']['levels']) == LEVELS, cfg['merging']['levels']
    if args.finalize:
        return finalize(args, cfg)

    print('1. HR templates')
    build_templates(args, cfg)
    levels, principal = prepare(args, cfg)
    for name, level in levels.items():
        tag = '  <- image synthesised from this level (IT)' if name == principal else ''
        print(f'    level {name:14s} {len(level.labels):4d} labels{tag}')

    import torch
    gpus = args.gpus or [str(i) for i in range(torch.cuda.device_count())]
    devices = [f'cuda:{g}' for g in gpus] or ['cpu']
    print(f'\n2. {args.n} images ({args.start}..{args.start + args.n - 1}) on {devices}')
    t0 = time.time()
    argv = [sys.executable, '-u', __file__, '--config', args.config, '--out', args.out,
            '--models', args.models, '--labels', args.labels,
            '-n', str(args.n), '--start', str(args.start), '--seed', str(args.seed),
            '--dtype', args.dtype, '--ids', *map(str, args.ids), '--names', *args.names,
            '--n-workers', str(len(devices))] + (['--overwrite'] if args.overwrite else [])
    procs = [subprocess.Popen(argv + ['--worker', str(w), '--device', dev])
             for w, dev in enumerate(devices)]
    codes = [p.wait() for p in procs]
    print(f'    done in {(time.time() - t0) / 60:.1f} min, worker exit codes {codes}')

    print('\n3. dataset.json')
    finalize(args, cfg)
    if any(codes):
        sys.exit('some workers failed: run the same command again to finish the missing cases')


if __name__ == '__main__':
    main()
