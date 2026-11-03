"""Fixed validation set + generator settings for training nnU-Net with nxgen.

The training images are generated on the fly by the trainer, but nnU-Net still needs a
dataset on disk to (1) plan the network (spacing, patch size, batch size, topology) and
(2) validate. This script writes that dataset: a few full synthetic volumes per template.

    python prepare_dataset.py raw [--n-per-template 4] [--seed 1000] [--device cuda:0]
        -> $nnUNet_raw/$DATASET_NAME/{imagesTr,labelsTr,plans,dataset.json,label_space.json,nxgen.json}
           case names: <template>_<j>, e.g. case-1_000

    (nnUNetv2_plan_and_preprocess)

    python prepare_dataset.py splits
        -> $nnUNet_preprocessed/$DATASET_NAME/splits_final.json   fold i validates on template i
           $nnUNet_preprocessed/$DATASET_NAME/nxgen.json          read by nnUNetTrainerNxGen
           $nnUNet_preprocessed/$DATASET_NAME/nxgen_config.json   generator config (absolute paths)
           $nnUNet_preprocessed/$DATASET_NAME/label_space.json
"""
import argparse
import json
import os
import shutil
from pathlib import Path

import torch

from nxgen import Generator, load_config
from nxgen.cli.generate import template_paths
from nxgen.core.io import save_image, save_json, save_labels
from nxgen.core.rng import image_seed


def template_name(path):
    return Path(path).name.split('_hr')[0].split('.')[0]


def make_raw(args):
    raw = Path(os.environ['nnUNet_raw']) / args.name
    for sub in ('imagesTr', 'labelsTr', 'plans'):
        (raw / sub).mkdir(parents=True, exist_ok=True)

    cfg = load_config(args.config)
    if cfg['patch']['size'] is not None:
        raise ValueError('the validation set needs full volumes: set patch.size to null in the config')
    templates = template_paths([args.templates])
    if not templates:
        raise FileNotFoundError(f'no templates in {args.templates} (run 00_build_templates.sh)')

    ls_path = raw / 'label_space.json'
    gen = Generator(templates, cfg, str(ls_path) if ls_path.exists() else None, args.device)
    if not ls_path.exists():
        gen.labels.save(str(ls_path))

    cases = {}
    for t, path in enumerate(templates):
        name = template_name(path)
        cases[name] = []
        for j in range(args.n_per_template):
            case = f'{name}_{j:03d}'
            cases[name].append(case)
            files = [raw / 'imagesTr' / f'{case}_0000.nii.gz', raw / 'labelsTr' / f'{case}.nii.gz',
                     raw / 'plans' / f'{case}.json']
            if all(f.exists() for f in files) and not args.overwrite:
                print(f'{case}: exists, skipped')
                continue
            # seeds of the validation set: image_seed(--seed, t * 1000 + j), independent of the training stream
            s = gen(image_seed(args.seed, t * 1000 + j), template=t)
            save_image(s.image, s.affine, str(files[0]))
            save_labels(s.seg, s.affine, str(files[1]))
            save_json(dict(case=case, plan=s.plan, normalization=s.normalization), str(files[2]))
            print(f'{case}: {list(s.seg.shape)} voxels')
            del s
            torch.cuda.empty_cache()

    with open(raw / 'dataset.json', 'w') as f:
        json.dump(dict(channel_names={'0': 'synthetic'},
                       labels={n: i for i, n in enumerate(gen.labels.names)},
                       numTraining=sum(len(c) for c in cases.values()), file_ending='.nii.gz',
                       original_label_values=gen.labels.values), f, indent=2)
    # everything the trainer needs to rebuild the same generator
    shutil.copy(cfg['labels']['tree'], raw / 'tree.json')
    save_json(cfg, str(raw / 'nxgen_config.json'))
    save_json(dict(templates=[os.path.abspath(p) for p in templates],
                   template_names=list(cases), cases=cases, val_seed=args.seed,
                   config=str(raw / 'nxgen_config.json'), label_space=str(ls_path)),
              str(raw / 'nxgen.json'))
    print(f'Wrote {raw}')


def make_splits(args):
    raw = Path(os.environ['nnUNet_raw']) / args.name
    pre = Path(os.environ['nnUNet_preprocessed']) / args.name
    if not pre.exists():
        raise FileNotFoundError(f'{pre} does not exist: run nnUNetv2_plan_and_preprocess first')
    info = json.load(open(raw / 'nxgen.json'))

    # copies, so training does not depend on the raw folder
    for f in ('nxgen_config.json', 'label_space.json', 'tree.json'):
        shutil.copy(raw / f, pre / f)
    cfg = json.load(open(pre / 'nxgen_config.json'))
    cfg['labels']['tree'] = str(pre / 'tree.json')
    json.dump(cfg, open(pre / 'nxgen_config.json', 'w'), indent=2)

    # fold i: the generator uses every template but template i, validation = images of template i
    names = info['template_names']
    all_cases = sorted(c for cs in info['cases'].values() for c in cs)
    splits, folds = [], {}
    for i, name in enumerate(names):
        val = sorted(info['cases'][name])
        splits.append({'train': [c for c in all_cases if c not in set(val)], 'val': val})
        folds[i] = name
        print(f'fold {i}: held-out template = {name} ({len(val)} validation cases), '
              f'generator templates = {[n for n in names if n != name]}')
    json.dump(splits, open(pre / 'splits_final.json', 'w'), indent=2)

    info = dict(info, config=str(pre / 'nxgen_config.json'), label_space=str(pre / 'label_space.json'),
                fold_holdout=folds)
    json.dump(info, open(pre / 'nxgen.json', 'w'), indent=2)
    print(f'Wrote {pre / "splits_final.json"} and {pre / "nxgen.json"}')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('stage', choices=['raw', 'splits'])
    p.add_argument('--name', default=os.environ.get('DATASET_NAME'))
    p.add_argument('--config', default=os.environ.get('NXGEN_CONFIG'))
    p.add_argument('--templates', default=os.environ.get('NXGEN_TEMPLATES'))
    p.add_argument('--n-per-template', type=int, default=4)
    p.add_argument('--seed', type=int, default=1000, help='seed of the validation set')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--overwrite', action='store_true')
    args = p.parse_args()
    make_raw(args) if args.stage == 'raw' else make_splits(args)


if __name__ == '__main__':
    main()
