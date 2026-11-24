"""Write a leave-one-template-out 5-fold split for one nnU-Net dataset.

The synthetic images come from 5 label templates (case-1..case-5; image i uses
template i % 5). A random split would put images of the same anatomy in train
and val, so fold k validates on every image of template k and trains on the rest.
The template of each image is read from its generator plan (plans/synth_XXXXX.json),
so all the datasets (same images, different GT) get exactly the same split.

    python make_splits.py --dataset Dataset102_nxb18
"""
import argparse
import json
import os
from collections import defaultdict
from pathlib import Path


def template_name(path):
    return Path(path).name.split('_')[0]          # case-1_hr0.2mm.nii.gz -> case-1


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--dataset', default=os.environ.get('DATASET_NAME'))
    p.add_argument('--plans_dir', default=os.environ.get('PLANS_DIR'))
    args = p.parse_args()

    raw = Path(os.environ['nnUNet_raw']) / args.dataset
    pre = Path(os.environ['nnUNet_preprocessed']) / args.dataset
    plans = Path(args.plans_dir)

    cases = sorted(f.name[:-len('.nii.gz')] for f in (raw / 'labelsTr').glob('*.nii.gz'))
    n_declared = json.loads((raw / 'dataset.json').read_text())['numTraining']
    assert len(cases) == n_declared, f'{len(cases)} labels but numTraining={n_declared}: rerun generate_data/generate.py --finalize'

    by_template = defaultdict(list)
    for c in cases:
        plan = json.loads((plans / f'{c}.json').read_text())
        assert plan['case'] == c, (plan['case'], c)
        by_template[template_name(plan['template'])].append(c)

    templates = sorted(by_template)
    splits = []
    for t in templates:
        val = set(by_template[t])
        splits.append({'train': [c for c in cases if c not in val], 'val': sorted(val)})
        print(f'fold {len(splits) - 1}: val = {t} ({len(val)} cases), train = {len(cases) - len(val)}')

    pre.mkdir(parents=True, exist_ok=True)
    (pre / 'splits_final.json').write_text(json.dumps(splits, indent=2))
    (pre / 'splits_templates.json').write_text(json.dumps(dict(enumerate(templates)), indent=2))
    print(f'Wrote {pre / "splits_final.json"}')


if __name__ == '__main__':
    main()
