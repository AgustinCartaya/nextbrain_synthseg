"""Share the preprocessed images of the datasets (same images, different GT).

nnU-Net stores each preprocessed case as <case>.b2nd (image), <case>_seg.b2nd (GT)
and <case>.pkl (properties, incl. label sampling locations). Only the image is the
same for all the datasets, so it is kept once in a shared store and every dataset
gets a relative symlink to it; GT and properties stay in each dataset.

    data_shared/shared_preprocessed/<data_identifier>/<case>.b2nd                 the images (once)
    models/<net>/nnUNet_preprocessed/<dataset>/<data_identifier>/<case>.b2nd -> symlink

The first dataset processed fills the store. For the next ones the image depends
only on the plan, so the plan fields that affect it must match the store's, and
the image shapes (all cases) and contents (a few cases) are checked before linking.
Run after nnUNetv2_plan_and_preprocess (01_preprocess.sh does it).

    python share_images.py --dataset Dataset102_nxb18
"""
import argparse
import json
import os
from pathlib import Path

import blosc2
import numpy as np

# plan fields that determine the preprocessed image (not the GT)
IMAGE_FIELDS = ['data_identifier', 'spacing', 'normalization_schemes', 'use_mask_for_norm',
                'resampling_fn_data', 'resampling_fn_data_kwargs']
N_CONTENT_CHECKS = 3


def image_plan(plans_file, configuration):
    plans = json.loads(plans_file.read_text())
    cfg = plans['configurations'][configuration]
    return {'transpose_forward': plans['transpose_forward'], **{k: cfg[k] for k in IMAGE_FIELDS}}


def load(path):
    return blosc2.open(urlpath=str(path), mode='r')


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--dataset', default=os.environ.get('DATASET_NAME'))
    p.add_argument('--plans', default=os.environ.get('PLANS'))
    p.add_argument('--configuration', default='3d_fullres')
    p.add_argument('--store', default=os.environ.get('SHARED_PREPROCESSED'))
    args = p.parse_args()

    pre = Path(os.environ['nnUNet_preprocessed']) / args.dataset
    plan = image_plan(pre / f'{args.plans}.json', args.configuration)
    src = pre / plan['data_identifier']
    store = Path(args.store) / plan['data_identifier']
    store.mkdir(parents=True, exist_ok=True)

    # the store remembers the plan its images were made with
    plan_file = store / 'image_plan.json'
    if plan_file.exists():
        ref = json.loads(plan_file.read_text())
        diff = {k: (ref[k], plan[k]) for k in plan if ref.get(k) != plan[k]}
        assert not diff, f'{args.dataset}: image plan differs from the shared store (store, here): {diff}'
    else:
        plan_file.write_text(json.dumps(plan, indent=2))

    images = sorted(f for f in src.glob('*.b2nd') if not f.name.endswith('_seg.b2nd') and not f.is_symlink())
    n_moved = n_linked = n_checked = 0
    for f in images:
        shared = store / f.name
        if shared.exists():
            a, b = load(f), load(shared)
            assert a.shape == b.shape, f'{f.name}: shape {a.shape} != shared {b.shape}'
            if n_checked < N_CONTENT_CHECKS:
                assert np.array_equal(a[:], b[:]), f'{f.name}: content differs from the shared image'
                n_checked += 1
            f.unlink()
            n_linked += 1
        else:
            f.rename(shared)
            n_moved += 1
        f.symlink_to(os.path.relpath(shared, src))

    n_total = len(list(src.glob('*.pkl')))
    n_links = sum(f.is_symlink() for f in src.glob('*.b2nd'))
    assert n_links == n_total, f'{n_links} linked images for {n_total} cases'
    assert all(f.exists() for f in src.glob('*.b2nd')), 'dangling symlink'
    print(f'{args.dataset}: {n_moved} images moved to {store}, {n_linked} replaced by links '
          f'({n_checked} compared voxel by voxel); {n_total} cases linked')


if __name__ == '__main__':
    main()
