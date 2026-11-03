"""Look at the training batches without training: speed, shapes, labels, and NIfTI files.

    python check_batches.py [--fold 0] [--plans nnUNetResEncUNetMPlans] [-n 20] [--save 2] [--device cuda:0]

Builds the same GeneratorBatches as nnUNetTrainerNxGen (same plan, same fold, same seeds),
times `n` batches and writes the first `--save` batches to check_batches/ in nnU-Net axes
(image after z-score, full-resolution target). Open them with the same viewer as a
preprocessed case: axes are nnU-Net's, not the NIfTI's.
"""
import argparse
import json
import os
import sys
import time

import nibabel as nib
import numpy as np
import torch
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'trainers'))
from nxgen_batches import build_from_plans, ds_scales_from_plans  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--fold', default='0')
    p.add_argument('--plans', default='nnUNetResEncUNetMPlans')
    p.add_argument('--configuration', default='3d_fullres')
    p.add_argument('-n', type=int, default=20)
    p.add_argument('--save', type=int, default=2)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--out', default='check_batches')
    args = p.parse_args()

    base = os.path.join(os.environ['nnUNet_preprocessed'], os.environ['DATASET_NAME'])
    plans = json.load(open(os.path.join(base, args.plans + '.json')))
    cm = PlansManager(plans).get_configuration(args.configuration)
    fold = args.fold if args.fold == 'all' else int(args.fold)
    batches = build_from_plans(base, plans, args.configuration, fold, cm.batch_size,
                               ds_scales_from_plans(cm.pool_op_kernel_sizes), (0, 1, 2),
                               args.device, prefetch=0)
    names = batches.generator.labels.names
    os.makedirs(args.out, exist_ok=True)
    counts = np.zeros(len(names), np.int64)
    times = []
    for b in range(args.n):
        t0 = time.perf_counter()
        batch = next(batches)
        times.append(time.perf_counter() - t0)
        seg = batch['target'][0]
        counts += torch.bincount(seg.long().flatten(), minlength=len(names)).cpu().numpy()
        if b == 0:
            print('data', tuple(batch['data'].shape), batch['data'].dtype,
                  f"mean {batch['data'].mean():.3f} std {batch['data'].std():.3f}")
            print('target', [tuple(t.shape) for t in batch['target']])
        if b < args.save:
            aff = np.diag(list(cm.spacing) + [1.0])
            for j in range(batch['data'].shape[0]):
                nib.save(nib.Nifti1Image(batch['data'][j, 0].cpu().numpy(), aff),
                         os.path.join(args.out, f'batch{b}_{j}_image.nii.gz'))
                nib.save(nib.Nifti1Image(seg[j, 0].cpu().numpy().astype(np.uint8), aff),
                         os.path.join(args.out, f'batch{b}_{j}_seg.nii.gz'))
    print(f'{args.n} batches of {cm.batch_size}: first {times[0]:.2f} s, then '
          f'{np.mean(times[1:]):.3f} s/batch (mean), peak GPU memory '
          f'{torch.cuda.max_memory_allocated(args.device) / 2**30:.1f} GB')
    frac = counts / counts.sum()
    print('label fractions:', ', '.join(f'{n} {f:.3f}' for n, f in zip(names, frac)))


if __name__ == '__main__':
    main()
