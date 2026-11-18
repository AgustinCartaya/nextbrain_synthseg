#!/usr/bin/env python3
"""Held-out section test: the only comparison with a ground truth.

For every three consecutive usable sections (k-1, k, k+1) of every block, section k is hidden and
predicted from its two neighbours with the two rules used to fill the space between sections:

    nearest   the rule of build_3d_segmentation.py: copy the nearest section. Half-way between the two
              neighbours both are equally near, so the score is the mean of "copy k-1" and "copy k+1".
    interp    the rule of build_3d_segmentation_v2.py: shape-based interpolation at t = 0.5.

The prediction is compared with the real section k. restore.py does not act between sections, so
"v1 + restore" behaves like "nearest" in this test.

Limits: the hidden section is half-way (the hardest position for both rules) and the two neighbours
are twice as far apart as in the real reconstruction (0.67-0.76 mm for the densely sectioned blocks,
which is the real spacing of the other blocks; 1.3-1.6 mm for the others).

Outputs (outputs/methods_comparison/tables):
    heldout_blocks.csv   one row per case, block and rule: pixel accuracy
    heldout_labels.csv   one row per case, label and rule: Dice

Usage:
    python heldout_sections.py
"""
import glob
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import nibabel as nib
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import build_3d_segmentation_v2 as V2  # noqa: E402
from compare_methods import TABLES, load_names, case_dirs  # noqa: E402

STRIDE = 2            # in-plane stride: 0.2 mm pixels, the resolution of the output grid
CAP_MM = 1.0          # default --sdt-cap-mm
MIN_FRAC = 0.3        # default --min-section-frac
NLAB = 4096


def interpolate(sa, sb, shape, pmin, t=0.5):
    """Same rule as reconstruct_block in build_3d_segmentation_v2.py, for one whole section."""
    best_val = np.full(shape, -np.inf, np.float32)
    best_lab = np.zeros(shape, np.uint16)
    for l in sorted(set(sa) | set(sb)):
        if l in sa and l in sb:
            va, vb = sa[l], sb[l]
        elif l in sa:
            va = sa[l]; vb = -np.maximum(np.abs(va), pmin)
        else:
            vb = sb[l]; va = -np.maximum(np.abs(vb), pmin)
        val = (1 - t) * va + t * vb
        better = val > best_val
        best_val[better] = val[better]
        best_lab[better] = l
    return best_lab


def one_block(args):
    case, b, path = args
    img = nib.load(path)
    full = np.asanyarray(img.dataobj).astype(np.uint16)      # one block of P85-18 is stored as float64
    if full.ndim == 2 or full.shape[2] < 3:
        return None
    pix = float(np.linalg.norm(img.affine[:3, 0])) * STRIDE
    dz = float(np.linalg.norm(img.affine[:3, 2]))
    area = (full > 0).sum(axis=(0, 1))
    if not (area > 0).any():
        return None
    usable = area >= MIN_FRAC * np.median(area[area > 0])
    lab = np.ascontiguousarray(full[::STRIDE, ::STRIDE, :])
    del full
    cache = {}

    def sdt(k):
        if k not in cache:
            cache[k] = V2.section_sdt(lab[:, :, k], pix, CAP_MM)
        return cache[k]

    acc = {r: dict(n_true=np.zeros(NLAB, np.int64), n_pred=np.zeros(NLAB, np.int64), inter=np.zeros(NLAB, np.int64))
           for r in ('nearest', 'interp')}
    tot = dict(nearest=[0, 0, 0], interp=[0, 0, 0])      # correct, labelled-in-truth, labelled-in-truth-or-prediction
    n_test = 0
    for k in range(1, lab.shape[2] - 1):
        if not (usable[k - 1] and usable[k] and usable[k + 1]):
            continue
        n_test += 1
        truth = lab[:, :, k]
        preds = {'interp': [interpolate(sdt(k - 1), sdt(k + 1), truth.shape, np.float32(pix))],
                 'nearest': [lab[:, :, k - 1], lab[:, :, k + 1]]}
        cache.pop(k - 1, None)
        for r, plist in preds.items():
            w = 2 // len(plist)                          # interp counted twice so both rules have the same weight
            for p in plist:
                ok = p == truth
                a = acc[r]
                a['n_true'] += w * np.bincount(truth.ravel(), minlength=NLAB)
                a['n_pred'] += w * np.bincount(p.ravel(), minlength=NLAB)
                a['inter'] += w * np.bincount(truth[ok], minlength=NLAB)
                tot[r][0] += w * int((ok & (truth > 0)).sum())
                tot[r][1] += w * int((truth > 0).sum())
                tot[r][2] += w * int(((truth > 0) | (p > 0)).sum())
    if not n_test:
        return None
    blocks = [dict(case=case, block=int(b), rule=r, dz_mm=dz, heldout_spacing_mm=2 * dz, sections_tested=n_test,
                   accuracy_pct=100 * t[0] / max(t[1], 1), accuracy_union_pct=100 * t[0] / max(t[2], 1),
                   correct=t[0], labelled=t[1]) for r, t in tot.items()]
    return case, dz, blocks, acc


def main():
    names = load_names()
    jobs = []
    for case, cd in case_dirs().items():
        for p in sorted(glob.glob(os.path.join(cd, 'labels_blocks', '*', 'LABELS.nii.gz'))):
            jobs.append((case, os.path.basename(os.path.dirname(p)), p))
    brows, lab_acc = [], {}
    with ProcessPoolExecutor(48) as ex:
        for res in ex.map(one_block, jobs, chunksize=1):
            if res is None:
                continue
            case, dz, blocks, acc = res
            brows += blocks
            group = 'dense (held-out spacing ~0.7 mm)' if dz < 0.5 else 'sparse (held-out spacing ~1.4 mm)'
            for r, a in acc.items():
                d = lab_acc.setdefault((case, group, r), {k: np.zeros(NLAB, np.int64) for k in a})
                for k in a:
                    d[k] += a[k]
    lrows = []
    for (case, group, r), a in lab_acc.items():
        for l in np.flatnonzero(a['n_true'][1:]) + 1:
            den = a['n_true'][l] + a['n_pred'][l]
            lrows.append(dict(case=case, blocks=group, rule=r, label=int(l), name=names.get(int(l), '?'),
                              true_px=int(a['n_true'][l] // 2), dice=2 * a['inter'][l] / den if den else np.nan))
    pd.DataFrame(brows).to_csv(os.path.join(TABLES, 'heldout_blocks.csv'), index=False)
    pd.DataFrame(lrows).to_csv(os.path.join(TABLES, 'heldout_labels.csv'), index=False)
    b = pd.DataFrame(brows)
    b['blocks'] = np.where(b.dz_mm < 0.5, 'dense', 'sparse')
    g = b.groupby(['blocks', 'rule'])[['correct', 'labelled']].sum()
    print((100 * g.correct / g.labelled).round(2).to_string())


if __name__ == '__main__':
    main()
