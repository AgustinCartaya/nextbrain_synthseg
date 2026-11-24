#!/usr/bin/env python3
"""
Theoretical check of the common space, from the two conversion scripts.

Every nxbsimplified label is sent to nxb16 (nxbsimplified_to_nxb16.label_mapping) and to
nxbsupersynth (nxbsimplified_to_nxbsupersynth.label_mapping), then to the common space. Writes:

  crosswalk.csv           every (nxb16 class, nxbsupersynth class) pair with the NextBrain labels
                          in it and their mean volume (mm3) over the nxbsimplified label maps
  common_space_check.csv  per common class: volume on each side, volume on which they disagree,
                          and the best possible dice (dice between the two perfect conversions of
                          the same nxbsimplified map, ignored voxels excluded)

It also checks that common_space.yaml covers every class of the nxb16 and nxbsupersynth trees.

    python crosswalk.py                 # volumes from data/outputs/nxbsimplified/labels
    python crosswalk.py --seg-dir ''    # labels only, no volumes
"""
import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import nibabel as nib
import numpy as np

HERE = Path(__file__).resolve().parent
SIMPLIFIED = HERE.parents[3]                       # .../simplified_trees_and_labels
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(SIMPLIFIED))
from to_common_space import DEFAULT_CS, load_common_space  # noqa: E402

DEFAULT_SEG_DIR = SIMPLIFIED / 'data/outputs/nxbsimplified/labels'
TREE_SIMPLIFIED = SIMPLIFIED / 'data/outputs/nxbsimplified/nextbrain_tree_allen_gyral_simplified.json'
TREE_16 = SIMPLIFIED / 'data/outputs/nxb16/nextbrain_tree_nxb16.json'
TREE_SS = SIMPLIFIED / 'data/outputs/nxbsupersynth/nextbrain_tree_nxbsupersynth.json'


def mean_volumes(seg_dir):
    """{nxbsimplified label: mean volume over the cases (mm3)}."""
    files = sorted(p for p in Path(seg_dir).glob('*.nii*')
                   if not p.name.endswith(('_steps.nii.gz', '_removed.nii.gz')))
    vol = defaultdict(float)
    for f in files:
        img = nib.load(f)
        vv = float(np.prod(img.header.get_zooms()[:3]))
        u, n = np.unique(np.rint(np.asanyarray(img.dataobj)).astype(np.int64), return_counts=True)
        for lab, k in zip(u.tolist(), n.tolist()):
            vol[lab] += k * vv / len(files)
        print(f'  volumes: {f.name}', flush=True)
    return vol


def node_names(tree_path):
    with open(tree_path) as f:
        nodes = json.load(f)['nodes']
    return {n['labels'][0]: k for k, n in nodes.items() if n.get('labels')}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--common-space', type=Path, default=DEFAULT_CS)
    p.add_argument('--seg-dir', default=str(DEFAULT_SEG_DIR), help="'' = no volumes")
    p.add_argument('--out-dir', type=Path, default=HERE)
    args = p.parse_args()

    import nxbsimplified_to_nxb16 as s16                     # noqa: E402
    import nxbsimplified_to_nxbsupersynth as sss             # noqa: E402

    cs = load_common_space(args.common_space)
    m16, mss = s16.label_mapping(), sss.label_mapping(side='R')
    n16, nss = node_names(TREE_16), node_names(TREE_SS)
    with open(TREE_SIMPLIFIED) as f:
        labels = sorted({l for n in json.load(f)['nodes'].values() for l in n.get('labels', [])})
    vol = mean_volumes(args.seg_dir) if args.seg_dir else defaultdict(float)

    for ont, names in (('nxb16', n16), ('nxbsupersynth', nss)):
        miss = sorted(set(names) - set(cs['lut'][ont]))
        if miss:
            raise ValueError(f'{ont} classes not in common_space.yaml: {[names[i] for i in miss]}')
    print('common_space.yaml covers every nxb16 and nxbsupersynth class')

    # crosswalk.csv
    cells = defaultdict(list)
    for l in labels:
        cells[(m16[l], mss[l])].append(l)
    tot16, totss = defaultdict(float), defaultdict(float)
    for (a, b), labs in cells.items():
        v = sum(vol[l] for l in labs)
        tot16[a] += v
        totss[b] += v
    out = args.out_dir / 'crosswalk.csv'
    with open(out, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['nxb16_id', 'nxb16_name', 'nxbsupersynth_id', 'nxbsupersynth_name',
                    'common_id_from_nxb16', 'common_id_from_nxbsupersynth', 'volume_mm3',
                    'pct_of_nxb16_class', 'pct_of_nxbsupersynth_class', 'nxbsimplified_labels'])
        for (a, b), labs in sorted(cells.items()):
            v = sum(vol[l] for l in labs)
            w.writerow([a, n16[a], b, nss[b], cs['lut']['nxb16'][a], cs['lut']['nxbsupersynth'][b],
                        round(v, 1), round(100 * v / tot16[a], 2) if tot16[a] else '',
                        round(100 * v / totss[b], 2) if totss[b] else '',
                        ' '.join(map(str, labs))])
    print(f'saved {out}')

    # common_space_check.csv
    ign = cs['ignore_label']
    c16 = {l: cs['lut']['nxb16'][m16[l]] for l in labels}
    css = {l: cs['lut']['nxbsupersynth'][mss[l]] for l in labels}
    valid = [l for l in labels if c16[l] != ign and css[l] != ign]
    out = args.out_dir / 'common_space_check.csv'
    with open(out, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['common_id', 'common_name', 'status', 'vol_nxb16_mm3', 'vol_nxbsupersynth_mm3',
                    'vol_only_nxb16_mm3', 'vol_only_nxbsupersynth_mm3', 'best_dice',
                    'labels_only_in_nxb16_side', 'labels_only_in_nxbsupersynth_side'])
        print(f'  {"common class":34s} {"status":12s} best dice')
        for c in cs['classes']:
            k = c['id']
            A = {l for l in valid if c16[l] == k}
            B = {l for l in valid if css[l] == k}
            va, vb = sum(vol[l] for l in A), sum(vol[l] for l in B)
            vab = sum(vol[l] for l in A & B)
            dice = 2 * vab / (va + vb) if va + vb else float('nan')
            w.writerow([k, c['name'], c['status'], round(va, 1), round(vb, 1), round(va - vab, 1),
                        round(vb - vab, 1), round(dice, 4), ' '.join(map(str, sorted(A - B))),
                        ' '.join(map(str, sorted(B - A)))])
            print(f'  {c["name"]:34s} {c["status"]:12s} {dice:.4f}  only16 {sorted(A - B)}  '
                  f'onlySS {sorted(B - A)}')
    ignored = [l for l in labels if l not in valid]
    print(f'  ignored NextBrain labels ({sum(vol[l] for l in ignored):.0f} mm3): {ignored}')
    print(f'saved {out}')


if __name__ == '__main__':
    main()
