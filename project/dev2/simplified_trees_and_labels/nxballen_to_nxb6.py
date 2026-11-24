#!/usr/bin/env python3
"""
NextBrain -> nxb6: group the labels into 6 classes + background.

    0 Background, 1 White Matter, 2 Gray Matter, 3 Cerebellum White Matter,
    4 Cerebellum Gray Matter, 5 Deep Gray Matter, 6 Brainstem

Inputs
------
  --grouping   nextbrain_to_nxb6.yaml: {"Combined Structures": [{<class>: [labels]}, ...]}
               New label id = position of the class in the file. The first class must be the
               background (it must contain label 0). A label can be in only one class.
  --seg-dir    label maps to relabel (default: output of nxballen_to_nxbsimplified.py)

Step 1: tree
------------
  A new flat tree, built only from the grouping (no input tree):

      root
       |- background                      level 1, labels [0], null
       |- white_matter                    level 1, labels [1], w_mu = [1, 0, 0, 0, 0, 0]
       |- ...                             one node per class, labels [new id]

  One principal colour per class (colors = WM, GM, WMC, GMC, DGM, BS; w_mu = one-hot), i.e. the
  6 classes are sampled independently. The description of every node lists the original labels
  merged into it.

Step 2: label maps
------------------
  Every case is relabelled with nxb_label_mapping.map_segmentation
  (libraries/new_nextbrain_utils). A label of a map that is not in the grouping raises an error.

Outputs (--out-dir)
-------------------
  nextbrain_tree_nxb6.json      nxb6 tree
  labels/<case>.nii.gz          relabelled label maps (uint8, values 0..6)

Usage
-----
    python nxballen_to_nxb6.py                  # both steps with the default paths
    python nxballen_to_nxb6.py --step tree      # only step 1
    python nxballen_to_nxb6.py --step labels    # only step 2
"""
import argparse
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import nibabel as nib
import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
sys.path.insert(0, str(PROJECT / 'libraries'))
sys.path.insert(0, str(PROJECT / 'libraries/new_nextbrain_utils'))
from trees.tree_allen import AllenLabelTree  # noqa: E402
from nxb_label_mapping import load_groups, map_segmentation  # noqa: E402

DEFAULT_GROUPING = HERE / 'data/inptus/nxb6/nextbrain_to_nxb6.yaml'
DEFAULT_SEG_DIR = HERE / 'data/outputs/nxbsimplified/labels'
DEFAULT_OUT_DIR = HERE / 'data/outputs/nxb6_from_nxbsimplified'
TREE_NAME = 'nextbrain_tree_nxb6.json'

CONTRASTS = ['T1', 'T2', 'FLAIR', 'PD']
# principal colour of every class (background excluded)
COLORS = {'White Matter': 'WM', 'Gray Matter': 'GM', 'Cerebellum White Matter': 'WMC',
          'Cerebellum Gray Matter': 'GMC', 'Deep Gray Matter': 'DGM', 'Brainstem': 'BS'}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def node_name(cls):
    """'Deep Gray Matter' -> 'deep_gray_matter'."""
    return re.sub(r'[^0-9a-z]+', '_', cls.lower()).strip('_')


def load_grouping(path):
    """{class name: [original labels]} in file order (position = new label id), checked."""
    groups = {name: [int(l) for l in labs] for name, labs in load_groups(path).items()}
    labels = [l for labs in groups.values() for l in labs]
    if len(labels) != len(set(labels)):
        raise ValueError('A label appears in more than one class')
    if 0 not in next(iter(groups.values())):
        raise ValueError('The first class must be the background (label 0)')
    return groups


# ================================================================================================ #
# Step 1: tree                                                                                     #
# ================================================================================================ #

def step_tree(groups, out_dir):
    """Flat tree: root -> one level-1 node per class (the first one is the null background)."""
    log('Step 1: tree')
    (bg, bg_labels), *classes = groups.items()
    colors = [COLORS[cls] for cls, _ in classes]
    tree = AllenLabelTree.new(
        colors, contrasts=CONTRASTS, version='nextbrain-nxb6-0.2',
        description=f'NextBrain labels grouped into {len(classes)} classes + background, one '
                    f'principal colour per class. Flat tree built with nxballen_to_nxb6.py.')
    tree.add_node(node_name(bg), tree.root, labels=[0], null=True,
                  description=f'{bg}. Original labels: {bg_labels}')
    for new_id, (cls, labs) in enumerate(classes, start=1):
        tree.add_node(node_name(cls), tree.root, labels=[new_id],
                      w_mu=[float(c == COLORS[cls]) for c in colors],
                      description=f'{cls}. Original labels: {labs}')
    tree.validate()

    out_dir.mkdir(parents=True, exist_ok=True)
    tree.to_json(out_dir / TREE_NAME)
    for i, (cls, labs) in enumerate(groups.items()):
        log(f'  {i} {cls:24s} {len(labs):3d} labels')
    log(f'  saved {out_dir / TREE_NAME}')


# ================================================================================================ #
# Step 2: label maps                                                                               #
# ================================================================================================ #

def relabel_case(seg_path, out_dir, groups):
    img = nib.load(seg_path)
    seg = np.rint(np.asanyarray(img.dataobj))
    try:
        out = map_segmentation(seg, groups, unknown='error',
                               zooms=img.header.get_zooms()[:3]).astype(np.uint8)
    except ValueError as e:
        raise ValueError(f'{seg_path.name}: {e}') from None
    hdr = img.header.copy()
    hdr.set_data_dtype(np.uint8)
    hdr['scl_slope'], hdr['scl_inter'] = 1, 0
    nib.save(nib.Nifti1Image(out, img.affine, hdr), out_dir / seg_path.name)
    return seg_path.name, np.unique(out).tolist()


def step_labels(seg_dir, groups, out_dir, jobs):
    log(f'Step 2: label maps ({seg_dir})')
    files = sorted(p for p in seg_dir.glob('*.nii*')
                   if not p.name.endswith(('_steps.nii.gz', '_removed.nii.gz')))
    out_labels = out_dir / 'labels'
    out_labels.mkdir(parents=True, exist_ok=True)
    log(f'  {len(files)} cases, {jobs} in parallel')
    with ProcessPoolExecutor(jobs) as ex:
        for name, present in ex.map(relabel_case, files, [out_labels] * len(files),
                                    [groups] * len(files)):
            log(f'  {name}: labels {present}')
    log(f'  saved in {out_labels}')


# ================================================================================================ #

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--grouping', type=Path, default=DEFAULT_GROUPING, help='nextbrain_to_nxb6.yaml')
    p.add_argument('--seg-dir', type=Path, default=DEFAULT_SEG_DIR, help='label maps to relabel')
    p.add_argument('--out-dir', type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument('--step', choices=['all', 'tree', 'labels'], default='all')
    p.add_argument('--jobs', type=int, default=5, help='cases processed in parallel (default 5)')
    args = p.parse_args()

    groups = load_grouping(args.grouping)
    if args.step in ('all', 'tree'):
        step_tree(groups, args.out_dir)
    if args.step in ('all', 'labels'):
        step_labels(args.seg_dir, groups, args.out_dir, args.jobs)


if __name__ == '__main__':
    main()
