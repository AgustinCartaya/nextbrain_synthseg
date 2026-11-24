#!/usr/bin/env python3
"""
SimplifiedNextBrain + CSF -> nxb18: group the labels into 18 classes + background.

Same as nxbsimplified_to_nxb16.py, plus two classes: Ventricles (label 50, background in nxb16)
and CSF (label 24, added by other_codes/add_csf_to_labels). CSF is an 8th principal colour.

Inputs
------
  --grouping   nextbrain_to_nxb18.yaml: {"Combined Structures": [{<class>: [labels]}, ...]}
               New label id = position of the class in the file. The first class must be the
               background (it must contain label 0). A label can be in only one class.
               Same as nxb16/GMM_label_grouping_corrected.yaml, with label 50 moved from the
               background to "Ventricles" and a new "CSF" class with label 24.
  --weights    GMM_weights.yaml: {"Recipe ... for WM, GM, ...": [{<class>: [[w1, ..., wN]]}, ...]}
               Colour weights of every class (same class names as the grouping).
  --seg-dir    label maps to relabel (default: nxbsimplified maps with ventricles and CSF, output of
               other_codes/add_csf_to_labels/combined_approach)

Step 1: tree
------------
  A new flat tree, built only from the grouping and the weights (no input tree):

      root
       |- background                      level 1, labels [0], null
       |- white_matter                    level 1, labels [1], w_mu = GMM weights
       |- ...                             one node per class, labels [new id]

  colors  = the principal colours of GMM_weights.yaml (WM, GM, WMC, GMC, CA, PU, PA, CSF).
  w_mu    = the GMM weights of the class. Some classes extrapolate beyond the principal colours
            (e.g. Substantia Nigra = 1.5625 WM - 0.5625 GM, darker than WM): negative weights
            are kept in w_mu.
  w_sigma = w_mu when all its weights are >= 0. Otherwise the negative weights are set to 0
            and the rest renormalised (a negative weight could give a negative sigma).
  The description of every node lists the original labels merged into it.

Step 2: label maps
------------------
  Every case is relabelled with nxb_label_mapping.map_segmentation
  (libraries/new_nextbrain_utils). A label of a map that is not in the grouping raises an error.

Outputs (--out-dir)
-------------------
  nextbrain_tree_nxb18.json     nxb18 tree
  labels/<case>.nii.gz          relabelled label maps (uint8, values 0..18)

Usage
-----
    python nxbsimplified_csf_to_nxb18.py              # both steps with the default paths
    python nxbsimplified_csf_to_nxb18.py --step tree  # only step 1
    python nxbsimplified_csf_to_nxb18.py --step labels # only step 2
"""
import argparse
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import nibabel as nib
import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
sys.path.insert(0, str(PROJECT / 'libraries'))
sys.path.insert(0, str(PROJECT / 'libraries/new_nextbrain_utils'))
from trees.tree_allen import AllenLabelTree  # noqa: E402
from nxb_label_mapping import load_groups, map_segmentation  # noqa: E402

DEFAULT_GROUPING = HERE / 'data/inptus/nxb18/nextbrain_to_nxb18.yaml'
DEFAULT_WEIGHTS = HERE / 'data/inptus/nxb18/GMM_weights.yaml'
DEFAULT_SEG_DIR = HERE / 'data/outputs/nxbsimplified_csf/labels'
DEFAULT_OUT_DIR = HERE / 'data/outputs/nxb18'
TREE_NAME = 'nextbrain_tree_nxb18.json'

CONTRASTS = ['T1', 'T2', 'FLAIR', 'PD']


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def node_name(cls):
    """'Substantia Nigra and STN' -> 'substantia_nigra_and_stn'."""
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


def load_weights(path):
    """(colors, {class name: weights}). Colors are read from the key: '... for WM, GM, ...'."""
    with open(path) as f:
        (key, rows), = yaml.safe_load(f).items()
    colors = [c.strip() for c in key.split(' for ', 1)[1].rstrip(':').split(',')]
    weights = {}
    for row in rows:
        (name, w), = row.items()
        weights[name] = [float(x) for x in (w[0] if isinstance(w[0], list) else w)]
        if len(weights[name]) != len(colors):
            raise ValueError(f'{name}: {len(weights[name])} weights, expected {len(colors)} {colors}')
    return colors, weights


def label_mapping(grouping=DEFAULT_GROUPING):
    """
    {nxbsimplified label: nxb18 label} (the rule of step 2), for the online generator
    (generator_double_tree), which merges the augmented label maps on the fly.
    """
    groups = load_grouping(Path(grouping))
    return {lab: new_id for new_id, labs in enumerate(groups.values()) for lab in labs}


# ================================================================================================ #
# Step 1: tree                                                                                     #
# ================================================================================================ #

def step_tree(groups, colors, weights, out_dir):
    """Flat tree: root -> one level-1 node per class (the first one is the null background)."""
    log('Step 1: tree')
    (bg, bg_labels), *classes = groups.items()
    tree = AllenLabelTree.new(
        colors, contrasts=CONTRASTS, version='nextbrain-nxb18-0.2',
        description=f'NextBrain labels grouped into {len(classes)} classes + background, colour '
                    f'weights from GMM_weights.yaml. Flat tree built with nxbsimplified_csf_to_nxb18.py.')
    tree.add_node(node_name(bg), tree.root, labels=[0], null=True,
                  description=f'{bg}. Original labels: {bg_labels}')
    for new_id, (cls, labs) in enumerate(classes, start=1):
        w_mu = weights[cls]
        w_sigma = None                           # None = same as w_mu
        if min(w_mu) < 0:
            w = np.clip(w_mu, 0, None)
            w_sigma = (w / w.sum()).tolist()
        tree.add_node(node_name(cls), tree.root, labels=[new_id], w_mu=w_mu, w_sigma=w_sigma,
                      description=f'{cls}. Original labels: {labs}')
    tree.validate()

    out_dir.mkdir(parents=True, exist_ok=True)
    tree.to_json(out_dir / TREE_NAME)
    for i, (cls, labs) in enumerate(groups.items()):
        log(f'  {i:2d} {cls:42s} {len(labs):3d} labels')
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
    p.add_argument('--grouping', type=Path, default=DEFAULT_GROUPING, help='nextbrain_to_nxb18.yaml')
    p.add_argument('--weights', type=Path, default=DEFAULT_WEIGHTS, help='GMM_weights.yaml')
    p.add_argument('--seg-dir', type=Path, default=DEFAULT_SEG_DIR, help='label maps to relabel')
    p.add_argument('--out-dir', type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument('--step', choices=['all', 'tree', 'labels'], default='all')
    p.add_argument('--jobs', type=int, default=5, help='cases processed in parallel (default 5)')
    args = p.parse_args()

    groups = load_grouping(args.grouping)
    colors, weights = load_weights(args.weights)
    missing = [cls for cls in list(groups)[1:] if cls not in weights]
    if missing:
        raise ValueError(f'No weights for classes {missing} in {args.weights}')
    if args.step in ('all', 'tree'):
        step_tree(groups, colors, weights, args.out_dir)
    if args.step in ('all', 'labels'):
        step_labels(args.seg_dir, groups, args.out_dir, args.jobs)


if __name__ == '__main__':
    main()
