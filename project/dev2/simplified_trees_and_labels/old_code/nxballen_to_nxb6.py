#!/usr/bin/env python3
"""
NextBrain -> nxb6: group the labels into 6 classes + background.

    0 Background, 1 White Matter, 2 Gray Matter, 3 Cerebellum White Matter,
    4 Cerebellum Gray Matter, 5 Deep Gray Matter, 6 Brainstem

Inputs
------
  --tree       simplified AllenLabelTree JSON (output of allennxb_to_simplifiednxb.py)
  --seg-dir    label maps to relabel
  --grouping   nextbrain_to_nxb6.yaml: {"Combined Structures": [{<class>: [labels]}, ...]}
               The first class must be the background (label 0). Every label of the tree must
               be in exactly one class. Grouped labels that are not in the tree (e.g. labels
               removed by the simplification) are only reported.

New label ids: the position of the class in the grouping file (Background = 0, then 1..6).

Step 1: tree
------------
  Flat tree (the classes cut across the Allen hierarchy):

      root
       |- background                      level 1, labels [0], null
       |- white_matter                    level 1, labels [1], w_mu = [1, 0, 0, 0, 0, 0]
       |- ...                             one node per class, labels [new id]

  There is no colour-weights file for nxb6: every class gets its own principal colour
  (colors = WM, GM, WMC, GMC, DGM, BS; w_mu = one-hot), i.e. the 6 classes are sampled
  independently. r_max = null (level-1 nodes are always visible). Contrasts are copied from the
  input tree. The original labels and Allen-NextBrain nodes of every class are saved in
  tree_report.json.

Step 2: label maps
------------------
  Every case is relabelled with next_brain_mapping.map_segmentation
  (libraries/new_nextbrain_utils). Labels of a map that are not in the grouping raise an error.

Outputs (--out-dir)
-------------------
  nextbrain_tree_nxb6.json      nxb6 tree
  tree_report.json              per class: new id, original labels and their tree nodes
  labels/<case>.nii.gz          relabelled label maps (uint8, values 0..6)

Usage
-----
    python allennxb_to_nxb6.py                  # both steps with the default paths
    python allennxb_to_nxb6.py --step tree      # only step 1
    python allennxb_to_nxb6.py --step labels    # only step 2
"""
import argparse
import json
import re
import sys
import time
import warnings
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
# DEFAULT_SEG_DIR = HERE / 'data/outputs/simplifiednxb/labels'
TREE_NAME = 'nextbrain_tree_nxb6.json'

# DEFAULT_TREE = PROJECT / 'ontology/data/outputs/nextbrain_tree_allen_gyral.json'
# DEFAULT_SEG_DIR = PROJECT / 'atlas_subjects/data/outputs/restored/rec_nn_0.2mm'
# DEFAULT_OUT_DIR = HERE / 'data/outputs/nxb6'

DEFAULT_TREE = HERE / 'data/outputs/nxbsimplified/nextbrain_tree_allen_gyral_simplified.json'
DEFAULT_SEG_DIR = HERE / 'data/outputs/nxbsimplified/labels'
DEFAULT_OUT_DIR = HERE / 'data/outputs/nxb6_from_nxbsimplified'

# principal colour of every class (background excluded)
COLORS = {'White Matter': 'WM', 'Gray Matter': 'GM', 'Cerebellum White Matter': 'WMC',
          'Cerebellum Gray Matter': 'GMC', 'Deep Gray Matter': 'DGM', 'Brainstem': 'BS'}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_tree(path):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')          # r_max warnings of level-1 nodes
        return AllenLabelTree.from_json(path)


def node_name(cls):
    """'Deep Gray Matter' -> 'deep_gray_matter'."""
    return re.sub(r'[^0-9a-z]+', '_', cls.lower()).strip('_')


def load_grouping(path):
    """{class name: [original labels]} in file order (position = new label id), checked."""
    groups = {name: [int(l) for l in labs] for name, labs in load_groups(path).items()}
    seen = {}
    for name, labs in groups.items():
        for lab in labs:
            if lab in seen:
                raise ValueError(f'Label {lab} in both {seen[lab]!r} and {name!r}')
            seen[lab] = name
    first = next(iter(groups))
    if 0 not in groups[first]:
        raise ValueError(f'The first class ({first!r}) must be the background (label 0)')
    return groups


# ================================================================================================ #
# Step 1: tree                                                                                     #
# ================================================================================================ #

def build_nxb6_tree(src_tree, groups, grouping_name):
    """Flat tree: root -> one level-1 node per class (the first one is the null background)."""
    classes = list(groups)[1:]
    missing = [c for c in classes if c not in COLORS]
    if missing:
        raise ValueError(f'No principal colour for classes {missing} (see COLORS)')
    colors = [COLORS[c] for c in classes]
    tree = AllenLabelTree.new(
        colors, contrasts=src_tree.contrasts, version='nextbrain-nxb6-0.1',
        description=(f'NextBrain labels grouped into {len(classes)} classes + background '
                     f'({grouping_name}), one principal colour per class. Flat tree (the classes '
                     f'cut across the Allen hierarchy). Built from {src_tree.version} with '
                     f'allennxb_to_nxb6.py.'))
    for new_id, (cls, labs) in enumerate(groups.items()):
        if new_id == 0:
            tree.add_node(node_name(cls), tree.root, labels=[0], null=True,
                          description=f'{cls}. Original labels: {labs}')
            continue
        w_mu = [float(c == COLORS[cls]) for c in colors]
        tree.add_node(node_name(cls), tree.root, labels=[new_id], w_mu=w_mu,
                      description=f'{cls}. Original labels: {labs}')
    tree.validate()
    return tree, colors


def step_tree(tree_path, grouping_path, out_dir):
    log(f'Step 1: tree ({tree_path.name})')
    src = load_tree(tree_path)
    groups = load_grouping(grouping_path)

    # every label of the source tree must be grouped
    tree_labels = set(src.label_to_node)
    grouped = {l for labs in groups.values() for l in labs}
    if tree_labels - grouped:
        raise ValueError(f'Tree labels not in the grouping: {sorted(tree_labels - grouped)}')
    not_in_tree = sorted(grouped - tree_labels)
    if not_in_tree:
        log(f'  {len(not_in_tree)} grouped labels not in the tree: {not_in_tree}')

    tree, colors = build_nxb6_tree(src, groups, grouping_path.name)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_tree = out_dir / TREE_NAME
    tree.to_json(out_tree)
    load_tree(out_tree)  # re-read: the saved file is a valid tree

    l2n = src.label_to_node
    report = dict(input_tree=str(tree_path), grouping=str(grouping_path), output=str(out_tree),
                  colors=colors, grouped_labels_not_in_tree=not_in_tree,
                  classes=[dict(new_label=i, name=cls, node=node_name(cls),
                                w_mu=tree.nodes[node_name(cls)].w_mu,
                                original_labels={str(l): l2n.get(l) for l in labs})
                           for i, (cls, labs) in enumerate(groups.items())])
    with open(out_dir / 'tree_report.json', 'w') as f:
        json.dump(report, f, indent=1)

    log(f'  {len(tree_labels)} tree labels -> {len(groups)} classes, colors {colors}')
    for i, (cls, labs) in enumerate(groups.items()):
        n = tree.nodes[node_name(cls)]
        extra = ' (null)' if n.null else f'  w_mu {n.w_mu}'
        log(f'  {i:2d} {cls:28s} {len(labs):3d} labels{extra}')
    log(f'  saved {out_tree}')
    return out_tree


# ================================================================================================ #
# Step 2: label maps                                                                               #
# ================================================================================================ #

def relabel_case(seg_path, out_dir, groups):
    t0 = time.time()
    img = nib.load(seg_path)
    seg = np.asanyarray(img.dataobj)
    if np.issubdtype(seg.dtype, np.floating):
        seg = np.rint(seg)
    n_in = len(np.unique(seg))
    try:
        out = map_segmentation(seg, groups, unknown='error').astype(np.uint8)
    except ValueError as e:
        raise ValueError(f'{seg_path.name}: {e}') from None
    hdr = img.header.copy()
    hdr.set_data_dtype(np.uint8)
    hdr['scl_slope'], hdr['scl_inter'] = 1, 0
    out_dir.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(out, img.affine, hdr), out_dir / seg_path.name)
    return seg_path.name, n_in, np.unique(out).tolist(), time.time() - t0


def step_labels(seg_dir, grouping_path, out_dir, jobs):
    log(f'Step 2: label maps ({seg_dir})')
    files = sorted(p for p in seg_dir.glob('*.nii*')
                   if not p.name.endswith(('_steps.nii.gz', '_removed.nii.gz')))
    groups = load_grouping(grouping_path)
    out_labels = out_dir / 'labels'
    log(f'  {len(files)} cases, {jobs} in parallel')
    with ProcessPoolExecutor(jobs) as ex:
        for name, n_in, present, sec in ex.map(relabel_case, files, [out_labels] * len(files),
                                               [groups] * len(files)):
            log(f'  {name}: {n_in} labels -> {len(present)} {present} in {sec:.0f} s')
    log(f'  saved in {out_labels}')


# ================================================================================================ #

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--tree', type=Path, default=DEFAULT_TREE, help='simplified AllenLabelTree JSON')
    p.add_argument('--seg-dir', type=Path, default=DEFAULT_SEG_DIR, help='label maps to relabel')
    p.add_argument('--grouping', type=Path, default=DEFAULT_GROUPING, help='nextbrain_to_nxb6.yaml')
    p.add_argument('--out-dir', type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument('--step', choices=['all', 'tree', 'labels'], default='all')
    p.add_argument('--jobs', type=int, default=5, help='cases processed in parallel (default 5)')
    args = p.parse_args()

    if args.step in ('all', 'tree'):
        step_tree(args.tree, args.grouping, args.out_dir)
    if args.step in ('all', 'labels'):
        step_labels(args.seg_dir, args.grouping, args.out_dir, args.jobs)


if __name__ == '__main__':
    main()
