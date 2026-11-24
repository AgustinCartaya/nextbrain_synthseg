#!/usr/bin/env python3
"""
SimplifiedNextBrain -> SimplifiedNextBrain + CSF: add the ventricles and the CSF to the
nxbsimplified tree, and take the label maps that already contain them.

Inputs
------
  --tree      nxbsimplified tree (output of nxballen_to_nxbsimplified.py)
  --seg-dir   nxbsimplified label maps with ventricles (50) and CSF (24) added
              (output of other_codes/add_csf_to_labels, combined_approach)

Step 1: tree
------------
  nxballen_to_nxbsimplified.py removed label 50 and pruned the ventricle branch. This step puts
  back a level-1 node, `ventricles_of_forebrain_l1`, with the colour FV. The fields of that node
  (wr_mu, wr_sigma, allen_*) come from the original Allen tree. It is inserted in the root's
  children just after `white_matter_of_forebrain_l1`:

      root
       |- background
       |- gray_matter_of_forebrain_l1
       |- white_matter_of_forebrain_l1
       |- ventricles_of_forebrain_l1      level 1, w_mu = one-hot FV               (new)
       |   |- ventricles                  level 2, labels [50], r_max 2.105        (new)
       |   |- csf                         level 2, labels [24], r_max null         (new)
       |- surface_structures_of_forebrain_l1
       |- ...

  Both leaves inherit the colour FV from their parent and get their own random component
  (wr_mu = 3, as the other leaves), so they have similar but different intensities.
  `ventricles` r_max = 2.105 is the value of the ventricle nodes in the original Allen tree.
  CSF has no thickness measurement, so it is left visible at every resolution (r_max null).

Step 2: label maps
------------------
  The label maps are copied unchanged. Every label of every map must be in the new tree
  (error otherwise).

Outputs (--out-dir)
-------------------
  nextbrain_tree_allen_gyral_simplified_csf.json   tree with ventricles and CSF
  labels/<case>.nii.gz                             label maps (copied)

Usage
-----
    python nxbsimplified_to_nxbsimplified_csf.py                  # both steps with the default paths
    python nxbsimplified_to_nxbsimplified_csf.py --step tree      # only step 1
    python nxbsimplified_to_nxbsimplified_csf.py --step labels    # only step 2
"""
import argparse
import shutil
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
from trees.tree_allen import AllenLabelTree  # noqa: E402

DEFAULT_TREE = HERE / 'data/outputs/nxbsimplified/nextbrain_tree_allen_gyral_simplified.json'
DEFAULT_SEG_DIR = PROJECT / 'other_codes/add_csf_to_labels/data/outputs/combined_approach/labels'
DEFAULT_OUT_DIR = HERE / 'data/outputs/nxbsimplified_csf'
TREE_NAME = 'nextbrain_tree_allen_gyral_simplified_csf.json'

VENTRICLE_LABEL = 50        # NextBrainLabels.yaml: anterior_horn_of_lateral_ventricle
CSF_LABEL = 24              # FreeSurfer LUT: CSF (unused in NextBrain)
COLOR = 'FV'                # principal colour of both nodes
PARENT = 'ventricles_of_forebrain_l1'
INSERT_AFTER = 'white_matter_of_forebrain_l1'

# level-1 node, as in the original Allen tree (ontology/data/outputs/nextbrain_tree_allen_gyral.json)
PARENT_FIELDS = dict(wr_mu=4.0, wr_sigma=0.5, r_max=None, allen_id=10595, allen_acronym='FV',
                     allen_name='ventricles of forebrain', allen_color='231F20')
LEAVES = {
    'ventricles': dict(labels=[VENTRICLE_LABEL], wr_mu=3.0, wr_sigma=0.5, r_max=2.105,
                       description=f'Ventricles (label {VENTRICLE_LABEL}), added by '
                                   f'other_codes/add_csf_to_labels'),
    'csf': dict(labels=[CSF_LABEL], wr_mu=3.0, wr_sigma=0.5, r_max=None,
                description=f'CSF (label {CSF_LABEL}), added by other_codes/add_csf_to_labels'),
}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_tree(path):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')          # r_max warnings of level-1 nodes
        return AllenLabelTree.from_json(path)


# ================================================================================================ #
# Step 1: tree                                                                                     #
# ================================================================================================ #

def add_csf_to_tree(tree):
    """Add `ventricles_of_forebrain_l1` (colour FV) with the leaves `ventricles` and `csf` (in place)."""
    if COLOR not in tree.colors:
        raise ValueError(f'Colour {COLOR!r} not in the tree colours {tree.colors}')
    taken = sorted({VENTRICLE_LABEL, CSF_LABEL} & set(tree.label_to_node))
    if taken:
        raise ValueError(f'Labels {taken} are already in the tree')
    root = tree.nodes[tree.root]
    if INSERT_AFTER not in root.children:
        raise ValueError(f'{INSERT_AFTER!r} is not a child of the root')

    w_mu = [float(c == COLOR) for c in tree.colors]
    tree.add_node(PARENT, tree.root, w_mu=w_mu, **PARENT_FIELDS)
    root.children.remove(PARENT)                                 # add_node appends: move it
    root.children.insert(root.children.index(INSERT_AFTER) + 1, PARENT)
    for name, fields in LEAVES.items():
        tree.add_node(name, PARENT, **fields)
    tree.validate()


def step_tree(tree_path, out_dir):
    log(f'Step 1: tree ({tree_path.name})')
    tree = load_tree(tree_path)
    n_nodes, n_labels = len(tree.nodes), len(tree.label_to_node)

    add_csf_to_tree(tree)
    tree.version = f'{tree.version}-csf'
    tree.description = (f'{tree.description} CSF (nxbsimplified_to_nxbsimplified_csf.py): '
                        f'{PARENT} (colour {COLOR}) added back with the leaves ventricles '
                        f'(label {VENTRICLE_LABEL}) and csf (label {CSF_LABEL}).')

    out_dir.mkdir(parents=True, exist_ok=True)
    out_tree = out_dir / TREE_NAME
    tree.to_json(out_tree)
    log(f'  nodes {n_nodes} -> {len(tree.nodes)}, labels {n_labels} -> {len(tree.label_to_node)}')
    log(f'  root children: {tree.nodes[tree.root].children}')
    log(f'  saved {out_tree}')
    return out_tree


# ================================================================================================ #
# Step 2: label maps                                                                               #
# ================================================================================================ #

def copy_case(seg_path, out_dir):
    present = np.unique(np.asanyarray(nib.load(seg_path).dataobj)).astype(int).tolist()
    shutil.copy2(seg_path, out_dir / seg_path.name)
    return seg_path.name, present


def step_labels(seg_dir, tree_path, out_dir, jobs):
    log(f'Step 2: label maps ({seg_dir})')
    files = sorted(p for p in seg_dir.glob('*.nii*')
                   if not p.name.endswith(('_steps.nii.gz', '_removed.nii.gz')))
    if not files:
        raise FileNotFoundError(f'No label maps in {seg_dir}')
    tree_labels = set(load_tree(tree_path).label_to_node)
    out_labels = out_dir / 'labels'
    out_labels.mkdir(parents=True, exist_ok=True)
    log(f'  {len(files)} cases, {jobs} in parallel')

    with ProcessPoolExecutor(jobs) as ex:
        for name, present in ex.map(copy_case, files, [out_labels] * len(files)):
            not_in_tree = sorted(set(present) - tree_labels)
            if not_in_tree:
                raise ValueError(f'{name}: labels not in the tree: {not_in_tree}')
            log(f'  {name}: {len(present)} labels, ventricles {VENTRICLE_LABEL in present}, '
                f'CSF {CSF_LABEL in present}')
    log(f'  saved in {out_labels}')


# ================================================================================================ #

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--tree', type=Path, default=DEFAULT_TREE, help='nxbsimplified tree JSON')
    p.add_argument('--seg-dir', type=Path, default=DEFAULT_SEG_DIR,
                   help='label maps with ventricles and CSF')
    p.add_argument('--out-dir', type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument('--step', choices=['all', 'tree', 'labels'], default='all')
    p.add_argument('--jobs', type=int, default=5, help='cases processed in parallel (default 5)')
    args = p.parse_args()

    out_tree = args.out_dir / TREE_NAME
    if args.step in ('all', 'tree'):
        out_tree = step_tree(args.tree, args.out_dir)
    if args.step in ('all', 'labels'):
        if not out_tree.exists():
            sys.exit(f'{out_tree} not found: run --step tree first')
        step_labels(args.seg_dir, out_tree, args.out_dir, args.jobs)


if __name__ == '__main__':
    main()
