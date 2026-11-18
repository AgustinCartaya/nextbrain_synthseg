#!/usr/bin/env python3
"""
SimplifiedNextBrain -> 16NextBrain: group the labels into 16 classes + background.

IMPORTANT: corrected grouping file (label 611)
----------------------------------------------
The original data/inptus/16nxb/GMM_label_grouping.yaml puts
    846  white_matter_of_the_cerebellum  in  "Cerebellum White Matter"
    611  white_matter_of_hindbrain       in  "White Brainstem nuclei"
but in NextBrain the cerebellar white matter is painted with 611: the ontology
(data/basics/nextbrain_ontology_v1.csv) says 611 is the "*label used for WM of the cerebellum",
and 846 is only in the NextBrain LUT ("added, missing from v0") with 0 voxels in every case.
With the original file, class 3 (Cerebellum White Matter) is empty in all the label maps and
the whole cerebellar WM goes to class 12 (White Brainstem nuclei, 2.2M voxels in case-1 vs
0.1-0.5M for the other nucleus classes).
The default grouping is therefore GMM_label_grouping_corrected.yaml: a copy of the original
with 611 moved to "Cerebellum White Matter" (846 is kept there as well, harmless).

Inputs
------
  --tree       simplified AllenLabelTree JSON (output of allennxb_to_simplifiednxb.py)
  --seg-dir    simplified label maps (simplifiednxb/labels)
  --grouping   GMM_label_grouping_corrected.yaml (see above):
               {"Combined Structures": [{<class>: [labels]}, ...]}
               The first class must be the background. Every label of the tree must be in
               exactly one class.
  --weights    GMM_weights.yaml: {"Recipe ... for WM, GM, ...": [{<class>: [[w1, ..., wN]]}, ...]}
               Colour weights of every class (same class names as the grouping).

New label ids: the position of the class in the grouping file (Background = 0, then 1..16).

Step 1: tree
------------
  The classes cut across the Allen hierarchy (e.g. "White Matter" or "Gray Brainstem nuclei"
  take labels from several Allen level-1 classes), so the hierarchy cannot be kept: the 16nxb
  tree is flat.

      root
       |- background                      level 1, labels [0], null
       |- white_matter                    level 1, labels [1], w_mu = GMM weights
       |- ...                             one node per class, labels [new id]

  colors = the principal colours of GMM_weights.yaml (WM, GM, WMC, GMC, CA, PU, PA).
  w_mu   = the GMM weights of the class. Some classes extrapolate beyond the principal colours
           (e.g. Substantia Nigra = 1.5625 WM - 0.5625 GM, darker than WM): negative weights
           are kept in w_mu.
  w_sigma = w_mu when all its weights are >= 0. Otherwise the negative weights are set to 0
           and the rest renormalised (a negative weight could give a negative sigma).
  r_max  = null (level-1 nodes are always visible). Contrasts are copied from the input tree.
  The original labels and Allen-NextBrain nodes of every class are saved in tree_report.json.

Step 2: label maps
------------------
  Every case is relabelled with a lookup table (original label -> new id). Labels of a map
  that are not in the grouping raise an error.

Outputs (--out-dir)
-------------------
  nextbrain_tree_16nxb.json     16nxb tree
  tree_report.json              per class: new id, original labels and their tree nodes
  labels/<case>.nii.gz          relabelled label maps (uint8, values 0..16)

Usage
-----
    python allennxb_to_16nxb.py                  # both steps with the default paths
    python allennxb_to_16nxb.py --step tree      # only step 1
    python allennxb_to_16nxb.py --step labels    # only step 2
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
import yaml

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
sys.path.insert(0, str(PROJECT / 'libraries'))
from trees.tree_allen import AllenLabelTree  # noqa: E402

DEFAULT_TREE = HERE / 'data/outputs/simplifiednxb/nextbrain_tree_allen_gyral_simplified.json'
DEFAULT_SEG_DIR = HERE / 'data/outputs/simplifiednxb/labels'
DEFAULT_GROUPING = HERE / 'data/inptus/16nxb/GMM_label_grouping_corrected.yaml'  # 611 fixed, see top
DEFAULT_WEIGHTS = HERE / 'data/inptus/16nxb/GMM_weights.yaml'
DEFAULT_OUT_DIR = HERE / 'data/outputs/16nxb'
TREE_NAME = 'nextbrain_tree_16nxb.json'


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_tree(path):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')          # r_max warnings of level-1 nodes
        return AllenLabelTree.from_json(path)


def node_name(cls):
    """'Substantia Nigra and STN' -> 'substantia_nigra_and_stn'."""
    return re.sub(r'[^0-9a-z]+', '_', cls.lower()).strip('_')


def load_grouping(path):
    """[(class name, [original labels]), ...] in file order (index = new label id)."""
    with open(path) as f:
        d = yaml.safe_load(f)
    groups = [next(iter(g.items())) for g in d['Combined Structures']]
    groups = [(name, [int(l) for l in labs]) for name, labs in groups]
    seen = {}
    for name, labs in groups:
        for lab in labs:
            if lab in seen:
                raise ValueError(f'Label {lab} in both {seen[lab]!r} and {name!r}')
            seen[lab] = name
    if 0 not in groups[0][1]:
        raise ValueError(f'The first class ({groups[0][0]!r}) must be the background (label 0)')
    return groups


def load_weights(path):
    """(colors, {class name: weights}). Colors are read from the key: '... for WM, GM, ...'."""
    with open(path) as f:
        d = yaml.safe_load(f)
    (key, rows), = d.items()
    colors = [c.strip() for c in key.split(' for ', 1)[1].rstrip(':').split(',')]
    weights = {}
    for row in rows:
        (name, w), = row.items()
        w = [float(x) for x in (w[0] if isinstance(w[0], list) else w)]
        if len(w) != len(colors):
            raise ValueError(f'{name}: {len(w)} weights, expected {len(colors)} {colors}')
        weights[name] = w
    return colors, weights


def label_lut(groups):
    """Lookup table original label -> new id (-1 = not grouped)."""
    lut = np.full(max(l for _, labs in groups for l in labs) + 1, -1, np.int32)
    for new_id, (_, labs) in enumerate(groups):
        lut[labs] = new_id
    return lut


# ================================================================================================ #
# Step 1: tree                                                                                     #
# ================================================================================================ #

def build_16nxb_tree(src_tree, groups, colors, weights, grouping_name):
    """Flat tree: root -> one level-1 node per class (the first one is the null background)."""
    tree = AllenLabelTree.new(
        colors, contrasts=src_tree.contrasts, version='nextbrain-16nxb-0.1',
        description=(f'NextBrain labels grouped into {len(groups) - 1} classes + background '
                     f'({grouping_name}), colour weights from GMM_weights.yaml. Flat tree '
                     f'(the classes cut across the Allen hierarchy). Built from {src_tree.version} '
                     f'with allennxb_to_16nxb.py.'))
    for new_id, (cls, labs) in enumerate(groups):
        if new_id == 0:
            tree.add_node(node_name(cls), tree.root, labels=[0], null=True,
                          description=f'{cls}. Original labels: {labs}')
            continue
        if cls not in weights:
            raise ValueError(f'No weights for class {cls!r} in the weights file')
        w_mu = weights[cls]
        w_sigma = None
        if min(w_mu) < 0:
            w = np.clip(w_mu, 0, None)
            w_sigma = (w / w.sum()).tolist()
        tree.add_node(node_name(cls), tree.root, labels=[new_id], w_mu=w_mu, w_sigma=w_sigma,
                      description=f'{cls}. Original labels: {labs}')
    tree.validate()
    return tree


def step_tree(tree_path, grouping_path, weights_path, out_dir):
    log(f'Step 1: tree ({tree_path.name})')
    src = load_tree(tree_path)
    groups = load_grouping(grouping_path)
    colors, weights = load_weights(weights_path)

    # every label of the source tree must be grouped, and nothing else
    tree_labels = set(src.label_to_node)
    grouped = {l for _, labs in groups for l in labs}
    if tree_labels - grouped:
        raise ValueError(f'Tree labels not in the grouping: {sorted(tree_labels - grouped)}')
    if grouped - tree_labels:
        raise ValueError(f'Grouped labels not in the tree: {sorted(grouped - tree_labels)}')

    tree = build_16nxb_tree(src, groups, colors, weights, grouping_path.name)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_tree = out_dir / TREE_NAME
    tree.to_json(out_tree)
    load_tree(out_tree)  # re-read: the saved file is a valid tree

    l2n = src.label_to_node
    report = dict(input_tree=str(tree_path), grouping=str(grouping_path), weights=str(weights_path),
                  output=str(out_tree), colors=colors,
                  classes=[dict(new_label=i, name=cls, node=node_name(cls),
                                w_mu=None if i == 0 else weights[cls],
                                original_labels={str(l): l2n[l] for l in labs})
                           for i, (cls, labs) in enumerate(groups)])
    with open(out_dir / 'tree_report.json', 'w') as f:
        json.dump(report, f, indent=1)

    log(f'  {len(tree_labels)} labels -> {len(groups)} classes, colors {colors}')
    for i, (cls, labs) in enumerate(groups):
        n = tree.nodes[node_name(cls)]
        extra = ' (null)' if n.null else f'  w_mu {n.w_mu}' + (f'  w_sigma {[round(x, 3) for x in n.w_sigma]}'
                                                               if n.w_sigma else '')
        log(f'  {i:2d} {cls:42s} {len(labs):3d} labels{extra}')
    log(f'  saved {out_tree}')
    return out_tree


# ================================================================================================ #
# Step 2: label maps                                                                               #
# ================================================================================================ #

def relabel_case(seg_path, out_dir, lut):
    t0 = time.time()
    img = nib.load(seg_path)
    seg = np.asanyarray(img.dataobj)
    if np.issubdtype(seg.dtype, np.floating):
        seg = np.rint(seg)
    seg = seg.astype(np.int64)
    present = np.unique(seg)
    bad = present[(present >= len(lut)) | (lut[np.minimum(present, len(lut) - 1)] < 0)]
    if bad.size:
        raise ValueError(f'{seg_path.name}: labels not in the grouping: {bad.tolist()}')
    out = lut[seg].astype(np.uint8)
    hdr = img.header.copy()
    hdr.set_data_dtype(np.uint8)
    hdr['scl_slope'], hdr['scl_inter'] = 1, 0
    out_dir.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(out, img.affine, hdr), out_dir / seg_path.name)
    return seg_path.name, len(present), np.unique(out).tolist(), time.time() - t0


def step_labels(seg_dir, grouping_path, out_dir, jobs):
    log(f'Step 2: label maps ({seg_dir})')
    files = sorted(p for p in seg_dir.glob('*.nii*')
                   if not p.name.endswith(('_steps.nii.gz', '_removed.nii.gz')))
    lut = label_lut(load_grouping(grouping_path))
    out_labels = out_dir / 'labels'
    log(f'  {len(files)} cases, {jobs} in parallel')
    with ProcessPoolExecutor(jobs) as ex:
        for name, n_in, present, sec in ex.map(relabel_case, files, [out_labels] * len(files),
                                               [lut] * len(files)):
            log(f'  {name}: {n_in} labels -> {len(present)} {present} in {sec:.0f} s')
    log(f'  saved in {out_labels}')


# ================================================================================================ #

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--tree', type=Path, default=DEFAULT_TREE, help='simplified AllenLabelTree JSON')
    p.add_argument('--seg-dir', type=Path, default=DEFAULT_SEG_DIR, help='simplified label maps')
    p.add_argument('--grouping', type=Path, default=DEFAULT_GROUPING, help='GMM_label_grouping.yaml')
    p.add_argument('--weights', type=Path, default=DEFAULT_WEIGHTS, help='GMM_weights.yaml')
    p.add_argument('--out-dir', type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument('--step', choices=['all', 'tree', 'labels'], default='all')
    p.add_argument('--jobs', type=int, default=5, help='cases processed in parallel (default 5)')
    args = p.parse_args()

    if args.step in ('all', 'tree'):
        step_tree(args.tree, args.grouping, args.weights, args.out_dir)
    if args.step in ('all', 'labels'):
        step_labels(args.seg_dir, args.grouping, args.out_dir, args.jobs)


if __name__ == '__main__':
    main()
