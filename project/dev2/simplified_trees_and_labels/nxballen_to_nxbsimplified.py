#!/usr/bin/env python3
"""
---------- MODIFIED TREE ----------
AllenNextBrain -> SimplifiedNextBrain: remove a list of labels from the ontology tree and from
the label maps.

Inputs
------
  --tree               AllenLabelTree JSON (ontology/data/outputs/nextbrain_tree_allen_gyral.json)
  --labels-to-remove   YAML {fill: [labels]} read with nxb_label_filling.load_groups:
                           null: [386, 401, ...]    # filled with the nearest kept region
                           0:    [50]               # filled with 0
                           3:    [2001, 2002, ...]  # filled with 3
                       Every listed label (of any group) is removed from the tree. Constant fills
                       are applied first, so their values compete in the nearest-region fill.
  --seg-dir            folder with the label maps
                       (atlas_subjects/data/outputs/adria/recon_holes_0.2mm_improved)

Step 1: tree simplification
---------------------------
  1. The labels to remove are taken out of the `labels` list of the nodes that carry them.
  2. Pruning: every node left with no label in its whole subtree (itself + descendants) is
     removed, bottom-up. This removes the leaves of the removed labels and also the parents
     that only grouped removed labels (e.g. a nucleus group whose nuclei are all removed,
     possibly up to a level-1 class). The root and the nodes that still have labels are kept.
     Nodes left with a single child are NOT collapsed: the hierarchy above the remaining
     labels is unchanged.
  The result is validated (AllenLabelTree checks parent/children/levels/labels).

Step 2: label maps
------------------
  For every case:
  1. Every voxel with a label to remove is set to 0. A map of the removed voxels is saved
     (value = the original label, 0 elsewhere): the voxels that have to be filled.
  2. The removed voxels are filled with nxb_label_filling.fill_labels
     (libraries/new_nextbrain_utils, see its docstring for the details): constant groups get
     their value, `null` labels get the label of the nearest kept region (Euclidean distance in
     mm). Background (0) does not compete by default, so removed structures are filled with
     tissue; --background-competes lets voxels closer to the background become background.
  3. Check: every label of the output exists in the simplified tree (error otherwise).

Outputs (--out-dir)
-------------------
  <tree name>_simplified.json           simplified tree
  tree_report.json                      removed labels / pruned nodes
  labels/<case>.nii.gz                  simplified label map
  labels/<case>_removed.nii.gz          removed voxels (original label, 0 elsewhere)   [only with --write-report]
  labels/<case>_report.json             per removed label: voxels and value(s) used     [only with --write-report]

Usage
-----
    python nxballen_to_nxbsimplified.py                  # both steps with the default paths
    python nxballen_to_nxbsimplified.py --step tree      # only step 1
    python nxballen_to_nxbsimplified.py --step labels    # only step 2
"""
import argparse
import json
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
from nxb_label_filling import fill_labels, load_groups  # noqa: E402

DEFAULT_TREE = PROJECT / 'ontology/data/outputs/nextbrain_tree_allen_gyral.json'
DEFAULT_LABELS = HERE / 'data/inptus/nxbsimplified/labels_to_remove.yaml'
DEFAULT_SEG_DIR = PROJECT / 'atlas_subjects/data/outputs/restored_adria/recon_holes_0.2mm_improved'
DEFAULT_OUT_DIR = HERE / 'data/outputs/nxbsimplified'

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_labels_to_remove(path):
    """(groups {fill: [labels]}, sorted list of all the labels to remove). fill = None
    (nearest kept region) or an int (constant value)."""
    groups = load_groups(path)
    return groups, sorted(l for labs in groups.values() for l in labs)


def load_tree(path):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')          # r_max warnings of level-1 nodes
        return AllenLabelTree.from_json(path)


# ================================================================================================ #
# Step 1: tree simplification                                                                      #
# ================================================================================================ #

def remove_labels_from_tree(tree, labels):
    """
    Remove `labels` from `tree` (in place) and prune the nodes left without labels.
    Returns a report dict: removed labels with their node, pruned nodes.
    """
    labels = set(labels)
    l2n = tree.label_to_node
    missing = sorted(labels - set(l2n))
    if missing:
        raise ValueError(f'Labels not in the tree: {missing}')

    # 1. take the labels out of their nodes
    removed = {lab: l2n[lab] for lab in sorted(labels)}
    for node in tree.nodes.values():
        node.labels = [lab for lab in node.labels if lab not in labels]

    # 2. prune, bottom-up (reverse breadth-first order: children before parents)
    pruned = []
    for name in reversed(tree.order()):
        node = tree.nodes[name]
        if name == tree.root or node.labels or node.children:
            continue
        parent = tree.nodes[node.parent]
        parent.children.remove(name)
        del tree.nodes[name]
        pruned.append(dict(name=name, level=node.level, parent=node.parent))

    tree.validate()
    return dict(removed_labels={str(k): v for k, v in removed.items()}, pruned_nodes=pruned)


def step_tree(tree_path, labels, out_dir):
    log(f'Step 1: tree simplification ({tree_path.name})')
    tree = load_tree(tree_path)
    n_nodes, n_labels = len(tree.nodes), len(tree.label_to_node)

    report = remove_labels_from_tree(tree, labels)
    tree.version = f'{tree.version}-nxbsimplified'
    tree.description = (f'{tree.description} Simplified (nxballen_to_nxbsimplified.py): '
                        f'{len(labels)} labels removed and filled (nearest region or constant); '
                        f'nodes left without labels pruned.')

    out_dir.mkdir(parents=True, exist_ok=True)
    out_tree = out_dir / f'{tree_path.stem}_simplified.json'
    tree.to_json(out_tree)
    report.update(input=str(tree_path), output=str(out_tree),
                  nodes_before=n_nodes, nodes_after=len(tree.nodes),
                  labels_before=n_labels, labels_after=len(tree.label_to_node))
    with open(out_dir / 'tree_report.json', 'w') as f:
        json.dump(report, f, indent=1)

    log(f'  nodes {n_nodes} -> {len(tree.nodes)} ({len(report["pruned_nodes"])} pruned), '
        f'labels {n_labels} -> {len(tree.label_to_node)}')
    for p in report['pruned_nodes']:
        log(f'    pruned (level {p["level"]}): {p["name"]}')
    log(f'  saved {out_tree}')
    return out_tree


# ================================================================================================ #
# Step 2: label maps                                                                               #
# ================================================================================================ #

def simplify_case(seg_path, out_dir, groups, margin_mm, background_competes, save_report):
    t0 = time.time()
    img = nib.load(seg_path)
    seg = np.asanyarray(img.dataobj)
    case = seg_path.name.replace('.nii.gz', '').replace('.nii', '')

    # 1-2. remove the labels and fill them: constant values, then nearest kept region
    out, fill_report = fill_labels(seg, groups, img.header.get_zooms()[:3], margin_mm,
                                   background_competes)
    report = {**fill_report['constant'], **fill_report['nearest']}
    n_removed = sum(r['voxels'] for r in report.values())

    # 3. save
    out_dir.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(out.astype(img.get_data_dtype()), img.affine, img.header),
             out_dir / f'{case}.nii.gz')
    if save_report:
        # map of the removed voxels (original label, 0 elsewhere)
        removed_map = np.where(np.isin(seg, [l for labs in groups.values() for l in labs]), seg, 0)
        nib.save(nib.Nifti1Image(removed_map.astype(img.get_data_dtype()), img.affine, img.header),
                 out_dir / f'{case}_removed.nii.gz')
        with open(out_dir / f'{case}_report.json', 'w') as f:
            json.dump(dict(input=str(seg_path), removed_voxels=n_removed,
                           still_unfilled=fill_report.get('still_unfilled', 0),
                           labels=report), f, indent=1)
    return case, n_removed, len(report), np.unique(out).tolist(), time.time() - t0


def step_labels(seg_dir, groups, tree_path, out_dir, jobs, margin_mm, background_competes,
                save_report):
    log(f'Step 2: label maps ({seg_dir})')
    files = sorted(p for p in seg_dir.glob('*.nii*')
                   if not p.name.endswith(('_steps.nii.gz', '_removed.nii.gz')))
    tree_labels = set(load_tree(tree_path).label_to_node)
    out_labels = out_dir / 'labels'
    n_labels = sum(len(labs) for labs in groups.values())
    log(f'  {len(files)} cases, {n_labels} labels to remove, {jobs} in parallel')

    with ProcessPoolExecutor(jobs) as ex:
        futs = [ex.submit(simplify_case, f, out_labels, groups, margin_mm, background_competes,
                          save_report) for f in files]
        for fut in futs:
            case, n_vox, n_lab, present, sec = fut.result()
            log(f'  {case}: {n_lab} labels removed ({n_vox:,} voxels) in {sec:.0f} s')
            not_in_tree = sorted(set(present) - tree_labels - {0})
            if not_in_tree:
                raise ValueError(f'{case}: labels not in the simplified tree: {not_in_tree}')
    log(f'  saved in {out_labels}')


# ================================================================================================ #

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--tree', type=Path, default=DEFAULT_TREE, help='AllenLabelTree JSON')
    p.add_argument('--labels-to-remove', type=Path, default=DEFAULT_LABELS, help='YAML {fill: [labels]}')
    p.add_argument('--seg-dir', type=Path, default=DEFAULT_SEG_DIR, help='folder with the label maps')
    p.add_argument('--out-dir', type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument('--step', choices=['all', 'tree', 'labels'], default='all')
    p.add_argument('--jobs', type=int, default=5, help='cases processed in parallel (default 5)')
    p.add_argument('--margin-mm', type=float, default=2.0,
                   help='initial margin around each label bbox (default 2 mm, doubled if needed)')
    p.add_argument('--background-competes', action='store_true',
                   help='background (0) also competes in the nearest-region fill')
    p.add_argument('--write-report', action='store_true',
                   help='also save <case>_removed.nii.gz and <case>_report.json')
    args = p.parse_args()

    groups, labels = load_labels_to_remove(args.labels_to_remove)
    out_tree = args.out_dir / f'{args.tree.stem}_simplified.json'
    if args.step in ('all', 'tree'):
        out_tree = step_tree(args.tree, labels, args.out_dir)
    if args.step in ('all', 'labels'):
        if not out_tree.exists():
            sys.exit(f'{out_tree} not found: run --step tree first')
        step_labels(args.seg_dir, groups, out_tree, args.out_dir, args.jobs, args.margin_mm,
                    args.background_competes, args.write_report)


if __name__ == '__main__':
    main()
