#!/usr/bin/env python3
"""
AllenNextBrain -> SimplifiedNextBrain: remove a list of labels from the ontology tree and from
the label maps.

Inputs
------
  --tree               AllenLabelTree JSON (ontology/data/outputs/nextbrain_tree_allen_gyral.json)
  --labels-to-remove   JSON whose keys are the label ids to remove, e.g. {"386": null, "401": null}
  --seg-dir            folder with the label maps (atlas_subjects/data/outputs/restored/nn)

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
  2. For every removed label:
     a. Bounding box of that label (+ a margin) to restrict the search and memory.
     b. Distance from every voxel of the label to the neighbouring regions. Instead of one
        distance map per neighbouring region and an argmin over them, one Euclidean distance
        transform (scipy.ndimage.distance_transform_edt, in mm, with return_indices) gives
        for every voxel the nearest voxel of ANY kept region, i.e. the same nearest label.
        "Kept region" = any label that is not removed. Background (0) does not compete by
        default (the removed structure is filled with tissue); use --background-competes to
        let voxels closer to the background than to tissue become background.
     c. The voxel gets the label of its nearest kept region.
     The margin grows (x2) until every voxel of the label has its nearest kept voxel within
     the margin, so the result is the same as computing it on the whole volume.
  3. Check: every label of the output exists in the simplified tree.

Outputs (--out-dir)
-------------------
  <tree name>_simplified.json           simplified tree
  tree_report.json                      removed labels / pruned nodes
  labels/<case>.nii.gz                  simplified label map
  labels/<case>_removed.nii.gz          removed voxels (original label, 0 elsewhere)
  labels/<case>_report.json             per removed label: voxels and labels used to fill them

Usage
-----
    python allennxb_to_simplifiednxb.py                  # both steps with the default paths
    python allennxb_to_simplifiednxb.py --step tree      # only step 1
    python allennxb_to_simplifiednxb.py --step labels    # only step 2
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
from scipy import ndimage as ndi

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
sys.path.insert(0, str(PROJECT / 'libraries'))
from trees.tree_allen import AllenLabelTree  # noqa: E402

DEFAULT_TREE = PROJECT / 'ontology/data/outputs/nextbrain_tree_allen_gyral.json'
DEFAULT_LABELS = HERE / 'data/inptus/simplifiednxb/labels_to_remove.json'
DEFAULT_SEG_DIR = PROJECT / 'atlas_subjects/data/outputs/restored/nn'
DEFAULT_OUT_DIR = HERE / 'data/outputs/simplifiednxb'


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_labels_to_remove(path):
    """Label ids to remove (keys of the JSON file)."""
    with open(path) as f:
        d = json.load(f)
    if any(v is not None for v in d.values()):
        raise ValueError(f'{path}: only null values are supported (label -> removed and filled '
                         f'with the nearest region)')
    return sorted(int(k) for k in d)


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
    tree.version = f'{tree.version}-simplifiednxb'
    tree.description = (f'{tree.description} Simplified (allennxb_to_simplifiednxb.py): '
                        f'{len(labels)} labels removed and filled with the nearest region; '
                        f'nodes left without labels pruned.')

    out_dir.mkdir(parents=True, exist_ok=True)
    out_tree = out_dir / f'{tree_path.stem}_simplified.json'
    tree.to_json(out_tree)
    report.update(input=str(tree_path), output=str(out_tree),
                  nodes_before=n_nodes, nodes_after=len(tree.nodes),
                  labels_before=n_labels, labels_after=len(tree.label_to_node))
    with open(out_dir / 'tree_report.json', 'w') as f:
        json.dump(report, f, indent=1)

    load_tree(out_tree)  # re-read: the saved file is a valid tree
    log(f'  nodes {n_nodes} -> {len(tree.nodes)} ({len(report["pruned_nodes"])} pruned), '
        f'labels {n_labels} -> {len(tree.label_to_node)}')
    for p in report['pruned_nodes']:
        log(f'    pruned (level {p["level"]}): {p["name"]}')
    log(f'  saved {out_tree}')
    return out_tree


# ================================================================================================ #
# Step 2: label maps                                                                               #
# ================================================================================================ #

def fill_label(seg, removed_map, lab, bbox, zooms, margin_mm, background_competes):
    """
    Fill the voxels of one removed label with the label of the nearest kept region.

    seg          label map, the removed labels are already 0 (modified in place)
    removed_map  original label of the removed voxels, 0 elsewhere
    lab          the removed label to fill
    bbox         tuple of slices, bounding box of `lab` (from ndi.find_objects)
    Returns (number of voxels, {fill label: number of voxels}).
    """
    zooms = np.asarray(zooms, float)
    margin = margin_mm
    while True:
        pad = np.ceil(margin / zooms).astype(int)
        box = tuple(slice(max(0, s.start - p), min(n, s.stop + p))
                    for s, p, n in zip(bbox, pad, seg.shape))
        sub = seg[box]
        rm = removed_map[box]
        target = rm == lab
        # kept regions: any voxel with a label that is not removed (+ background if it competes)
        known = (rm == 0) if background_competes else (sub > 0)
        if known.any():
            # EDT of "not known": distance (mm) to the nearest known voxel and its index
            dist, ind = ndi.distance_transform_edt(~known, sampling=zooms, return_indices=True)
            d_max = float(dist[target].max())
            whole = all(b.start == 0 and b.stop == n for b, n in zip(box, seg.shape))
            # every voxel of the label is >= margin from the box border, so a nearest voxel
            # found at distance <= margin is the true (whole-volume) nearest voxel
            if d_max <= margin or whole:
                src = tuple(i[target] for i in ind)
                values = sub[src]
                sub[target] = values
                fill, cnt = np.unique(values, return_counts=True)
                return int(target.sum()), {int(f): int(c) for f, c in zip(fill, cnt)}
        elif all(b.start == 0 and b.stop == n for b, n in zip(box, seg.shape)):
            return int(target.sum()), {}         # nothing kept in the whole volume
        margin *= 2


def simplify_case(seg_path, out_dir, labels, margin_mm, background_competes):
    t0 = time.time()
    img = nib.load(seg_path)
    seg = np.asanyarray(img.dataobj)
    if np.issubdtype(seg.dtype, np.floating):
        seg = np.rint(seg).astype(np.int32)
    seg = seg.copy()
    zooms = img.header.get_zooms()[:3]
    case = seg_path.name.replace('.nii.gz', '').replace('.nii', '')

    # 1. remove the labels, keep a map of what was removed
    removed_any = np.isin(seg, labels)
    removed_map = np.where(removed_any, seg, 0).astype(seg.dtype)
    seg[removed_any] = 0

    # 2. fill every removed label from its nearest kept region
    bboxes = ndi.find_objects(removed_map)       # bboxes[l - 1] = bbox of label l (or None)
    report = {}
    for lab in labels:
        if lab > len(bboxes) or bboxes[lab - 1] is None:
            continue
        n_vox, filled = fill_label(seg, removed_map, lab, bboxes[lab - 1], zooms, margin_mm,
                                   background_competes)
        report[str(lab)] = dict(voxels=n_vox, filled_with=filled)

    # 3. save
    out_dir.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(seg.astype(img.get_data_dtype()), img.affine, img.header),
             out_dir / f'{case}.nii.gz')
    nib.save(nib.Nifti1Image(removed_map.astype(img.get_data_dtype()), img.affine, img.header),
             out_dir / f'{case}_removed.nii.gz')
    with open(out_dir / f'{case}_report.json', 'w') as f:
        json.dump(dict(input=str(seg_path), removed_voxels=int(removed_any.sum()),
                       still_unfilled=int((removed_any & (seg == 0)).sum()),
                       labels=report), f, indent=1)
    return case, int(removed_any.sum()), len(report), np.unique(seg).tolist(), time.time() - t0


def step_labels(seg_dir, labels, tree_path, out_dir, jobs, margin_mm, background_competes):
    log(f'Step 2: label maps ({seg_dir})')
    files = sorted(p for p in seg_dir.glob('*.nii*')
                   if not p.name.endswith(('_steps.nii.gz', '_removed.nii.gz')))
    tree_labels = set(load_tree(tree_path).label_to_node)
    out_labels = out_dir / 'labels'
    log(f'  {len(files)} cases, {len(labels)} labels to remove, {jobs} in parallel')

    with ProcessPoolExecutor(jobs) as ex:
        futs = [ex.submit(simplify_case, f, out_labels, labels, margin_mm, background_competes)
                for f in files]
        for fut in futs:
            case, n_vox, n_lab, present, sec = fut.result()
            not_in_tree = sorted(set(present) - tree_labels)
            log(f'  {case}: {n_lab} labels removed ({n_vox:,} voxels) in {sec:.0f} s' +
                (f'  WARNING labels not in the simplified tree: {not_in_tree}' if not_in_tree else ''))
    log(f'  saved in {out_labels}')


# ================================================================================================ #

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--tree', type=Path, default=DEFAULT_TREE, help='AllenLabelTree JSON')
    p.add_argument('--labels-to-remove', type=Path, default=DEFAULT_LABELS, help='JSON, keys = label ids')
    p.add_argument('--seg-dir', type=Path, default=DEFAULT_SEG_DIR, help='folder with the label maps')
    p.add_argument('--out-dir', type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument('--step', choices=['all', 'tree', 'labels'], default='all')
    p.add_argument('--jobs', type=int, default=5, help='cases processed in parallel (default 5)')
    p.add_argument('--margin-mm', type=float, default=2.0,
                   help='initial margin around each label bbox (default 2 mm, doubled if needed)')
    p.add_argument('--background-competes', action='store_true',
                   help='background (0) also competes in the nearest-region fill')
    args = p.parse_args()

    labels = load_labels_to_remove(args.labels_to_remove)
    out_tree = args.out_dir / f'{args.tree.stem}_simplified.json'
    if args.step in ('all', 'tree'):
        out_tree = step_tree(args.tree, labels, args.out_dir)
    if args.step in ('all', 'labels'):
        if not out_tree.exists():
            sys.exit(f'{out_tree} not found: run --step tree first')
        step_labels(args.seg_dir, labels, out_tree, args.out_dir, args.jobs, args.margin_mm,
                    args.background_competes)


if __name__ == '__main__':
    main()
