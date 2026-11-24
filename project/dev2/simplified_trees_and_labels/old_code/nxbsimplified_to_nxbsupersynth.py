#!/usr/bin/env python3
"""
SimplifiedNextBrain -> SuperSynthNextBrain: convert the labels to SuperSynth labels with
nextbrain-utils (libraries/nextbrain-utils, nextbrain_utils/to_supersynth.py) and build the
matching tree.

How nextbrain_utils.to_supersynth works
---------------------------------------
  get_nextbrain2supersynth_map(side, claustrum) builds a lookup table NextBrain id -> SuperSynth
  id from lut/NextBrainToSuperSynth.yaml ({SuperSynth class: [NextBrain labels]}):
    * lateralised classes (Thalamus, Putamen, ...) get the Left-/Right- id of the given side;
    * the cortex parcels (NextBrain 2001-2035, ctx-rh-*) are NOT in the YAML: they keep their
      aparc id, 20xx on the right side, 10xx on the left side;
    * claustrum=False maps the Claustrum to Cerebral-White-Matter;
    * any other label not in the YAML keeps its NextBrain id (identity).
  to_supersynth(image, side) applies that table to a label map. A NextBrain case is one
  hemisphere, so every case is converted with one side.

IMPORTANT: corrected mapping file
---------------------------------
  The YAML shipped with nextbrain-utils has five problems for our labels (simplified tree):
    611  white_matter_of_hindbrain is mapped to Brain-Stem, but NextBrain paints the cerebellar
         white matter with 611 (ontology: "label used for WM of the cerebellum"; 846 is unused).
         Same problem as in GMM_label_grouping.yaml (see allennxb_to_16nxb.py).
    238, 240  are in both Cerebral-Cortex and Amygdala. The library loop lets the last one
         (Amygdala) win; the corrected file keeps them only in Amygdala.
    3    cerebral_gyri_and_lobules_l2 (unparcellated cortex) is not in the YAML: it kept id 3,
         which is Left-Cerebral-Cortex even for a right hemisphere.
    114  optic_nerve is not in the YAML: it kept id 114, which is not a SuperSynth label.
         The corrected file puts it in Optic-Chiasm (same fibre tract, continuous with the
         chiasm): only the classes of the original YAML are used, no new class.
    2000-2035  cortical parcels (ctx-*, aparc) are not in the YAML: they kept their aparc id
         (20xx, or 10xx on the left side). SuperSynth has no cortical parcellation, so the
         corrected file maps them all to Cerebral-Cortex: the whole cortex is one label
         (3 on the left side, 42 on the right side).
  The default mapping is therefore data/inptus/nxbsupersynth/NextBrainToSuperSynth_corrected.yaml
  (copy of the library file with these fixes, see its header). The library itself is not
  modified: its module path PATH_MAPPING is pointed to the corrected file.

Hemisphere side
---------------
  By default ALL cases are converted as LEFT hemispheres (--side L), so that every case uses
  the same label ids (Left-*, e.g. the whole cortex = 3) whatever hemisphere it really is.
  The tree then only carries the left ids. The images are not flipped: only the label ids
  are homogenised.
  (Detected anatomy: case-1..3 are right hemispheres, case-4..5 left.)

  --side R     all cases as right hemispheres (Right-*, e.g. the whole cortex = 42)
  --side auto  the side of every case is detected from the anatomy: the thalamus lies on the
               medial side, so if it is on the +x half of the cortex extent the medial side faces
               +x (= right), i.e. a LEFT hemisphere (RAS). The tree carries both ids.
               --sides case-1=R ... overrides the detection for some cases.

Step 1: tree
------------
  The SuperSynth classes cut across the Allen hierarchy (Thalamus, Hippocampus, Brain-Stem...
  take labels from several Allen level-1 classes), so the tree is flat:

      root
       |- background                  level 1, labels [0], null
       |- cerebral_white_matter       level 1, labels [2]              (Left id with --side L)
       |- cerebral_cortex             level 1, labels [3]              (whole cortex, parcels included)
       |- thalamus                    level 1, labels [10]
       |- brain_stem                  level 1, labels [16]             (not lateralised)
       |- ...

  Every node carries the ids of the chosen side (both sides with --side auto).
  colors = one colour per level-1 class (one-hot w_mu): independent intensities per class.
  tree_report.json lists the original labels of every class.
  (With a mapping that keeps the aparc parcels, e.g. the library YAML, they become level-2
  children of cerebral_cortex, inheriting its colour, with wr_mu / wr_sigma / r_max copied from
  the simplified tree.)

Step 2: label maps
------------------
  Every case is converted with nextbrain_utils.to_supersynth(image, side, claustrum). The
  output is checked: every label must be in the SuperSynth tree (catches unmapped labels).

Outputs (--out-dir)
-------------------
  nextbrain_tree_nxbsupersynth.json   SuperSynth tree
  tree_report.json                    per class: ids (L, R) and original NextBrain labels
  labels/<case>.nii.gz                SuperSynth label maps
  labels/sides.json                   side used for every case (and how it was found)

Usage
-----
    python allennxb_to_nxbsupersynth.py                       # both steps, default paths
    python allennxb_to_nxbsupersynth.py --step tree           # only step 1
    python allennxb_to_nxbsupersynth.py --side auto           # side detected per case
    python allennxb_to_nxbsupersynth.py --claustrum           # keep the claustrum as a class
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
sys.path.insert(0, str(PROJECT / 'libraries/nextbrain-utils'))
from nextbrain_utils import to_supersynth as ts  # noqa: E402
from nextbrain_utils.io import load_lut, load_yaml  # noqa: E402
from trees.tree_allen import AllenLabelTree  # noqa: E402

DEFAULT_TREE = HERE / 'data/outputs/nxbsimplified/nextbrain_tree_allen_gyral_simplified.json'
DEFAULT_MAPPING = HERE / 'data/inptus/nxbsupersynth/NextBrainToSuperSynth_corrected.yaml'
# DEFAULT_MAPPING = HERE / 'data/inptus/nxbsupersynth/NextBrainToSuperSynth.yaml'
TREE_NAME = 'nextbrain_tree_nxbsupersynth.json'
PARCELS = range(2001, 2036)         # NextBrain cortex parcels (ctx-rh-*), used by detect_side

DEFAULT_SEG_DIR = HERE / 'data/outputs/nxbsimplified/labels'
DEFAULT_OUT_DIR = HERE / 'data/outputs/nxbsupersynth'

# DEFAULT_SEG_DIR = 'study_conversions/data/outputs/nxbsimplified/labels'
# DEFAULT_OUT_DIR = HERE / 'study_conversions/data/outputs/nxbsupersynth'


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_tree(path):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')          # r_max warnings of level-1 nodes
        return AllenLabelTree.from_json(path)


def use_mapping(path):
    """Point nextbrain_utils.to_supersynth to our (corrected) mapping YAML."""
    ts.PATH_MAPPING = str(path)


def node_name(name):
    """'Left-Cerebral-White-Matter' -> 'cerebral_white_matter', 'ctx-rh-bankssts' -> 'ctx_bankssts'."""
    name = re.sub(r'^(Left|Right)-', '', name)
    name = re.sub(r'^ctx-[lr]h-', 'ctx-', name)
    return re.sub(r'[^0-9a-z]+', '_', name.lower()).strip('_')


# ================================================================================================ #
# Step 1: tree                                                                                     #
# ================================================================================================ #

def supersynth_classes(src, claustrum):
    """
    Group the labels of the simplified tree by their (Left id, Right id) SuperSynth pair.
    Returns {(id_L, id_R): [original labels]}, in increasing order of the left id.
    """
    map_l = ts.get_nextbrain2supersynth_map('L', claustrum)
    map_r = ts.get_nextbrain2supersynth_map('R', claustrum)
    classes = {}
    for lab in sorted(src.label_to_node):
        classes.setdefault((int(map_l[lab]), int(map_r[lab])), []).append(lab)
    return dict(sorted(classes.items()))


def side_ids(pair, side):
    """Ids of a (Left id, Right id) pair carried by the tree for `side` ('L', 'R' or 'auto')."""
    return [pair[0]] if side == 'L' else [pair[1]] if side == 'R' else sorted(set(pair))


def build_supersynth_tree(src, classes, mapping_name, side):
    supersynth = load_lut(ts.PATH_SUPERSYNTH).name2label()
    id2name = {v: k for k, v in supersynth.items()}
    l2n = src.label_to_node

    level1 = {pair: labs for pair, labs in classes.items() if pair[0] not in range(1001, 1036)}
    parcels = {pair: labs for pair, labs in classes.items() if pair[0] in range(1001, 1036)}
    unknown = [pair for pair in level1 if pair[0] not in id2name or pair[1] not in id2name]
    if unknown:
        raise ValueError(f'SuperSynth ids not in {ts.PATH_SUPERSYNTH}: {unknown} '
                         f'(NextBrain labels {[level1[p] for p in unknown]}): fix the mapping YAML')

    names = {pair: node_name(id2name[pair[0]]) for pair in level1}
    colors = [n for pair, n in names.items() if pair != (0, 0)]
    tree = AllenLabelTree.new(
        colors, contrasts=src.contrasts, version='nextbrain-nxbsupersynth-0.1',
        description=(f'NextBrain labels converted to SuperSynth with nextbrain_utils.to_supersynth '
                     f'({mapping_name}). Flat tree (the classes cut across the Allen hierarchy), one '
                     f'colour per class; label ids of side {side} (auto = both sides). Built from '
                     f'{src.version} with allennxb_to_nxbsupersynth.py.'))
    for pair, labs in level1.items():
        ids = side_ids(pair, side)
        desc = f'{" / ".join(id2name[i] for i in ids)}. Original labels: {labs}'
        if pair == (0, 0):
            tree.add_node('background', tree.root, labels=[0], null=True, description=desc)
            continue
        w = [0.0] * len(colors)
        w[colors.index(names[pair])] = 1.0
        tree.add_node(names[pair], tree.root, labels=ids, w_mu=w, description=desc)

    if parcels:
        cortex = tree.nodes.get('cerebral_cortex')
        if cortex is None:
            raise ValueError('Cortex parcels but no Cerebral-Cortex class')
        for pair, labs in parcels.items():
            orig = src.nodes[l2n[labs[0]]]
            tree.add_node(node_name(orig.name), cortex.name, labels=side_ids(pair, side),
                          wr_mu=orig.wr_mu, wr_sigma=orig.wr_sigma, r_max=orig.r_max,
                          description=f'aparc parcel. Original labels: {labs}')
    tree.validate()
    return tree, id2name


def step_tree(tree_path, mapping_path, claustrum, out_dir, side):
    log(f'Step 1: tree ({tree_path.name}, mapping {mapping_path.name})')
    use_mapping(mapping_path)
    src = load_tree(tree_path)
    classes = supersynth_classes(src, claustrum)
    tree, id2name = build_supersynth_tree(src, classes, mapping_path.name, side)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_tree = out_dir / TREE_NAME
    tree.to_json(out_tree)
    load_tree(out_tree)  # re-read: the saved file is a valid tree

    l2n = src.label_to_node
    report = dict(input_tree=str(tree_path), mapping=str(mapping_path), claustrum=claustrum, side=side,
                  output=str(out_tree), colors=tree.colors,
                  classes=[dict(left=l, right=r, name=id2name.get(l, f'aparc {l}'),
                                original_labels={str(x): l2n[x] for x in labs})
                           for (l, r), labs in classes.items()])
    with open(out_dir / 'tree_report.json', 'w') as f:
        json.dump(report, f, indent=1)

    n_parcels = sum(l in range(1001, 1036) for l, _ in classes)
    log(f'  {len(src.label_to_node)} labels -> {len(classes) - n_parcels} classes'
        + (f' + {n_parcels} aparc parcels' if n_parcels else ''))
    for (l, r), labs in classes.items():
        if l not in range(1001, 1036):
            log(f'    L {l:4d} / R {r:4d}  {id2name[l]:32s} {len(labs):3d} labels')
    log(f'  saved {out_tree}')
    return out_tree


# ================================================================================================ #
# Step 2: label maps                                                                               #
# ================================================================================================ #

def detect_side(seg, affine, thalamus_labels):
    """
    'L' or 'R' from the anatomy: the thalamus is medial. Position of the thalamus centroid along
    world x within the cortex (parcels) extent: > 0.5 -> medial side towards +x -> left hemisphere.
    """
    sub = seg[::2, ::2, ::2]
    aff = affine.copy()
    aff[:3, :3] *= 2

    def world_x(mask):
        idx = np.argwhere(mask)
        return idx @ aff[0, :3] + aff[0, 3]

    xt, xc = world_x(np.isin(sub, thalamus_labels)), world_x(np.isin(sub, list(PARCELS)))
    if xt.size == 0 or xc.size == 0:
        raise ValueError('No thalamus or cortex parcels: cannot detect the side, use --sides')
    lo, hi = np.percentile(xc, [1, 99])
    pos = float((xt.mean() - lo) / (hi - lo))
    return ('L' if pos > 0.5 else 'R'), pos


def convert_case(seg_path, out_dir, mapping_path, claustrum, side, valid_ids):
    t0 = time.time()
    use_mapping(mapping_path)                    # also in the worker process
    img = nib.load(seg_path)
    how = 'fixed (--side / --sides)'
    if side is None:
        thal = load_yaml(mapping_path)['Thalamus']
        side, pos = detect_side(np.asanyarray(img.dataobj), img.affine, thal)
        how = f'detected: thalamus at {pos:.2f} of the cortex x-extent'

    out = ts.to_supersynth(img, side=side, claustrum=claustrum, save=False)
    dat = np.asanyarray(out.dataobj)
    present = np.unique(dat).tolist()
    bad = sorted(set(present) - valid_ids)
    if bad:
        raise ValueError(f'{seg_path.name}: labels not in the SuperSynth tree: {bad}')

    out_dir.mkdir(parents=True, exist_ok=True)
    hdr = img.header.copy()
    hdr.set_data_dtype(np.int16)
    hdr['scl_slope'], hdr['scl_inter'] = 1, 0
    nib.save(nib.Nifti1Image(dat.astype(np.int16), img.affine, hdr), out_dir / seg_path.name)
    return seg_path.name, side, how, len(present), time.time() - t0


def parse_sides(items):
    sides = {}
    for it in items or []:
        case, side = it.split('=')
        side = side.upper()[0]
        if side not in 'LR':
            raise ValueError(f'--sides {it}: side must be L or R')
        sides[case] = side
    return sides


def step_labels(seg_dir, mapping_path, claustrum, out_dir, jobs, side, sides):
    log(f'Step 2: label maps ({seg_dir})')
    tree_path = out_dir / TREE_NAME
    if not tree_path.exists():
        sys.exit(f'{tree_path} not found: run --step tree first')
    valid_ids = set(load_tree(tree_path).label_to_node)
    files = sorted(p for p in seg_dir.glob('*.nii*')
                   if not p.name.endswith(('_steps.nii.gz', '_removed.nii.gz')))
    out_labels = out_dir / 'labels'
    log(f'  {len(files)} cases, {jobs} in parallel')

    case = lambda p: p.name.replace('.nii.gz', '').replace('.nii', '')
    used = {}
    with ProcessPoolExecutor(jobs) as ex:
        futs = [ex.submit(convert_case, f, out_labels, mapping_path, claustrum,
                          side if side != 'auto' else sides.get(case(f)), valid_ids) for f in files]
        for fut in futs:
            name, side, how, n, sec = fut.result()
            used[name] = dict(side=side, how=how)
            log(f'  {name}: side {side} ({how}), {n} labels, {sec:.0f} s')
    with open(out_labels / 'sides.json', 'w') as f:
        json.dump(used, f, indent=1)
    log(f'  saved in {out_labels}')


# ================================================================================================ #

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--tree', type=Path, default=DEFAULT_TREE, help='simplified AllenLabelTree JSON')
    p.add_argument('--seg-dir', type=Path, default=DEFAULT_SEG_DIR, help='simplified label maps')
    p.add_argument('--mapping', type=Path, default=DEFAULT_MAPPING,
                   help='NextBrain -> SuperSynth YAML (default: the corrected copy, see top)')
    p.add_argument('--out-dir', type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument('--step', choices=['all', 'tree', 'labels'], default='all')
    p.add_argument('--claustrum', action='store_true',
                   help='keep the claustrum as a class (default: merged into the cerebral WM, as in the library)')
    p.add_argument('--side', choices=['L', 'R', 'auto'], default='R',
                   help='side used for ALL cases (default L, homogenised ids) or auto = detected per case')
    p.add_argument('--sides', nargs='*', default=None, metavar='CASE=L|R',
                   help='with --side auto: hemisphere of some cases, e.g. case-1=R (others detected)')
    p.add_argument('--jobs', type=int, default=5, help='cases processed in parallel (default 5)')
    args = p.parse_args()

    if args.step in ('all', 'tree'):
        step_tree(args.tree, args.mapping, args.claustrum, args.out_dir, args.side)
    if args.step in ('all', 'labels'):
        step_labels(args.seg_dir, args.mapping, args.claustrum, args.out_dir, args.jobs,
                    args.side, parse_sides(args.sides))


if __name__ == '__main__':
    main()
