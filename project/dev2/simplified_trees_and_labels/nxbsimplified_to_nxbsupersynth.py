#!/usr/bin/env python3
"""
SimplifiedNextBrain -> SuperSynthNextBrain: convert the labels to SuperSynth labels with
nextbrain-utils (libraries/nextbrain-utils, nextbrain_utils/to_supersynth.py) and build the
matching tree.

How nextbrain_utils.to_supersynth works
---------------------------------------
  get_nextbrain2supersynth_map(side, claustrum) builds a lookup table NextBrain id -> SuperSynth
  id from a YAML {SuperSynth class: [NextBrain labels]} (module variable PATH_MAPPING):
    * lateralised classes (Thalamus, Putamen, ...) get the Left-/Right- id of the given side;
    * claustrum=False maps the Claustrum to Cerebral-White-Matter;
    * a label that is not in the YAML keeps its NextBrain id.
  to_supersynth(image, side, claustrum) applies that table to a label map. This script only
  points PATH_MAPPING to our mapping file; the library is not modified.

Mapping file (--mapping)
------------------------
  data/inptus/nxbsupersynth/NextBrainToSuperSynth_corrected.yaml: copy of the library YAML with
  these fixes (see its header):
    238, 240    were in both Cerebral-Cortex and Amygdala: kept only in Amygdala.
    3           unparcellated cortex (also the value used by nxballen_to_nxbsimplified.py to fill
                the right-hemisphere cortex) -> Cerebral-Cortex.
    114         optic nerve -> Optic-Chiasm.
    2000-2035   cortical parcels (aparc) -> Cerebral-Cortex (SuperSynth has no parcellation).

Hemisphere side (--side, default R)
-----------------------------------
  All the cases are converted with the same side, so that every case uses the same label ids
  (e.g. the whole cortex = 42 with R, 3 with L), whatever hemisphere it really is. The images
  are not flipped: only the label ids are homogenised.

Step 1: tree
------------
  A new flat tree, built only from the mapping file (no input tree). The labels of the YAML are
  grouped by the SuperSynth id the library gives them, so the classes are exactly those of the
  label maps (e.g. Claustrum merged into Cerebral-White-Matter without --claustrum):

      root
       |- background                  level 1, labels [0], null
       |- cerebral_white_matter       level 1, labels [41]     (Right id with --side R)
       |- cerebral_cortex             level 1, labels [42]
       |- ...                         one node per class, in increasing order of the left id
       |- brain_stem                  level 1, labels [16]     (not lateralised)

  One principal colour per class (colors = node names; w_mu = one-hot), i.e. the classes are
  sampled independently. The description of every node lists the NextBrain labels of the YAML
  merged into it (labels removed by the simplification included).

Step 2: label maps
------------------
  Every case is converted with nextbrain_utils.to_supersynth(image, side, claustrum) and saved
  as int16. Every label of the output must be in the tree (catches NextBrain labels missing
  from the YAML, which the library keeps unchanged).

Outputs (--out-dir)
-------------------
  nextbrain_tree_nxbsupersynth.json   SuperSynth tree
  labels/<case>.nii.gz                SuperSynth label maps

Usage
-----
    python nxbsimplified_to_nxbsupersynth.py                  # both steps, default paths
    python nxbsimplified_to_nxbsupersynth.py --step tree      # only step 1
    python nxbsimplified_to_nxbsupersynth.py --side L         # Left- ids for all the cases
    python nxbsimplified_to_nxbsupersynth.py --claustrum      # keep the claustrum as a class
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
sys.path.insert(0, str(PROJECT / 'libraries/nextbrain-utils'))
from nextbrain_utils import to_supersynth as ts  # noqa: E402
from nextbrain_utils.io import load_lut, load_yaml  # noqa: E402
from trees.tree_allen import AllenLabelTree  # noqa: E402

DEFAULT_MAPPING = HERE / 'data/inptus/nxbsupersynth/NextBrainToSuperSynth_corrected.yaml'
DEFAULT_SEG_DIR = HERE / 'data/outputs/nxbsimplified/labels'
DEFAULT_OUT_DIR = HERE / 'data/outputs/nxbsupersynth' 
TREE_NAME = 'nextbrain_tree_nxbsupersynth.json'

CONTRASTS = ['T1', 'T2', 'FLAIR', 'PD']


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def node_name(name):
    """'Left-Cerebral-White-Matter' -> 'cerebral_white_matter'."""
    return re.sub(r'[^0-9a-z]+', '_', re.sub(r'^(Left|Right)-', '', name).lower()).strip('_')


def label_mapping(mapping=DEFAULT_MAPPING, side='R', claustrum=False):
    """
    {nxbsimplified label: SuperSynth label} (the rule of step 2) for the labels of the mapping
    YAML (+ background), for the online generator (generator_double_tree), which merges the
    augmented label maps on the fly. Labels missing from the YAML are left out (the generator
    raises an error if a label map contains one).
    """
    ts.PATH_MAPPING = str(mapping)
    lut = ts.get_nextbrain2supersynth_map(side, claustrum)
    labels = {int(l) for labs in load_yaml(ts.PATH_MAPPING).values() for l in labs}
    return {0: 0, **{lab: int(lut[lab]) for lab in sorted(labels)}}


# ================================================================================================ #
# Step 1: tree                                                                                     #
# ================================================================================================ #

def supersynth_classes(claustrum):
    """
    {(Left id, Right id): [NextBrain labels]} for the labels of the mapping YAML, as the library
    maps them, in increasing order of the left id. Background (0, not in the YAML) included.
    """
    map_l = ts.get_nextbrain2supersynth_map('L', claustrum)
    map_r = ts.get_nextbrain2supersynth_map('R', claustrum)
    labels = sorted({int(l) for labs in load_yaml(ts.PATH_MAPPING).values() for l in labs})
    classes = {(0, 0): [0]}
    for lab in labels:
        classes.setdefault((int(map_l[lab]), int(map_r[lab])), []).append(lab)
    return dict(sorted(classes.items()))


def step_tree(side, claustrum, out_dir):
    """Flat tree: root -> one level-1 node per SuperSynth class (background = null)."""
    log(f'Step 1: tree ({Path(ts.PATH_MAPPING).name}, side {side})')
    id2name = load_lut(ts.PATH_SUPERSYNTH).label2name()
    classes = supersynth_classes(claustrum)
    ids = {pair: pair[0] if side == 'L' else pair[1] for pair in classes}
    names = {pair: node_name(id2name[i]) for pair, i in ids.items() if pair != (0, 0)}
    colors = list(names.values())

    tree = AllenLabelTree.new(
        colors, contrasts=CONTRASTS, version='nextbrain-nxbsupersynth-0.2',
        description=f'NextBrain labels converted to SuperSynth with nextbrain_utils.to_supersynth '
                    f'({Path(ts.PATH_MAPPING).name}), label ids of side {side}. Flat tree, one '
                    f'colour per class. Built with nxbsimplified_to_nxbsupersynth.py.')
    tree.add_node('background', tree.root, labels=[0], null=True,
                  description=f'{id2name[0]}. Original labels: [0]')
    for pair, name in names.items():
        tree.add_node(name, tree.root, labels=[ids[pair]], w_mu=[float(c == name) for c in colors],
                      description=f'{id2name[ids[pair]]}. Original labels: {classes[pair]}')
    tree.validate()

    out_dir.mkdir(parents=True, exist_ok=True)
    tree.to_json(out_dir / TREE_NAME)
    for pair, labs in classes.items():
        log(f'  {ids[pair]:4d} {id2name[ids[pair]]:32s} {len(labs):3d} labels')
    log(f'  saved {out_dir / TREE_NAME}')


# ================================================================================================ #
# Step 2: label maps                                                                               #
# ================================================================================================ #

def convert_case(seg_path, out_dir, mapping_path, side, claustrum, valid_ids):
    ts.PATH_MAPPING = str(mapping_path)          # module variable: set it in the worker too
    img = nib.load(seg_path)
    out = np.asanyarray(ts.to_supersynth(img, side=side, claustrum=claustrum).dataobj)
    present = np.unique(out).tolist()
    bad = sorted(set(present) - valid_ids)
    if bad:
        raise ValueError(f'{seg_path.name}: labels not in the SuperSynth tree: {bad} '
                         f'(NextBrain labels missing from the mapping YAML?)')
    hdr = img.header.copy()
    hdr.set_data_dtype(np.int16)
    hdr['scl_slope'], hdr['scl_inter'] = 1, 0
    nib.save(nib.Nifti1Image(out.astype(np.int16), img.affine, hdr), out_dir / seg_path.name)
    return seg_path.name, present


def step_labels(seg_dir, mapping_path, side, claustrum, out_dir, jobs):
    log(f'Step 2: label maps ({seg_dir}, side {side})')
    tree_path = out_dir / TREE_NAME
    if not tree_path.exists():
        sys.exit(f'{tree_path} not found: run --step tree first')
    valid_ids = set(AllenLabelTree.from_json(tree_path).label_to_node)
    files = sorted(p for p in seg_dir.glob('*.nii*')
                   if not p.name.endswith(('_steps.nii.gz', '_removed.nii.gz')))
    out_labels = out_dir / 'labels'
    out_labels.mkdir(parents=True, exist_ok=True)
    log(f'  {len(files)} cases, {jobs} in parallel')
    n = len(files)
    with ProcessPoolExecutor(jobs) as ex:
        for name, present in ex.map(convert_case, files, [out_labels] * n, [mapping_path] * n,
                                    [side] * n, [claustrum] * n, [valid_ids] * n):
            log(f'  {name}: {len(present)} labels')
    log(f'  saved in {out_labels}')


# ================================================================================================ #

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--mapping', type=Path, default=DEFAULT_MAPPING,
                   help='NextBrain -> SuperSynth YAML (default: the corrected copy, see top)')
    p.add_argument('--seg-dir', type=Path, default=DEFAULT_SEG_DIR, help='label maps to convert')
    p.add_argument('--out-dir', type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument('--step', choices=['all', 'tree', 'labels'], default='all')
    p.add_argument('--side', choices=['L', 'R'], default='R',
                   help='side whose label ids are used for ALL the cases (default R)')
    p.add_argument('--claustrum', action='store_true',
                   help='keep the claustrum as a class (default: merged into the cerebral WM)')
    p.add_argument('--jobs', type=int, default=5, help='cases processed in parallel (default 5)')
    args = p.parse_args()

    ts.PATH_MAPPING = str(args.mapping)          # point the library to our mapping file
    if args.step in ('all', 'tree'):
        step_tree(args.side, args.claustrum, args.out_dir)
    if args.step in ('all', 'labels'):
        step_labels(args.seg_dir, args.mapping, args.side, args.claustrum, args.out_dir, args.jobs)


if __name__ == '__main__':
    main()
