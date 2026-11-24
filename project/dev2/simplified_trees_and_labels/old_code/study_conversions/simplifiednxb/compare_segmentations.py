"""
Voxel-wise comparison of two label maps on the same grid:
  reference  = simplified NextBrain label map  (data/outputs/simplifiednxb/labels/case-N.nii.gz)
  prediction = NextBrain segmentation of the synthetic image (seg.<side>_resampled.nii.gz)

All metrics come from one sparse confusion table (pairs of (ref label, pred label) with their
voxel counts), so any grouping of labels (tree levels) is a cheap re-aggregation.
"""
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from scipy import ndimage as ndi

HERE = Path(__file__).resolve().parent
SIMP_DIR = HERE / 'data/outputs/simplifiednxb'
REF_DIR = SIMP_DIR / 'labels'
TREE_PATH = SIMP_DIR / 'nextbrain_tree_allen_gyral_simplified.json'
METRICS_DIR = SIMP_DIR / 'metrics'
GEN_DIR = Path('/home/agustin/phd/synthseg_nextbrain/project/dev2/generator/test_generators/'
               'test_tree_generator/data/outputs/deterministic')
SEG_DIR = GEN_DIR / 'segmentation_nextbrain'
IMG_DIR = GEN_DIR / 'images'

CASES = [1, 2, 3, 4, 5]
K = 1 << 16  # pair encoding: ref * K + pred


def case_side(case):
    d = SEG_DIR / f'case-{case}.nii.gz'
    return 'right' if (d / 'seg.right_resampled.nii.gz').exists() else 'left'


def paths(case):
    side = case_side(case)
    return dict(ref=REF_DIR / f'case-{case}.nii.gz',
                pred=SEG_DIR / f'case-{case}.nii.gz' / f'seg.{side}_resampled.nii.gz',
                img=IMG_DIR / f'case-{case}.nii.gz',
                side=side)


def load_tree():
    return json.load(open(TREE_PATH))


def label_names():
    """label id -> name. Tree leaf names, completed with NextBrain's lut (e.g. labels the tree removed)."""
    names = {}
    lut = SEG_DIR / 'case-1.nii.gz' / 'lut.txt'
    for line in open(lut):
        p = line.split()
        if p:
            names[int(p[0])] = p[1]
    for node, v in load_tree()['nodes'].items():
        for lab in v['labels']:
            names.setdefault(lab, node)
    names[0] = 'background'
    return names


def label_ancestors():
    """label id -> {level: ancestor node name} following the simplified tree."""
    nodes = load_tree()['nodes']
    out = {}
    for node, v in nodes.items():
        chain, n = {}, node
        while n is not None:
            chain[nodes[n]['level']] = n
            n = nodes[n]['parent']
        for lab in v['labels']:
            out[lab] = chain
    return out


def load_case(case):
    p = paths(case)
    ref_img, pred_img = nib.load(p['ref']), nib.load(p['pred'])
    assert ref_img.shape == pred_img.shape and np.allclose(ref_img.affine, pred_img.affine), case
    ref = np.asanyarray(ref_img.dataobj).astype(np.int32)
    pred = np.rint(np.asanyarray(pred_img.dataobj)).astype(np.int32)
    return ref, pred, ref_img.affine, float(np.prod(ref_img.header.get_zooms()[:3]))


def confusion(ref, pred):
    """DataFrame (ref, pred, n) of every label pair with at least one voxel (background pair excluded)."""
    m = (ref > 0) | (pred > 0)
    codes, n = np.unique(ref[m].astype(np.int64) * K + pred[m], return_counts=True)
    return pd.DataFrame({'ref': codes // K, 'pred': codes % K, 'n': n})


def centroids_mm(lab, labels, affine):
    com = np.array(ndi.center_of_mass(np.ones(lab.shape, np.uint8), lab, labels))
    return com @ affine[:3, :3].T + affine[:3, 3]


def dice_table(conf, vox_mm3, group=None):
    """Per-label (or per group, if `group` maps label -> group) volumes and dice from the confusion table."""
    c = conf.copy()
    if group is not None:
        c['ref'] = c['ref'].map(lambda x: group.get(x, 'background' if x == 0 else f'?{x}'))
        c['pred'] = c['pred'].map(lambda x: group.get(x, 'background' if x == 0 else f'?{x}'))
        c = c.groupby(['ref', 'pred'], as_index=False)['n'].sum()
    bg = 'background' if group is not None else 0
    vr = c.groupby('ref')['n'].sum()
    vp = c.groupby('pred')['n'].sum()
    inter = c[c.ref == c.pred].set_index('ref')['n']
    labs = sorted((set(vr.index) | set(vp.index)) - {bg}, key=str)
    df = pd.DataFrame(index=pd.Index(labs, name='label'))
    df['vox_ref'] = vr.reindex(labs).fillna(0).astype(int)
    df['vox_pred'] = vp.reindex(labs).fillna(0).astype(int)
    df['vox_inter'] = inter.reindex(labs).fillna(0).astype(int)
    df['vol_ref_mm3'] = df.vox_ref * vox_mm3
    df['vol_pred_mm3'] = df.vox_pred * vox_mm3
    df['vol_diff_rel'] = (df.vox_pred - df.vox_ref) / df.vox_ref.where(df.vox_ref > 0)
    df['dice'] = 2 * df.vox_inter / (df.vox_ref + df.vox_pred)
    return df


def dominant(conf, src, dst):
    """For every `src` label: the `dst` label holding most of its voxels and that fraction."""
    c = conf[conf[src] > 0]
    tot = c.groupby(src)['n'].sum()
    top = c.sort_values('n', ascending=False).drop_duplicates(src).set_index(src)
    return top[dst], top['n'] / tot


def compare_case(case):
    ref, pred, affine, vox = load_case(case)
    conf = confusion(ref, pred)
    names = label_names()

    df = dice_table(conf, vox)
    labs = df.index.to_numpy()
    in_ref, in_pred = df.vox_ref > 0, df.vox_pred > 0
    df.insert(0, 'name', [names.get(l, '?') for l in labs])
    df['presence'] = np.select([in_ref & in_pred, in_ref], ['both', 'only_ref'], 'only_pred')

    # centroids (mm, world) and their displacement
    cr = pd.DataFrame(centroids_mm(ref, labs[in_ref], affine), index=labs[in_ref], columns=['x', 'y', 'z'])
    cp = pd.DataFrame(centroids_mm(pred, labs[in_pred], affine), index=labs[in_pred], columns=['x', 'y', 'z'])
    for a in 'xyz':
        df[f'c_ref_{a}'] = cr[a].reindex(labs)
        df[f'c_pred_{a}'] = cp[a].reindex(labs)
    df['centroid_dist_mm'] = np.sqrt(sum((df[f'c_ref_{a}'] - df[f'c_pred_{a}']) ** 2 for a in 'xyz'))

    # where do the voxels of each label go (ref -> pred) and come from (pred -> ref)
    d, f = dominant(conf, 'ref', 'pred')
    df['ref_to_pred_label'] = d.reindex(labs)
    df['ref_to_pred_frac'] = f.reindex(labs)
    d, f = dominant(conf, 'pred', 'ref')
    df['pred_from_ref_label'] = d.reindex(labs)
    df['pred_from_ref_frac'] = f.reindex(labs)
    for col in ['ref_to_pred_label', 'pred_from_ref_label']:
        df[col.replace('_label', '_name')] = df[col].map(lambda x: names.get(int(x), '?') if pd.notna(x) else None)
    df.insert(0, 'case', case)

    # tree-level dice. level 'leaf' = tree leaf (merges labels that share a leaf, e.g. 611+846);
    # levels 1-3 = ancestors in the simplified tree (leaves shallower than the level keep their leaf)
    anc = label_ancestors()
    levels = []
    for lev in ('leaf', 1, 2, 3):
        g = {l: a[max(a)] if lev == 'leaf' else a.get(lev, a[max(a)]) for l, a in anc.items()}
        t = dice_table(conf, vox, g)
        t.insert(0, 'level', lev)
        t.insert(0, 'case', case)
        levels.append(t)

    fg = (ref > 0) | (pred > 0)
    summary = dict(case=case, side=paths(case)['side'],
                   n_labels_ref=int(in_ref.sum()), n_labels_pred=int(in_pred.sum()),
                   n_only_ref=int((df.presence == 'only_ref').sum()),
                   n_only_pred=int((df.presence == 'only_pred').sum()),
                   brain_dice=float(2 * ((ref > 0) & (pred > 0)).sum() / ((ref > 0).sum() + (pred > 0).sum())),
                   voxel_agreement_fg=float((ref[fg] == pred[fg]).mean()),
                   mean_dice_shared=float(df.loc[df.presence == 'both', 'dice'].mean()),
                   median_dice_shared=float(df.loc[df.presence == 'both', 'dice'].median()),
                   vol_weighted_dice=float(np.average(df.dice.fillna(0), weights=df.vox_ref + df.vox_pred)))
    conf.insert(0, 'case', case)
    return dict(labels=df.reset_index(), levels=pd.concat(levels).reset_index(), confusion=conf,
                summary=summary)


def flag_misplaced(df, max_dice=0.2):
    """Labels present in both maps but not where they should be.

    r_eq_mm       radius of the sphere with the reference volume (size scale of the structure)
    rel_shift     centroid displacement / r_eq_mm
    misplaced     dice < max_dice AND the predicted centroid lies outside that sphere (rel_shift > 1)
    relabelled    the reference voxels mostly go to another label AND the predicted voxels mostly
                  come from another label (the label exists in both maps, but marks other tissue)
    """
    df = df.copy()
    both = df.presence == 'both'
    df['r_eq_mm'] = (3 * df.vol_ref_mm3 / (4 * np.pi)) ** (1 / 3)
    df['rel_shift'] = df.centroid_dist_mm / df.r_eq_mm
    df['misplaced'] = both & (df.dice < max_dice) & (df.rel_shift > 1)
    df['relabelled'] = both & (df.ref_to_pred_label != df.label) & (df.pred_from_ref_label != df.label)
    return df


def swaps(df):
    """Mutual confusions: label a -> mostly b in the prediction and label b -> mostly a."""
    d = df[df.presence == 'both'].set_index(['case', 'label'])
    out = []
    for (case, a), r in d.iterrows():
        b = r.ref_to_pred_label
        if pd.notna(b) and b != a and a < b and (case, b) in d.index and d.loc[(case, b), 'ref_to_pred_label'] == a:
            out.append(dict(case=case, label_a=a, name_a=r['name'], label_b=int(b), name_b=d.loc[(case, b), 'name'],
                            frac_a_to_b=r.ref_to_pred_frac, frac_b_to_a=d.loc[(case, b), 'ref_to_pred_frac'],
                            dice_a=r.dice, dice_b=d.loc[(case, b), 'dice']))
    return pd.DataFrame(out)
