"""
Voxel-wise comparison of two SuperSynth label maps on the same grid:
  REF  = NextBrain labels converted to SuperSynth (allennxb_to_supersynthnxb.py): data/labels/case-N.nii.gz
  SEG  = SuperSynth segmentation of the synthetic image:
         .../deterministic/segmentation_nextbrain/case-N.nii.gz/SuperSynth/segmentation_resampled.nii.gz
  FINE = simplified NextBrain label map the REF was converted from (../simplifiednxb/data/labels/case-N.nii.gz),
         used to trace every disagreement back to the original NextBrain structure.

Hemisphere ids
--------------
REF uses the RIGHT ids for every case (data/labels/sides.json: side R for all cases), while SuperSynth labels each
hemisphere with its anatomical side (cases 4-5 are left hemispheres -> Left-* ids). SEG is therefore harmonised before
the comparison: every lateralised SuperSynth id is mapped to its Right-* id (Left-Thalamus 10 -> Right-Thalamus 49, ...).
The raw side information is kept in the summary (voxels with Left vs Right ids).

All metrics come from one sparse confusion table (pairs (ref, seg) with their voxel counts).
"""
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from scipy import ndimage as ndi

HERE = Path(__file__).resolve().parent
DATA = HERE / 'data'
REF_DIR = DATA / 'labels'
TREE_PATH = DATA / 'nextbrain_tree_supersynthnxb.json'
TREE_REPORT = DATA / 'tree_report.json'
METRICS_DIR = DATA / 'metrics'
FINE_DIR = HERE.parent / 'simplifiednxb' / 'data' / 'labels'
GEN_DIR = Path('/home/agustin/phd/synthseg_nextbrain/project/dev2/generator/test_generators/'
               'test_tree_generator/data/outputs/deterministic')
SEG_DIR = GEN_DIR / 'segmentation_nextbrain'
IMG_DIR = GEN_DIR / 'images'
FS_LUT = Path('/home/code/freesurfer/freesurfer-nextbrain/FreeSurferColorLUT.txt')
NXB_LUT = SEG_DIR / 'case-1.nii.gz' / 'lut.txt'

CASES = [1, 2, 3, 4, 5]
K = 1 << 16  # pair encoding: ref * K + seg

# Left -> Right ids of the lateralised SuperSynth labels (FreeSurfer LUT)
LEFT_TO_RIGHT = {2: 41, 3: 42, 4: 43, 5: 44, 7: 46, 8: 47, 10: 49, 11: 50, 12: 51, 13: 52, 17: 53, 18: 54,
                 26: 58, 28: 60, 30: 62, 31: 63, 819: 820, 821: 822, 843: 844, 865: 866, 869: 870}


def paths(case):
    return dict(ref=REF_DIR / f'case-{case}.nii.gz',
                seg=SEG_DIR / f'case-{case}.nii.gz' / 'SuperSynth' / 'segmentation_resampled.nii.gz',
                fine=FINE_DIR / f'case-{case}.nii.gz',
                img=IMG_DIR / f'case-{case}.nii.gz')


def _read_lut(path):
    names, colors = {}, {}
    for line in open(path):
        p = line.split()
        if len(p) >= 5 and p[0].isdigit():
            names[int(p[0])] = p[1]
            colors[int(p[0])] = tuple(int(x) for x in p[2:5])
    return names, colors


def label_names():
    """SuperSynth / FreeSurfer id -> name."""
    names, _ = _read_lut(FS_LUT)
    names[0] = 'background'
    return names


def label_colors():
    return _read_lut(FS_LUT)[1]


def fine_names():
    """NextBrain (simplified) id -> name."""
    names, _ = _read_lut(NXB_LUT)
    for node, v in json.load(open(HERE.parent / 'simplifiednxb' / 'data' /
                                  'nextbrain_tree_allen_gyral_simplified.json'))['nodes'].items():
        for lab in v['labels']:
            names.setdefault(lab, node)
    names[0] = 'background'
    return names


def class_composition():
    """SuperSynth class (right id) -> {NextBrain id: name} from tree_report.json."""
    rep = json.load(open(TREE_REPORT))
    return {c['right']: {int(k): v for k, v in c['original_labels'].items()} for c in rep['classes']}


def harmonise(seg):
    lut = np.arange(max(int(seg.max()), max(LEFT_TO_RIGHT)) + 1)
    for l, r in LEFT_TO_RIGHT.items():
        lut[l] = r
    return lut[seg]


def load_case(case, harmonised=True):
    p = paths(case)
    ref_img, seg_img = nib.load(p['ref']), nib.load(p['seg'])
    assert ref_img.shape == seg_img.shape and np.allclose(ref_img.affine, seg_img.affine), case
    ref = np.asanyarray(ref_img.dataobj).astype(np.int32)
    seg_raw = np.rint(np.asanyarray(seg_img.dataobj)).astype(np.int32)
    seg = harmonise(seg_raw) if harmonised else seg_raw
    return ref, seg, seg_raw, ref_img.affine, float(np.prod(ref_img.header.get_zooms()[:3]))


def load_fine(case):
    return np.asanyarray(nib.load(paths(case)['fine']).dataobj).astype(np.int32)


def confusion(ref, seg):
    m = (ref > 0) | (seg > 0)
    codes, n = np.unique(ref[m].astype(np.int64) * K + seg[m], return_counts=True)
    return pd.DataFrame({'ref': codes // K, 'pred': codes % K, 'n': n})


def centroids_mm(lab, labels, affine):
    com = np.array(ndi.center_of_mass(np.ones(lab.shape, np.uint8), lab, labels))
    return com @ affine[:3, :3].T + affine[:3, 3]


def dice_table(conf, vox_mm3):
    vr = conf.groupby('ref')['n'].sum()
    vp = conf.groupby('pred')['n'].sum()
    inter = conf[conf.ref == conf.pred].set_index('ref')['n']
    labs = sorted((set(vr.index) | set(vp.index)) - {0})
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
    c = conf[conf[src] > 0]
    tot = c.groupby(src)['n'].sum()
    top = c.sort_values('n', ascending=False).drop_duplicates(src).set_index(src)
    return top[dst], top['n'] / tot


def compare_case(case):
    ref, seg, seg_raw, affine, vox = load_case(case)
    conf = confusion(ref, seg)
    names = label_names()

    df = dice_table(conf, vox)
    labs = df.index.to_numpy()
    in_ref, in_pred = df.vox_ref > 0, df.vox_pred > 0
    df.insert(0, 'name', [names.get(l, '?') for l in labs])
    df['presence'] = np.select([in_ref & in_pred, in_ref], ['both', 'only_ref'], 'only_pred')

    cr = pd.DataFrame(centroids_mm(ref, labs[in_ref], affine), index=labs[in_ref], columns=['x', 'y', 'z'])
    cp = pd.DataFrame(centroids_mm(seg, labs[in_pred], affine), index=labs[in_pred], columns=['x', 'y', 'z'])
    for a in 'xyz':
        df[f'c_ref_{a}'] = cr[a].reindex(labs)
        df[f'c_pred_{a}'] = cp[a].reindex(labs)
    df['centroid_dist_mm'] = np.sqrt(sum((df[f'c_ref_{a}'] - df[f'c_pred_{a}']) ** 2 for a in 'xyz'))

    d, f = dominant(conf, 'ref', 'pred')
    df['ref_to_pred_label'] = d.reindex(labs)
    df['ref_to_pred_frac'] = f.reindex(labs)
    d, f = dominant(conf, 'pred', 'ref')
    df['pred_from_ref_label'] = d.reindex(labs)
    df['pred_from_ref_frac'] = f.reindex(labs)
    for col in ['ref_to_pred_label', 'pred_from_ref_label']:
        df[col.replace('_label', '_name')] = df[col].map(lambda x: names.get(int(x), '?') if pd.notna(x) else None)
    df.insert(0, 'case', case)

    # fine (NextBrain) label -> SEG class: traces the disagreements back to the original structures
    fine = load_fine(case)
    assert fine.shape == ref.shape, case
    m = (fine > 0) | (seg > 0)
    codes, n = np.unique(fine[m].astype(np.int64) * K + seg[m], return_counts=True)
    fine_conf = pd.DataFrame({'case': case, 'fine': codes // K, 'pred': codes % K, 'n': n})
    # expected class of every fine label = its REF class (REF = mapping(FINE))
    codes, n = np.unique(fine[fine > 0].astype(np.int64) * K + ref[fine > 0], return_counts=True)
    fine_map = pd.DataFrame({'fine': codes // K, 'ref': codes % K, 'n': n})

    fg = (ref > 0) | (seg > 0)
    left_ids, right_ids = list(LEFT_TO_RIGHT), list(LEFT_TO_RIGHT.values())
    n_left, n_right = int(np.isin(seg_raw, left_ids).sum()), int(np.isin(seg_raw, right_ids).sum())
    summary = dict(case=case,
                   seg_side='left' if n_left > n_right else 'right',
                   seg_vox_left_ids=n_left, seg_vox_right_ids=n_right,
                   seg_contralateral_mm3=min(n_left, n_right) * vox,
                   n_labels_ref=int(in_ref.sum()), n_labels_pred=int(in_pred.sum()),
                   n_only_ref=int((df.presence == 'only_ref').sum()),
                   n_only_pred=int((df.presence == 'only_pred').sum()),
                   brain_dice=float(2 * ((ref > 0) & (seg > 0)).sum() / ((ref > 0).sum() + (seg > 0).sum())),
                   voxel_agreement_fg=float((ref[fg] == seg[fg]).mean()),
                   mean_dice_shared=float(df.loc[df.presence == 'both', 'dice'].mean()),
                   median_dice_shared=float(df.loc[df.presence == 'both', 'dice'].median()),
                   vol_weighted_dice=float(np.average(df.dice.fillna(0), weights=df.vox_ref + df.vox_pred)))
    conf.insert(0, 'case', case)
    return dict(labels=df.reset_index(), confusion=conf, fine_confusion=fine_conf, fine_map=fine_map,
                summary=summary)


def fine_origin(fine_conf, fine_map, vox_mm3=0.064):
    """Per NextBrain (fine) label and case: its REF class, the fraction of its voxels SEG gives to that class,
    and the SEG class that takes most of the rest."""
    exp = fine_map.sort_values('n', ascending=False).drop_duplicates('fine').set_index('fine')['ref']
    c = fine_conf[fine_conf.fine > 0].copy()
    c['expected'] = c.fine.map(exp)
    c['ok'] = c.pred == c.expected
    tot = c.groupby(['case', 'fine']).n.sum().rename('vox')
    ok = c[c.ok].groupby(['case', 'fine']).n.sum().reindex(tot.index).fillna(0).rename('vox_ok')
    wrong = c[~c.ok].sort_values('n', ascending=False).drop_duplicates(['case', 'fine']).set_index(['case', 'fine'])
    out = pd.concat([tot, ok], axis=1).reset_index()
    out['expected_class'] = out.fine.map(exp)
    out['vol_mm3'] = out.vox * vox_mm3
    out['vol_wrong_mm3'] = (out.vox - out.vox_ok) * vox_mm3
    out['frac_ok'] = out.vox_ok / out.vox
    key = list(zip(out.case, out.fine))
    out['main_wrong_class'] = [wrong['pred'].get(k, np.nan) for k in key]
    out['main_wrong_frac'] = [wrong['n'].get(k, 0) / v for k, v in zip(key, out.vox)]
    return out


def flag_misplaced(df, max_dice=0.2):
    """misplaced = present in both, dice < max_dice and the SEG centroid outside the sphere with the REF volume."""
    df = df.copy()
    both = df.presence == 'both'
    df['r_eq_mm'] = (3 * df.vol_ref_mm3 / (4 * np.pi)) ** (1 / 3)
    df['rel_shift'] = df.centroid_dist_mm / df.r_eq_mm
    df['misplaced'] = both & (df.dice < max_dice) & (df.rel_shift > 1)
    df['relabelled'] = both & (df.ref_to_pred_label != df.label) & (df.pred_from_ref_label != df.label)
    return df


def swaps(df):
    d = df[df.presence == 'both'].set_index(['case', 'label'])
    out = []
    for (case, a), r in d.iterrows():
        b = r.ref_to_pred_label
        if pd.notna(b) and b != a and a < b and (case, b) in d.index and d.loc[(case, b), 'ref_to_pred_label'] == a:
            out.append(dict(case=case, label_a=a, name_a=r['name'], label_b=int(b), name_b=d.loc[(case, b), 'name'],
                            frac_a_to_b=r.ref_to_pred_frac, frac_b_to_a=d.loc[(case, b), 'ref_to_pred_frac'],
                            dice_a=r.dice, dice_b=d.loc[(case, b), 'dice']))
    return pd.DataFrame(out, columns=['case', 'label_a', 'name_a', 'label_b', 'name_b', 'frac_a_to_b', 'frac_b_to_a',
                                      'dice_a', 'dice_b'])
