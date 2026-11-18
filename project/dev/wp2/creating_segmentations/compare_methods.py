#!/usr/bin/env python3
"""Measure the three candidate hemisphere segmentations of every NextBrain case.

Methods (built by run_methods_comparison.sh, in outputs/methods_comparison/<method>/):
    v1          build_3d_segmentation.py (nearest section)
    v1_restore  v1 followed by dev/wp2/fill_holes/manual/restore.py
    v2          build_3d_segmentation_v2.py
    v2_restore  v2 followed by restore.py (extra combination)

There is no ground truth, so the script measures things that can be checked:
    volumes.csv           voxels / mm3 of every label, overlap (Dice) with v1
    global_metrics.csv    labelled volume, residual gaps, holes, surface, agreement with the MRI...
    filled_regions.csv    what each method does with the unlabelled regions of v1 (gaps, sulci, holes)
    added_voxels.csv      voxels labelled by a method and not by v1: are they plausible in the MRI?
    added_distance.csv    distance from the added voxels to the nearest label of v1
    section_fidelity.csv  agreement of each volume with the original histology sections
    section_fidelity_labels.csv   the same, per label
    runtime.csv           wall time and peak memory, from the logs

Usage:
    python compare_methods.py                 # all cases found, one process per case
    python compare_methods.py --cases P57-16  # one case
"""
import argparse
import glob
import itertools
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import nibabel as nib
import numpy as np
import pandas as pd
from scipy import ndimage as ndi

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'outputs', 'methods_comparison')
TABLES = os.path.join(OUT, 'tables')
PER_CASE = os.path.join(TABLES, 'per_case')      # one file per case and table, merged at the end
RESTORE_DIR = '/home/agustin/phd/synthseg_nextbrain/project/dev/wp2/fill_holes/manual'
LUT_FILE = ('/home/agustin/phd/synthseg_nextbrain/project/dev/wp1/allen_ontology/nextbrain-utils/'
            'nextbrain_utils/lut/NextBrainLUT.txt')
RES = 0.2
METHODS = {'v1': 'v1/{case}_seg_0.2mm.nii.gz',
           'v1_restore': 'v1_restore/{case}_seg_0.2mm_restored.nii.gz',
           'v2': 'v2/{case}_seg_0.2mm_v2.nii.gz',
           'v2_restore': 'v2_restore/{case}_seg_0.2mm_v2_restored.nii.gz'}   # extra: restore.py applied to v2
MAX_RUN_MM = 2.4          # "thin" unlabelled run (same value as restore.py and the v2 notebook)
AXES = {0: 'LR', 1: 'AP', 2: 'IS'}
L1_NAMES = {0: 'Background', 1: 'Brainstem', 2: 'CSF', 3: 'CerebellarGM', 4: 'CerebellarWM',
            5: 'CerebralGM', 6: 'CerebralWM', 7: 'DeepGM', 8: 'Other'}

sys.path.insert(0, RESTORE_DIR)
import restore as R  # noqa: E402  (bg_runs and the label -> L1 class table)


def load_names(path=LUT_FILE):
    names = {}
    for line in open(path):
        p = line.split()
        if len(p) >= 2 and p[0].isdigit():
            names[int(p[0])] = p[1]
    return names


def case_dirs():
    out = {}
    for d in sorted(glob.glob(os.path.join(HERE, '..', 'NextBrain-Case-*', '*', 'labels_blocks'))):
        cd = os.path.dirname(d)
        out[os.path.basename(cd)] = cd
    return out


def load_methods(case):
    return {m: np.asanyarray(nib.load(os.path.join(OUT, f.format(case=case))).dataobj)
            for m, f in METHODS.items()}


def short_runs(seg, is_cortex, is_wm, axis, max_len):
    """Unlabelled voxels in a run of <= max_len voxels along `axis`, bounded by tissue at both ends.
    Returns boolean masks by the class of the two bounding labels."""
    prev, nxt = R.bg_runs(seg, axis)
    L = seg.shape[axis]
    ok = (seg == 0) & (prev >= 0) & (nxt < L) & ((nxt - prev - 1) <= max_len)
    a = np.take_along_axis(seg, np.clip(prev, 0, L - 1).astype(np.intp), axis)
    b = np.take_along_axis(seg, np.clip(nxt, 0, L - 1).astype(np.intp), axis)
    del prev, nxt
    ca, cb = is_cortex[a], is_cortex[b]
    wm = ok & is_wm[a] & is_wm[b]
    return dict(cortex=ok & ca & cb, noncortex=ok & ~ca & ~cb, mixed=ok & (ca ^ cb), wm=wm)


def mri_on_grid(mri, shape, affine):
    """MRI intensities and MRI mask on the output grid. The mask is defined as in build_3d_segmentation_v2.py."""
    M = np.linalg.inv(mri.affine) @ affine
    data = np.asanyarray(mri.dataobj).astype(np.float32)
    inten = ndi.affine_transform(data, M[:3, :3], M[:3, 3], output_shape=shape, order=1)
    tissue = ndi.affine_transform((data > 0).astype(np.float32), M[:3, :3], M[:3, 3], output_shape=shape, order=1) >= 0.5
    return inten, tissue


def process_case(case, case_dir):
    t0 = time.time()
    names = load_names()
    segs = load_methods(case)
    ref = nib.load(os.path.join(OUT, METHODS['v1'].format(case=case)))
    shape, affine = ref.shape, ref.affine
    vox_mm3 = RES ** 3
    nlab = int(max(s.max() for s in segs.values())) + 1
    l1 = R.load_l1_lut(nlab)
    l1 = np.concatenate([l1, np.full(max(nlab - len(l1), 0), 8, np.uint8)])[:nlab]
    is_cortex = np.isin(l1, R.CORTEX_L1); is_cortex[0] = False
    is_wm = np.isin(l1, (4, 6)); is_wm[0] = False
    max_len = int(round(MAX_RUN_MM / RES))
    v1 = segs['v1']

    # ---- volumes per label ------------------------------------------------------------------
    counts = {m: np.bincount(s.ravel(), minlength=nlab) for m, s in segs.items()}
    rows = []
    for m, s in segs.items():
        same = s == v1
        inter = np.bincount(v1[same], minlength=nlab)
        for l in np.flatnonzero((counts['v1'] > 0) | (counts[m] > 0)):
            if l == 0:
                continue
            den = counts['v1'][l] + counts[m][l]
            rows.append(dict(case=case, method=m, label=int(l), name=names.get(int(l), '?'),
                             l1=L1_NAMES[int(l1[l])], voxels=int(counts[m][l]), mm3=counts[m][l] * vox_mm3,
                             dice_v1=2 * inter[l] / den if den else np.nan))
    pd.DataFrame(rows).to_csv(os.path.join(PER_CASE, f'{case}_volumes.csv'), index=False)

    # ---- MRI on the output grid, intensity model of every label (from v1) -------------------
    mri, tissue = mri_on_grid(nib.load(os.path.join(case_dir, 'mri_rotated', 'mri.nii.gz')), shape, affine)
    # intensities are only trusted away from the border of the MRI mask (partial volume with the zero background)
    core = ndi.binary_erosion(tissue, iterations=2)
    sel = (v1 > 0) & core
    n1 = np.bincount(v1[sel], minlength=nlab).astype(np.float64)
    s1 = np.bincount(v1[sel], weights=mri[sel], minlength=nlab)
    s2 = np.bincount(v1[sel], weights=mri[sel].astype(np.float64) ** 2, minlength=nlab)
    mean = s1 / np.maximum(n1, 1)
    std = np.sqrt(np.maximum(s2 / np.maximum(n1, 1) - mean ** 2, 0))
    std[n1 < 50] = np.nan                    # too few voxels to say anything
    del sel

    def plausible(lab, where):
        """Fraction of the voxels `where` whose MRI intensity is within 2 SD of the mean of their label."""
        where = where & core
        if not where.any():
            return np.nan
        ll = lab[where]; z = (mri[where] - mean[ll]) / std[ll]
        known = np.isfinite(z)
        return float((np.abs(z[known]) <= 2).mean()) if known.any() else np.nan

    # ---- regions of v1 that a method may fill: thin runs by class, enclosed holes ------------
    region = {k: np.zeros(shape, bool) for k in ('gap_noncortex', 'run_cortex', 'run_mixed')}
    for ax in AXES:
        r = short_runs(v1, is_cortex, is_wm, ax, max_len)
        region['gap_noncortex'] |= r['noncortex']; region['run_cortex'] |= r['cortex']; region['run_mixed'] |= r['mixed']
        del r
    region['run_cortex'] &= ~region['gap_noncortex']
    region['run_mixed'] &= ~(region['gap_noncortex'] | region['run_cortex'])
    region['enclosed_hole'] = ndi.binary_fill_holes(v1 > 0) & (v1 == 0)
    assigned = region['gap_noncortex'] | region['run_cortex'] | region['run_mixed'] | region['enclosed_hole']
    region['other_inside_mri'] = (v1 == 0) & tissue & ~assigned
    region['other_outside_mri'] = (v1 == 0) & ~tissue & ~assigned
    del assigned

    grows, frows, arows = [], [], []
    for m, s in segs.items():
        lab = s > 0
        g = dict(case=case, method=m, n_labels=int((counts[m][1:] > 0).sum()),
                 labelled_cm3=lab.sum() * vox_mm3 / 1000,
                 labels_lost_vs_v1=int(((counts['v1'][1:] > 0) & (counts[m][1:] == 0)).sum()))
        both = lab & (v1 > 0)
        g['added_cm3'] = (lab & (v1 == 0)).sum() * vox_mm3 / 1000
        g['removed_cm3'] = (~lab & (v1 > 0)).sum() * vox_mm3 / 1000
        g['relabelled_cm3'] = (both & (s != v1)).sum() * vox_mm3 / 1000
        g['same_label_where_both_pct'] = 100 * (s[both] == v1[both]).mean()
        del both
        # residual thin unlabelled runs, per axis and class
        for ax, axn in AXES.items():
            r = short_runs(s, is_cortex, is_wm, ax, max_len)
            for k, v in r.items():
                g[f'runs_{axn}_{k}_vox'] = int(v.sum())
            del r
        g['enclosed_hole_vox'] = int((ndi.binary_fill_holes(lab) & ~lab).sum())
        # surface between different labels (background included), in cm2, per axis
        for ax, axn in AXES.items():
            a = np.moveaxis(s, ax, 0)
            g[f'faces_{axn}_cm2'] = int((a[1:] != a[:-1]).sum()) * RES ** 2 / 100
        # agreement with the MRI
        sel = lab & core
        n = np.bincount(s[sel], minlength=nlab).astype(np.float64)
        a1 = np.bincount(s[sel], weights=mri[sel], minlength=nlab)
        a2 = np.bincount(s[sel], weights=mri[sel].astype(np.float64) ** 2, minlength=nlab)
        within = (a2 - a1 ** 2 / np.maximum(n, 1)).sum()
        total = a2.sum() - a1.sum() ** 2 / n.sum()
        g['mri_variance_explained'] = 1 - within / total
        g['labels_where_mri_zero_cm3'] = (lab & ~tissue).sum() * vox_mm3 / 1000
        g['mri_tissue_unlabelled_cm3'] = (~lab & tissue).sum() * vox_mm3 / 1000
        g['plausible_all_labelled_pct'] = 100 * plausible(s, sel)
        del sel
        grows.append(g)

        # what the method does with every unlabelled region of v1
        for k, reg in region.items():
            f = reg & lab
            row = dict(case=case, method=m, region=k, region_vox=int(reg.sum()), filled_vox=int(f.sum()))
            row['filled_pct'] = 100 * row['filled_vox'] / max(row['region_vox'], 1)
            row['filled_plausible_pct'] = 100 * plausible(s, f)
            row['filled_as_cortex_pct'] = 100 * is_cortex[s[f]].mean() if f.any() else np.nan
            frows.append(row)
        # voxels added / removed with respect to v1, by class of the label
        added = lab & (v1 == 0)
        for cname, cmask in (('cortex', is_cortex), ('noncortex', ~is_cortex)):
            for change, w, src in (('added', added & cmask[s], s), ('removed', ~lab & (v1 > 0) & cmask[v1], v1)):
                nw = max(int(w.sum()), 1)
                arows.append(dict(case=case, method=m, change=change, label_class=cname, vox=int(w.sum()),
                                  mri_zero_pct=100 * (w & ~tissue).sum() / nw,
                                  mri_edge_pct=100 * (w & tissue & ~core).sum() / nw,
                                  plausible_pct=100 * plausible(src, w)))
        del added, lab
    pd.DataFrame(grows).to_csv(os.path.join(PER_CASE, f'{case}_global.csv'), index=False)
    pd.DataFrame(frows).to_csv(os.path.join(PER_CASE, f'{case}_filled.csv'), index=False)
    pd.DataFrame(arows).to_csv(os.path.join(PER_CASE, f'{case}_added.csv'), index=False)
    del region, mri, tissue, core

    del segs
    section_fidelity(case, case_dir)
    return f'{case}: done in {time.time() - t0:.0f} s'


def section_fidelity(case, case_dir):
    """Agreement of the three volumes with the original histology sections: every labelled pixel of
    every section (0.2 mm in-plane) is looked up in the three volumes."""
    names = load_names()
    segs = load_methods(case)
    ref = nib.load(os.path.join(OUT, METHODS['v1'].format(case=case)))
    shape, affine = ref.shape, ref.affine
    nlab = int(max(s.max() for s in segs.values())) + 1
    blocks_dir = os.path.join(case_dir, 'labels_blocks')
    blocks = sorted((b for b in os.listdir(blocks_dir)
                     if os.path.exists(os.path.join(blocks_dir, b, 'LABELS.nii.gz'))), key=int)
    acc = {(m, u): dict(n=np.zeros(nlab, np.int64), match=np.zeros(nlab, np.int64), lost=np.zeros(nlab, np.int64))
           for m in METHODS for u in ('usable', 'partial')}
    for b in blocks:
        img = nib.load(os.path.join(blocks_dir, b, 'LABELS.nii.gz'))
        full = np.asanyarray(img.dataobj).astype(np.uint16)   # one block of P85-18 is stored as float64
        if full.ndim == 2:
            full = full[..., None]
        area = (full > 0).sum(axis=(0, 1))
        usable = area >= 0.3 * np.median(area[area > 0]) if (area > 0).any() else np.zeros(len(area), bool)
        lab = full[::2, ::2, :]
        x, y, z = np.nonzero(lab)
        if not len(x):
            continue
        A = img.affine
        ras = np.stack([2 * x, 2 * y, z], 1) @ A[:3, :3].T + A[:3, 3]
        idx = np.rint((ras - affine[:3, 3]) / RES).astype(np.int64)
        ok = np.all((idx >= 0) & (idx < np.array(shape)), axis=1)
        idx, sec, us = idx[ok], lab[x[ok], y[ok], z[ok]], usable[z[ok]]
        for m, s in segs.items():
            got = s[idx[:, 0], idx[:, 1], idx[:, 2]]
            for u, w in (('usable', us), ('partial', ~us)):
                a = acc[(m, u)]
                a['n'] += np.bincount(sec[w], minlength=nlab)
                a['match'] += np.bincount(sec[w & (got == sec)], minlength=nlab)
                a['lost'] += np.bincount(sec[w & (got == 0)], minlength=nlab)
    srows, lrows = [], []
    for (m, u), a in acc.items():
        n = a['n'].sum()
        srows.append(dict(case=case, method=m, sections=u, pixels=int(n),
                          same_label_pct=100 * a['match'].sum() / max(n, 1),
                          unlabelled_pct=100 * a['lost'].sum() / max(n, 1),
                          other_label_pct=100 * (n - a['match'].sum() - a['lost'].sum()) / max(n, 1)))
        if u == 'usable':
            for l in np.flatnonzero(a['n']):
                lrows.append(dict(case=case, method=m, label=int(l), name=names.get(int(l), '?'),
                                  pixels=int(a['n'][l]), same_label_pct=100 * a['match'][l] / a['n'][l],
                                  unlabelled_pct=100 * a['lost'][l] / a['n'][l]))
    pd.DataFrame(srows).to_csv(os.path.join(PER_CASE, f'{case}_sections.csv'), index=False)
    pd.DataFrame(lrows).to_csv(os.path.join(PER_CASE, f'{case}_sections_labels.csv'), index=False)
    return f'{case}: sections done'


def added_distance(case):
    """How far from the labels of v1 are the voxels that each method adds? (distance in mm to the nearest
    labelled voxel of v1). Short distances = a thin gap was closed; long ones = labels were extended far
    from any real section."""
    segs = load_methods(case)
    v1 = segs['v1']
    nz = [np.flatnonzero((v1 > 0).any(axis=tuple(a for a in range(3) if a != ax))) for ax in range(3)]
    box = tuple(slice(max(n[0] - 30, 0), n[-1] + 31) for n in nz)
    dist = ndi.distance_transform_edt(v1[box] == 0).astype(np.float32) * RES
    l1 = R.load_l1_lut(int(max(s.max() for s in segs.values())) + 1)
    rows = []
    for m in ('v1_restore', 'v2', 'v2_restore'):
        s = segs[m][box]
        add = (s > 0) & (v1[box] == 0)
        for region, sel in (('all', add), ('cerebellum', add & np.isin(l1[s], (3, 4))),
                            ('brainstem', add & (l1[s] == 1)), ('cerebrum', add & np.isin(l1[s], (5, 6, 7)))):
            d = dist[sel]
            if not d.size:
                continue
            rows.append(dict(case=case, method=m, region=region, added_cm3=d.size * RES ** 3 / 1000,
                             median_mm=float(np.median(d)), p90_mm=float(np.percentile(d, 90)),
                             p99_mm=float(np.percentile(d, 99)), max_mm=float(d.max()),
                             beyond_1mm_cm3=(d > 1).sum() * RES ** 3 / 1000, beyond_2mm_cm3=(d > 2).sum() * RES ** 3 / 1000))
    pd.DataFrame(rows).to_csv(os.path.join(PER_CASE, f'{case}_distance.csv'), index=False)
    return f'{case}: distances done'


def runtimes(cases):
    rows = []
    for case in cases:
        for m in METHODS:
            p = os.path.join(OUT, 'logs', f'{case}_{m}.log')
            if not os.path.exists(p):
                continue
            t = open(p).read()
            w = re.search(r'Elapsed \(wall clock\) time.*?: ([\d:.]+)', t)
            r = re.search(r'Maximum resident set size \(kbytes\): (\d+)', t)
            sec = None
            if w:
                parts = [float(v) for v in w.group(1).split(':')]
                sec = sum(v * 60 ** i for i, v in enumerate(reversed(parts)))
            rows.append(dict(case=case, method=m, wall_s=sec, peak_gb=int(r.group(1)) / 1e6 if r else None))
    df = pd.DataFrame(rows)
    # restore needs the reconstruction first: report the total
    for case in cases:
        for base in ('v1', 'v2'):
            a = df[(df.case == case) & (df.method == base)].wall_s
            sel = (df.case == case) & (df.method == base + '_restore')
            if len(a) and sel.any():
                df.loc[sel, 'wall_total_s'] = df.loc[sel, 'wall_s'] + a.iloc[0]
    df['wall_total_s'] = df.get('wall_total_s', df.wall_s).fillna(df.wall_s)
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--cases', nargs='*', default=None)
    ap.add_argument('--merge-only', action='store_true', help='only merge the per-case tables')
    ap.add_argument('--sections-only', action='store_true', help='only compute section_fidelity*.csv')
    ap.add_argument('--distance-only', action='store_true', help='only compute added_distance.csv')
    args = ap.parse_args()
    os.makedirs(PER_CASE, exist_ok=True)
    dirs = case_dirs()
    cases = args.cases or list(dirs)
    if not args.merge_only:
        with ProcessPoolExecutor(len(cases)) as ex:
            if args.sections_only:
                for msg in ex.map(section_fidelity, cases, [dirs[c] for c in cases]):
                    print(msg, flush=True)
            else:
                if not args.distance_only:
                    for msg in ex.map(process_case, cases, [dirs[c] for c in cases]):
                        print(msg, flush=True)
                for msg in ex.map(added_distance, cases):
                    print(msg, flush=True)
    for part, name in (('volumes', 'volumes'), ('global', 'global_metrics'), ('filled', 'filled_regions'),
                       ('added', 'added_voxels'), ('sections', 'section_fidelity'),
                       ('sections_labels', 'section_fidelity_labels'), ('distance', 'added_distance')):
        files = [os.path.join(PER_CASE, f'{c}_{part}.csv') for c in dirs]
        files = [f for f in files if os.path.exists(f)]
        if files:
            pd.concat([pd.read_csv(f) for f in files]).to_csv(os.path.join(TABLES, f'{name}.csv'), index=False)
    runtimes(list(dirs)).to_csv(os.path.join(TABLES, 'runtime.csv'), index=False)
    print('tables in', TABLES)


if __name__ == '__main__':
    main()
