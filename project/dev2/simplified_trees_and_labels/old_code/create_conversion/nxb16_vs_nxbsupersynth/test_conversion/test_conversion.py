#!/usr/bin/env python3
"""
Prove that to_common_space.py makes nxb16 and nxbsupersynth segmentations comparable.

Data (data/, one file per case, same names in every folder)
  nxb16/          nxb16 ground truth            (0.2 mm)
  nxbsupersynth/  nxbsupersynth ground truth    (0.2 mm, same grid as nxb16)
  supersynth/     real mri_super_synth output   (0.4 mm, Right- or Left- ids): resampled to the
                  ground-truth grid with nearest neighbour before any comparison

Tests
  0. coverage     every label of every map is in common_space.yaml (conversion with unknown=error)
  1. gt_vs_gt     nxb16 GT vs nxbsupersynth GT in the common space. Both come from the same
                  NextBrain map, so the dice must reach the best possible dice of
                  common_space_check.csv (|diff| <= 0.01) and be >= 0.99 except for 'approximate'.
  2. references   real SuperSynth vs nxbsupersynth GT and vs nxb16 GT, in the common space, on the
                  same voxels (comparable in all three pairs). If the common space works, the
                  two ground truths are interchangeable references: the dice of the same
                  SuperSynth map must be the same with both (|diff| <= 0.02 except 'approximate').
  3. native       real SuperSynth vs nxbsupersynth GT in the native SuperSynth classes (Right-
                  ids, no common space), for reference: what the merging hides or gains.

Outputs (results/)
  gt_vs_gt_dice.csv, supersynth_vs_nxbsupersynth_dice.csv, supersynth_vs_nxb16_dice.csv,
  supersynth_native_dice.csv (per case + mean), and results.md (summary with PASS / FAIL).

    python test_conversion.py [--jobs 5]
"""
import argparse
import csv
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import affine_transform

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from to_common_space import DEFAULT_CS, comparable_mask, load_common_space, to_common  # noqa: E402

DATA = HERE / 'data'
OUT = HERE / 'results'
CHECK = HERE / 'common_space_check.csv'

TOL_GT = 0.01          # test 1: |dice - best possible dice|
TOL_REF = 0.02         # test 2: |dice vs nxb16 GT - dice vs nxbsupersynth GT|

# SuperSynth Left- -> Right- ids (to harmonise the native comparison)
L2R = {2: 41, 3: 42, 4: 43, 5: 44, 7: 46, 8: 47, 10: 49, 11: 50, 12: 51, 13: 52, 17: 53, 18: 54,
       26: 58, 819: 820, 821: 822, 843: 844, 865: 866, 869: 870}
NATIVE = {41: 'cerebral_white_matter', 42: 'cerebral_cortex', 46: 'cerebellum_white_matter',
          47: 'cerebellum_cortex', 49: 'thalamus', 50: 'caudate', 51: 'putamen', 52: 'pallidum',
          16: 'brain_stem', 53: 'hippocampus', 54: 'amygdala', 58: 'accumbens_area',
          85: 'optic_chiasm', 820: 'hypothal_nomb', 822: 'fornix', 844: 'mammillarybody',
          866: 'basal_forebrain', 870: 'septalnuc'}


def resample_nn(img, ref):
    """Label map `img` resampled to the grid of `ref` (nearest neighbour, 0 outside)."""
    m = np.linalg.inv(img.affine) @ ref.affine
    data = np.rint(np.asanyarray(img.dataobj)).astype(np.int32)
    return affine_transform(data, m[:3, :3], m[:3, 3], output_shape=ref.shape, order=0,
                            mode='constant', cval=0)


def dice(a, b, labels):
    """{label: dice} between two label arrays (nan if the label is in neither)."""
    out = {}
    for k in labels:
        x, y = a == k, b == k
        s = int(x.sum()) + int(y.sum())
        out[k] = 2 * int((x & y).sum()) / s if s else float('nan')
    return out


def run_case(name):
    cs = load_common_space(DEFAULT_CS)
    ids = [c['id'] for c in cs['classes']]
    i16 = nib.load(DATA / 'nxb16' / name)
    inss = nib.load(DATA / 'nxbsupersynth' / name)
    if i16.shape != inss.shape or not np.allclose(i16.affine, inss.affine, atol=1e-4):
        raise ValueError(f'{name}: nxb16 and nxbsupersynth GT are not on the same grid')
    g16 = np.asanyarray(i16.dataobj)
    gnss = np.asanyarray(inss.dataobj)
    ss = resample_nn(nib.load(DATA / 'supersynth' / name), i16)

    # test 0: coverage (raises if a label is missing from common_space.yaml)
    c16 = to_common(g16, 'nxb16', cs, unknown='error')
    cnss = to_common(gnss, 'nxbsupersynth', cs, unknown='error')
    css = to_common(ss, 'supersynth', cs, unknown='error')

    # test 1
    m = comparable_mask(c16, cnss, 'nxb16', 'nxbsupersynth', cs)
    t1 = dice(c16[m], cnss[m], ids)
    t1['ignored_fraction'] = 1 - m.mean()

    # test 2: same voxels for both references
    m = (comparable_mask(css, cnss, 'supersynth', 'nxbsupersynth', cs)
         & comparable_mask(css, c16, 'supersynth', 'nxb16', cs))
    t2_nss = dice(css[m], cnss[m], ids)
    t2_16 = dice(css[m], c16[m], ids)
    t2_nss['ignored_fraction'] = t2_16['ignored_fraction'] = 1 - m.mean()

    # test 3: native SuperSynth classes
    gnss = gnss.astype(np.int32)
    lut = np.arange(max(int(ss.max()), int(gnss.max()), max(L2R)) + 1, dtype=np.int32)
    for l, r in L2R.items():
        lut[l] = r
    t3 = dice(lut[ss], lut[gnss], list(NATIVE))

    labels = {'ss': sorted(np.unique(ss).tolist())}
    return name, t1, t2_nss, t2_16, t3, labels


def write_csv(path, rows, keys, names):
    mean = {k: float(np.nanmean([r[1][k] for r in rows])) for k in keys}
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['case'] + names)
        for case, d in rows:
            w.writerow([case] + [round(d[k], 5) for k in keys])
        w.writerow(['mean'] + [round(mean[k], 5) for k in keys])
    return mean


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--jobs', type=int, default=5)
    args = p.parse_args()

    cs = load_common_space(DEFAULT_CS)
    with open(CHECK) as f:
        best = {int(r['common_id']): float(r['best_dice']) for r in csv.DictReader(f)}
    cases = sorted(f.name for f in (DATA / 'nxb16').glob('*.nii*'))
    for d in ('nxbsupersynth', 'supersynth'):
        miss = [c for c in cases if not (DATA / d / c).exists()]
        if miss:
            sys.exit(f'missing in data/{d}: {miss}')

    res = []
    with ProcessPoolExecutor(args.jobs) as ex:
        for r in ex.map(run_case, cases):
            print(f'{r[0]}: done (supersynth labels {r[5]["ss"]})', flush=True)
            res.append(r)

    OUT.mkdir(exist_ok=True)
    ids = [c['id'] for c in cs['classes']]
    keys = ids + ['ignored_fraction']
    names = [c['name'] for c in cs['classes']] + ['ignored_fraction']
    m1 = write_csv(OUT / 'gt_vs_gt_dice.csv', [(r[0], r[1]) for r in res], keys, names)
    m2n = write_csv(OUT / 'supersynth_vs_nxbsupersynth_dice.csv', [(r[0], r[2]) for r in res], keys, names)
    m2s = write_csv(OUT / 'supersynth_vs_nxb16_dice.csv', [(r[0], r[3]) for r in res], keys, names)
    m3 = write_csv(OUT / 'supersynth_native_dice.csv', [(r[0], r[4]) for r in res],
                   list(NATIVE), list(NATIVE.values()))

    # ---------------------------------------------------------------- report
    L = ['# Test results: nxb16 vs nxbsupersynth common space', '',
         f'{len(cases)} cases ({", ".join(cases)}). Generated by `test_conversion.py`. '
         'Dice values are means over the cases.', '',
         '## Test 0: coverage', '',
         'Every label of every map (nxb16 GT, nxbsupersynth GT, real SuperSynth) is in '
         '`common_space.yaml`: **PASS** (the conversion runs with `unknown=error`).', '',
         'SuperSynth labels found: ' + ', '.join(
             f'{r[0]} {r[5]["ss"]}' for r in res), '']
    all_ok = True

    L += ['## Test 1: nxb16 GT vs nxbsupersynth GT', '',
          f'Pass if |dice − best possible dice| ≤ {TOL_GT} and dice ≥ 0.99 (except "approximate"). '
          f'Ignored voxels: {100 * m1["ignored_fraction"]:.2f} %.', '',
          '| common class | status | best possible | dice | result |', '|---|---|---|---|---|']
    for c in cs['classes']:
        d, b = m1[c['id']], best[c['id']]
        ok = abs(d - b) <= TOL_GT and (d >= 0.99 or c['status'] == 'approximate')
        all_ok &= ok
        L.append(f'| {c["name"]} | {c["status"]} | {b:.4f} | {d:.4f} | {"PASS" if ok else "FAIL"} |')

    L += ['', '## Test 2: real SuperSynth against both ground truths', '',
          'Same SuperSynth map, same voxels, two references. Pass if '
          f'|dice vs nxb16 − dice vs nxbsupersynth| ≤ {TOL_REF} (except "approximate"; the class '
          'is skipped if SuperSynth does not have it). '
          f'Ignored voxels: {100 * m2n["ignored_fraction"]:.2f} %.', '',
          '| common class | status | vs nxbsupersynth GT | vs nxb16 GT | diff | result |',
          '|---|---|---|---|---|---|']
    for c in cs['classes']:
        a, b = m2n[c['id']], m2s[c['id']]
        if c['id'] not in cs['available']['supersynth']:
            L.append(f'| {c["name"]} | {c["status"]} | – | – | – | n/a (not in SuperSynth) |')
            continue
        ok = abs(a - b) <= TOL_REF or c['status'] == 'approximate'
        all_ok &= ok
        res_s = 'PASS' if abs(a - b) <= TOL_REF else ('expected (approximate)' if ok else 'FAIL')
        L.append(f'| {c["name"]} | {c["status"]} | {a:.4f} | {b:.4f} | {b - a:+.4f} | {res_s} |')

    L += ['', '## Test 3 (reference): real SuperSynth vs nxbsupersynth GT, native classes', '',
          'No common space, Right- ids. Shows which SuperSynth classes disagree before merging.', '',
          '| SuperSynth class | dice |', '|---|---|']
    for k, n in NATIVE.items():
        L.append(f'| {k} {n} | {m3[k]:.4f} |')

    L += ['', f'## Overall: **{"PASS" if all_ok else "FAIL"}**', '']
    (OUT / 'results.md').write_text('\n'.join(L))
    print('\n'.join(L))


if __name__ == '__main__':
    main()
