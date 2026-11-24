"""Evaluate the ventricles and the CSF of the nxb18 predictions on MRIxFields against SuperSynth.

The NextBrain references have no ventricle / CSF labels, so these two nxb18 labels are compared
with the SuperSynth segmentations of the same images (FreeSurfer label ids):

    ventricles: nxb18 17  <->  SuperSynth {4, 5, 14, 15, 43, 44}   (lateral + inferior lateral
                                                                   ventricles, 3rd, 4th ventricle;
                                                                   same set as the
                                                                   other_codes/add_csf_to_labels
                                                                   SS_VENTRICLE_LABELS)
    csf:        nxb18 18  <->  SuperSynth {24}

Caveat: SuperSynth CSF (24) is all the extra-cerebral CSF of a whole-head scan, while the
training CSF of nxb18 is the rim built by the combined approach; the CSF Dice measures agreement
of definitions as much as accuracy. The ventricles are the meaningful comparison.

    prediction : <pred_dir>/<contrast>/<field>/<case>.nii.gz        (nxb18, raw values 0..18)
    reference  : <gt_dir>/<contrast>/<field>/<case>_0000.nii.gz     (SuperSynth)

Usage (conda env nxb):
    python evaluate.py                                  # every contrast/field present in both trees
    python evaluate.py --contrasts T1w,FLAIR --fields 7T --workers 48

Results (--out_dir, default ./results): same tables as ../nextbrain_nxb18 (per_case.csv,
per_case_mean.csv, summary_*.csv, missing.csv) for the 2 structures (label 1 = ventricles,
2 = csf; the 'mean' rows average the two).
"""
import argparse
import importlib.util
import sys
import warnings
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from nibabel.processing import resample_from_to

HERE = Path(__file__).resolve().parent
NXB18_ROOT = HERE.parents[2]                       # models/nxb18
TRAIN_ROOT = NXB18_ROOT.parents[1]                 # fixed_tree_training_csf
PRED_DIR = NXB18_ROOT / "predictions" / "mrixfields"
GT_DIR = TRAIN_ROOT / "data_shared" / "test" / "mrixfields" / "segmentations" / "supersynth"

# shared helpers of the NextBrain evaluation (tables, case matching)
_spec = importlib.util.spec_from_file_location("ev_nb", HERE.parent / "nextbrain_nxb18" / "evaluate.py")
ev = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ev)

STRUCTURES = {1: "ventricles", 2: "csf"}
PRED_LABELS = {1: [17], 2: [18]}
SS_LABELS = {1: [4, 5, 14, 15, 43, 44], 2: [24]}


def to_classes(seg, groups):
    out = np.zeros(seg.shape, np.uint8)
    for k, labs in groups.items():
        out[np.isin(seg, labs)] = k
    return out


def evaluate_case(item):
    contrast, field, case, pred_path, gt_path = item
    try:
        pred_img = nib.load(pred_path)
        pred = to_classes(np.rint(np.asanyarray(pred_img.dataobj)).astype(np.int64), PRED_LABELS)
        gt_img = nib.load(gt_path)
        gt = to_classes(np.rint(np.asanyarray(gt_img.dataobj)).astype(np.int64), SS_LABELS)
        resampled = False
        if gt.shape != pred.shape or not np.allclose(gt_img.affine, pred_img.affine, atol=1e-3):
            gt = np.asanyarray(resample_from_to(nib.Nifti1Image(gt, gt_img.affine), pred_img,
                                                order=0).dataobj).astype(np.uint8)
            resampled = True
        voxel_ml = float(np.prod(pred_img.header.get_zooms()[:3])) / 1000.0
        conf = np.bincount(gt.ravel().astype(np.int64) * 3 + pred.ravel(), minlength=9).reshape(3, 3)
        rows = []
        for lab, name in STRUCTURES.items():
            tp = conf[lab, lab]
            v_gt, v_pred = conf[lab].sum(), conf[:, lab].sum()
            dice = 2.0 * tp / (v_gt + v_pred) if v_gt + v_pred > 0 else np.nan
            vg, vp = v_gt * voxel_ml, v_pred * voxel_ml
            rows.append(dict(
                contrast=contrast, field=field, case=case, subject=case.split("_")[0],
                label=lab, structure=name, dice=dice,
                vol_gt_ml=vg, vol_pred_ml=vp, vol_diff_ml=vp - vg, abs_vol_diff_ml=abs(vp - vg),
                rel_vol_diff_pct=100.0 * (vp - vg) / vg if vg > 0 else np.nan,
                abs_rel_vol_diff_pct=100.0 * abs(vp - vg) / vg if vg > 0 else np.nan,
                # where the predicted label falls in SuperSynth (other class / background), mL
                pred_on_ss_other_ml=conf[3 - lab, lab] * voxel_ml,
                pred_on_ss_background_ml=conf[0, lab] * voxel_ml,
                gt_resampled=resampled))
        return rows, None
    except Exception as e:
        return [], f"{contrast}/{field}/{case}: {type(e).__name__}: {e}"


def main():
    warnings.filterwarnings("ignore", "invalid value encountered", RuntimeWarning)
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--contrasts", help="comma-separated, e.g. T1w,T2w,FLAIR (default: all available in both)")
    p.add_argument("--fields", help="comma-separated, e.g. 3T,7T (default: all available in both)")
    p.add_argument("--pred_dir", type=Path, default=PRED_DIR)
    p.add_argument("--gt_dir", type=Path, default=GT_DIR)
    p.add_argument("--out_dir", type=Path, default=HERE / "results")
    p.add_argument("--workers", type=int, default=16, help="parallel cases (default 16)")
    args = p.parse_args()

    items, missing = ev.collect_items(args.pred_dir, args.gt_dir, ev.parse_list(args.contrasts),
                                      ev.parse_list(args.fields))
    rows, errors = [], []
    with Pool(args.workers) as pool:
        for i, (r, err) in enumerate(pool.imap_unordered(evaluate_case, items, chunksize=1), 1):
            rows += r
            if err:
                errors.append(err)
                print("ERROR", err, file=sys.stderr)
            if i % 50 == 0 or i == len(items):
                print(f"  {i}/{len(items)}", flush=True)
    if not rows:
        sys.exit("every case failed")
    df = pd.DataFrame(rows).sort_values(["contrast", "field", "case", "label"]).reset_index(drop=True)
    if df["gt_resampled"].any():
        print(f"note: {df.loc[df.gt_resampled, 'case'].nunique()} references were resampled to the prediction grid")
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    ev.write_tables(df, out, missing, errors, len(STRUCTURES))
    # per structure, not only the mean of the two
    s = ev.summarize(df, ["contrast", "field"])
    print("\nper structure (Dice mean, median volume difference %, Pearson r of volumes):")
    for r in s[s.label > 0].itertuples():
        print(f"  {r.contrast:6s} {r.field:4s} {r.structure:10s} Dice {r.dice_mean:.3f}  "
              f"vol diff {r.rel_vol_diff_pct_median:+.0f}%  r {r.vol_pearson_r:.2f}")
    print(f"\nresults in {out}" + (f"  ({len(errors)} errors, see missing.csv)" if errors else ""))


if __name__ == "__main__":
    main()
