"""Evaluate the nxb16 nnU-Net predictions on MRIxFields against the NextBrain segmentations.

The NextBrain segmentations (hundreds of labels) are mapped to the 16 nxb16 structures with
new_nextbrain_utils/nxb_label_mapping.map_segmentation(seg, "nxb16"), then every structure
(labels 1..16, background excluded) is compared with Dice and volumes.

    prediction : <pred_dir>/<contrast>/<field>/<case>.nii.gz        (e.g. sub-P0001_ses-3T_T1w.nii.gz)
    reference  : <gt_dir>/<contrast>/<field>/<case>_0000.nii.gz

Usage (conda env nxb):
    python evaluate.py                                  # every contrast/field present in both trees
    python evaluate.py --contrasts T1w,FLAIR --fields 7T
    python evaluate.py --workers 32 --cache_gt          # keep the mapped NextBrain segs for re-runs

Results (--out_dir, default ./results):
    per_case.csv                    one row per case x structure: dice, volumes (mL), volume errors
    per_case_mean.csv               one row per case: mean over the 16 structures
    summary_contrast_field.csv      per contrast x field x structure (+ "mean" = mean over structures)
    summary_contrast.csv            per contrast x structure (fields pooled)
    summary_field.csv               per field x structure (contrasts pooled)
    summary_overall.csv             per structure (everything pooled)
    missing.csv                     reference cases without a prediction (or vice versa)
Summaries give n_cases, mean/std/median of Dice, volumes and volume errors, and the Pearson r
between predicted and reference volumes. Dice is NaN when a structure is absent in both.
"""
import argparse
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
NXB16_ROOT = HERE.parents[2]                       # models/nxb16
TRAIN_ROOT = NXB16_ROOT.parents[1]                 # fixed_tree_training
DEV2 = TRAIN_ROOT.parents[1]                       # dev2
sys.path.insert(0, str(DEV2 / "libraries" / "new_nextbrain_utils"))
from nxb_label_mapping import load_groups, map_segmentation  # noqa: E402

PRED_DIR = NXB16_ROOT / "predictions" / "mrixfields"
GT_DIR = TRAIN_ROOT / "data_shared" / "test" / "mrixfields" / "segmentations" / "nextbrain"
STRUCTURES = {i: name for i, name in enumerate(load_groups("nxb16")) if i > 0}   # 1..16


def list_dirs(root):
    """{(contrast, field)} of every <root>/<contrast>/<field> folder that holds .nii.gz files."""
    return {(d.parent.name, d.name) for d in root.glob("*/*")
            if d.is_dir() and not d.parent.name.startswith("_") and any(d.glob("*.nii.gz"))}


def load_gt(gt_path, cache_path=None):
    """NextBrain segmentation mapped to nxb16 (uint8 array) and its image."""
    if cache_path is not None and cache_path.exists():
        img = nib.load(cache_path)
        return np.asanyarray(img.dataobj).astype(np.uint8), img
    img = nib.load(gt_path)
    seg = map_segmentation(np.asanyarray(img.dataobj), "nxb16", zooms=img.header.get_zooms()[:3])
    seg = seg.astype(np.uint8)
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        out = nib.Nifti1Image(seg, img.affine)
        out.set_data_dtype(np.uint8)
        nib.save(out, cache_path)
    return seg, img


def evaluate_case(item, cache_dir=None):
    contrast, field, case, pred_path, gt_path = item
    try:
        pred_img = nib.load(pred_path)
        pred = np.rint(np.asanyarray(pred_img.dataobj)).astype(np.int64)
        cache = cache_dir / contrast / field / f"{case}.nii.gz" if cache_dir else None
        gt, gt_img = load_gt(gt_path, cache)
        resampled = False
        if gt.shape != pred.shape or not np.allclose(gt_img.affine, pred_img.affine, atol=1e-3):
            # different grids: nearest-neighbour resampling of the reference onto the prediction
            gt = np.asanyarray(resample_from_to(nib.Nifti1Image(gt, gt_img.affine), pred_img,
                                                order=0).dataobj).astype(np.int64)
            resampled = True
        voxel_ml = float(np.prod(pred_img.header.get_zooms()[:3])) / 1000.0

        n = len(STRUCTURES) + 1
        pred_c = np.clip(pred.astype(np.int64), 0, n)          # values > 16 -> "other" bin n
        gt_c = np.clip(gt.astype(np.int64), 0, n)              # int64: no uint8 overflow below
        conf = np.bincount(gt_c.ravel() * (n + 1) + pred_c.ravel(),
                           minlength=(n + 1) ** 2).reshape(n + 1, n + 1)   # rows gt, cols pred
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
                gt_resampled=resampled))
        return rows, None
    except Exception as e:  # keep going, report at the end
        return [], f"{contrast}/{field}/{case}: {type(e).__name__}: {e}"


METRICS = ["dice", "vol_gt_ml", "vol_pred_ml", "vol_diff_ml", "abs_vol_diff_ml",
           "rel_vol_diff_pct", "abs_rel_vol_diff_pct"]


def summarize(df, keys):
    """mean/std/median of every metric + Pearson r(vol_pred, vol_gt) per keys x structure,
    plus a 'mean' structure row (per-case mean over the 16 structures, then summarized)."""
    case_keys = list(dict.fromkeys(keys + ["contrast", "field", "case"]))
    per_case_mean = (df.groupby(case_keys, as_index=False)[METRICS]
                     .mean().assign(label=0, structure="mean"))
    full = pd.concat([df, per_case_mean], ignore_index=True)
    g = full.groupby(keys + ["label", "structure"], sort=True)
    out = g[METRICS].agg(["mean", "std", "median"])
    out.columns = [f"{m}_{s}" for m, s in out.columns]
    out.insert(0, "n_cases", g["case"].nunique())
    out["vol_pearson_r"] = g[["vol_pred_ml", "vol_gt_ml"]].apply(
        lambda x: x["vol_pred_ml"].corr(x["vol_gt_ml"]) if len(x) > 1 else np.nan)
    out = out.reset_index()
    # put the 'mean' row last inside each group
    out["_o"] = out["label"].replace(0, 99)
    return out.sort_values(keys + ["_o"]).drop(columns="_o").reset_index(drop=True)


def parse_list(s):
    return None if s is None else [x.strip() for x in s.split(",") if x.strip()]


def main():
    warnings.filterwarnings("ignore", "invalid value encountered", RuntimeWarning)   # Pearson r with constant volumes
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--contrasts", help="comma-separated, e.g. T1w,T2w,FLAIR (default: all available in both)")
    p.add_argument("--fields", help="comma-separated, e.g. 3T,7T (default: all available in both)")
    p.add_argument("--pred_dir", type=Path, default=PRED_DIR)
    p.add_argument("--gt_dir", type=Path, default=GT_DIR)
    p.add_argument("--out_dir", type=Path, default=HERE / "results")
    p.add_argument("--workers", type=int, default=16, help="parallel cases (default 16; ~8 s and a few GB per case)")
    p.add_argument("--cache_gt", nargs="?", const=HERE / "gt_nxb16_cache", type=Path, default=None,
                   help="save/reuse the NextBrain segs mapped to nxb16 (default folder: ./gt_nxb16_cache)")
    args = p.parse_args()

    pred_dirs, gt_dirs = list_dirs(args.pred_dir), list_dirs(args.gt_dir)
    common = pred_dirs & gt_dirs
    contrasts, fields = parse_list(args.contrasts), parse_list(args.fields)
    sel = sorted(d for d in common
                 if (contrasts is None or d[0] in contrasts) and (fields is None or d[1] in fields))
    for c in contrasts or []:
        if not any(d[0] == c for d in common):
            print(f"warning: contrast {c} not available in both trees", file=sys.stderr)
    for f in fields or []:
        if not any(d[1] == f for d in common):
            print(f"warning: field {f} not available in both trees", file=sys.stderr)
    if not sel:
        sys.exit(f"nothing to evaluate. predictions: {sorted(pred_dirs)}  nextbrain: {sorted(gt_dirs)}")

    items, missing = [], []
    for contrast, field in sel:
        preds = {f.name[:-7]: f for f in (args.pred_dir / contrast / field).glob("*.nii.gz")}
        gts = {f.name[:-12]: f for f in (args.gt_dir / contrast / field).glob("*_0000.nii.gz")}
        for case in sorted(gts.keys() - preds.keys()):
            missing.append(dict(contrast=contrast, field=field, case=case, missing="prediction"))
        for case in sorted(preds.keys() - gts.keys()):
            missing.append(dict(contrast=contrast, field=field, case=case, missing="nextbrain"))
        for case in sorted(preds.keys() & gts.keys()):
            items.append((contrast, field, case, preds[case], gts[case]))
        print(f"{contrast}/{field}: {len(preds.keys() & gts.keys())} cases "
              f"({len(gts.keys() - preds.keys())} without prediction, "
              f"{len(preds.keys() - gts.keys())} without nextbrain)")
    if not items:
        sys.exit("no case has both a prediction and a nextbrain segmentation")

    rows, errors = [], []
    with Pool(args.workers) as pool:
        for i, (r, err) in enumerate(pool.imap_unordered(
                partial(evaluate_case, cache_dir=args.cache_gt), items, chunksize=1), 1):
            rows += r
            if err:
                errors.append(err)
                print("ERROR", err, file=sys.stderr)
            if i % 20 == 0 or i == len(items):
                print(f"  {i}/{len(items)}", flush=True)
    if not rows:
        sys.exit("every case failed")

    df = pd.DataFrame(rows).sort_values(["contrast", "field", "case", "label"]).reset_index(drop=True)
    if df["gt_resampled"].any():
        print(f"note: {df.loc[df.gt_resampled, 'case'].nunique()} references were resampled to the prediction grid")

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "per_case.csv", index=False)
    (df.groupby(["contrast", "field", "case", "subject"], as_index=False)[METRICS].mean()
       .to_csv(out / "per_case_mean.csv", index=False))
    summarize(df, ["contrast", "field"]).to_csv(out / "summary_contrast_field.csv", index=False)
    summarize(df, ["contrast"]).to_csv(out / "summary_contrast.csv", index=False)
    summarize(df, ["field"]).to_csv(out / "summary_field.csv", index=False)
    summarize(df.assign(all="all"), ["all"]).drop(columns="all").to_csv(out / "summary_overall.csv", index=False)
    pd.DataFrame(missing + [dict(contrast="", field="", case=e, missing="error") for e in errors],
                 columns=["contrast", "field", "case", "missing"]).to_csv(out / "missing.csv", index=False)

    # short report: mean Dice over the 16 structures per contrast x field
    s = summarize(df, ["contrast", "field"])
    s = s[s.structure == "mean"].set_index(["contrast", "field"])
    print("\nmean Dice over the 16 structures (mean +- std over cases):")
    for (c, f), r in s.iterrows():
        print(f"  {c:6s} {f:4s} n={r.n_cases:4d}  Dice {r.dice_mean:.3f} +- {r.dice_std:.3f}   "
              f"|vol err| {r.abs_rel_vol_diff_pct_mean:.1f}%")
    print(f"\nresults in {out}" + (f"  ({len(errors)} errors, see missing.csv)" if errors else ""))


if __name__ == "__main__":
    main()
