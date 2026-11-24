"""Evaluate the nxb18 nnU-Net predictions on MRIxFields against the NextBrain segmentations.

Adapted from fixed_tree_training/models/nxb16/evaluation/mrixfields/nextbrain_nxb16/evaluate.py.
nxb18 = the 16 nxb16 structures + ventricles (17) + CSF (18). The NextBrain segmentations of
MRIxFields have no ventricle and no CSF labels (they are background), so only the 16 nxb16
structures are scored: the references are mapped with
new_nextbrain_utils/nxb_label_mapping.map_segmentation(seg, "nxb16") exactly as for nxb16, and a
predicted 17/18 counts as "not that structure". The scores are therefore directly comparable with
the nxb16 tables (compare_nxb16.py).

    prediction : <pred_dir>/<contrast>/<field>/<case>.nii.gz        (raw values 0..18)
    reference  : <gt_dir>/<contrast>/<field>/<case>_0000.nii.gz

Mapped references: read first from --gt_cache_ro (default: the nxb16 cache, read-only, identical
to recomputing them), then from --cache_gt (inside this folder) if given; otherwise computed.
Nothing is ever written outside this folder.

Usage (conda env nxb):
    python evaluate.py                                  # every contrast/field present in both trees
    python evaluate.py --contrasts T1w,FLAIR --fields 7T
    python evaluate.py --workers 48

Results (--out_dir, default ./results): the same tables as nxb16
    per_case.csv, per_case_mean.csv, summary_contrast_field.csv, summary_contrast.csv,
    summary_field.csv, summary_overall.csv, missing.csv
plus
    per_case_csf.csv        per case: predicted ventricles / CSF (mL) and how much of them lies on
                            NextBrain tissue (labels 1..16) vs NextBrain background
    confusion_<field>.npz   one 17 x 19 matrix per contrast: rows NextBrain 0..16, cols nxb18 0..18
                            (voxel counts summed over the cases)
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
NXB18_ROOT = HERE.parents[2]                       # models/nxb18
TRAIN_ROOT = NXB18_ROOT.parents[1]                 # fixed_tree_training_csf
DEV2 = TRAIN_ROOT.parents[1]                       # dev2
sys.path.insert(0, str(DEV2 / "libraries" / "new_nextbrain_utils"))
from nxb_label_mapping import load_groups, map_segmentation  # noqa: E402

PRED_DIR = NXB18_ROOT / "predictions" / "mrixfields"
GT_DIR = TRAIN_ROOT / "data_shared" / "test" / "mrixfields" / "segmentations" / "nextbrain"
NXB16_CACHE = (DEV2 / "models_training" / "fixed_tree_training" / "models" / "nxb16" / "evaluation"
               / "mrixfields" / "nextbrain_nxb16" / "gt_nxb16_cache")
STRUCTURES = {i: name for i, name in enumerate(load_groups("nxb16")) if i > 0}   # 1..16
VENT, CSF = 17, 18
N_GT, N_PRED = 17, 19                              # 0..16 reference, 0..18 prediction


def list_dirs(root):
    """{(contrast, field)} of every <root>/<contrast>/<field> folder that holds .nii.gz files."""
    return {(d.parent.name, d.name) for d in root.glob("*/*")
            if d.is_dir() and not d.parent.name.startswith("_") and any(d.glob("*.nii.gz"))}


def load_gt(gt_path, contrast, field, case, ro_cache=None, rw_cache=None):
    """NextBrain segmentation mapped to nxb16 (uint8 array) and its image."""
    for cache in (ro_cache, rw_cache):
        if cache is not None and (cache / contrast / field / f"{case}.nii.gz").exists():
            img = nib.load(cache / contrast / field / f"{case}.nii.gz")
            return np.asanyarray(img.dataobj).astype(np.uint8), img
    img = nib.load(gt_path)
    seg = map_segmentation(np.asanyarray(img.dataobj), "nxb16", zooms=img.header.get_zooms()[:3])
    seg = seg.astype(np.uint8)
    if rw_cache is not None:
        path = rw_cache / contrast / field / f"{case}.nii.gz"
        path.parent.mkdir(parents=True, exist_ok=True)
        out = nib.Nifti1Image(seg, img.affine)
        out.set_data_dtype(np.uint8)
        nib.save(out, path)
    return seg, img


def evaluate_case(item, ro_cache=None, rw_cache=None):
    contrast, field, case, pred_path, gt_path = item
    try:
        pred_img = nib.load(pred_path)
        pred = np.rint(np.asanyarray(pred_img.dataobj)).astype(np.int64)
        gt, gt_img = load_gt(gt_path, contrast, field, case, ro_cache, rw_cache)
        resampled = False
        if gt.shape != pred.shape or not np.allclose(gt_img.affine, pred_img.affine, atol=1e-3):
            # different grids: nearest-neighbour resampling of the reference onto the prediction
            gt = np.asanyarray(resample_from_to(nib.Nifti1Image(gt, gt_img.affine), pred_img,
                                                order=0).dataobj).astype(np.int64)
            resampled = True
        voxel_ml = float(np.prod(pred_img.header.get_zooms()[:3])) / 1000.0
        if pred.min() < 0 or pred.max() >= N_PRED or gt.max() >= N_GT:
            raise ValueError(f"labels out of range: pred {pred.min()}..{pred.max()}, gt max {gt.max()}")

        conf = np.bincount(gt.ravel().astype(np.int64) * N_PRED + pred.ravel(),
                           minlength=N_GT * N_PRED).reshape(N_GT, N_PRED)     # rows gt, cols pred
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
        csf = dict(contrast=contrast, field=field, case=case, subject=case.split("_")[0])
        for lab, name in ((VENT, "vent"), (CSF, "csf")):
            csf[f"{name}_pred_ml"] = conf[:, lab].sum() * voxel_ml
            csf[f"{name}_on_nb_background_ml"] = conf[0, lab] * voxel_ml
            csf[f"{name}_on_nb_tissue_ml"] = conf[1:, lab].sum() * voxel_ml
        return rows, csf, conf, None
    except Exception as e:  # keep going, report at the end
        return [], None, None, f"{contrast}/{field}/{case}: {type(e).__name__}: {e}"


METRICS = ["dice", "vol_gt_ml", "vol_pred_ml", "vol_diff_ml", "abs_vol_diff_ml",
           "rel_vol_diff_pct", "abs_rel_vol_diff_pct"]


def summarize(df, keys, metrics=METRICS):
    """mean/std/median of every metric + Pearson r(vol_pred, vol_gt) per keys x structure,
    plus a 'mean' structure row (per-case mean over the structures, then summarized)."""
    case_keys = list(dict.fromkeys(keys + ["contrast", "field", "case"]))
    per_case_mean = (df.groupby(case_keys, as_index=False)[metrics]
                     .mean().assign(label=0, structure="mean"))
    full = pd.concat([df, per_case_mean], ignore_index=True)
    g = full.groupby(keys + ["label", "structure"], sort=True)
    out = g[metrics].agg(["mean", "std", "median"])
    out.columns = [f"{m}_{s}" for m, s in out.columns]
    out.insert(0, "n_cases", g["case"].nunique())
    out["vol_pearson_r"] = g[["vol_pred_ml", "vol_gt_ml"]].apply(
        lambda x: x["vol_pred_ml"].corr(x["vol_gt_ml"]) if len(x) > 1 else np.nan)
    out = out.reset_index()
    # put the 'mean' row last inside each group
    out["_o"] = out["label"].replace(0, 99)
    return out.sort_values(keys + ["_o"]).drop(columns="_o").reset_index(drop=True)


def write_tables(df, out, missing, errors, n_struct):
    df.to_csv(out / "per_case.csv", index=False)
    (df.groupby(["contrast", "field", "case", "subject"], as_index=False)[METRICS].mean()
       .to_csv(out / "per_case_mean.csv", index=False))
    summarize(df, ["contrast", "field"]).to_csv(out / "summary_contrast_field.csv", index=False)
    summarize(df, ["contrast"]).to_csv(out / "summary_contrast.csv", index=False)
    summarize(df, ["field"]).to_csv(out / "summary_field.csv", index=False)
    summarize(df.assign(all="all"), ["all"]).drop(columns="all").to_csv(out / "summary_overall.csv", index=False)
    pd.DataFrame(missing + [dict(contrast="", field="", case=e, missing="error") for e in errors],
                 columns=["contrast", "field", "case", "missing"]).to_csv(out / "missing.csv", index=False)
    s = summarize(df, ["contrast", "field"])
    s = s[s.structure == "mean"].set_index(["contrast", "field"])
    print(f"\nmean Dice over the {n_struct} structures (mean +- std over cases):")
    for (c, f), r in s.iterrows():
        print(f"  {c:6s} {f:4s} n={r.n_cases:4d}  Dice {r.dice_mean:.3f} +- {r.dice_std:.3f}   "
              f"|vol err| {r.abs_rel_vol_diff_pct_mean:.1f}%")


def parse_list(s):
    return None if s is None else [x.strip() for x in s.split(",") if x.strip()]


def collect_items(pred_dir, gt_dir, contrasts, fields):
    """(items, missing): matched (contrast, field, case, pred, gt) and the unmatched cases."""
    pred_dirs, gt_dirs = list_dirs(pred_dir), list_dirs(gt_dir)
    common = pred_dirs & gt_dirs
    sel = sorted(d for d in common
                 if (contrasts is None or d[0] in contrasts) and (fields is None or d[1] in fields))
    for c in contrasts or []:
        if not any(d[0] == c for d in common):
            print(f"warning: contrast {c} not available in both trees", file=sys.stderr)
    for f in fields or []:
        if not any(d[1] == f for d in common):
            print(f"warning: field {f} not available in both trees", file=sys.stderr)
    if not sel:
        sys.exit(f"nothing to evaluate. predictions: {sorted(pred_dirs)}  reference: {sorted(gt_dirs)}")
    items, missing = [], []
    for contrast, field in sel:
        preds = {f.name[:-7]: f for f in (pred_dir / contrast / field).glob("*.nii.gz")}
        gts = {f.name[:-12]: f for f in (gt_dir / contrast / field).glob("*_0000.nii.gz")}
        for case in sorted(gts.keys() - preds.keys()):
            missing.append(dict(contrast=contrast, field=field, case=case, missing="prediction"))
        for case in sorted(preds.keys() - gts.keys()):
            missing.append(dict(contrast=contrast, field=field, case=case, missing="reference"))
        for case in sorted(preds.keys() & gts.keys()):
            items.append((contrast, field, case, preds[case], gts[case]))
        print(f"{contrast}/{field}: {len(preds.keys() & gts.keys())} cases "
              f"({len(gts.keys() - preds.keys())} without prediction, "
              f"{len(preds.keys() - gts.keys())} without reference)")
    if not items:
        sys.exit("no case has both a prediction and a reference")
    return items, missing


def main():
    warnings.filterwarnings("ignore", "invalid value encountered", RuntimeWarning)   # Pearson r with constant volumes
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--contrasts", help="comma-separated, e.g. T1w,T2w,FLAIR (default: all available in both)")
    p.add_argument("--fields", help="comma-separated, e.g. 3T,7T (default: all available in both)")
    p.add_argument("--pred_dir", type=Path, default=PRED_DIR)
    p.add_argument("--gt_dir", type=Path, default=GT_DIR)
    p.add_argument("--out_dir", type=Path, default=HERE / "results")
    p.add_argument("--workers", type=int, default=16, help="parallel cases (default 16)")
    p.add_argument("--gt_cache_ro", type=Path, default=NXB16_CACHE,
                   help="read-only cache of mapped references (default: the nxb16 one); 'none' to disable")
    p.add_argument("--cache_gt", nargs="?", const=HERE / "gt_nxb16_cache", type=Path, default=None,
                   help="save/reuse the references not in --gt_cache_ro here (default folder: ./gt_nxb16_cache)")
    args = p.parse_args()
    ro = None if str(args.gt_cache_ro).lower() == "none" else args.gt_cache_ro
    if args.cache_gt is not None and TRAIN_ROOT not in args.cache_gt.resolve().parents:
        sys.exit(f"--cache_gt must be inside {TRAIN_ROOT}")

    items, missing = collect_items(args.pred_dir, args.gt_dir, parse_list(args.contrasts),
                                   parse_list(args.fields))
    rows, csf_rows, errors = [], [], []
    conf = {}                                      # field -> contrast -> 17 x 19
    with Pool(args.workers) as pool:
        work = partial(evaluate_case, ro_cache=ro, rw_cache=args.cache_gt)
        for i, ((contrast, field, *_), (r, c, m, err)) in enumerate(
                zip(items, pool.imap(work, items, chunksize=1)), 1):
            rows += r
            if c:
                csf_rows.append(c)
                cf = conf.setdefault(field, {}).setdefault(contrast, np.zeros((N_GT, N_PRED), np.int64))
                cf += m
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
    write_tables(df, out, missing, errors, len(STRUCTURES))
    pd.DataFrame(csf_rows).sort_values(["contrast", "field", "case"]).to_csv(out / "per_case_csf.csv", index=False)
    for field, m in conf.items():
        np.savez(out / f"confusion_{field}.npz", **m)
    print(f"\nresults in {out}" + (f"  ({len(errors)} errors, see missing.csv)" if errors else ""))


if __name__ == "__main__":
    main()
