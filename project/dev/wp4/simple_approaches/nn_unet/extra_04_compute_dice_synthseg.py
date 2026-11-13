"""Compute per-label Dice between nnU-Net predictions and SynthSeg ground truth.

Pipeline for every (prediction, ground-truth) pair:
    1. Load the nnU-Net prediction (labels 0-8, see dataset.json).
    2. Load the SynthSeg ground truth (FreeSurfer ASeg ids) and merge it into
       the 8 L1 tissue classes (0-7) with `merge_synthseg_labels.merge_array`.
    3. Ignore label 8 ("other") of the prediction: those voxels are set to
       background (0), so label 8 gets no Dice score of its own.
    4. Compute the Dice of every foreground label 1-7 and their mean.

Pairing:
    Prediction:   <pred_dir>/<modality>/.../R_T1W_3T_0403.nii.gz
    Ground truth: <gt_dir>/<modality>/.../SR_T1W_3T_0403_synthseg_resampled.nii.gz
    A GT file matches a prediction when its name, with the
    '_synthseg_resampled' suffix removed, ends with the prediction name.
    Only cases that exist in both folders are evaluated.

Outputs (in --out_dir):
    dice_per_case.csv   one row per case: modality, case, dice per label, mean
    dice_summary.csv    mean / std / median of every label, per modality

Usage:
    python 04_compute_dice.py                       # default paths below
    python 04_compute_dice.py --workers 16
"""
import argparse
import glob
import importlib.util
import os
from multiprocessing import Pool

import nibabel as nib
import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Default paths
# --------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
PRED_DIR = os.path.join(HERE, "data/outputs/mrixfields")
GT_DIR = "/home/agustin/phd/miccai/miccai_2026/mri_x_fields/data/train_data/preprocessed/synthseg"
OUT_DIR = os.path.join(PRED_DIR, "results")
MERGE_SCRIPT = ("/home/agustin/phd/synthseg_nextbrain/project/dev/wp1/"
                "create_merged_segmentations/merge_synthseg_seg/merge_synthseg_labels.py")

GT_SUFFIX = "_synthseg_resampled"
IGNORED_PRED_LABEL = 8  # "other" in the nnU-Net dataset, absent from the GT


def load_merge_module(path=MERGE_SCRIPT):
    """Import merge_synthseg_labels.py from its absolute path."""
    spec = importlib.util.spec_from_file_location("merge_synthseg_labels", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


merge = load_merge_module()
# {label id: label name} for the foreground L1 classes (1..7)
LABELS = {v: k for k, v in merge.L1_LABELS.items() if v != 0}


# --------------------------------------------------------------------------
# File pairing
# --------------------------------------------------------------------------
def stem(path):
    """File name without the .nii / .nii.gz extension."""
    name = os.path.basename(path)
    for ext in (".nii.gz", ".nii"):
        if name.endswith(ext):
            return name[: -len(ext)]
    return name


def find_pairs(pred_dir, gt_dir):
    """Return a list of (modality, case, pred_path, gt_path) for matching cases.

    Modalities are the sub-folders of pred_dir (T1W, T2W, ...). Field-strength
    sub-folders may be named differently (3T vs 3.0T), so files are searched
    recursively inside each modality folder.
    """
    pairs = []
    for modality in sorted(os.listdir(pred_dir)):
        pred_mod, gt_mod = os.path.join(pred_dir, modality), os.path.join(gt_dir, modality)
        if not (os.path.isdir(pred_mod) and os.path.isdir(gt_mod)):
            continue

        gt_files = glob.glob(os.path.join(gt_mod, "**", f"*{GT_SUFFIX}.nii*"), recursive=True)
        gt_by_stem = {stem(f)[: -len(GT_SUFFIX)]: f for f in gt_files}

        for pred in sorted(glob.glob(os.path.join(pred_mod, "**", "*.nii*"), recursive=True)):
            case = stem(pred)
            matches = [f for s, f in gt_by_stem.items() if s.endswith(case)]
            if len(matches) == 1:
                pairs.append((modality, case, pred, matches[0]))
            elif len(matches) > 1:
                print(f"[skip] {modality}/{case}: ambiguous GT matches {matches}")
    return pairs


# --------------------------------------------------------------------------
# Dice
# --------------------------------------------------------------------------
def dice(pred, gt, label):
    """Dice of one label. NaN when the label is absent from both volumes."""
    p, g = pred == label, gt == label
    denom = p.sum() + g.sum()
    if denom == 0:
        return np.nan
    return 2.0 * np.logical_and(p, g).sum() / denom


def evaluate_pair(pair):
    """Compute the Dice of every L1 label for one (prediction, GT) pair."""
    modality, case, pred_path, gt_path = pair
    pred_img, gt_img = nib.load(pred_path), nib.load(gt_path)

    pred = np.rint(np.asanyarray(pred_img.dataobj)).astype(np.uint8)
    gt = merge.merge_array(np.asanyarray(gt_img.dataobj))  # ASeg ids -> L1 ids (0-7)

    if pred.shape != gt.shape:
        print(f"[skip] {modality}/{case}: shape mismatch {pred.shape} vs {gt.shape}")
        return None
    if not np.allclose(pred_img.affine, gt_img.affine, atol=1e-3):
        print(f"[warn] {modality}/{case}: affines differ")

    pred[pred == IGNORED_PRED_LABEL] = 0  # ignore the "other" label
    gt[gt == IGNORED_PRED_LABEL] = 0  # ignore the "other" label in GT too, if present

    # triack to avoid computing csf
    # pred[gt == 0] = 0

    row = {"modality": modality, "case": case}
    for label, name in LABELS.items():
        row[name] = dice(pred, gt, label)
    row["mean_dice"] = np.nanmean([row[n] for n in LABELS.values()])
    print(f"{modality}/{case}: mean dice = {row['mean_dice']:.4f}")
    return row


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pred_dir", default=PRED_DIR, help="nnU-Net predictions (one sub-folder per modality)")
    p.add_argument("--gt_dir", default=GT_DIR, help="SynthSeg ground truth (one sub-folder per modality)")
    p.add_argument("--out_dir", default=OUT_DIR, help="Where to write the CSV files")
    p.add_argument("--workers", type=int, default=8, help="Parallel processes")
    args = p.parse_args()

    pairs = find_pairs(args.pred_dir, args.gt_dir)
    print(f"Found {len(pairs)} prediction/GT pairs")
    if not pairs:
        return

    with Pool(args.workers) as pool:
        rows = [r for r in pool.map(evaluate_pair, pairs) if r is not None]

    df = pd.DataFrame(rows).sort_values(["modality", "case"])
    metric_cols = list(LABELS.values()) + ["mean_dice"]

    # Summary: mean / std / median of every label for each modality and overall
    summary = df.groupby("modality")[metric_cols].agg(["mean", "std", "median"])
    overall = df[metric_cols].agg(["mean", "std", "median"]).unstack().to_frame("ALL").T
    summary = pd.concat([summary, overall])
    summary.insert(0, "n_cases", list(df.groupby("modality").size()) + [len(df)])

    os.makedirs(args.out_dir, exist_ok=True)
    per_case_csv = os.path.join(args.out_dir, "dice_per_case.csv")
    summary_csv = os.path.join(args.out_dir, "dice_summary.csv")
    df.to_csv(per_case_csv, index=False, float_format="%.4f")
    summary.to_csv(summary_csv, float_format="%.4f")

    print("\nMean Dice per modality:")
    print(df.groupby("modality")[metric_cols].mean().round(4).to_string())
    print(f"\nSaved: {per_case_csv}\n       {summary_csv}")


if __name__ == "__main__":
    main()
