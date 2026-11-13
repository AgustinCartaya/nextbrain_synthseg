"""Compute per-label Dice between nnU-Net predictions and NextBrain ground truth.

For every prediction:
    1. Find its NextBrain GT. The relative path is the same, but the GT is a
       folder named after the case that always holds `seg_registered.nii.gz`:
           pred: <PRED_DIR>/T1W/3T/R_T1W_3T_0329.nii.gz
           GT:   <GT_DIR>/T1W/3T/R_T1W_3T_0329/seg_registered.nii.gz
    2. Merge the GT (NextBrain / Allen ids) into the L1 tissue classes with
       merge_labels.py, using the same ids as the nnU-Net (dataset.json).
    3. Compute the Dice of every label 1-8. `mean_dice` excludes:
         - "other" (8): absent from the SynthSeg GT and nearly empty here.
         - "csf_ventricles" (2): NextBrain labels the ventricles as background
           (only a few septum pellucidum voxels are CSF), so its Dice is ~0
           and meaningless. It is still reported in its own column.

Outputs (in OUT_DIR):
    dice_per_case.csv   one row per case: modality, case, dice per label, mean
    dice_summary.csv    mean / std / median of every label, per modality + ALL

Usage:
    python extra_04_compute_dice_nextbrain.py
    python extra_04_compute_dice_nextbrain.py --workers 16
"""
import argparse
import glob
import os
import sys
from multiprocessing import Pool

import nibabel as nib
import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
PRED_DIR = os.path.join(HERE, "data/outputs/mrixfields")
GT_DIR = "/home/agustin/phd/synthseg_nextbrain/project/dev/testing/segment_mrixfields/data/outputs"
OUT_DIR = os.path.join(PRED_DIR, "results/nextbrain_gt")
GT_NAME = "seg_registered.nii.gz"

ONTOLOGY_CSV = "/home/agustin/phd/synthseg_nextbrain/data/basics/nextbrain_ontology_v1.csv"
MERGE_DIR = "/home/agustin/phd/synthseg_nextbrain/project/dev/wp1/create_merged_segmentations"
sys.path.insert(0, MERGE_DIR)
import merge_labels as ml  # noqa: E402

# --------------------------------------------------------------------------
# Labels: NextBrain L1 name -> nnU-Net id (same ids as dataset.json)
# --------------------------------------------------------------------------
MAPPING = {
    "Background": 0,
    "Brainstem": 1,
    "CSF / ventricles": 2,
    "Cerebellar GM": 3,
    "Cerebellar WM": 4,
    "Cerebral GM": 5,
    "Cerebral WM": 6,
    "Deep GM": 7,
    "Other": 8,
}
# Column names in the CSV (same as the SynthSeg script, so extra_05 works)
LABELS = {
    1: "brainstem",
    2: "csf_ventricles",
    3: "cerebellar_gm",
    4: "cerebellar_wm",
    5: "cerebral_gm",
    6: "cerebral_wm",
    7: "deep_gm",
    8: "other",
}
MEAN_LABELS = [n for n in LABELS.values() if n not in ("other", "csf_ventricles")]

# Built once at import time (also available in the worker processes)
ONTOLOGY = ml.load_ontology(ONTOLOGY_CSV)
LABEL_MAP = ml.build_label_map(ONTOLOGY, "L1", MAPPING)


# --------------------------------------------------------------------------
# File pairing
# --------------------------------------------------------------------------
def find_pairs(pred_dir, gt_dir):
    """Return a list of (modality, case, pred_path, gt_path)."""
    pairs = []
    for pred in sorted(glob.glob(os.path.join(pred_dir, "*", "*", "*.nii.gz"))):
        rel = os.path.relpath(pred, pred_dir)            # T1W/3T/R_T1W_3T_0329.nii.gz
        modality = rel.split(os.sep)[0]                   # T1W
        case = os.path.basename(pred)[: -len(".nii.gz")]  # R_T1W_3T_0329
        gt = os.path.join(gt_dir, os.path.dirname(rel), case, GT_NAME)
        if os.path.exists(gt):
            pairs.append((modality, case, pred, gt))
        else:
            print(f"[skip] {modality}/{case}: no GT at {gt}")
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
    """Load one pair, merge the GT to L1 and compute the Dice of every label."""
    modality, case, pred_path, gt_path = pair
    pred_img, gt_img = nib.load(pred_path), nib.load(gt_path)

    pred = np.rint(pred_img.get_fdata()).astype(np.uint8)
    gt, unknown = ml.merge_segmentation(np.rint(gt_img.get_fdata()), ONTOLOGY, LABEL_MAP)
    if unknown:
        print(f"[warn] {modality}/{case}: GT labels not in the ontology (kept as-is): {unknown}")

    if pred.shape != gt.shape:
        print(f"[skip] {modality}/{case}: shape mismatch {pred.shape} vs {gt.shape}")
        return None
    if not np.allclose(pred_img.affine, gt_img.affine, atol=1e-3):
        print(f"[warn] {modality}/{case}: affines differ")

    row = {"modality": modality, "case": case}
    for label, name in LABELS.items():
        row[name] = dice(pred, gt, label)
    row["mean_dice"] = np.nanmean([row[n] for n in MEAN_LABELS])
    print(f"{modality}/{case}: mean dice = {row['mean_dice']:.4f}")
    return row


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pred_dir", default=PRED_DIR, help="nnU-Net predictions (<modality>/<field>/<case>.nii.gz)")
    p.add_argument("--gt_dir", default=GT_DIR, help="NextBrain outputs (<modality>/<field>/<case>/seg_registered.nii.gz)")
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
