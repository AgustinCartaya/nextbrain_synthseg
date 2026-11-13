"""Compute per-label Dice between the SynthSeg and NextBrain ground truths.

This measures how much the two reference segmentations agree with each other
(an "inter-GT" agreement), independent of the nnU-Net predictions.

For every NextBrain GT case:
    1. NextBrain GT:  <NB_DIR>/<modality>/3T/R_T1W_3T_0329/seg_registered.nii.gz
       merged into the L1 tissue classes with merge_labels.py (Allen ids -> 0-8).
       Label 8 ("other") is set to background: it has no SynthSeg counterpart.
    2. SynthSeg GT:   <SS_DIR>/<modality>/3.0T/SR_T1W_3T_0329_synthseg_resampled.nii.gz
       merged into L1 (ASeg ids -> 0-7) with merge_synthseg_labels.merge_array.
       Matched when its name, without '_synthseg_resampled', ends with the case.
    3. Compute the Dice of every label 1-7 (Dice is symmetric, so the choice of
       reference does not matter). `mean_dice` excludes "csf_ventricles" (2):
       NextBrain labels the ventricles as background, so its Dice is ~0 and
       meaningless. It is still reported in its own column.

Outputs (in OUT_DIR):
    dice_per_case.csv   one row per case: modality, case, dice per label, mean
    dice_summary.csv    mean / std / median of every label, per modality + ALL

Usage:
    python extra_04_compute_dice_synthseg_nextbrain_gt.py
    python extra_04_compute_dice_synthseg_nextbrain_gt.py --workers 16
"""
import argparse
import glob
import importlib.util
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
NB_DIR = "/home/agustin/phd/synthseg_nextbrain/project/dev/testing/segment_mrixfields/data/outputs"
SS_DIR = "/home/agustin/phd/miccai/miccai_2026/mri_x_fields/data/train_data/preprocessed/synthseg"
OUT_DIR = os.path.join(HERE, "data/outputs/mrixfields/results/nextbrain_synthseg_gt")
NB_NAME = "seg_registered.nii.gz"
SS_SUFFIX = "_synthseg_resampled"

ONTOLOGY_CSV = "/home/agustin/phd/synthseg_nextbrain/data/basics/nextbrain_ontology_v1.csv"
MERGE_DIR = "/home/agustin/phd/synthseg_nextbrain/project/dev/wp1/create_merged_segmentations"
SS_MERGE_SCRIPT = os.path.join(MERGE_DIR, "merge_synthseg_seg/merge_synthseg_labels.py")

sys.path.insert(0, MERGE_DIR)
import merge_labels as ml  # noqa: E402


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ss_merge = load_module(SS_MERGE_SCRIPT, "merge_synthseg_labels")

# --------------------------------------------------------------------------
# Labels: NextBrain L1 name -> id (same ids as SynthSeg L1 / nnU-Net)
# --------------------------------------------------------------------------
NB_MAPPING = {
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
OTHER_LABEL = 8
# {id: column name} for labels 1..7 (names from merge_synthseg_labels.L1_LABELS)
LABELS = {v: k for k, v in ss_merge.L1_LABELS.items() if v != 0}
MEAN_LABELS = [n for n in LABELS.values() if n != "csf_ventricles"]

ONTOLOGY = ml.load_ontology(ONTOLOGY_CSV)
NB_LABEL_MAP = ml.build_label_map(ONTOLOGY, "L1", NB_MAPPING)


# --------------------------------------------------------------------------
# File pairing
# --------------------------------------------------------------------------
def stem(path):
    name = os.path.basename(path)
    for ext in (".nii.gz", ".nii"):
        if name.endswith(ext):
            return name[: -len(ext)]
    return name


def find_pairs(nb_dir, ss_dir):
    """Return a list of (modality, case, nb_path, ss_path)."""
    pairs, ss_cache = [], {}
    for nb in sorted(glob.glob(os.path.join(nb_dir, "*", "*", "*", NB_NAME))):
        case_dir = os.path.dirname(nb)
        case = os.path.basename(case_dir)                               # R_T1W_3T_0329
        modality = os.path.relpath(case_dir, nb_dir).split(os.sep)[0]   # T1W

        if modality not in ss_cache:  # {stem without suffix: path}, e.g. SR_T1W_3T_0329
            files = glob.glob(os.path.join(ss_dir, modality, "**", f"*{SS_SUFFIX}.nii*"), recursive=True)
            ss_cache[modality] = {stem(f)[: -len(SS_SUFFIX)]: f for f in files}
        matches = [f for s, f in ss_cache[modality].items() if s.endswith(case)]

        if len(matches) == 1:
            pairs.append((modality, case, nb, matches[0]))
        elif len(matches) > 1:
            print(f"[skip] {modality}/{case}: ambiguous SynthSeg matches {matches}")
        else:
            print(f"[skip] {modality}/{case}: no SynthSeg GT found")
    return pairs


# --------------------------------------------------------------------------
# Dice
# --------------------------------------------------------------------------
def dice(a, b, label):
    """Dice of one label. NaN when the label is absent from both volumes."""
    p, g = a == label, b == label
    denom = p.sum() + g.sum()
    if denom == 0:
        return np.nan
    return 2.0 * np.logical_and(p, g).sum() / denom


def evaluate_pair(pair):
    modality, case, nb_path, ss_path = pair
    nb_img, ss_img = nib.load(nb_path), nib.load(ss_path)

    nb, unknown = ml.merge_segmentation(np.rint(nb_img.get_fdata()), ONTOLOGY, NB_LABEL_MAP)
    if unknown:
        print(f"[warn] {modality}/{case}: NextBrain labels not in the ontology (kept as-is): {unknown}")
    nb = np.asarray(nb).astype(np.uint8)
    nb[nb == OTHER_LABEL] = 0
    ss = ss_merge.merge_array(np.asanyarray(ss_img.dataobj))  # ASeg -> L1 (0-7)

    if nb.shape != ss.shape:
        print(f"[skip] {modality}/{case}: shape mismatch {nb.shape} vs {ss.shape}")
        return None
    if not np.allclose(nb_img.affine, ss_img.affine, atol=1e-3):
        print(f"[warn] {modality}/{case}: affines differ")

    row = {"modality": modality, "case": case}
    for label, name in LABELS.items():
        row[name] = dice(nb, ss, label)
    row["mean_dice"] = np.nanmean([row[n] for n in MEAN_LABELS])
    print(f"{modality}/{case}: mean dice = {row['mean_dice']:.4f}")
    return row


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--nb_dir", default=NB_DIR, help="NextBrain outputs (<modality>/<field>/<case>/seg_registered.nii.gz)")
    p.add_argument("--ss_dir", default=SS_DIR, help="SynthSeg GT (<modality>/<field>/*_synthseg_resampled.nii.gz)")
    p.add_argument("--out_dir", default=OUT_DIR, help="Where to write the CSV files")
    p.add_argument("--workers", type=int, default=8, help="Parallel processes")
    args = p.parse_args()

    pairs = find_pairs(args.nb_dir, args.ss_dir)
    print(f"Found {len(pairs)} NextBrain/SynthSeg pairs")
    if not pairs:
        return

    with Pool(args.workers) as pool:
        rows = [r for r in pool.map(evaluate_pair, pairs) if r is not None]

    df = pd.DataFrame(rows).sort_values(["modality", "case"])
    metric_cols = list(LABELS.values()) + ["mean_dice"]

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
