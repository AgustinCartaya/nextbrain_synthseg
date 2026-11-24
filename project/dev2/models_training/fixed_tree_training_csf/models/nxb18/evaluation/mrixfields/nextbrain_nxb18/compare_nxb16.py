"""nxb18 vs nxb16 on MRIxFields: paired comparison of the 16 shared structures against NextBrain.

Reads results/per_case.csv of this folder (evaluate.py) and of
fixed_tree_training/models/nxb16/evaluation/mrixfields/nextbrain_nxb16/results/ (read-only),
pairs them per case and structure and writes (results/):

    compare_nxb16_per_case.csv          case x structure: dice and volume of both nets, differences
    compare_nxb16_contrast_field.csv    contrast x field x structure (+ 'mean' = mean of the 16):
                                        mean Dice of each net, mean/std of the paired difference
                                        (nxb18 - nxb16), % of cases improved, Wilcoxon p
    compare_nxb16_contrast_field_group.csv   same, split by subject group (P / R)

    python compare_nxb16.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

HERE = Path(__file__).resolve().parent
DEV2 = HERE.parents[6]                          # ./mrixfields/evaluation/nxb18/models/fixed_tree_training_csf/models_training -> dev2
NXB16 = (DEV2 / "models_training/fixed_tree_training/models/nxb16/evaluation/mrixfields/"
         "nextbrain_nxb16/results/per_case.csv")
KEYS = ["contrast", "field", "case", "subject", "label", "structure"]


def paired(df, keys):
    def agg(g):
        d = g.dice_diff.dropna()
        p = wilcoxon(d).pvalue if len(d) > 5 and (d != 0).any() else np.nan
        return pd.Series(dict(n_cases=g.case.nunique(), dice_nxb16=g.dice_nxb16.mean(),
                              dice_nxb18=g.dice_nxb18.mean(), dice_diff_mean=d.mean(),
                              dice_diff_std=d.std(), pct_cases_improved=100 * (d > 0).mean(),
                              abs_rel_vol_err_nxb16=g.abs_rel_vol_diff_pct_nxb16.median(),
                              abs_rel_vol_err_nxb18=g.abs_rel_vol_diff_pct_nxb18.median(),
                              wilcoxon_p=p))
    case_keys = list(dict.fromkeys(keys + ["contrast", "field", "case"]))
    mean = (df.groupby(case_keys, as_index=False)
              [["dice_nxb16", "dice_nxb18", "dice_diff", "abs_rel_vol_diff_pct_nxb16",
                "abs_rel_vol_diff_pct_nxb18"]].mean().assign(label=0, structure="mean"))
    full = pd.concat([df, mean], ignore_index=True)
    out = full.groupby(keys + ["label", "structure"]).apply(agg, include_groups=False).reset_index()
    out["_o"] = out.label.replace(0, 99)
    return out.sort_values(keys + ["_o"]).drop(columns="_o").reset_index(drop=True)


def main():
    cols = KEYS + ["dice", "vol_pred_ml", "abs_rel_vol_diff_pct"]
    a = pd.read_csv(NXB16)[cols]
    b = pd.read_csv(HERE / "results" / "per_case.csv")[cols]
    df = a.merge(b, on=KEYS, suffixes=("_nxb16", "_nxb18"))
    print(f"paired rows: {len(df)} ({df.case.nunique()} cases); nxb16 only: "
          f"{len(a) - len(df)}, nxb18 only: {len(b) - len(df)}")
    df["dice_diff"] = df.dice_nxb18 - df.dice_nxb16
    df["group"] = df.subject.str[4]
    out = HERE / "results"
    df.to_csv(out / "compare_nxb16_per_case.csv", index=False)
    s = paired(df, ["contrast", "field"])
    s.to_csv(out / "compare_nxb16_contrast_field.csv", index=False)
    paired(df, ["contrast", "field", "group"]).to_csv(out / "compare_nxb16_contrast_field_group.csv", index=False)
    print("\nmean Dice of the 16 structures, nxb16 -> nxb18 (paired difference):")
    for r in s[s.label == 0].itertuples():
        print(f"  {r.contrast:6s} {r.field:4s} n={int(r.n_cases):4d}  {r.dice_nxb16:.3f} -> {r.dice_nxb18:.3f}  "
              f"({r.dice_diff_mean:+.3f} +- {r.dice_diff_std:.3f}, {r.pct_cases_improved:.0f}% improved, "
              f"p={r.wilcoxon_p:.1e})")


if __name__ == "__main__":
    main()
