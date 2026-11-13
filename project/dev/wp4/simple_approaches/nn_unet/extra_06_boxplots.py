"""Boxplots comparing the three Dice evaluations on MRIxFields.

    nnU-Net vs SynthSeg GT       results/synthseg_gt/dice_per_case.csv
    nnU-Net vs NextBrain GT      results/nextbrain_gt/dice_per_case.csv
    SynthSeg GT vs NextBrain GT  results/nextbrain_synthseg_gt/dice_per_case.csv

`csf_ventricles` is left out everywhere: NextBrain labels the ventricles as
background, so its Dice is ~0 in the two NextBrain comparisons. The mean Dice
is recomputed over the same 6 tissue labels for all three comparisons, so they
are comparable (the stored synthseg_gt mean also includes CSF).

Outputs (in results/boxplots/, PNG + PDF):
    01_mean_dice_per_modality      mean Dice, one group per modality + ALL
    02_structures_all_modalities   per-structure Dice, all modalities pooled
    03_structures_per_modality     per-structure Dice, one panel per modality
    04_csf_nnunet_vs_synthseg      CSF / ventricles (only meaningful vs SynthSeg)
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "data/outputs/mrixfields/results")
OUT = os.path.join(RES, "boxplots")

COMPARISONS = [  # (folder, label, color)
    ("synthseg_gt", "nnU-Net vs SynthSeg GT", "#2a78d6"),
    ("nextbrain_gt", "nnU-Net vs NextBrain GT", "#eb6834"),
    ("nextbrain_synthseg_gt", "SynthSeg GT vs NextBrain GT", "#1baf7a"),
]
STRUCTURES = {
    "brainstem": "Brainstem",
    "cerebellar_gm": "Cerebellar GM",
    "cerebellar_wm": "Cerebellar WM",
    "cerebral_gm": "Cerebral GM",
    "cerebral_wm": "Cerebral WM",
    "deep_gm": "Deep GM",
}
MODALITIES = ["T1W", "T2FLAIR", "T2W"]

plt.rcParams.update({
    "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#52514e", "axes.labelcolor": "#0b0b0b",
    "xtick.color": "#52514e", "ytick.color": "#52514e",
    "figure.facecolor": "white", "axes.facecolor": "white",
})


def load():
    dfs = {}
    for folder, label, _ in COMPARISONS:
        df = pd.read_csv(os.path.join(RES, folder, "dice_per_case.csv"))
        df["mean6"] = df[list(STRUCTURES)].mean(axis=1)
        dfs[label] = df
    return dfs


def grouped_boxplot(ax, groups, data, ylabel="Dice"):
    """data[label][group] -> array. One cluster per group, one box per comparison."""
    n = len(COMPARISONS)
    width = 0.8 / n
    for i, (_, label, color) in enumerate(COMPARISONS):
        pos = np.arange(len(groups)) + (i - (n - 1) / 2) * width
        vals = [data[label][g] for g in groups]
        bp = ax.boxplot(
            vals, positions=pos, widths=width * 0.85, patch_artist=True,
            showfliers=True, medianprops=dict(color="#0b0b0b", lw=1.5),
            whiskerprops=dict(color=color, lw=1.2), capprops=dict(color=color, lw=1.2),
            flierprops=dict(marker="o", ms=3, mfc=color, mec="white", mew=0.5, alpha=0.7),
            boxprops=dict(edgecolor=color, lw=1.2),
        )
        for b in bp["boxes"]:
            b.set_facecolor(color + "59")  # ~35% alpha fill
        for p, v in zip(pos, vals):  # mean marker
            ax.scatter(p, np.mean(v), marker="D", s=14, color="white",
                       edgecolor="#0b0b0b", lw=0.8, zorder=5)
    ax.set_xticks(np.arange(len(groups)))
    ax.set_xticklabels(groups)
    ax.set_xlim(-0.5, len(groups) - 0.5)
    ax.set_ylabel(ylabel)
    ax.yaxis.grid(True, color="#e5e4e0", lw=0.8)
    ax.set_axisbelow(True)
    for x in np.arange(len(groups) - 1) + 0.5:
        ax.axvline(x, color="#e5e4e0", lw=0.8)


def legend(fig, extra_note=True):
    handles = [Patch(facecolor=c + "59", edgecolor=c, label=l) for _, l, c in COMPARISONS]
    if extra_note:
        handles.append(plt.Line2D([], [], marker="D", ls="", mfc="white",
                                  mec="#0b0b0b", label="mean (line = median)"))
    fig.legend(handles=handles, loc="upper center", ncol=len(handles),
               frameon=False, bbox_to_anchor=(0.5, 1.0))


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("saved", name)


def main():
    os.makedirs(OUT, exist_ok=True)
    dfs = load()

    # 1. Mean Dice per modality
    groups = MODALITIES + ["ALL"]
    data = {l: {**{m: df.loc[df.modality == m, "mean6"].values for m in MODALITIES},
                "ALL": df["mean6"].values} for l, df in dfs.items()}
    fig, ax = plt.subplots(figsize=(9, 4.8))
    grouped_boxplot(ax, groups, data, "Mean Dice (6 tissue labels)")
    ax.set_title("Mean Dice per modality", pad=8, loc="left", fontweight="bold")
    legend(fig)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save(fig, "01_mean_dice_per_modality")

    # 2. Per structure, all modalities pooled
    names = list(STRUCTURES.values()) + ["Mean"]
    cols = list(STRUCTURES) + ["mean6"]
    data = {l: {n: df[c].values for n, c in zip(names, cols)} for l, df in dfs.items()}
    fig, ax = plt.subplots(figsize=(13, 5))
    grouped_boxplot(ax, names, data)
    ax.set_title("Dice per structure — all modalities (n=428)", pad=8,
                 loc="left", fontweight="bold")
    legend(fig)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save(fig, "02_structures_all_modalities")

    # 3. Per structure, one panel per modality
    fig, axes = plt.subplots(3, 1, figsize=(13, 13), sharey=True)
    for ax, m in zip(axes, MODALITIES):
        data = {l: {n: df.loc[df.modality == m, c].values for n, c in zip(names, cols)}
                for l, df in dfs.items()}
        grouped_boxplot(ax, names, data)
        n = (dfs[COMPARISONS[0][1]].modality == m).sum()
        ax.set_title(f"{m} (n={n})", loc="left", fontweight="bold")
    legend(fig)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    save(fig, "03_structures_per_modality")

    # 4. CSF / ventricles: only nnU-Net vs SynthSeg is meaningful
    ss = dfs[COMPARISONS[0][1]]
    color = COMPARISONS[0][2]
    vals = [ss.loc[ss.modality == m, "csf_ventricles"].values for m in MODALITIES]
    vals.append(ss["csf_ventricles"].values)
    fig, ax = plt.subplots(figsize=(6, 4.2))
    bp = ax.boxplot(vals, patch_artist=True, widths=0.5,
                    medianprops=dict(color="#0b0b0b", lw=1.5),
                    whiskerprops=dict(color=color), capprops=dict(color=color),
                    boxprops=dict(edgecolor=color),
                    flierprops=dict(marker="o", ms=3, mfc=color, mec="white", mew=0.5))
    for b in bp["boxes"]:
        b.set_facecolor(color + "59")
    ax.set_xticks(range(1, 5))
    ax.set_xticklabels(MODALITIES + ["ALL"])
    ax.set_ylabel("Dice")
    ax.yaxis.grid(True, color="#e5e4e0", lw=0.8)
    ax.set_axisbelow(True)
    ax.set_title("CSF / ventricles — nnU-Net vs SynthSeg GT", loc="left",
                 fontweight="bold")
    fig.tight_layout()
    save(fig, "04_csf_nnunet_vs_synthseg")


if __name__ == "__main__":
    main()
