import sys, pandas as pd
from pathlib import Path
root = Path(sys.argv[1] if len(sys.argv) > 1 else "data/outputs/mrixfields/results")
NAMES = {"brainstem":"Brainstem","csf_ventricles":"CSF / ventricles","cerebellar_gm":"Cerebellar GM",
         "cerebellar_wm":"Cerebellar WM","cerebral_gm":"Cerebral GM","cerebral_wm":"Cerebral WM",
         "deep_gm":"Deep GM","mean_dice":"**Mean Dice**"}
f3 = lambda x: f"{x:.3f}"

def summary_tables(d):
    s = pd.read_csv(d/"dice_summary.csv", header=[0,1], index_col=0)
    mods = list(s.index)
    n = s[("n_cases", s.columns[0][1])] if False else s.iloc[:,0]
    out = []
    head = "| Structure | " + " | ".join(f"{m} (n={int(n[m])})" for m in mods) + " |"
    sep = "|---|" + "---:|"*len(mods)
    out += ["**Mean ± std**", "", head, sep]
    for k, name in NAMES.items():
        out.append(f"| {name} | " + " | ".join(f"{f3(s.loc[m,(k,'mean')])} ± {f3(s.loc[m,(k,'std')])}" for m in mods) + " |")
    out += ["", "**Median**", "", head, sep]
    for k, name in NAMES.items():
        out.append(f"| {name} | " + " | ".join(f3(s.loc[m,(k,'median')]) for m in mods) + " |")
    return out

def per_case(d):
    df = pd.read_csv(d/"dice_per_case.csv")
    cols = list(NAMES)
    out = []
    for mod, g in df.groupby("modality", sort=False):
        out += [f"<details><summary>{mod} ({len(g)} cases)</summary>", "",
                "| Case | " + " | ".join(NAMES[c] for c in cols) + " |",
                "|---|" + "---:|"*len(cols)]
        for _, r in g.iterrows():
            out.append(f"| {r['case']} | " + " | ".join(f3(r[c]) for c in cols) + " |")
        out += ["", "</details>", ""]
    return out

L = [f"# nnU-Net — MRIxFields Dice results ({root.name})", ""]
for title, d, note in [("Without CSF (default)", root, "`csf_ventricles` evaluated as ventricles only."),
                       ("With CSF", root/"with_csf", "`csf_ventricles` includes extra-cerebral CSF.")]:
    if not (d/"dice_summary.csv").exists():  # e.g. nextbrain_gt has no with_csf/
        continue
    L += [f"## {title}", "", f"Source: `{d.relative_to(root.parent)}/`. {note}", ""]
    L += summary_tables(d) + ["", f"### Per case — {title.lower()}", ""] + per_case(d)
(root/"results.md").write_text("\n".join(L) + "\n")
print(root/"results.md")
