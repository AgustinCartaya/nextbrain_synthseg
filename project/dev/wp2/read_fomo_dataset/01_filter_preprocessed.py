"""Filter filtered_fomo.csv to the images that have a NextBrain segmentation.

Each derivative segmentation
    <derivatives>/<dataset>/<participant_id>/<session_id>/anat/<stem>dseg.nii.gz
is matched to the original image whose filename (from `abs_path`) is <stem>.nii.gz,
in the same dataset/participant/session. This works at image level, so sessions with
several runs (or several files with the same contrast) keep only the processed image.

Output: the input CSV columns plus `nextbrain_dseg` (segmentation path), one row per
processed image. `abs_path` is the original image path.
"""

import argparse
import re
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
DEFAULT_INPUT = HERE / "data/inputs/filtered_fomo.csv"
DEFAULT_DERIV = Path("/home/Data/FOMO-300K/derivatives/nextbrain")
DEFAULT_OUTPUT = HERE / "data/outputs/filtered_fomo_preprocessed.csv"

MATCH_KEYS = ["dataset", "participant_id", "session_id", "src_file"]
DSEG_RE = re.compile(r"_(T1w|T2w|FLAIR)dseg\.nii\.gz$")


def collect_derivatives(deriv_root: Path) -> pd.DataFrame:
    rows = []
    for f in deriv_root.glob("*/sub-*/ses-*/anat/*dseg.nii.gz"):
        if not DSEG_RE.search(f.name):
            continue
        dataset, participant, session = f.parts[-5:-2]
        rows.append({
            "dataset": dataset,
            "participant_id": participant,
            "session_id": session,
            "src_file": f.name.replace("dseg.nii.gz", ".nii.gz"),
            "nextbrain_dseg": str(f),
        })
    return pd.DataFrame(rows, columns=MATCH_KEYS + ["nextbrain_dseg"])


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    p.add_argument("--derivatives", type=Path, default=DEFAULT_DERIV)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = p.parse_args()

    df = pd.read_csv(args.input)
    df["src_file"] = df["abs_path"].map(lambda s: Path(s).name)
    deriv = collect_derivatives(args.derivatives)

    out = df.merge(deriv, on=MATCH_KEYS, how="inner", validate="one_to_one")
    out = out.drop(columns="src_file")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)

    print(f"Input rows:                {len(df)}")
    print(f"Derivative segmentations:  {len(deriv)}")
    print(f"Output rows:               {len(out)} -> {args.output}")
    print(out["contrast"].value_counts().to_string())
    n_unmatched = len(deriv) - len(out)
    if n_unmatched:
        print(f"Note: {n_unmatched} segmentations have no matching image in the input CSV")


if __name__ == "__main__":
    main()
