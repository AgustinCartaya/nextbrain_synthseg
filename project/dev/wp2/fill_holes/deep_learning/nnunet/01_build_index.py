#!/usr/bin/env python
"""
Step 1 — index the FOMO scans and split them by subject.

For every row of paths.scans_csv:
  * read the voxel size of the original T1w (`input_file`) from its NIfTI header
    (header only, in parallel; the result for ALL scans is cached in
    paths.headers_csv, so re-running with another threshold is instantaneous)
  * check that seg.left.nii.gz / seg.right.nii.gz exist in `nextbrain_dir`
Then keep the scans with voxel size <= data.max_voxel_mm in all three axes,
assign each SUBJECT to train / val / test and write paths.index_csv.

Nothing is written outside this project.

  python 01_build_index.py --config configs/default.yaml
  python 01_build_index.py --set data.max_voxel_mm=0.8        # stricter filter
  python 01_build_index.py --config configs/smoke.yaml --limit 40
"""
import os
from concurrent.futures import ThreadPoolExecutor

import nibabel as nib
import numpy as np
import pandas as pd

from lib.config import config_from_args, parser
from lib.data import subject_split
from lib.labels import assert_writable


def read_voxel_size(path):
    """(vx, vy, vz) in mm from a NIfTI header, or NaNs if the file cannot be read."""
    try:
        return tuple(float(z) for z in nib.load(path).header.get_zooms()[:3])
    except Exception:                                        # noqa: BLE001
        return (np.nan, np.nan, np.nan)


def main():
    p = parser(__doc__)
    p.add_argument("--limit", type=int, default=None,
                   help="only index a random subset of N scans (smoke test)")
    p.add_argument("--threads", type=int, default=32, help="parallel header reads")
    args = p.parse_args()
    cfg = config_from_args(args)
    d, paths = cfg["data"], cfg["paths"]

    scans = pd.read_csv(paths["scans_csv"])
    if args.limit:
        scans = scans.sample(n=min(args.limit, len(scans)), random_state=0)
    scans = scans[["dataset", "subject", "session", "input_file", "nextbrain_dir"]].reset_index(drop=True)

    # ---- voxel sizes: reuse the cache, read only the missing headers ----------
    cached = {}
    if os.path.exists(paths["headers_csv"]):
        old = pd.read_csv(paths["headers_csv"])
        cached = {f: (x, y, z) for f, x, y, z in zip(old["input_file"], old["vx"], old["vy"], old["vz"])}
    todo = [f for f in scans["input_file"] if f not in cached]
    print(f"{len(scans)} scans, {len(todo)} headers to read")
    with ThreadPoolExecutor(args.threads) as pool:
        for i, (f, size) in enumerate(zip(todo, pool.map(read_voxel_size, todo))):
            cached[f] = size
            if (i + 1) % 2000 == 0:
                print(f"  {i + 1}/{len(todo)}", flush=True)
    scans[["vx", "vy", "vz"]] = [cached[f] for f in scans["input_file"]]
    scans["max_vox"] = scans[["vx", "vy", "vz"]].max(axis=1, skipna=False)
    for hemi in ("left", "right"):
        scans[f"has_{hemi}"] = [os.path.isfile(os.path.join(nd, f"seg.{hemi}.nii.gz"))
                                for nd in scans["nextbrain_dir"]]

    os.makedirs(os.path.dirname(assert_writable(paths["headers_csv"])), exist_ok=True)
    scans.to_csv(paths["headers_csv"], index=False)

    # ---- filter + subject split ------------------------------------------------
    keep = (scans["max_vox"] <= d["max_voxel_mm"] + d["voxel_tolerance"]) & \
           (scans["has_left"] | scans["has_right"])
    index = scans[keep].copy()
    index["split"] = [subject_split(ds, sub, d["split_seed"], d["val_fraction"], d["test_fraction"])
                      for ds, sub in zip(index["dataset"], index["subject"])]
    os.makedirs(os.path.dirname(assert_writable(paths["index_csv"])), exist_ok=True)
    index.to_csv(paths["index_csv"], index=False)

    # ---- summary -----------------------------------------------------------------
    print(f"unreadable T1w headers: {int(scans['max_vox'].isna().sum())}")
    print(f"kept {len(index)}/{len(scans)} scans with voxel size <= {d['max_voxel_mm']} mm "
          f"-> {paths['index_csv']}")
    for split, g in index.groupby("split"):
        print(f"  {split:5s}: {len(g):6d} scans, {g.groupby(['dataset', 'subject']).ngroups:6d} subjects, "
              f"{int(g['has_left'].sum() + g['has_right'].sum()):6d} hemispheres")
    bins = [0, 0.5, 0.7, 0.8, 0.9, 0.999, 10]
    print("largest voxel size of the kept scans (mm):")
    print(pd.cut(index["max_vox"], bins).value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
