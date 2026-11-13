#!/usr/bin/env python
"""
Restore all real NextBrain reconstructions (full resolution) and write QC figures.

  python run_all.py                       # cases 1-5 -> results/restored/
  python run_all.py --cases 4 --workers 1

Outputs per case: case-N_restored.nii.gz (full label set), case-N_restored_steps.nii.gz
(1 gap, 2 crack, 3 hole, 4 shift), case-N_restored_report.json, QC png files.
"""
import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import nibabel as nib
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                       # manual/
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
import restore as R   # noqa: E402
import qc             # noqa: E402

TEST = os.path.join(os.path.dirname(ROOT), "data/inputs/recon_holes_0.2mm")


def one(case, out_dir):
    path = os.path.join(TEST, f"case-{case}.nii.gz")
    out_img, steps_img, rep = R.restore_nifti(path, verbose=True)
    stem = os.path.join(out_dir, f"case-{case}_restored")
    nib.save(out_img, stem + ".nii.gz")
    nib.save(steps_img, stem + "_steps.nii.gz")
    with open(stem + "_report.json", "w") as f:
        json.dump(rep, f, indent=1)

    seg = np.asarray(nib.load(path).dataobj)
    res = np.asarray(out_img.dataobj)
    steps = np.asarray(steps_img.dataobj)
    lut = R.load_l1_lut(int(max(seg.max(), res.max())))
    a, b = lut[seg], lut[res]
    fg = seg > 0
    at = [int(np.mean(np.nonzero(fg.any(axis=tuple(x for x in range(3) if x != ax))))) for ax in range(3)]
    qc.qc_figure(a, b, steps, stem + "_qc.png", at=at, title=f"case-{case} (0.2 mm)")
    at2 = list(at)
    at2[0] = at[0] - 40 if at[0] > 60 else at[0] + 40
    qc.qc_figure(a, b, steps, stem + "_qc_sagittal2.png", at=at2, views=("sagittal",),
                 title=f"case-{case}, another sagittal slice")
    # zoom on the centre of the sagittal slice
    K, J = seg.shape[2], seg.shape[1]
    crop = {"sagittal": (slice(K // 2 - 150, K // 2 + 150), slice(J // 2 - 200, J // 2 + 200))}
    qc.qc_figure(a, b, steps, stem + "_qc_zoom.png", at=at2, views=("sagittal",), crop=crop,
                 title=f"case-{case}, zoom 60 x 80 mm")
    summary = {k: rep.get(k) for k in ("gap_filled", "crack_filled", "hole_filled", "changed_voxels",
                                       "block_translations_vox", "seconds")}
    summary["n_planes"] = len(rep.get("planes", []))
    return case, summary


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cases", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    p.add_argument("--out", default=os.path.join(HERE, "results/restored"))
    p.add_argument("--workers", type=int, default=5)
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    with ProcessPoolExecutor(a.workers) as ex:
        for case, s in ex.map(one, a.cases, [a.out] * len(a.cases)):
            print(f"case-{case}: {json.dumps(s)}", flush=True)


if __name__ == "__main__":
    main()
