#!/usr/bin/env python
"""
Score the naive baseline (../baseline_naive.py) with the same synthetic corruption as
benchmark.py, and measure how many sulci it closes on the clean volume.

  python compare_naive.py <FOMO seg.left.nii.gz> [...] --closing-iter 3 6
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                       # manual/
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(os.path.dirname(ROOT), "image_corruption"))
import restore as R                 # noqa: E402
import benchmark as B               # noqa: E402
from baseline_naive import naive    # noqa: E402
from corrupt_nifti import corrupt_array   # noqa: E402


def run(source, ds=1, seed=1, closing_iter=3):
    gt, z, aff, name = B.load_reference(source, ds)
    lut = R.load_l1_lut(int(gt.max()))
    rng = np.random.default_rng(seed)
    x, gap, shifts, planes, bidx = B.synthetic_blocks(gt, z, rng)
    x2, _, _ = corrupt_array(x, aff, (-6, 6), "coronal", rng, thickness=0.4, p_section=0, p_block=0,
                             n_blobs=(15, 25), radius_mm=(1.0, 4.0), thin_prob=0.6, thin_sections=(1, 3),
                             p_slab=0, irregularity=0.5)
    holes = x2 != x
    x = x2
    corrupted = x != gt
    y, _ = naive(x, closing_iter)
    sul = B.sulcal_background(gt, lut)
    y0, _ = naive(gt, closing_iter)
    out = dict(name=name, closing_iter=closing_iter)
    for k, m in {"gaps": gap & corrupted, "holes_cracks": holes & corrupted}.items():
        out[k] = round(100 * (1 - (lut[y] != lut[gt])[m].sum() / max(m.sum(), 1)), 1)
    out["sulcal_filled_pct_clean"] = round(100 * ((y0 > 0) & sul).sum() / sul.sum(), 1)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("source", nargs="+", help="test case numbers (1-5) and/or paths to clean label maps")
    p.add_argument("--closing-iter", type=int, nargs="+", default=[3, 6])
    p.add_argument("--downsample", type=int, default=1)
    p.add_argument("--seed", type=int, default=1)
    a = p.parse_args()
    for s in a.source:
        for ci in a.closing_iter:
            print(json.dumps(run(s, a.downsample, a.seed, ci)), flush=True)


if __name__ == "__main__":
    main()
