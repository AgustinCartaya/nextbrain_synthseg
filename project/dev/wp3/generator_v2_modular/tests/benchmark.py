"""
Benchmark: time and peak GPU memory of every step, on real templates.

    python tests/benchmark.py --templates data/templates \
        --config configs/nextbrain.json --device cuda:0 [-n 5] \
        [--patch 160 160 160] [--worst-case]

Prints, per step, the median time and the largest peak memory over `n`
samples, then the median total time per sample. `--patch` overrides
patch.size (full volume by default). `--worst-case` forces motion and a
degraded acquisition in every sample (the slowest path).

Separate from the tests: needs real templates and a GPU, and takes minutes.
"""

import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nxgen.cli.generate import template_paths  # noqa: E402
from nxgen.core.config import load_config  # noqa: E402
from nxgen.core.recorder import Recorder  # noqa: E402
from nxgen.online.acquisition import describe  # noqa: E402
from nxgen.pipeline import Generator  # noqa: E402


def main():
    p = argparse.ArgumentParser(description='Benchmark the generator.')
    p.add_argument('--templates', nargs='+', required=True)
    p.add_argument('--config', required=True)
    p.add_argument('--device', default='cuda')
    p.add_argument('-n', type=int, default=5)
    p.add_argument('--patch', type=int, nargs=3, default=None)
    p.add_argument('--worst-case', action='store_true')
    args = p.parse_args()

    over = {}
    if args.patch:
        over['patch'] = {'size': args.patch}
    if args.worst_case:
        over['motion'] = {'prob': 1.0}
        over['acquisition'] = {'lowres_prob': 1.0}
    cfg = load_config(args.config, **over)
    gen = Generator(template_paths(args.templates), cfg, device=args.device)
    print(f'{len(gen.labels)} labels, factor {gen.factor}, output voxel '
          f'{gen.voxel_mm} mm, patch {cfg["patch"]["size"] or "full volume"}')

    gen(0)                                            # warm-up (CUDA init)
    timings, totals = {}, []
    for seed in range(1, args.n + 1):
        rec = Recorder(profile=True)
        torch.cuda.synchronize(args.device)
        t0 = time.perf_counter()
        s = gen(seed, recorder=rec)
        torch.cuda.synchronize(args.device)
        totals.append(time.perf_counter() - t0)
        for name, t in rec.timings.items():
            timings.setdefault(name, []).append(t)
        print(f'  seed {seed}: {totals[-1]:.3f} s, shape {list(s.seg.shape)},'
              f' motion {s.plan["motion"] is not None}, acquisition '
              f'{describe(s.plan["acquisition"], gen.voxel_mm)[0]}')

    print(f'\n{"step":16s}{"median s":>10s}{"max peak GB":>13s}')
    for name, ts in timings.items():
        peak = max(t['peak_gb'] or 0 for t in ts)
        print(f'{name:16s}{np.median([t["time_s"] for t in ts]):10.3f}'
              f'{peak:13.2f}')
    print(f'{"total":16s}{np.median(totals):10.3f}   (without profiling '
          f'syncs it is slightly faster)')


if __name__ == '__main__':
    main()
