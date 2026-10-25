"""
Figure of every step of one or more samples (for visual inspection).

    python examples/plot_steps.py --templates data/templates \
        --config configs/nextbrain.json --device cuda:0 \
        --seeds 0 1 2 --out data/outputs/steps.png [--worst-case]

One row per seed, one column per recorded step, middle slice of each
volume (the slice of the axis given by --axis). Label maps are shown with
a random colour per label. `--worst-case` forces motion and a degraded
acquisition, to see every artefact.
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nxgen.cli.generate import template_paths  # noqa: E402
from nxgen.core.config import load_config  # noqa: E402
from nxgen.core.recorder import Recorder  # noqa: E402
from nxgen.online.acquisition import describe  # noqa: E402
from nxgen.pipeline import Generator  # noqa: E402

STEPS = ['morphology', 'spatial', 'pv_seg', 'pv_max', 'synthesis', 'bias',
         'motion', 'lowres', 'resolution', 'image']
LABEL_STEPS = ('morphology', 'spatial', 'pv_seg')


def middle_slice(vol, axis):
    """Middle slice of a 3D array along `axis`, transposed for display."""
    vol = np.asarray(vol)
    return np.take(vol, vol.shape[axis] // 2, axis=axis).T


def main():
    p = argparse.ArgumentParser(description='Plot the steps of samples.')
    p.add_argument('--templates', nargs='+', required=True)
    p.add_argument('--config', required=True)
    p.add_argument('--device', default='cuda')
    p.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    p.add_argument('--axis', type=int, default=2)
    p.add_argument('--out', default='steps.png')
    p.add_argument('--worst-case', action='store_true')
    args = p.parse_args()

    over = {}
    if args.worst_case:
        over = dict(motion={'prob': 1.0}, acquisition={'lowres_prob': 1.0})
    gen = Generator(template_paths(args.templates),
                    load_config(args.config, **over), device=args.device)
    colors = np.random.default_rng(0).random((len(gen.labels), 3))
    colors[0] = 0

    fig, axes = plt.subplots(len(args.seeds), len(STEPS),
                             figsize=(2.2 * len(STEPS),
                                      2.6 * len(args.seeds)),
                             squeeze=False)
    for row, seed in enumerate(args.seeds):
        rec = Recorder(steps=STEPS)
        s = gen(seed, recorder=rec)
        acq = s.plan['acquisition']
        spacing = [round(x, 1) for x in acq['spacing_mm']] if acq else '-'
        for col, name in enumerate(STEPS):
            ax = axes[row, col]
            ax.axis('off')
            if name not in rec.steps:
                ax.set_title(f'{name}\n(not applied)', fontsize=8)
                continue
            sl = middle_slice(rec.steps[name], args.axis)
            if name in LABEL_STEPS:
                ax.imshow(colors[sl], origin='lower', interpolation='nearest')
            else:
                ax.imshow(sl, cmap='gray', origin='lower',
                          interpolation='nearest')
            ax.set_title(name, fontsize=8)
        axes[row, 0].text(
            -0.1, 0.5, f'seed {seed}\n{describe(acq, gen.voxel_mm)[0]} '
            f'{spacing}\nmotion '
            f'{s.plan["motion"] is not None}', transform=axes[row, 0].transAxes,
            ha='right', va='center', fontsize=7)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out, dpi=110)
    print(f'saved {args.out}')


if __name__ == '__main__':
    main()
