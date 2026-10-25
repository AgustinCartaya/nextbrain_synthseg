"""
Example of the v1 tree generator, with every generation parameter
adjustable from the command line.

Inputs (defaults, in data/inputs/ and configs/):
    seg.nii.gz                 label map used as template
    T1w.nii.gz                 matching real T1w (only for the figure)
    synthseg_ctx_tree.json     label tree (intensity model)
    configs/freesurfer.json    base config (morphology rules, contrasts)

Outputs (--out, default data/outputs/):
    templates/<seg>_hr0.25mm.nii.gz   super-resolved template (cached, uint16)
    synth_XXXX_img.nii.gz             synthetic images, 0.75 mm, [0, 1]
    synth_XXXX_seg.nii.gz             matching labels (raw values, uint16)
    steps.png                         the generator step by step
    samples.png                       real T1w vs synthetic samples
    config_used.json                  the final config

Parameters are applied in this order (later wins):
    1. TreeGeneratorConfig defaults (see `--print-config`)
    2. --config JSON
    3. one flag per config field, e.g.
         --lowres-prob 1 --lowres-max-aniso 6 --noise-space highres
         --noise-sigma 0.02 0.05            (ranges: two values)
         --contrasts T1 T2                  (lists: several values)
         --resolution null                  (None)
         --morpho-pairs '[["cerebral_wm", "cortex"]]'   (JSON)

Usage:
    python run_example.py [-n 4] [--device cuda] [--seed 0] [field flags]
    python run_example.py --print-config
    python run_example.py --help          lists every field with its meaning
"""

import argparse
import json
import os
from dataclasses import fields

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import nibabel as nib  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import tree_generator as tg  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
INPUTS = os.path.join(HERE, 'data', 'inputs')


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------

def _parse_value(tokens):
    """
    Command-line tokens of one config field -> python value.

    Each token is read as JSON when possible (numbers, true/false, null,
    lists), otherwise kept as a string. One token -> that value; several
    tokens -> a tuple (ranges, lists of contrasts, ...).
    """
    def one(tok):
        try:
            return json.loads(tok)
        except json.JSONDecodeError:
            return tok
    values = [one(t) for t in tokens]
    return values[0] if len(values) == 1 else tuple(values)


def build_parser():
    """
    Argument parser: run options + one flag per TreeGeneratorConfig field
    (generated from the dataclass, so it never gets out of sync).
    """
    p = argparse.ArgumentParser(
        description=__doc__.split('\n')[1],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    run = p.add_argument_group('run options')
    run.add_argument('--seg', nargs='+',
                     default=[os.path.join(INPUTS, 'case-1.nii.gz')],
                     help='template label map(s)')
    run.add_argument('--t1', default=os.path.join(INPUTS, 'case-1_l1.nii.gz'),
                     help='real image shown next to the samples ("" = none)')
    run.add_argument('--tree',
                     default=os.path.join(INPUTS, 'nextbrain_complete_tree.json'),
                     help='label tree JSON')
    run.add_argument('--config',
                     default=os.path.join(INPUTS, 'nextbrain_configs.json'),
                     help='base config JSON ("" = dataclass defaults)')
    run.add_argument('--out', default=os.path.join(HERE, 'data', 'outputs'),
                     help='output directory')
    run.add_argument('-n', type=int, default=4, help='number of samples')
    run.add_argument('--device', default='cuda' if torch.cuda.is_available()
                     else 'cpu')
    run.add_argument('--seed', type=int, default=0)
    run.add_argument('--index-labels', action='store_true',
                     help='save label indices instead of raw label values')
    run.add_argument('--no-figures', action='store_true',
                     help='only save the volumes')
    run.add_argument('--print-config', action='store_true',
                     help='print the final config and exit')

    gen = p.add_argument_group(
        'generation parameters (TreeGeneratorConfig fields; default = '
        'value from --config, else the dataclass default)')
    for f in fields(tg.TreeGeneratorConfig):
        default = f.default
        gen.add_argument(
            '--' + f.name.replace('_', '-'), dest=f.name, nargs='+',
            default=None, metavar='V',
            help=f'{f.metadata.get("help", "")} [dataclass: {default!r}]')
    return p


def make_config(args):
    """
    Final config: dataclass defaults < --config JSON < field flags.

    Returns
    -------
    TreeGeneratorConfig
    """
    d = {}
    if args.config:
        with open(args.config) as fh:
            d = json.load(fh)
    for f in fields(tg.TreeGeneratorConfig):
        tokens = getattr(args, f.name)
        if tokens is not None:
            value = _parse_value(tokens)
            if f.name == 'contrasts' and isinstance(value, (str, tuple)):
                value = [value] if isinstance(value, str) else list(value)
            d[f.name] = value
    return tg.TreeGeneratorConfig.from_dict(d)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def world_to_index(affine, point):
    """World coordinates (mm) -> voxel index of `affine`'s grid."""
    return np.round(np.linalg.inv(affine) @ np.append(point, 1))[:3].astype(int)


def axial(vol, affine, point):
    """Axial slice of `vol` through world `point`, oriented for display."""
    vol = np.asarray(vol)
    k = int(np.clip(world_to_index(affine, point)[2], 0, vol.shape[-1] - 1))
    return np.rot90(vol[..., k])


def coronal(vol, affine, point):
    """Coronal slice of `vol` through world `point` (shows axial slices)."""
    vol = np.asarray(vol)
    j = int(np.clip(world_to_index(affine, point)[1], 0, vol.shape[1] - 1))
    return np.rot90(vol[:, j, :])


def show(ax, img, title, cmap='gray', **kw):
    """imshow without axes."""
    ax.imshow(img, cmap=cmap, interpolation='nearest', **kw)
    ax.set_title(title, fontsize=8)
    ax.axis('off')


def random_label_cmap(n, seed=0):
    """Random colormap for label indices (background black)."""
    rng = np.random.default_rng(seed)
    colors = rng.uniform(0.15, 1, (n, 3))
    colors[0] = 0
    return matplotlib.colors.ListedColormap(colors)


def acq_title(acq):
    """Short description of a simulated acquisition."""
    if acq['mode'] == 'none':
        return 'native 0.4 mm'
    if acq['mode'] == 'iso':
        return f'iso {acq["spacing_mm"][0]:.1f} mm'
    a = acq['axis']
    return (f'axis {a}: {acq["spacing_mm"][a]:.1f} mm '
            f'(thick {acq["thickness_mm"][a]:.1f})')


def steps_figure(gen, point, lkw, path):
    """
    One sample step by step. The acquisition is forced to 5 mm axial
    slices (axis 2) and motion is on, so every step is visible.
    """
    r = gen.cfg.target_voxel
    sp, th, sg = [r] * 3, [r] * 3, [0.] * 3
    # sp[2], th[2] = 5., 4.
    # sg[2] = tg.sigma_thick_vox(4., 1., r)
    # acq = dict(mode='aniso', axis=2, spacing_mm=sp, thickness_mm=th,
    #            alpha=[1.] * 3, sigma_vox=sg)
    
    # sp[2], th[2] = 1.5, 1.5
    sg[2] = tg.sigma_thick_vox(sp[2], 1., r)
    acq = dict(mode='aniso', axis=2, spacing_mm=sp, thickness_mm=th,
            alpha=[1.] * 3, sigma_vox=sg)
    img, seg, aff, info = gen(template_index=0, acquisition=acq, motion=True,
                              return_steps=True)
    st = info['steps']
    lab, aff_hr = gen.templates[0][0].cpu(), gen.templates[0][1]
    wm_index = gen.index.get(2, 1)   # left WM if present
    wm = info['group_of'][wm_index]
    panels = [
        (axial(lab, aff_hr, point), 'super-res (0.25 mm)', lkw),
        (axial(st['morphology'], aff_hr, point), '+ erosion/dilation', lkw),
        (axial(st['deformed'], aff_hr, point), '+ affine/elastic', lkw),
        (axial(st['pv'][wm], aff, point),
         f'PV: {info["class_names"][wm]}', dict(cmap='magma')),
        (axial(st['tree'][0], aff, point),
         f'tree @ {info["resolution"]:g} mm\n'
         f'({len(info["class_names"])} classes)', {}),
        (axial(st['bias'][0], aff, point), '+ bias field', {}),
        (axial(st['motion'][0], aff, point), '+ motion', {}),
        (coronal(st['motion'][0], aff, point), 'before low-res (coronal)', {}),
        (coronal(st['resolution'][0], aff, point),
         f'+ {acq_title(acq)}\n+ noise (coronal)', {}),
        (coronal(img[0].cpu(), aff, point), 'final, [0, 1] (coronal)', {}),
        (coronal(seg.cpu(), aff, point), 'target (0.4 mm)', lkw),
    ]
    fig, axes = plt.subplots(1, len(panels), figsize=(2.3 * len(panels), 3.4))
    for ax, (im, title, kw) in zip(axes, panels):
        show(ax, im, title, **kw)
    fig.tight_layout()
    # fig.savefig(path, dpi=150)
    fig.savefig(path, dpi=300)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = build_parser().parse_args()
    cfg = make_config(args)
    if args.print_config:
        print(json.dumps(cfg.to_dict(), indent=2))
        return
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, 'config_used.json'), 'w') as fh:
        json.dump(cfg.to_dict(), fh, indent=2)
    tg.seed_everything(args.seed)

    # --- offline: super-resolution (raw labels kept, cached) -------------
    templates = tg.build_templates(args.seg, args.tree, cfg,
                                   os.path.join(args.out, 'templates'),
                                   args.device)
    gen = tg.TreeGenerator(templates, args.tree, cfg, device=args.device)
    lab_hr, aff_hr = gen.templates[0]
    print(f'HR template: {tuple(lab_hr.shape)} @ {abs(aff_hr[0, 0]):g} mm, '
          f'{gen.nb_labels} labels, {len(gen.morpho_pairs)} morphology pairs')

    # display point: centre of the foreground of the first template
    centre = torch.nonzero(lab_hr > 0).float().mean(0).cpu().numpy()
    point = (aff_hr @ np.append(centre, 1))[:3]
    lkw = dict(cmap=random_label_cmap(gen.nb_labels), vmin=0,
               vmax=gen.nb_labels - 1)

    if not args.no_figures:
        steps_figure(gen, point, lkw, os.path.join(args.out, 'steps.png'))

    # return  # only the steps figure for now
    # --- samples -----------------------------------------------------------
    has_t1 = bool(args.t1) and not args.no_figures
    ncol = args.n + int(has_t1)
    if not args.no_figures:
        fig, axes = plt.subplots(3, ncol, figsize=(2.6 * ncol, 8),
                                 squeeze=False)
    if has_t1:
        t1 = nib.load(args.t1)
        show(axes[0, 0], axial(t1.get_fdata(), t1.affine, point), 'real T1w')
        show(axes[1, 0], coronal(t1.get_fdata(), t1.affine, point), '')
        axes[2, 0].axis('off')
    for i in range(args.n):
        img, seg, aff, info = gen()
        name = f'synth_{i:04d}'
        tg.save_sample(img, seg, aff,
                       os.path.join(args.out, f'{name}_img.nii.gz'),
                       os.path.join(args.out, f'{name}_seg.nii.gz'),
                       labels=None if args.index_labels else gen.labels)
        acq = info['acquisition']
        print(f'saved {name}: {tuple(seg.shape)}, contrast '
              f'{info["contrast"]}, {acq_title(acq)}, tree @ '
              f'{info["resolution"]:.2f} mm -> '
              f'{len(info["class_names"])} generation classes')
        if args.no_figures:
            continue
        c = i + int(has_t1)
        show(axes[0, c], axial(img[0].cpu(), aff, point),
             f'{name} ({info["contrast"]})\n{acq_title(acq)}')
        show(axes[1, c], coronal(img[0].cpu(), aff, point),
             f'{len(info["class_names"])} classes')
        show(axes[2, c], axial(seg.cpu(), aff, point), '', **lkw)
    if not args.no_figures:
        fig.tight_layout()
        # fig.savefig(os.path.join(args.out, 'samples.png'), dpi=150)
        fig.savefig(os.path.join(args.out, 'samples.png'), dpi=300)
        plt.close(fig)
    print(f'outputs in {args.out}')


if __name__ == '__main__':
    main()
