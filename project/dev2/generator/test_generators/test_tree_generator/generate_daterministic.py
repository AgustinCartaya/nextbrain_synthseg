"""
One synthetic image per subject, without any augmentation, with chosen
principal colors (e.g. a T1w contrast).

    conda activate nxb
    python generate_daterministic.py --device cuda:0

For every label map of --subjects the script writes, in --out:

    images/<case>.nii.gz   synthetic image (normalised to [0, 1])
    labels/<case>.nii.gz   segmentation on the grid of the image, with the
                           raw label values of the input
    plans/<case>.json      every parameter used (can be rendered again)

What is applied: the tree Gaussians (with their per-voxel sigma), Gaussian
acquisition noise inside the brain (--noise; the background stays 0, which
keeps the files small) and the normalisation. What is NOT applied: morphology, spatial deformation, low
resolution, bias field, motion. By default the image has
the voxel of the label maps and no partial volume (every voxel = its label);
--pv-sigma > 0 smooths the label boundaries, --voxel > label voxel averages
the labels over blocks (partial volume).

Intensities
-----------
--colors is a JSON file {color name: {"mu": .., "sigma": ..}} with one
entry per principal color of the tree (0-255 scale). A level-1 node of the
tree gets the mu/sigma of its color; the nodes below add the random
component of the tree (wr_mu, wr_sigma), scaled by --jitter:

    --jitter 0   every label has exactly the mean of its principal color
    --jitter 1   the tree model (structures of one tissue differ slightly)

The same --seed gives the same intensities for every subject (the noise
field differs per subject but is fixed by --seed too).
"""

import argparse
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
INPUTS = os.path.join(HERE, 'data', 'inputs')
OUTPUTS = os.path.join(HERE, 'data', 'outputs', 'deterministic')
GENERATOR = os.path.normpath(os.path.join(HERE, '..', '..', 'tree_generator'))

def build_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('--subjects', default=os.path.join(INPUTS, 'labels'),
        help='folder with one label map (.nii.gz) per subject')
    p.add_argument('--out', default=OUTPUTS)
    p.add_argument('--tree', default=os.path.join(
        INPUTS, 'nextbrain_tree_allen_gyral_simplified.json'))
    p.add_argument('--colors', default=os.path.join(INPUTS, 'colors_t1w.json'))
    p.add_argument('--generator', default=GENERATOR,
                   help='folder that holds nxgen/')
    p.add_argument('--voxel', type=float, default=0.4,
                   help='output voxel (mm), an integer multiple of the '
                        'voxel of the label maps (default: that voxel)')
    p.add_argument('--pv-sigma', type=float, default=0.3,
                   help='only when --voxel = voxel of the label maps: std '
                        '(voxels) of the partial-volume window; 0 = no '
                        'partial volume')
    p.add_argument('--contrast', default='T1',
                   help="contrast for the visibility of the tree nodes "
                        "(r_max); 'none' = default r_max")
    p.add_argument('--jitter', type=float, default=1.0,
                   help='scale of the random component of the tree nodes')
    p.add_argument('--noise', type=float, default=0.02,
                   help='std of the acquisition noise inside the brain, on '
                        'the image / 255 (WM ~ 0.78); added to the sigma of '
                        'every label: sqrt(sigma^2 + (255 noise)^2); 0 = none')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--dtype', default='uint8', choices=['uint8', 'uint16', 'float32'],
                   help='image file type; uint8/uint16 store [0, 1] with a scale factor '
                        '(uint8: -80%% size, error <= 0.002, well below the noise)')
    p.add_argument('--device', default='cuda:3')
    p.add_argument('--overwrite', action='store_true')
    return p


class ScaledNormal:
    """Random generator whose standard normal draws are scaled."""

    def __init__(self, rng, scale):
        self.rng, self.scale = rng, scale

    def standard_normal(self, size=None):
        return self.scale * self.rng.standard_normal(size)


def load_colors(path, tree):
    """(C, V): mean and std of every principal color, in the tree order."""
    with open(path) as f:
        colors = {k: v for k, v in json.load(f).items()
                  if not k.startswith('_')}
    missing = [c for c in tree.colors if c not in colors]
    unknown = [c for c in colors if c not in tree.colors]
    if missing or unknown:
        raise ValueError(f'{path}: missing colors {missing}, unknown colors '
                         f'{unknown}; the tree has {tree.colors}')
    C = np.array([colors[c]['mu'] for c in tree.colors], float)
    V = np.array([colors[c]['sigma'] for c in tree.colors], float)
    return C, V


def tree_params(gen, C, V, contrast, jitter, seed):
    """
    The 'tree' section of a plan (same content as
    `nxgen.online.tree_intensities.sample`) with given principal colors.
    """
    from nxgen.core.rng import module_rng, new_seed
    from nxgen.online.tree_intensities import node_stats

    cfg = gen.cfg['tree']
    rng = module_rng(seed, 'tree')
    res = gen.voxel_mm
    stats = node_stats(gen.tree, C, V, ScaledNormal(rng, jitter), res,
                       contrast, cfg['min_sigma'])
    rep = gen.tree.representative(res, contrast)
    l2n = gen.tree.label_to_node
    mu, sigma, class_of, class_names = [], [], [], []
    for value in gen.labels.values:
        node = l2n.get(value)
        name = rep[node] if node is not None else 'unassigned'
        m, s = stats[node] if node is not None else (0.0, 0.0)
        if name not in class_names:
            class_names.append(name)
        mu.append(m)
        sigma.append(s)
        class_of.append(class_names.index(name))
    return dict(colors_mu=C.tolist(), colors_sigma=V.tolist(),
                resolution_mm=res, contrast=contrast, mu=mu, sigma=sigma,
                class_of=class_of, class_names=class_names,
                intensity_scale=float(cfg['intensity_scale']),
                noise_seed=new_seed(rng))


def main(argv=None):
    args = build_parser().parse_args(argv)
    sys.path.insert(0, os.path.abspath(args.generator))
    import nibabel as nib
    from nxgen.core.config import load_config
    from nxgen.core.io import save_image, save_json, save_labels
    from nxgen.core.rng import image_seed, module_rng, new_seed
    from nxgen.pipeline import Generator

    paths = sorted(glob.glob(os.path.join(args.subjects, '*.nii*')))
    if not paths:
        raise SystemExit(f'no label map in {args.subjects}')
    hr = float(nib.load(paths[0]).header.get_zooms()[0])
    voxel = round(hr, 6) if args.voxel is None else args.voxel
    factor = int(round(voxel / hr))
    if factor > 1 or args.pv_sigma <= 0:     # factor 1: one label per voxel
        pv = dict(method='downsample')
    else:
        pv = dict(method='smooth', smooth_sigma_vox=args.pv_sigma)
    cfg = load_config(
        grid=dict(target_voxel_mm=voxel),
        labels=dict(tree=os.path.abspath(args.tree), unassigned='background'),
        partial_volume=pv, normalization=dict(method='minmax'))

    for sub in ('images', 'labels', 'plans'):
        os.makedirs(os.path.join(args.out, sub), exist_ok=True)
    gen = Generator(paths, cfg, device=args.device)
    contrast = None if args.contrast.lower() == 'none' else args.contrast
    C, V = load_colors(args.colors, gen.tree)
    tree = tree_params(gen, C, V, contrast, args.jitter, args.seed)
    print(f'{len(paths)} subjects, voxel {gen.voxel_mm} mm (labels '
          f'{gen.hr_voxel_mm} mm), {len(tree["class_names"])} generation '
          f'classes')
    for name, mu, sd in zip(gen.tree.colors, C, V):
        print(f'    {name:18s} mu {mu:6.1f}  sigma {sd:4.1f}')

    for i, path in enumerate(paths):
        case = os.path.basename(path).split('.nii')[0]
        files = [os.path.join(args.out, 'images', f'{case}.nii.gz'),
                 os.path.join(args.out, 'labels', f'{case}.nii.gz'),
                 os.path.join(args.out, 'plans', f'{case}.json')]
        if not args.overwrite and all(os.path.exists(f) for f in files):
            print(f'{case}: exists, skipped')
            continue
        # full volume of subject i; every augmentation is removed
        plan = gen.sample_plan(args.seed, template=i, start=[0, 0, 0])
        for key in ('morphology', 'spatial', 'acquisition', 'bias', 'motion'):
            plan[key] = None
        plan['noise'] = None
        # acquisition noise: an independent Gaussian of std 255 * noise per
        # voxel, folded into the per-voxel sigma of every brain label
        # (background, mu = 0, untouched); own noise field per subject
        n = 255.0 * args.noise
        plan['tree'] = dict(
            tree, acquisition_noise=args.noise,
            sigma=[float(np.hypot(sd, n)) if mu > 0 else sd
                   for mu, sd in zip(tree['mu'], tree['sigma'])],
            noise_seed=new_seed(module_rng(image_seed(args.seed, i), 'noise')))
        s = gen.render(plan)
        save_image(s.image, s.affine, files[0], args.dtype)
        save_labels(gen.labels.to_raw(s.seg), s.affine, files[1])
        save_json(dict(case=case, source=os.path.abspath(path), plan=s.plan,
                       normalization=s.normalization), files[2])
        print(f'{case}: {list(s.seg.shape)} voxels -> {files[0]}')


if __name__ == '__main__':
    main()
