"""
Test of the tree generator (nxgen) with the inputs of data/inputs.

    conda activate nxb
    python generate.py                      # 5 images, seed 0, cuda:0
    python generate.py -n 20 --seed 3 --device cuda:2
    python generate.py --config data/inputs/other_config.json
    python generate.py --templates data/outputs/templates   # reuse built templates, skip step 1

Steps
-----
1. Offline (cached): every label map of data/inputs/templates/ -> HR template in
   data/outputs/templates/. Rebuilt only if the map or the 'template' / 'labels'
   settings change.
2. Online: n images, image i with seed (seed, i), cycling over the templates.

Outputs (--out, default data/outputs/random/)
---------------------------------------------
    images/synth_<i>.nii.gz  synthetic image in [0, 1] (uint8 + scale factor by default, --dtype)
    labels/synth_<i>.nii.gz  segmentation on the grid of the image, raw label values (as the tree)
    plans/synth_<i>.json     every random parameter of the image (to replay it)
    label_space.json         index <-> raw value and name

Configuration: data/inputs/config.json holds every parameter. A config only needs the values
that differ from the defaults; every key is documented in
../../tree_generator/nxgen/core/config.py (DEFAULTS) and ../../tree_generator/docs/config.md.
The tree is read with `load_tree`, so its class (plain or Allen) comes from its "tree_type".
"""

import argparse
import glob
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / 'tree_generator'))          # dev2/generator/tree_generator

import torch  # noqa: E402

from nxgen import Generator, load_config  # noqa: E402
from nxgen.core.io import save_image, save_json, save_labels  # noqa: E402
from nxgen.core.rng import image_seed  # noqa: E402
from nxgen.core.tree import load_tree  # noqa: E402
from nxgen.offline.template import build_template  # noqa: E402

INPUTS = HERE / 'data' / 'inputs'
OUTPUTS = HERE / 'data' / 'outputs'/ 'random'


def build_templates(cfg, tree, seg_dir, out_dir, device='cpu'):
    """Step 1: label maps -> HR templates (reused when nothing changed)."""
    paths = sorted(glob.glob(str(seg_dir / '*.nii*')))
    if not paths:
        sys.exit(f'No label maps in {seg_dir}')
    allowed = None
    if cfg['labels']['unassigned'] == 'background':          # labels not in the tree -> background
        allowed = list(tree.label_to_node)
    t = cfg['template']
    out = []
    for path in paths:
        tpl, record = build_template(path, str(out_dir), t['hr_voxel_mm'], t['sigma_mm'],
                                     t['margin_mm'], allowed, cfg['labels']['background'], device)
        if record['removed_labels'] or record['vanished_labels']:
            print(f'    removed labels {record["removed_labels"]}, '
                  f'vanished labels {record["vanished_labels"]}')
        out.append(tpl)
    return out


def main():
    p = argparse.ArgumentParser(description='Generate synthetic images with the tree generator.')
    p.add_argument('--config', default=str(INPUTS / 'config.json'))
    p.add_argument('--labels', default=str(INPUTS / 'labels'), help='folder of label maps')
    p.add_argument('--templates', default=str(INPUTS / 'labels'),
                   help='folder of already built HR templates (*.nii.gz); skips step 1')
    p.add_argument('--out', default=str(OUTPUTS))
    p.add_argument('-n', type=int, default=5, help='number of images')
    p.add_argument('--seed', type=int, default=0, help='run seed')
    p.add_argument('--dtype', default='uint8', choices=['uint8', 'uint16', 'float32'],
                   help='image file type; uint8/uint16 store [0, 1] with a scale factor '
                        '(uint8: -80%% size, error <= 0.002, well below the noise)')
    p.add_argument('--device', default='cuda:3' if torch.cuda.is_available() else 'cpu')
    args = p.parse_args()

    out = Path(args.out)
    cfg = load_config(args.config)
    tree = load_tree(cfg['labels']['tree'])
    print(f'Tree: {type(tree).__name__} {tree.version!r}, {len(tree.nodes)} nodes, '
          f'{len(tree.label_to_node)} labels, colors {tree.colors}')

    if args.templates:
        templates = sorted(glob.glob(str(Path(args.templates) / '*.nii*')))
        if not templates:
            sys.exit(f'No templates in {args.templates}')
        print(f'\n1. HR templates: {len(templates)} found in {args.templates} (not rebuilt)')
    else:
        print('\n1. HR templates')
        templates = build_templates(cfg, tree, Path(args.labels), out / 'templates')

    print('\n2. Images')
    t0 = time.time()
    gen = Generator(templates, cfg, device=args.device)
    print(f'Generator ready in {time.time() - t0:.1f} s: {len(gen.templates)} templates, '
          f'{len(gen.labels.values)} output labels, HR {gen.hr_voxel_mm} mm -> {gen.voxel_mm} mm')
    for sub in ('images', 'labels', 'plans'):
        (out / sub).mkdir(parents=True, exist_ok=True)
    gen.labels.save(str(out / 'label_space.json'))

    for i in range(args.n):
        t0 = time.time()
        s = gen(image_seed(args.seed, i), template=i % len(gen.templates))
        if args.device.startswith('cuda'):
            torch.cuda.synchronize()
        dt = time.time() - t0
        case = f'synth_{i:05d}'
        save_image(s.image, s.affine, str(out / 'images' / f'{case}.nii.gz'), args.dtype)
        save_labels(gen.labels.to_raw(s.seg), s.affine, str(out / 'labels' / f'{case}.nii.gz'))
        save_json(dict(case=case, run_seed=args.seed, index=i,
                       template=templates[i % len(templates)],
                       plan=s.plan, normalization=s.normalization),
                  str(out / 'plans' / f'{case}.json'))
        acq = s.plan['acquisition']
        spacing = [round(x, 2) for x in acq['spacing_mm']] if acq else 'none'
        print(f'synth_{i:05d}: {list(s.seg.shape)} voxels, {dt:.2f} s, '
              f'contrast {acq.get("contrast") if acq else None}, spacing {spacing} mm, '
              f'{len(s.plan["tree"]["class_names"])} generation classes')
    print(f'\nSaved in {out}/{{images,labels,plans}}')


if __name__ == '__main__':
    main()
