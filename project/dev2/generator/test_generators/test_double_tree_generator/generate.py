"""
Test of the double-tree generator (generator_double_tree/nxgen).

    conda activate nxb
    python generate.py                          # 5 images, seed 0
    python generate.py -n 20 --seed 3 --device cuda:2
    python generate.py --build-templates        # rebuild HR templates from L first

Inputs (data/inputs/config.json)
--------------------------------
    L    simplified_trees_and_labels/data/outputs/nxbsimplified/labels   (--labels)
         already at 0.2 mm: used directly as HR templates (--build-templates
         builds them with the 'template' section instead)
    MT   labels.tree          nextbrain_tree_allen_gyral_simplified.json (morphology)
    SM   merging.levels       nxb_all (identity), nxb16, nxbsupersynth (merging scripts)
    SMP  merging.principal    nxb16
    IT   tree of the principal level: nextbrain_tree_nxb16.json (intensities)

Per image: morphology (MT) -> spatial -> merge into every level -> partial volume of
every level (one GT each, 0.4 mm) -> image synthesised from the nxb16 level with IT.

Outputs (--out, default data/outputs/random/)
---------------------------------------------
    images/synth_<i>.nii.gz            synthetic image in [0, 1]
    labels/<level>/synth_<i>.nii.gz    GT of every level, raw label values of its tree
    plans/synth_<i>.json               every random parameter of the image
    label_spaces/<level>.json          index <-> raw value and name of every level
"""

import argparse
import glob
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / 'generator_double_tree'))

import torch  # noqa: E402

from nxgen import Generator, load_config  # noqa: E402
from nxgen.core.io import save_image, save_json, save_labels  # noqa: E402
from nxgen.core.rng import image_seed  # noqa: E402
from nxgen.core.tree import load_tree  # noqa: E402
from nxgen.offline.template import build_template  # noqa: E402

INPUTS = HERE / 'data' / 'inputs'
OUTPUTS = HERE / 'data' / 'outputs' / 'random'
LABELS = HERE.parents[2] / 'simplified_trees_and_labels' / 'data' / 'outputs' / 'nxbsimplified' / 'labels'


def build_templates(cfg, tree, seg_dir, out_dir, device='cpu'):
    """Label maps -> HR templates (reused when nothing changed)."""
    paths = sorted(glob.glob(str(seg_dir / '*.nii*')))
    if not paths:
        sys.exit(f'No label maps in {seg_dir}')
    allowed = list(tree.label_to_node) if cfg['labels']['unassigned'] == 'background' else None
    t = cfg['template']
    return [build_template(p, str(out_dir), t['hr_voxel_mm'], t['sigma_mm'], t['margin_mm'],
                           allowed, cfg['labels']['background'], device)[0] for p in paths]


def main():
    p = argparse.ArgumentParser(description='Generate synthetic images with the double-tree generator.')
    p.add_argument('--config', default=str(INPUTS / 'config.json'))
    p.add_argument('--labels', default=str(LABELS), help='folder of label maps L')
    p.add_argument('--build-templates', action='store_true',
                   help='build HR templates from --labels (else the maps are the templates)')
    p.add_argument('--out', default=str(OUTPUTS))
    p.add_argument('-n', type=int, default=5, help='number of images')
    p.add_argument('--seed', type=int, default=0, help='run seed')
    p.add_argument('--dtype', default='uint8', choices=['uint8', 'uint16', 'float32'])
    p.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    args = p.parse_args()

    out = Path(args.out)
    cfg = load_config(args.config)
    tree = load_tree(cfg['labels']['tree'])
    print(f'MT: {tree.version!r}, {len(tree.nodes)} nodes, {len(tree.label_to_node)} labels')

    if args.build_templates:
        print('\n1. HR templates')
        templates = build_templates(cfg, tree, Path(args.labels), out / 'templates')
    else:
        templates = sorted(glob.glob(str(Path(args.labels) / '*.nii*')))
        if not templates:
            sys.exit(f'No label maps in {args.labels}')
        print(f'\n1. HR templates: {len(templates)} label maps of {args.labels} (used as is)')

    print('\n2. Images')
    t0 = time.time()
    gen = Generator(templates, cfg, device=args.device)
    print(f'Generator ready in {time.time() - t0:.1f} s: {len(gen.templates)} templates, '
          f'HR {gen.hr_voxel_mm} mm -> {gen.voxel_mm} mm')
    for name, level in gen.levels.items():
        tag = '  <- principal (IT)' if name == gen.principal else ''
        print(f'    level {name:14s} {len(level.labels):4d} labels, tree {level.tree.version!r}{tag}')
    for sub in ['images', 'plans', 'label_spaces'] + [f'labels/{n}' for n in gen.levels]:
        (out / sub).mkdir(parents=True, exist_ok=True)
    for name, level in gen.levels.items():
        level.labels.save(str(out / 'label_spaces' / f'{name}.json'))

    for i in range(args.n):
        t0 = time.time()
        s = gen(image_seed(args.seed, i), template=i % len(gen.templates))
        if args.device.startswith('cuda'):
            torch.cuda.synchronize()
        dt = time.time() - t0
        case = f'synth_{i:05d}'
        save_image(s.image, s.affine, str(out / 'images' / f'{case}.nii.gz'), args.dtype)
        for name, seg in s.segs.items():
            save_labels(gen.levels[name].labels.to_raw(seg), s.affine,
                        str(out / 'labels' / name / f'{case}.nii.gz'))
        save_json(dict(case=case, run_seed=args.seed, index=i,
                       template=templates[i % len(templates)], principal=gen.principal,
                       plan=s.plan, normalization=s.normalization),
                  str(out / 'plans' / f'{case}.json'))
        acq = s.plan['acquisition']
        spacing = [round(x, 2) for x in acq['spacing_mm']] if acq else 'none'
        present = {n: len(torch.unique(g)) for n, g in s.segs.items()}
        print(f'{case}: {list(s.seg.shape)} voxels, {dt:.2f} s, '
              f'contrast {acq.get("contrast") if acq else None}, spacing {spacing} mm, '
              f'labels present {present}')
    print(f'\nSaved in {out}/{{images,labels/<level>,plans,label_spaces}}')


if __name__ == '__main__':
    main()
