"""
Generate n synthetic (image, label) pairs with the tree generator and save
them as an nnU-Net v2 raw dataset, with the generation parameters of every
image.

Inputs (defaults, in data/inputs/):
    changed_recon_0.2mm_level1/*.nii.gz   label maps used as templates
    l1_nextbrain_tree.json                label tree (intensity model)
    generator_config.json                 TreeGeneratorConfig

Templates come from one of two sources:
    --seg-dir (default)   label maps, cropped with a background margin
                          (template_margin in the config: mm, voxels, per
                          axis or 'auto'), super-resolved once to
                          target_voxel / factor mm and cached in the
                          dataset folder (templates/, with a .json build
                          record), reused on later runs with the same
                          settings and rebuilt otherwise
    --hr-template-dir     precomputed HR templates (*.nii.gz, already at
                          target_voxel / factor mm, e.g. the templates/
                          folder of an earlier dataset, or label maps
                          already at that voxel size). The
                          super-resolution step is skipped and --seg-dir
                          is ignored. The voxel size is checked and the
                          shape padded with background to a multiple of
                          factor; the labels must match --tree. With
                          --save-hr-template-dir, the padded templates
                          are saved in the dataset folder (templates/).

Outputs (--out, default data/outputs/):
    Dataset<ID>_<NAME>/
        templates/<case>_hr<voxel>mm.nii.gz  super-resolved templates (cached);
                                             with --hr-template-dir, the
                                             padded templates, only if
                                             --save-hr-template-dir
        imagesTr/<prefix>_XXXX_0000.nii.gz   synthetic image, float32 [0, 1]
        labelsTr/<prefix>_XXXX.nii.gz        labels (consecutive indices)
        specs/<prefix>_XXXX.json             generation parameters (incl. the
                                             path of the template used)
        dataset.json                         nnU-Net dataset description
        generator_config.json, tree.json     copies of the inputs used

Every image i is generated from its own seed (derived from --seed and i,
stored in its spec), so a single image can be regenerated exactly with
`--start i -n 1 --overwrite` and the same inputs. Existing cases are
skipped, so an interrupted run can be resumed with the same command.

Usage:
    python generate_images.py -n 100 [--device cuda:1] [--seed 0]
                              [--template-device cpu]

    # every flag
    python generate_images.py \
        -n 5 \
        --seg-dir data/inputs/changed_recon_0.2mm_level1 \
        --tree data/inputs/l1_nextbrain_tree.json \
        --config data/inputs/generator_config2.json \
        --out data/outputs \
        --dataset-id 2 \
        --dataset-name NextBrain_L1 \
        --prefix synth \
        --device cuda:0 \
        --template-device cpu \
        --seed 0 \
        --start 0 \
        --overwrite
        
# Same but differetn   

# 0.4 mm images from 0.2 mm maps, with crop + 16 mm margin"
    python generate_images.py \
        -n 2 \
        --seg-dir data/inputs/changed_recon_0.2mm_level1_small \
        --tree data/inputs/l1_nextbrain_tree.json \
        --config data/inputs/generator_config2.json \
        --out data/outputs \
        --dataset-id 2 \
        --dataset-name NextBrain_L1 \
        --prefix synth \
        --device cuda:0 \
        --template-device cpu \
        --seed 0 \
        --start 0 \
        --overwrite  
        
# Use .4mm as target and use .2mm HR templates
    python generate_images.py \
        -n 2 \
        --hr-template-dir data/inputs/changed_recon_0.2mm_level1_small \
        --save-hr-template-dir \
        --tree data/inputs/l1_nextbrain_tree.json \
        --config data/inputs/generator_config2.json \
        --out data/outputs \
        --dataset-id 3 \
        --dataset-name NextBrain_L1 \
        --prefix synth \
        --device cuda:0 \
        --template-device cpu \
        --seed 0 \
        --start 0 \
        --overwrite
        
        
# 0.4 mm images from 0.2 mm maps, with crop + auto quartil 1 margin"
    python generate_images.py \
        -n 2 \
        --seg-dir data/inputs/changed_recon_0.2mm_level1 \
        --tree data/inputs/l1_nextbrain_tree.json \
        --config data/inputs/generator_config3.json \
        --out data/outputs \
        --dataset-id 4 \
        --dataset-name NextBrain_L1 \
        --prefix synth \
        --device cuda:0 \
        --template-device cpu \
        --seed 0 \
        --start 0 \
        --overwrite  

        
        
    # reuse the HR templates of Dataset001 (no super-resolution)
    python generate_images.py -n 100 --dataset-id 2 \
        --hr-template-dir data/outputs/Dataset001_NextBrain_L1/templates
        
        



    python generate_images.py \
        -n 2 \
        --seg-dir /home/agustin/phd/synthseg_nextbrain/project/dev/testing/data/inputs/changed_recon_0.2mm_level1 \
        --tree /home/agustin/phd/synthseg_nextbrain/project/dev/testing/data/inputs/l1_nextbrain_tree.json \
        --config /home/agustin/phd/synthseg_nextbrain/project/dev/testing/data/inputs/generator_config.json \
        --out /home/agustin/phd/synthseg_nextbrain/project/dev/testing/data/outputs \
        --dataset-id 0 \
        --dataset-name NextBrain_L1 \
        --prefix synth \
        --device cuda:0 \
        --template-device cpu \
        --seed 0 \
        --start 0 \
        --overwrite  


    python generate_images.py \
        -n 2 \
        --seg-dir data/inputs/changed_recon_0.2mm \
        --tree data/inputs/nextbrain_complete_tree.json \
        --config data/inputs/nextbrain_configs.json \
        --out data/outputs \
        --dataset-id 0 \
        --dataset-name NextBrain_complete \
        --prefix synth \
        --device cuda:0 \
        --template-device cuda:0 \
        --seed 0 \
        --start 0 \
        --overwrite  
"""

import argparse
import glob
import json
import os
import random
import shutil
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, 'src'))
import tree_generator as tg  # noqa: E402

INPUTS = os.path.join(HERE, 'data', 'inputs')
cc = tg.cc


# ---------------------------------------------------------------------------
# Generator that records what it draws
# ---------------------------------------------------------------------------

class RecordingTreeGenerator(tg.TreeGenerator):
    """
    TreeGenerator whose random choices are all drawn from one seed per
    image and returned with the sample.

    `TreeGenerator.__call__` already reports the acquisition and the tree
    intensities. On top of that, this class draws the template, the noise
    std, the bias and motion flags itself (and fixes them in the
    transforms), and logs the morphology operations.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._bias_field = self.bias.subtransform   # inside MaybeTransform
        self._morpho_log = []

    def _morphology(self, lab):
        """Same as `TreeGenerator._morphology`, logging each operation."""
        cfg = self.cfg
        lo, hi = cfg.morpho_iter
        for spec_a, spec_b, a, b in self.morpho_pairs:
            if random.random() > cfg.morpho_prob:
                continue
            if random.random() < 0.5:
                grow, shrink, names = a, b, (spec_a, spec_b)
            else:
                grow, shrink, names = b, a, (spec_b, spec_a)
            n_iter = random.randint(lo, hi)
            self._morpho_log.append(dict(grow=names[0], shrink=names[1],
                                         n_iter=n_iter))
            box = tg._bbox(self._isin(lab, grow) | self._isin(lab, shrink),
                           margin=n_iter + 1)
            if box is None:
                continue
            sub = lab[box]
            for _ in range(n_iter):
                val = torch.where(self._isin(sub, grow), sub.float() + 1, 0.)
                nbr = torch.nn.functional.max_pool3d(
                    val[None, None], 3, stride=1, padding=1)[0, 0]
                take = self._isin(sub, shrink) & (nbr > 0)
                sub = torch.where(take, (nbr - 1).to(sub.dtype), sub)
                del val, nbr, take
            lab[box] = sub
            del sub
        return lab

    def generate(self, seed):
        """
        One sample from `seed`.

        Returns
        -------
        img, seg, affine : as `TreeGenerator.__call__`
        prm : dict of every drawn parameter (JSON serialisable)
        """
        cfg = self.cfg
        tg.seed_everything(seed)
        self.scheme.rng = np.random.default_rng(seed)

        template_index = random.randrange(len(self.templates))
        noise_sigma = random.uniform(*cfg.noise_sigma)
        self.noise = cc.GaussianNoiseTransform(noise_sigma)
        if cfg.noise_space == 'lowres':
            self.lowres.noise = self.noise
        bias = random.random() < cfg.bias_prob
        self.bias = self._bias_field if bias else cc.IdentityTransform()
        motion = random.random() < cfg.motion_prob
        self._morpho_log = []

        img, seg, affine, info = self(template_index=template_index,
                                      motion=motion)

        names = self.label_names()
        group_of = np.asarray(info['group_of'])
        prm = dict(
            seed=int(seed),
            template_index=template_index,
            contrast=info['contrast'],
            tree_resolution_mm=float(info['resolution']),
            colors=list(self.tree.colors),
            colors_mu=info['colors_mu'],
            colors_sigma=info['colors_sigma'],
            acquisition=info['acquisition'],
            morphology=self._morpho_log,
            bias_field=bias,
            motion=motion,
            noise_sigma=noise_sigma,
            noise_space=cfg.noise_space,
            labels=[dict(index=i, value=int(v), name=names[i],
                         mu=float(info['mus'][i]),
                         sigma=float(info['sigmas'][i]),
                         generation_class=int(group_of[i]))
                    for i, v in enumerate(self.labels)],
            generation_classes=info['class_names'],
        )
        return img, seg, affine, prm


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def _to_json(o):
    """json.dump fallback for numpy values."""
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.generic):
        return o.item()
    raise TypeError(f'{type(o)} is not JSON serialisable')


def image_seed(seed, i):
    """Independent 32-bit seed of image `i` of a run with `seed`."""
    return int(np.random.SeedSequence([seed, i]).generate_state(1)[0])


def load_hr_template(path, cfg):
    """
    Load a precomputed HR template and pad it with background to a multiple
    of `cfg.factor` (at the high end, so the affine is unchanged).

    Raises ValueError if its voxel size is not `target_voxel / factor`.
    Templates built with --seg-dir (with a `.json` build record next to
    them) report the margin they were built with. For others (e.g. raw
    label maps), warns if the background around the brain is smaller than
    the margin the spatial deformation can need (`tg.auto_margin_mm`): the
    brain may then be cut at the border of some images.
    """
    lab, affine = tg.load_template(path)
    hr_voxel = cfg.target_voxel / cfg.factor
    voxel = np.sqrt((np.asarray(affine)[:3, :3] ** 2).sum(0))
    if np.any(np.abs(voxel - hr_voxel) > 1e-3):
        raise ValueError(f'{path}: voxel {voxel.round(4).tolist()} mm, but '
                         f'the config expects HR templates at {hr_voxel:g} mm '
                         f'(target_voxel / factor)')
    fg = (lab != cfg.background).numpy()
    idx = np.argwhere(fg)
    have = np.minimum(idx.min(0), np.array(fg.shape) - 1 - idx.max(0))
    have = have * hr_voxel
    record = path.removesuffix('.gz').removesuffix('.nii') + '.json'
    built = {}
    if os.path.exists(record):
        with open(record) as f:
            built = json.load(f).get('settings', {})
    if 'template_margin' in built:
        print(f'{os.path.basename(path)}: built with template_margin '
              f'{built["template_margin"]} (background around the brain: '
              f'{np.round(have, 1).tolist()} mm)')
    elif np.any(have < (need := tg.auto_margin_mm(fg, hr_voxel, cfg))):
        print(f'WARNING {os.path.basename(path)}: background margin '
              f'{np.round(have, 1).tolist()} mm, the deformation can need '
              f'up to {np.round(need, 1).tolist()} mm: the brain may be cut '
              f'at the border. Use --seg-dir (template_margin) to add it.')
    pad = [-n % cfg.factor for n in lab.shape]
    if any(pad):
        print(f'{os.path.basename(path)}: padded {tuple(lab.shape)} by '
              f'{tuple(pad)} to a multiple of factor={cfg.factor}')
        lab = torch.nn.functional.pad(
            lab, [0, pad[2], 0, pad[1], 0, pad[0]], value=cfg.background)
    return lab, affine


def write_dataset_json(out_dir, names, labels):
    """nnU-Net v2 dataset.json, numTraining = cases present in labelsTr."""
    n = len(glob.glob(os.path.join(out_dir, 'labelsTr', '*.nii.gz')))
    tg.write_nnunet_dataset_json(out_dir, n, names, labels)
    return n


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description=__doc__.split('\n')[1],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('-n', type=int, required=True, help='number of images')
    p.add_argument('--seg-dir',
                   default=os.path.join(INPUTS, 'changed_recon_0.2mm_level1'),
                   help='folder of the template label maps (*.nii.gz)')
    p.add_argument('--hr-template-dir',
                   help='folder of precomputed HR templates (*.nii.gz); '
                        'skips the super-resolution and ignores --seg-dir')
    p.add_argument('--save-hr-template-dir', action='store_true',
                   help='with --hr-template-dir: save the padded templates '
                        'in the dataset folder (templates/)')
    p.add_argument('--tree',
                   default=os.path.join(INPUTS, 'l1_nextbrain_tree.json'))
    p.add_argument('--config',
                   default=os.path.join(INPUTS, 'generator_config.json'))
    p.add_argument('--out', default=os.path.join(HERE, 'data', 'outputs'))
    p.add_argument('--dataset-id', type=int, default=1)
    p.add_argument('--dataset-name', default='NextBrain_L1')
    p.add_argument('--prefix', default='synth', help='case name prefix')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available()
                   else 'cpu')
    p.add_argument('--template-device', default='cpu',
                   help='device of the one-off template super-resolution '
                        '(~70 GB peak per 0.1 mm template)')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--start', type=int, default=0,
                   help='index of the first image')
    p.add_argument('--overwrite', action='store_true',
                   help='regenerate cases that already exist')
    args = p.parse_args()

    cfg = tg.TreeGeneratorConfig.from_json(args.config)
    seg_dir = args.hr_template_dir or args.seg_dir
    segs = sorted(glob.glob(os.path.join(seg_dir, '*.nii.gz')))
    if not segs:
        raise FileNotFoundError(f'no *.nii.gz in {seg_dir}')

    ds = os.path.join(args.out,
                      f'Dataset{args.dataset_id:03d}_{args.dataset_name}')
    for sub in ('imagesTr', 'labelsTr', 'specs'):
        os.makedirs(os.path.join(ds, sub), exist_ok=True)
    with open(os.path.join(ds, 'generator_config.json'), 'w') as f:
        json.dump(cfg.to_dict(), f, indent=2)
    shutil.copy(args.tree, os.path.join(ds, 'tree.json'))

    # --- offline: super-resolved templates (precomputed or cached) --------
    if args.hr_template_dir:
        print(f'Loading {len(segs)} precomputed HR templates from '
              f'{args.hr_template_dir}')
        templates = [load_hr_template(path, cfg) for path in segs]
        if args.save_hr_template_dir:
            tpl_dir = os.path.join(ds, 'templates')
            os.makedirs(tpl_dir, exist_ok=True)
            suffix = f'_hr{cfg.target_voxel / cfg.factor:g}mm'
            for path, (lab, affine) in zip(segs, templates):
                name = os.path.basename(path).removesuffix('.gz')
                name = name.removesuffix('.nii').removesuffix(suffix)
                out = os.path.join(tpl_dir, name + suffix)
                tg.save_labels(lab.numpy(), affine, out + '.nii.gz')
                # a different build record than --seg-dir, so that a later
                # --seg-dir run on this dataset rebuilds instead of reusing
                with open(out + '.json', 'w') as f:
                    json.dump(dict(source=os.path.abspath(path),
                                   settings=dict(hr_template_dir=True),
                                   shape=list(lab.shape)), f, indent=2)
            print(f'Saved {len(segs)} padded HR templates in {tpl_dir}')

    else:
        if args.save_hr_template_dir:
            print('--save-hr-template-dir ignored: the templates built from '
                  '--seg-dir are always saved in templates/')
        templates = tg.build_templates(segs, args.tree, cfg,
                                       os.path.join(ds, 'templates'),
                                       args.template_device)
    gen = RecordingTreeGenerator(templates, args.tree, cfg,
                                 device=args.device)
    names = gen.label_names()
    print(f'{len(segs)} templates, {gen.nb_labels} labels: {names}')

    # --- samples --------------------------------------------------------------
    for i in range(args.start, args.start + args.n):
        case = f'{args.prefix}_{i:04d}'
        img_path = os.path.join(ds, 'imagesTr', f'{case}_0000.nii.gz')
        seg_path = os.path.join(ds, 'labelsTr', f'{case}.nii.gz')
        spec_path = os.path.join(ds, 'specs', f'{case}.json')
        if os.path.exists(spec_path) and not args.overwrite:
            print(f'{case} exists, skipped')
            continue

        t0 = time.time()
        img, seg, affine, prm = gen.generate(image_seed(args.seed, i))
        # labels as indices: consecutive, as nnU-Net requires
        tg.save_sample(img, seg, affine, img_path, seg_path)
        spec = dict(
            case=case,
            image=os.path.relpath(img_path, ds),
            label=os.path.relpath(seg_path, ds),
            run_seed=args.seed,
            image_number=i,
            template=os.path.abspath(segs[prm['template_index']]),
            shape=list(seg.shape),
            voxel_mm=cfg.target_voxel,
            affine=affine,
            config='generator_config.json',
            **prm,
        )
        with open(spec_path, 'w') as f:
            json.dump(spec, f, indent=2, default=_to_json)

        acq = prm['acquisition']
        print(f'[{i - args.start + 1}/{args.n}] {case}: '
              f'{os.path.basename(spec["template"])}, '
              f'{tuple(seg.shape)}, acquisition {acq["mode"]} '
              f'{[round(s, 2) for s in acq["spacing_mm"]]} mm, '
              f'{len(prm["generation_classes"])} classes, '
              f'{time.time() - t0:.1f} s')
        del img, seg

    n = write_dataset_json(ds, names, gen.labels)
    print(f'{n} cases in {ds}')


if __name__ == '__main__':
    main()
