"""
Generate n synthetic (image, label) pairs with the tree generator and save
them as an nnU-Net v2 raw dataset, with the generation parameters of every
image.

Inputs (defaults, in data/inputs/):
    changed_recon_0.2mm_level1/*.nii.gz   label maps used as templates
    l1_nextbrain_tree.json                label tree (intensity model)
    generator_config.json                 TreeGeneratorConfig

Outputs (--out, default data/outputs/):
    templates/<case>_hr<voxel>mm.nii.gz   super-resolved templates (cached)
    Dataset<ID>_<NAME>/
        imagesTr/<prefix>_XXXX_0000.nii.gz   synthetic image, float32 [0, 1]
        labelsTr/<prefix>_XXXX.nii.gz        labels (consecutive indices)
        specs/<prefix>_XXXX.json             generation parameters
        dataset.json                         nnU-Net dataset description
        generator_config.json, tree.json     copies of the inputs used

Every image i is generated from its own seed (derived from --seed and i,
stored in its spec), so a single image can be regenerated exactly with
`--start i -n 1 --overwrite` and the same inputs. Existing cases are
skipped, so an interrupted run can be resumed with the same command.

Usage:
    python generate_images.py -n 100 [--device cuda:1] [--seed 0]
                              [--template-device cpu]
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
    p.add_argument('--tree',
                   default=os.path.join(INPUTS, 'l1_nextbrain_tree.json'))
    p.add_argument('--config',
                   default=os.path.join(INPUTS, 'generator_config.json'))
    p.add_argument('--out', default=os.path.join(HERE, 'data', 'outputs'))
    p.add_argument('--dataset-id', type=int, default=1)
    p.add_argument('--dataset-name', default='NextBrainL1Synth')
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
    segs = sorted(glob.glob(os.path.join(args.seg_dir, '*.nii.gz')))
    if not segs:
        raise FileNotFoundError(f'no *.nii.gz in {args.seg_dir}')

    ds = os.path.join(args.out,
                      f'Dataset{args.dataset_id:03d}_{args.dataset_name}')
    for sub in ('imagesTr', 'labelsTr', 'specs'):
        os.makedirs(os.path.join(ds, sub), exist_ok=True)
    with open(os.path.join(ds, 'generator_config.json'), 'w') as f:
        json.dump(cfg.to_dict(), f, indent=2)
    shutil.copy(args.tree, os.path.join(ds, 'tree.json'))

    # --- offline: super-resolved templates (cached) -----------------------
    templates = tg.build_templates(segs, args.tree, cfg,
                                   os.path.join(args.out, 'templates'),
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
