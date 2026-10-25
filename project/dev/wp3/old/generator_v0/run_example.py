"""
Example: tree generator on the SIAM example template, with the SynthSeg
tree that includes the cortical parcels.

Inputs (not copied, read from the sibling folders):
    ../augmentation/pipeline/data/inputs/{T1w,seg}.nii.gz
    ../simple_generator/data/inputs/synthseg_ctx_tree.json
    configs/freesurfer.json         morphology rules written as tree nodes

Outputs (data/outputs/):
    templates/seg_hr0.25mm.nii.gz   super-resolved template (cached, raw labels)
    synth_XXXX_img.nii.gz           synthetic images, 0.75 mm, [0, 1]
    synth_XXXX_seg.nii.gz           matching labels (raw label values)
    steps.png                       the generator step by step
    samples.png                     real T1w vs synthetic samples

Usage:
    python run_example.py [-n 4] [--device cuda] [--seed 0]
"""

import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torch

import tree_generator as tg

HERE = os.path.dirname(os.path.abspath(__file__))
SEG = os.path.join(HERE, 'data', 'inputs',
                   'seg.nii.gz')
T1 = os.path.join(HERE, 'data', 'inputs',
                  'T1w.nii.gz')
TREE = os.path.join(HERE, 'data', 'inputs',
                    'synthseg_ctx_tree.json')
CONFIG = os.path.join(HERE, 'configs', 'freesurfer.json')
OUTPUTS = os.path.join(HERE, 'data', 'outputs')


def world_to_index(affine, point):
    return np.round(np.linalg.inv(affine) @ np.append(point, 1))[:3].astype(int)


def axial(vol, affine, point):
    """Axial slice of `vol` through world `point`, oriented for display."""
    k = int(np.clip(world_to_index(affine, point)[2], 0, vol.shape[-1] - 1))
    return np.rot90(np.asarray(vol)[..., k])


def show(ax, img, title, cmap='gray', **kw):
    ax.imshow(img, cmap=cmap, interpolation='nearest', **kw)
    ax.set_title(title, fontsize=9)
    ax.axis('off')


def random_label_cmap(n, seed=0):
    rng = np.random.default_rng(seed)
    colors = rng.uniform(0.15, 1, (n, 3))
    colors[0] = 0
    return matplotlib.colors.ListedColormap(colors)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('-n', type=int, default=4)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available()
                   else 'cpu')
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()
    os.makedirs(OUTPUTS, exist_ok=True)
    tg.seed_everything(args.seed)
    cfg = tg.TreeGeneratorConfig.from_json(CONFIG)

    # --- 1-2. offline: super-resolution (raw labels kept) -----------------
    templates = tg.build_templates([SEG], TREE, cfg,
                                   os.path.join(OUTPUTS, 'templates'),
                                   args.device)
    lab_hr, aff_hr = templates[0]
    print(f'HR template: {tuple(lab_hr.shape)} @ {aff_hr[0, 0]:g} mm')

    gen = tg.TreeGenerator(templates, TREE, cfg, device=args.device)
    print(f'{gen.nb_labels} labels, {len(gen.morpho_pairs)} morphology pairs')

    t1 = nib.load(T1)
    seg = nib.load(SEG)
    seg_raw = np.asarray(seg.dataobj).astype(int)
    seg_idx = np.vectorize(lambda v: gen.index.get(v, 0))(seg_raw)
    # display slice: through the thalami
    thal = np.argwhere(np.isin(seg_raw, [10, 49])).mean(0)
    point = (seg.affine @ np.append(thal, 1))[:3]
    lkw = dict(cmap=random_label_cmap(gen.nb_labels), vmin=0,
               vmax=gen.nb_labels - 1)

    # --- 3-7. one sample, step by step (mirrors TreeGenerator.__call__) ---
    with torch.no_grad():
        lab = gen.templates[0][0]
        C, V, res, contrast = gen._sample_contrast()
        mus, sigmas = gen.scheme.compute_stats(C, V, res, contrast)
        group_of, class_names = gen.scheme.label_set_lut(res, contrast)
        lab_m = gen._morphology(lab.clone())
        lab_d = gen._deform(lab_m)
        img_tree, seg_lr, pv = gen._synthesize(lab_d, mus, sigmas, group_of,
                                               return_pv=True)
        img_tree = img_tree / cfg.intensity_scale
        img_bias = gen.bias(img_tree)
        gen.cfg.motion_prob, prob = 1.0, gen.cfg.motion_prob
        img_motion = gen._motion(img_bias)
        gen.cfg.motion_prob = prob
        img_final = gen.normalize(gen.noise(img_motion))
    f = cfg.factor
    aff_lr = aff_hr @ np.array([[f, 0, 0, (f - 1) / 2], [0, f, 0, (f - 1) / 2],
                                [0, 0, f, (f - 1) / 2], [0, 0, 0, 1]])
    wm = group_of[gen.index[2]]   # generation class of label 2 (left WM)
    steps = [
        (axial(seg_idx, seg.affine, point), 'labels (1 mm)', lkw),
        (axial(lab.cpu(), aff_hr, point), 'super-res (0.25 mm)', lkw),
        (axial(lab_m.cpu(), aff_hr, point), '+ erosion/dilation', lkw),
        (axial(lab_d.cpu(), aff_hr, point), '+ affine/elastic', lkw),
        (axial(pv[wm].cpu(), aff_lr, point), f'PV map: {class_names[wm]}',
         dict(cmap='magma')),
        (axial(img_tree[0].cpu(), aff_lr, point),
         f'tree intensities ({len(class_names)} classes)', {}),
        (axial(img_bias[0].cpu(), aff_lr, point), '+ bias field', {}),
        (axial(img_motion[0].cpu(), aff_lr, point), '+ motion', {}),
        (axial(img_final[0].cpu(), aff_lr, point), '+ noise, [0, 1]', {}),
    ]
    fig, axes = plt.subplots(1, len(steps), figsize=(2.6 * len(steps), 3.2))
    for ax, (im, title, kw) in zip(axes, steps):
        show(ax, im, title, **kw)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUTS, 'steps.png'), dpi=150)
    plt.close(fig)
    del lab_m, lab_d, pv

    # --- samples -----------------------------------------------------------
    fig, axes = plt.subplots(2, args.n + 1, figsize=(2.8 * (args.n + 1), 6))
    show(axes[0, 0], axial(t1.get_fdata(), t1.affine, point), 'real T1w')
    show(axes[1, 0], axial(seg_idx, seg.affine, point), 'labels', **lkw)
    for i in range(args.n):
        img, lab_i, aff, info = gen()
        name = f'synth_{i:04d}'
        tg.save_sample(img, lab_i, aff,
                       os.path.join(OUTPUTS, f'{name}_img.nii.gz'),
                       os.path.join(OUTPUTS, f'{name}_seg.nii.gz'),
                       labels=gen.labels)
        title = (f'{name}\n{info["contrast"]}, '
                 f'{len(info["class_names"])} classes')
        show(axes[0, i + 1], axial(img[0].cpu(), aff, point), title)
        show(axes[1, i + 1], axial(lab_i.cpu(), aff, point), '', **lkw)
        print(f'saved {name}: {tuple(lab_i.shape)} @ {aff[0, 0]:g} mm, '
              f'contrast {info["contrast"]}, '
              f'{len(info["class_names"])} generation classes')
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUTS, 'samples.png'), dpi=150)
    plt.close(fig)
    print(f'outputs in {OUTPUTS}')


if __name__ == '__main__':
    main()
