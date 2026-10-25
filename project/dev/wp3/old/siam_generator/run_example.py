"""
Example: SIAM generator on data/inputs/{T1w,seg}.nii.gz.

Outputs (data/outputs/):
    template_hr0.25mm.nii.gz      super-resolved tissue template (cached)
    synth_XXXX_img.nii.gz         synthetic images, 0.75 mm, [0, 1]
    synth_XXXX_seg.nii.gz         matching tissue labels
    synth_0000_pv.nii.gz          partial-volume maps of the first sample
    steps.png                     the generator step by step
    samples.png                   real T1w vs synthetic samples

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

import data_augmentation_pipeline as dap

HERE = os.path.dirname(os.path.abspath(__file__))
INPUTS = os.path.join(HERE, 'data', 'inputs')
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument('-n', type=int, default=4)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available()
                   else 'cpu')
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()
    os.makedirs(OUTPUTS, exist_ok=True)
    dap.seed_everything(args.seed)
    cfg = dap.SIAMConfig()

    # --- 1-2. offline: grouping + super-resolution ------------------------
    hr_path = os.path.join(OUTPUTS, 'template_hr0.25mm.nii.gz')
    if os.path.exists(hr_path):
        lab_hr, aff_hr = dap.load_template(hr_path)
    else:
        print('Super-resolving template (once) ...')
        lab_hr, aff_hr = dap.prepare_template(
            os.path.join(INPUTS, 'seg.nii.gz'), hr_path,
            cfg.target_voxel, cfg.factor, device=args.device)
    print(f'HR template: {tuple(lab_hr.shape)} @ {aff_hr[0, 0]:g} mm')

    gen = dap.SIAMGenerator([(lab_hr, aff_hr)], cfg, device=args.device)

    t1 = nib.load(os.path.join(INPUTS, 'T1w.nii.gz'))
    seg = nib.load(os.path.join(INPUTS, 'seg.nii.gz'))
    seg_grouped = dap.group_labels(np.asarray(seg.dataobj))
    # display slice: through the thalami
    thal = np.argwhere(seg_grouped == dap.T['thalamus']).mean(0)
    point = (seg.affine @ np.append(thal, 1))[:3]

    # --- 3-7. one sample, step by step (mirrors SIAMGenerator.__call__) ---
    with torch.no_grad():
        lab = gen.templates[0][0]
        lab_m = gen._morphology(lab.clone())
        lab_d = gen._deform(lab_m)
        pv = gen._partial_volume(lab_d)
        img_gmm = gen.gmm(pv)
        img_bias = gen.bias(img_gmm)
        gen.cfg.motion_prob, prob = 1.0, gen.cfg.motion_prob
        img_motion = gen._motion(img_bias)
        gen.cfg.motion_prob = prob
        img_final = gen.normalize(gen.noise(img_motion))
    f = cfg.factor
    aff_lr = aff_hr @ np.array([[f, 0, 0, (f - 1) / 2], [0, f, 0, (f - 1) / 2],
                                [0, 0, f, (f - 1) / 2], [0, 0, 0, 1]])
    nlab = cfg.nb_labels - 1
    lkw = dict(cmap='tab20', vmin=0, vmax=19)
    steps = [
        (axial(seg_grouped, seg.affine, point), 'grouped labels (1 mm)', lkw),
        (axial(lab.cpu(), aff_hr, point), 'super-res (0.25 mm)', lkw),
        (axial(lab_m.cpu(), aff_hr, point), '+ erosion/dilation', lkw),
        (axial(lab_d.cpu(), aff_hr, point), '+ affine/elastic', lkw),
        (axial(pv[dap.T['gm']].cpu(), aff_lr, point), 'PV map: GM (0.75 mm)',
         dict(cmap='magma')),
        (axial(img_gmm[0].cpu(), aff_lr, point), 'GMM w/ partial volume', {}),
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
    show(axes[1, 0], axial(seg_grouped, seg.affine, point), 'labels', **lkw)
    for i in range(args.n):
        img, lab_i, pv, aff = gen()
        name = f'synth_{i:04d}'
        dap.save_sample(img, lab_i, aff,
                        os.path.join(OUTPUTS, f'{name}_img.nii.gz'),
                        os.path.join(OUTPUTS, f'{name}_seg.nii.gz'))
        if i == 0:
            nib.save(nib.Nifti1Image(
                pv.movedim(0, -1).cpu().numpy().astype(np.float32), aff),
                os.path.join(OUTPUTS, f'{name}_pv.nii.gz'))
        show(axes[0, i + 1], axial(img[0].cpu(), aff, point), name)
        show(axes[1, i + 1], axial(lab_i.cpu(), aff, point), '', **lkw)
        print(f'saved {name}: {tuple(lab_i.shape)} @ {aff[0, 0]:g} mm')
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUTS, 'samples.png'), dpi=150)
    plt.close(fig)
    print(f'outputs in {OUTPUTS}')


if __name__ == '__main__':
    main()
