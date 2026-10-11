"""
End-to-end example with a SynthSeg / FreeSurfer segmentation.

    python synthseg_example.py <segmentation.nii.gz> [synthseg_tree.json] [FreeSurferColorLUT.txt]

Without arguments it uses training_seg_01.nii.gz (from the SynthSeg repo, data/training_label_maps).
Needs tree_sampling.py, tree_plot.py and sampling_scheme_v2.py in the same folder.

Outputs (in ./synthseg_example_out/):
  synth_<res>.nii.gz        synthetic images (on the segmentation grid) for each resolution
  synthetic_images.png      axial / coronal slices: segmentation, generation classes, images
  tree_<res>.png            tree with the nodes visible at each resolution
"""

import os
import sys

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from scipy.ndimage import gaussian_filter, zoom

from tree_sampling import LabelTree, TreeSampling
from tree_plot import plot_tree, _node_color_freesurfer
from matplotlib.colors import ListedColormap


LUT = 'FreeSurferColorLUT.txt'


# TREE = 'data/inputs/synthseg_tree.json'
# SEG = "data/inputs/SP_T1W_3T_0006_synthseg_resampled.nii.gz"  # from SynthSeg repo, data/training_label_maps
# # OUT = 'data/outputs/example_synthseg_random'
# OUT = 'data/outputs/example_synthseg'

TREE = 'data/inputs/synthseg_ctx_tree.json'
SEG = "data/inputs/sub-ADNI002S0295_ses-m000_space-MNI_T1wdseg_synthsr.nii.gz"  # from SynthSeg repo, data/training_label_maps
# OUT = 'data/outputs/example_synthseg_ctx_random'
OUT = 'data/outputs/example_synthseg_ctx'


CONTRAST = 'T1'
# Voxel sizes to simulate (mm). A tuple is an anisotropic stack (here 5 mm axial slices).
# RESOLUTIONS = [1.0, 1.5, 3.0, (1.0, 1.0, 5.0)]
RESOLUTIONS = [1.0, 3]
SEED = 3
RANDOM_CONTRAST = False    # True: SynthSeg-like random principal colors; False: T1-like for readability
os.makedirs(OUT, exist_ok=True)


def res_name(r):
    return 'x'.join(f'{v:g}' for v in np.atleast_1d(r)) + 'mm'


# ------------------------------------------------------------------------------------------------ #
# 1. Load segmentation and tree                                                                     #
# ------------------------------------------------------------------------------------------------ #

nii = nib.as_closest_canonical(nib.load(SEG))      # reorient to RAS: axes = (x: L->R, y: P->A, z: I->S)
seg = np.asarray(nii.dataobj).astype(int)
spacing = np.array(nii.header.get_zooms()[:3], dtype=float)


# Crop to the non-background bounding box (+ margin) to save time
nz = np.argwhere(seg > 0)
lo, hi = np.maximum(nz.min(0) - 4, 0), np.minimum(nz.max(0) + 5, seg.shape)
seg = seg[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
affine = nii.affine.copy()
affine[:3, 3] = nii.affine[:3, :3] @ lo + nii.affine[:3, 3]
print(f'Segmentation {SEG}: shape {seg.shape}, spacing {spacing}, {len(np.unique(seg))} labels')

tree = LabelTree.from_json(TREE)
labels = np.unique(seg)
scheme = TreeSampling(labels, tree, seed=SEED, std_noise=3, verbose=True)

### print the tree at the finest resolution (1 mm) to see which nodes are visible
# tree.print_tree(resolution=1, contrast=CONTRAST)

# verify that all the labels in seg are included in the tree
print("tree has ", len(tree.label_to_node.keys()), " labels")
print("segmentation has ", len(labels), " labels")
tree_labels = set(tree.label_to_node.keys())
if not set(labels).issubset(tree_labels):
    missing = set(labels) - tree_labels
    raise ValueError(f'Labels {missing} are in the segmentation but not in the tree. '
                     f'Check that the tree is compatible with the segmentation.')

# ------------------------------------------------------------------------------------------------ #
# 2. Acquisition model: bias field + blur + downsample + resample back to the segmentation grid     #
# ------------------------------------------------------------------------------------------------ #

def bias_field(shape, rng, strength=0.3):
    small = rng.normal(0, strength, (4, 4, 4))
    return np.exp(zoom(small, np.array(shape) / 4, order=3))


def acquire(img, resolution, rng):
    """Simulates an acquisition at `resolution` mm (scalar or per axis) and returns it on the input grid."""
    res = np.broadcast_to(np.atleast_1d(resolution).astype(float), (3,))
    img = img * bias_field(img.shape, rng)
    if np.all(res <= spacing):
        return img
    factor = spacing / res                              # < 1 along coarse axes
    sigma = 0.42 * res / spacing * (res > spacing)      # Gaussian PSF ~ one-voxel FWHM
    low = zoom(gaussian_filter(img, sigma), factor, order=1)
    up = zoom(low, np.array(img.shape) / np.array(low.shape), order=1)
    return np.clip(up, 0, None)


# ------------------------------------------------------------------------------------------------ #
# 3. Generate one image per resolution with the same contrast (same principal colors)               #
# ------------------------------------------------------------------------------------------------ #

rng = np.random.default_rng(SEED)
if RANDOM_CONTRAST:
    C = rng.uniform(20, 230, len(tree.colors))   # principal means  (order = tree.colors)
    V = rng.uniform(3, 12, len(tree.colors))     # principal sigmas
else:
    C = np.array([110., 170., 40.])              # GM, WM, CSF  ~ T1
    V = np.array([6., 5., 8.])
print('Principal colors', dict(zip(tree.colors, np.round(C).astype(int))))

results = []
for r in RESOLUTIONS:
    scheme.rng = np.random.default_rng(SEED)   # same draws -> visible nodes keep their color
    synth = scheme.sample_image(seg, C, V, resolution=r, contrast=CONTRAST)
    # image = synth
    # image = acquire(synth, r, np.random.default_rng(SEED + 1))
    # nib.save(nib.Nifti1Image(image.astype(np.float32), affine), f'{OUT}/synth_{res_name(r)}.nii.gz')
    nib.save(nib.Nifti1Image(synth.astype(np.float32), affine), f'{OUT}/synth_{res_name(r)}.nii.gz')

    # Generation classes: voxel -> index of the node whose Gaussian it uses
    rep = tree.representative(r, CONTRAST)
    order = {n: i for i, n in enumerate(tree.order())}
    lut = {lab: order[rep[tree.label_to_node[lab]]] for lab in labels}
    classes = scheme.convert_labelmap(seg, lut)
    n_classes = len(set(lut.values()))
    print(f'{res_name(r):>12}: {n_classes} generation classes')
    # results.append((r, synth, image, classes, n_classes))
    results.append((r, synth, classes, n_classes))

    plot_tree(tree, r, CONTRAST, colors='freesurfer', lut=LUT, save=f'{OUT}/tree_{res_name(r)}.png')
    plt.close('all')


# ------------------------------------------------------------------------------------------------ #
# 4. Figure: segmentation, generation classes and images (axial + coronal)                          #
# ------------------------------------------------------------------------------------------------ #

# Slices through the centre of the thalami (labels 10 / 49), so deep GM is visible
thal = np.argwhere(np.isin(seg, [10, 49]))
cx, cy, cz = thal.mean(0).astype(int) if len(thal) else np.array(seg.shape) // 2


def axial(v):
    return np.rot90(v[:, :, cz])
def coronal(v):
    return np.rot90(v[:, cy, :])
def sagittal(v):
    return np.rot90(v[cx+10, :, :])


# FreeSurfer colours; at each resolution collapsed nodes take the colour of their visible parent
order = {n: i for i, n in enumerate(tree.order())}
null_nodes = {n for n in tree.order() if tree.nodes[n].null}


def class_cmap(r):
    col = _node_color_freesurfer(tree, LUT, resolution=r, contrast=CONTRAST)
    return ListedColormap([(0, 0, 0)] + [col[n] for n in tree.order()[1:]])


# vmax = np.percentile(results[0][2], 99.5)
vmax = np.percentile(results[0][1], 99.5)

# fig, axes = plt.subplots(len(results), 5, figsize=(18, 3.9 * len(results)))
fig, axes = plt.subplots(len(results), 6, figsize=(18, 3.9 * len(results)))
# for row, (r, synth, image, classes, n_classes) in enumerate(results):
for row, (r, synth, classes, n_classes) in enumerate(results):
    shown = np.where(np.isin(classes, [order[n] for n in null_nodes]), 0, classes)
    panels = [
        (axial(shown), class_cmap(r), f'{res_name(r)}: {n_classes} generation classes', len(order) - 1),
        (coronal(shown), class_cmap(r), 'generation classes (coronal)', len(order) - 1),
        (sagittal(shown), class_cmap(r), 'generation classes (sagittal)', len(order) - 1),
        (axial(synth), 'gray', 'synthetic (axial)', vmax),
        (coronal(synth), 'gray', 'synthetic (coronal)', vmax),
        (sagittal(synth), 'gray', 'synthetic (sagittal)', vmax),      
        # (coronal(image), 'gray', f'acquired at {res_name(r)} (coronal)', vmax),
        # (axial(image), 'gray', f'acquired at {res_name(r)} (axial)', vmax),
    ]
    for ax, (data, cmap, title, vm) in zip(axes[row], panels):
        ax.imshow(data, cmap=cmap, interpolation='nearest', vmin=0, vmax=vm)
        ax.set_title(title, fontsize=10)
        ax.axis('off')
fig.tight_layout()
fig.savefig(f'{OUT}/synthetic_images.png', dpi=80)
print(f'Saved results in {OUT}/')
