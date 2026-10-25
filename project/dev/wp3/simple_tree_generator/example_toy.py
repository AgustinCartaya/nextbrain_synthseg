"""
Toy example for tree_sampling.py: nested squares + a small tree.

Put this file in the same folder as tree_sampling.py and sampling_scheme_v2.py and run:
    python toy_example.py
It writes toy_tree.json, toy_resolutions.png and toy_random_draws.png.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, zoom

from tree_sampling import LabelTree, TreeSampling

GRID = 0.33  # mm per pixel of the label map (target grid)


# ------------------------------------------------------------------------------------------------ #
# 1. The tree                                                                                       #
# ------------------------------------------------------------------------------------------------ #
#   root
#   ├── background      (null color)                       label 0
#   ├── white_matter    1.0*WM                             label 1
#   ├── csf             1.0*CSF                            label 3
#   └── nucleus         0.6*GM + 0.4*WM                    label 10   (always visible, level 1)
#       ├── part_A      r_max = 1.0 mm                     label 11
#       │   ├── A1      r_max = 0.5 mm                     label 111
#       │   └── A2      r_max = 0.5 mm (T2: 0.8 mm)        label 112
#       └── part_B      r_max = 2.0 mm                     label 12

tree = LabelTree.new(colors=['GM', 'WM', 'CSF'], contrasts=['T1', 'T2'], version='toy')
tree.add_node('background', 'root', labels=[0], null=True)
tree.add_node('white_matter', 'root', labels=[1], w_mu=[0, 1, 0], wr_mu=3, wr_sigma=0.5)
tree.add_node('csf', 'root', labels=[3], w_mu=[0, 0, 1], wr_mu=3, wr_sigma=0.5)
tree.add_node('nucleus', 'root', labels=[10], w_mu=[0.6, 0.4, 0], wr_mu=5, wr_sigma=1)
tree.add_node('part_A', 'nucleus', labels=[11], wr_mu=15, wr_sigma=1, r_max=1.0)
tree.add_node('A1', 'part_A', labels=[111], wr_mu=15, wr_sigma=1, r_max=0.5)
tree.add_node('A2', 'part_A', labels=[112], wr_mu=15, wr_sigma=1, r_max={'default': 0.5, 'T2': 0.8})
tree.add_node('part_B', 'nucleus', labels=[12], wr_mu=15, wr_sigma=1, r_max=2.0)
tree.validate()

# Normally you write the JSON by hand (or with a script) and just load it:
tree.to_json('data/inputs/toy_tree.json')
tree = LabelTree.from_json('data/inputs/toy_tree.json')


# ------------------------------------------------------------------------------------------------ #
# 2. A label map of nested squares                                                                  #
# ------------------------------------------------------------------------------------------------ #

def square(lm, y0, y1, x0, x1, label):
    lm[y0:y1, x0:x1] = label

seg = np.zeros((240, 240), dtype=int)      # 80 x 80 mm at 0.33 mm
square(seg, 15, 225, 15, 225, 1)           # white matter
square(seg, 30, 70, 160, 210, 3)           # csf
square(seg, 80, 210, 30, 210, 10)          # nucleus (its own label shows as a ring)
square(seg, 95, 195, 45, 125, 11)          # part_A (ring around A1/A2)
square(seg, 110, 180, 55, 115, 111)        # A1
square(seg, 128, 162, 70, 100, 112)        # A2, nested inside A1
square(seg, 110, 180, 140, 195, 12)        # part_B

labels = np.unique(seg)


# ------------------------------------------------------------------------------------------------ #
# 3. Simple acquisition model: blur + downsample to r mm + resample back to the 0.33 mm grid         #
# ------------------------------------------------------------------------------------------------ #

def acquire(img, resolution):
    if resolution <= GRID:
        return img
    factor = GRID / resolution
    sigma = 0.42 * resolution / GRID                     # Gaussian PSF ~ FWHM of one voxel
    low = zoom(gaussian_filter(img, sigma), factor, order=1)
    return zoom(low, np.array(img.shape) / np.array(low.shape), order=1)


def principal_colors(rng):
    """Random principal colors, SynthSeg-style: C ~ U(0, 255), V ~ U(2, 15)."""
    return rng.uniform(0, 255, 3), rng.uniform(2, 15, 3)


# ------------------------------------------------------------------------------------------------ #
# 4. Figure 1: same draw, increasing voxel size                                                     #
# ------------------------------------------------------------------------------------------------ #

resolutions = [0.33, 0.7, 1.5, 3.0]
contrast = 'T1'
C = np.array([110., 170., 40.])            # fixed colors so only the resolution changes
V = np.array([6., 4., 8.])

fig, axes = plt.subplots(len(resolutions), 3, figsize=(11, 3.6 * len(resolutions)))
for row, r in enumerate(resolutions):
    # Same seed -> visible nodes keep their color across rows
    scheme = TreeSampling(labels, tree, seed=0, std_noise=2, verbose=False)
    synth = scheme.sample_image(seg, C, V, resolution=r, contrast=contrast)

    # Map each label to the node whose Gaussian it uses at this resolution
    rep = tree.representative(r, contrast)
    groups = sorted(set(rep[tree.label_to_node[l]] for l in labels))
    node_color = {n: i for i, n in enumerate(tree.order())}      # fixed color per node across rows
    group_map = np.vectorize(lambda l: node_color[rep[tree.label_to_node[l]]])(seg)

    print(f'\n=== r = {r} mm, {contrast}: {len(groups)} distinct Gaussians')
    tree.print_tree(r, contrast)

    axes[row, 0].imshow(group_map, cmap='tab10', vmin=0, vmax=9, interpolation='nearest')
    axes[row, 0].set_title(f'{r} mm: {len(groups)} generation classes')
    axes[row, 1].imshow(synth, cmap='gray', vmin=0, vmax=255)
    axes[row, 1].set_title('synthetic image on 0.33 mm grid')
    axes[row, 2].imshow(acquire(synth, r), cmap='gray', vmin=0, vmax=255)
    axes[row, 2].set_title(f'network input ({r} mm, resampled)')

for ax in axes.ravel():
    ax.axis('off')
fig.tight_layout()

os.makedirs('data/outputs/toy_example', exist_ok=True)
fig.savefig('data/outputs/toy_example/toy_resolutions.png', dpi=90)


# ------------------------------------------------------------------------------------------------ #
# 5. Figure 2: random contrasts at 0.33 mm (shared structure, different appearance)                  #
# ------------------------------------------------------------------------------------------------ #

rng = np.random.default_rng(1)
fig, axes = plt.subplots(1, 4, figsize=(14, 3.8))
for i, ax in enumerate(axes):
    Ci, Vi = principal_colors(rng)
    scheme = TreeSampling(labels, tree, seed=100 + i, std_noise=2, verbose=False)
    ax.imshow(scheme.sample_image(seg, Ci, Vi, resolution=0.33, contrast='T1'), cmap='gray')
    ax.set_title(f'draw {i}: C = {np.round(Ci).astype(int)}')
    ax.axis('off')
fig.tight_layout()
fig.savefig('data/outputs/toy_example/toy_random_draws.png', dpi=90)


# ------------------------------------------------------------------------------------------------ #
# 6. What WP4 would use: merged label sets at a given resolution                                    #
# ------------------------------------------------------------------------------------------------ #

for r, c in [(0.7, 'T1'), (0.7, 'T2'), (1.5, 'T1')]:
    print(f'\nlabel groups at {r} mm, {c}:', tree.label_groups(r, c))




# ------------------------------------------------------------------------------------------------ #
# 7. Plot the tree at a specific resolution and contrast                                             #
# ------------------------------------------------------------------------------------------------ #
import tree_plot
tree_plot.plot_tree('data/inputs/toy_tree.json', resolution=0.7, contrast='T1', save='data/outputs/toy_example/arbol_0.7mm.png')