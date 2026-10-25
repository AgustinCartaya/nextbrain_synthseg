"""
Shared setup of the notebooks: paths, device, plotting helpers.

Every notebook starts with `from _setup import *`. This module only defines
constants and small display helpers; nothing here changes how the generator
works.

Device: the GPU with the most free memory, or the CPU. Set the environment
variable NXGEN_DEVICE (e.g. 'cuda:1' or 'cpu') before starting Jupyter to
choose it yourself.
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

NOTEBOOKS = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(NOTEBOOKS)                       # generator_v2_modular
sys.path.insert(0, ROOT)                                # import nxgen
sys.path.insert(0, os.path.join(ROOT, 'tests'))         # import phantoms

CONFIG = os.path.join(ROOT, 'configs', 'nextbrain.json')
CONFIG_PATCH = os.path.join(ROOT, 'configs', 'nextbrain_patch.json')
TREE = os.path.join(ROOT, 'data', 'inputs', 'nextbrain_complete_tree.json')
SOURCE_MAP = os.path.normpath(os.path.join(
    ROOT, '..', 'generator_v2', 'data', 'inputs', 'case-1.nii.gz'))
TEMPLATE_DIR = os.path.join(ROOT, 'data', 'templates')
TEMPLATE = os.path.join(TEMPLATE_DIR, 'case-1_hr0.2mm.nii.gz')
OUTPUTS = os.path.join(ROOT, 'data', 'outputs', 'notebooks')
os.makedirs(OUTPUTS, exist_ok=True)


def pick_device():
    """NXGEN_DEVICE, else the GPU with the most free memory, else 'cpu'."""
    if os.environ.get('NXGEN_DEVICE'):
        return os.environ['NXGEN_DEVICE']
    if not torch.cuda.is_available():
        return 'cpu'
    free = [torch.cuda.mem_get_info(i)[0]
            for i in range(torch.cuda.device_count())]
    return f'cuda:{int(np.argmax(free))}'


DEVICE = pick_device()


def ensure_template():
    """Build the case-1 HR template if it is not there yet (~30 s)."""
    if not os.path.exists(TEMPLATE):
        from nxgen.cli import build_templates
        build_templates.main(['--config', CONFIG, '--seg', SOURCE_MAP,
                              '--out', TEMPLATE_DIR, '--device', DEVICE])
    return TEMPLATE


def label_colors(n, seed=0):
    """(n, 3) random colours, index 0 (background) black."""
    c = np.random.default_rng(seed).uniform(0.15, 1, (n, 3))
    c[0] = 0
    return c


def to_numpy(x):
    """Tensor or array -> numpy array."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def mid(vol, axis=2, index=None):
    """Slice of a 3D volume (middle by default), transposed for display."""
    vol = to_numpy(vol)
    if index is None:
        index = vol.shape[axis] // 2
    return np.take(vol, index, axis=axis).T


def show(vols, titles=None, axis=2, index=None, labels=None, size=3.0,
         cmap='gray', vmin=None, vmax=None, colorbar=False, suptitle=None):
    """
    Show one slice of several volumes side by side.

    vols : list of 3D tensors/arrays (or 2D arrays, shown as they are)
    labels : list of bools, True = label map (shown with random colours);
        default: integer volumes are label maps
    """
    n = len(vols)
    titles = titles or [''] * n
    fig, axes = plt.subplots(1, n, figsize=(size * n, size * 1.1),
                             squeeze=False)
    for i, (v, ax) in enumerate(zip(vols, axes[0])):
        v = to_numpy(v)
        sl = v.T if v.ndim == 2 else mid(v, axis, index)
        is_label = labels[i] if labels else np.issubdtype(v.dtype, np.integer)
        if is_label:
            colors = label_colors(int(v.max()) + 1)
            ax.imshow(colors[sl], origin='lower', interpolation='nearest')
        else:
            im = ax.imshow(sl, cmap=cmap, origin='lower', vmin=vmin,
                           vmax=vmax, interpolation='nearest')
            if colorbar:
                fig.colorbar(im, ax=ax, fraction=0.046)
        ax.set_title(titles[i], fontsize=9)
        ax.axis('off')
    if suptitle:
        fig.suptitle(suptitle)
    fig.tight_layout()
    plt.show()


def gb(nbytes):
    """Bytes -> GB string."""
    return f'{nbytes / 1e9:.2f} GB'
