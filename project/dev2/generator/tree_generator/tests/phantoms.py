"""
Small synthetic inputs for the tests: a toy label tree and a spherical
"brain" label map, fast enough to run everything on the CPU in seconds.

Phantom (concentric spheres, raw label values):
    0   background            outside
    24  csf                   shell, radius 10-13 voxels
    3   gm                    shell, radius 6-10, half x <  centre
    42  gm_part (child of gm) shell, radius 6-10, half x >= centre
    2   wm                    ball, radius < 6

Tree: colors [GM, WM, CSF]; gm_part is visible only at resolution
<= 1 mm (r_max = 1), so at coarser resolutions it shares gm's Gaussian.
"""

import os
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from nxgen.core.io import save_labels  # noqa: E402

BG, CSF, GM, GM_PART, WM = 0, 24, 3, 42, 2


class Skip(Exception):
    """Raised by a test that cannot run here (missing optional dependency)."""


TREE = {
    'version': 'test', 'description': 'phantom tree',
    'colors': ['GM', 'WM', 'CSF'], 'contrasts': ['T1', 'T2'], 'root': 'root',
    'nodes': {
        'root': {'parent': None, 'children': ['background', 'csf', 'gm', 'wm'],
                 'level': 0, 'labels': []},
        'background': {'parent': 'root', 'children': [], 'level': 1,
                       'labels': [BG], 'null': True},
        'csf': {'parent': 'root', 'children': [], 'level': 1, 'labels': [CSF],
                'w_mu': [0, 0, 1], 'wr_mu': 5.0, 'wr_sigma': 0.5},
        'gm': {'parent': 'root', 'children': ['gm_part'], 'level': 1,
               'labels': [GM], 'w_mu': [1, 0, 0], 'wr_mu': 5.0,
               'wr_sigma': 0.5},
        'gm_part': {'parent': 'gm', 'children': [], 'level': 2,
                    'labels': [GM_PART], 'wr_mu': 20.0, 'wr_sigma': 0.5,
                    'r_max': {'default': 1.0, 'T2': 2.0}},
        'wm': {'parent': 'root', 'children': [], 'level': 1, 'labels': [WM],
               'w_mu': [0, 1, 0], 'wr_mu': 5.0, 'wr_sigma': 0.5},
    },
}


def tree():
    """The phantom LabelTree."""
    from nxgen.core.tree import LabelTree
    return LabelTree.from_dict(TREE)


def sphere_labels(shape=(40, 40, 40), scale=1.0):
    """
    Phantom label map (raw values). `scale` multiplies the radii (use 2 for
    a map at half the voxel size).
    """
    c = (np.asarray(shape) - 1) / 2
    x, y, z = np.meshgrid(*[np.arange(s) for s in shape], indexing='ij')
    r = np.sqrt((x - c[0]) ** 2 + (y - c[1]) ** 2 + (z - c[2]) ** 2) / scale
    lab = np.full(shape, BG, np.int64)
    lab[r < 13] = CSF
    lab[r < 10] = GM
    lab[(r < 10) & (x >= c[0])] = GM_PART
    lab[r < 6] = WM
    return lab


def workdir():
    """A new temporary folder (deleted at exit)."""
    d = tempfile.TemporaryDirectory()
    _KEEP.append(d)
    return d.name


_KEEP = []


def write_inputs(folder, voxel_mm=0.5, shape=(40, 40, 40)):
    """
    Write the phantom template (at `voxel_mm`) and the tree JSON.

    Returns
    -------
    template_path, tree_path
    """
    import json
    affine = np.diag([voxel_mm] * 3 + [1.0])
    tpl = os.path.join(folder, 'phantom.nii.gz')
    save_labels(sphere_labels(shape), affine, tpl)
    tree_path = os.path.join(folder, 'tree.json')
    with open(tree_path, 'w') as f:
        json.dump(TREE, f)
    return tpl, tree_path


def config(tree_path, **sections):
    """
    Config for the phantom: 0.5 mm templates, 1 mm output, everything
    enabled. `sections` override keys, e.g. config(t, motion={'prob': 0}).
    """
    from nxgen.core.config import load_config
    base = dict(
        grid={'target_voxel_mm': 1.0},
        labels={'tree': tree_path},
        morphology={'pairs': [['gm', 'wm'], ['csf', 'gm']], 'prob': 1.0,
                    'n_iter': {'randint': [1, 2]}},
        acquisition={'lowres_prob': 0.5, 'max_iso_mm': 3.0,
                     'max_aniso_mm': 4.0},
        spatial={'elastic_dmax_mm': 2.0},
    )
    for name, values in sections.items():
        base.setdefault(name, {}).update(values)
    return load_config(**base)


def generator(device='cpu', **sections):
    """Generator on the phantom (new temporary files each call)."""
    from nxgen.pipeline import Generator
    tpl, tree_path = write_inputs(workdir())
    return Generator([tpl], config(tree_path, **sections), device=device)
