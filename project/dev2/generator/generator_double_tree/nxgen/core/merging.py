"""
Label levels: the augmented labels merged into coarser label sets.

The generator augments ONE fine label map (morphology + spatial, with the
morphological tree MT). Right after the augmentation (online, on the HR
grid) the fine labels are merged into every *level* of the config, and each
merged map gets its own partial volume -> one ground truth per level. The
principal level is the one the image is synthesised from (with its tree,
the intensity tree IT).

    "merging": {
        "levels": {
            "nxb_all": null,                                   # identity
            "nxb16":   {"script": ".../nxbsimplified_to_nxb16.py",
                        "tree":   ".../nextbrain_tree_nxb16.json"},
            "nxbsupersynth": {"script": "...", "tree": "...",
                              "options": {"side": "R"}}
        },
        "principal": "nxb16"
    }

A merging script must define `label_mapping(**options)` returning
{fine raw label: merged raw label}. It is called once, when the generator
starts; the result is folded into a lookup table *fine index -> level
index* (`Level.lut`), so the online merge is a single gather on the GPU:

    merged = level.lut[fine_hr_labels]

Every label of the fine label space must be in the mapping, and every
merged value must be in the level tree (errors at start-up, never silent).
"""

import hashlib
import importlib.util
import os
import sys
from dataclasses import dataclass

import torch

from .label_space import LabelSpace
from .tree import load_tree


@dataclass
class Level:
    """
    One output label level.

    name : str
    tree : LabelTree of the level (MT for an identity level)
    labels : LabelSpace of the level
    lut : (n_fine,) long tensor, fine index -> level index (None = identity)
    """
    name: str
    tree: object
    labels: LabelSpace
    lut: torch.Tensor = None

    def merge(self, fine):
        """Fine index map (int16 tensor) -> level index map (int16)."""
        if self.lut is None:
            return fine
        return self.lut[fine.long()]


def load_script(path):
    """Import a merging script from its path (cached per absolute path)."""
    path = os.path.abspath(path)
    name = '_nxgen_merge_' + hashlib.sha1(path.encode()).hexdigest()[:12]
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    if not hasattr(module, 'label_mapping'):
        raise ValueError(f'{path}: a merging script must define '
                         f'label_mapping(**options)')
    return module


def build_level(name, spec, fine_tree, fine_labels, output_labels='templates',
                device=None):
    """
    Level from its config entry.

    Parameters
    ----------
    name : str
    spec : None (identity: the fine labels and MT) or dict(script, tree,
        options)
    fine_tree : LabelTree (MT)
    fine_labels : LabelSpace of the fine labels (templates)
    output_labels : 'templates' (merged labels the templates can produce)
        or 'tree' (every label of the level tree)
    """
    if spec is None:
        return Level(name, fine_tree, fine_labels, None)
    mapping = load_script(spec['script']).label_mapping(
        **(spec.get('options') or {}))
    mapping = {int(k): int(v) for k, v in mapping.items()}
    missing = [v for v in fine_labels.values if v not in mapping]
    if missing:
        raise ValueError(f'Level {name!r}: labels {missing} are not in the '
                         f'mapping of {spec["script"]}')
    bg = fine_labels.background
    if mapping[bg] != bg:
        raise ValueError(f'Level {name!r}: the background {bg} must map to '
                         f'itself, got {mapping[bg]}')
    tree = load_tree(spec['tree'])
    merged = [mapping[v] for v in fine_labels.values]
    labels = LabelSpace.build(tree, set(merged), bg, output_labels, 'error')
    lut = torch.tensor([labels.index[m] for m in merged], dtype=torch.int16,
                       device=device)
    return Level(name, tree, labels, lut)


def build_levels(cfg, fine_tree, fine_labels, device=None):
    """
    {name: Level} (principal level last) and the principal name.

    Without levels in the config: one identity level 'all' (= the tree
    generator: MT is also the intensity tree).
    """
    m = cfg['merging']
    specs = m['levels'] or {'all': None}
    principal = m['principal'] or next(iter(specs))
    out_labels = cfg['labels']['output_labels']
    levels = {name: build_level(name, spec, fine_tree, fine_labels,
                                out_labels, device)
              for name, spec in specs.items() if name != principal}
    levels[principal] = build_level(principal, specs[principal], fine_tree,
                                    fine_labels, out_labels, device)
    return levels, principal
