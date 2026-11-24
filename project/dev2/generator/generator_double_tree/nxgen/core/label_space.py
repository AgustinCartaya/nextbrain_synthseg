"""
Label space: raw label values <-> output indices.

Inside the generator every label map holds *indices* 0..K-1 (int16), not
raw label values: index 0 is the background and index i is raw value
`values[i]`. This keeps lookup tables small (NextBrain raw values go up to
~2000) and gives the contiguous labels nnU-Net requires.

The label space is built once (from the tree and the templates) and saved
with each dataset as `label_space.json`. Later runs into the same dataset
load it instead of rebuilding it, so an index always means the same label
(v2 rebuilt it every run: problems_to_solve #2).
"""

import json

import numpy as np
import torch


class LabelSpace:
    """
    Raw values and names of the output labels.

    Parameters
    ----------
    values : list of int
        Raw label value of each index. values[0] is the background.
    names : list of str
        Unique name of each index.
    background : int
        Raw value of the background (= values[0]).
    """

    def __init__(self, values, names, background=0):
        if values[0] != background:
            raise ValueError('values[0] must be the background value')
        if len(set(values)) != len(values):
            raise ValueError('duplicated label values')
        if len(values) > np.iinfo(np.int16).max:
            raise ValueError(f'{len(values)} labels: too many for int16')
        self.values = [int(v) for v in values]
        self.names = list(names)
        self.background = int(background)
        self.index = {v: i for i, v in enumerate(self.values)}

    def __len__(self):
        return len(self.values)

    # -- construction -------------------------------------------------------

    @classmethod
    def build(cls, tree, template_values, background=0,
              output_labels='templates', unassigned='error'):
        """
        Label space from the tree and the labels found in the templates.

        Parameters
        ----------
        tree : LabelTree
        template_values : set of int
            Raw values present in the templates.
        background : int
        output_labels : 'templates' or 'tree'
            'templates': labels present in the templates (and in the
            tree); 'tree': every label of the tree.
        unassigned : 'error' or 'background'
            What to do with template labels that are not in the tree.
            'background' maps them to index 0 (see `lut`).
        """
        tree_values = set(tree.label_to_node)
        missing = sorted(set(template_values) - tree_values - {background})
        if missing and unassigned == 'error':
            raise ValueError(
                f'Template labels {missing} are not in the tree. Add them to '
                f"the tree or set labels.unassigned = 'background'.")
        if output_labels == 'tree':
            kept = tree_values
        else:
            kept = set(template_values) & tree_values
        values = [background] + sorted(kept - {background})
        return cls(values, _names(values, tree, background), background)

    # -- conversion -----------------------------------------------------------

    def lut(self, max_value, device=None):
        """
        Lookup table raw value -> index (int16 tensor of size max_value+1).

        Values that are not in the label space map to 0 (background).
        """
        table = torch.zeros(max(max_value, max(self.values)) + 1,
                            dtype=torch.int16)
        table[torch.as_tensor(self.values)] = torch.arange(
            len(self.values), dtype=torch.int16)
        return table.to(device)

    def to_raw(self, indices):
        """Index map (tensor or array) -> raw value map (int64 numpy)."""
        if isinstance(indices, torch.Tensor):
            indices = indices.cpu().numpy()
        return np.asarray(self.values, dtype=np.int64)[np.asarray(indices)]

    def resolve(self, group, tree):
        """
        Label group -> sorted list of indices.

        Parameters
        ----------
        group : str, int or list
            Tree node name (all the labels of its subtree), raw label
            value, or a list of those. Labels absent from the label space
            are ignored.
        tree : LabelTree
        """
        if isinstance(group, (list, tuple)):
            out = set()
            for g in group:
                out |= set(self.resolve(g, tree))
            return sorted(out)
        if isinstance(group, str):
            if group not in tree.nodes:
                raise ValueError(f'Unknown tree node {group!r}')
            raw = []
            for name in [group] + tree.descendants(group):
                raw += tree.nodes[name].labels
        else:
            raw = [int(group)]
        return sorted(self.index[v] for v in raw if v in self.index)

    # -- persistence ----------------------------------------------------------

    def to_dict(self):
        """JSON-serialisable description."""
        return dict(background=self.background, values=self.values,
                    names=self.names)

    @classmethod
    def from_dict(cls, d):
        return cls(d['values'], d['names'], d['background'])

    def save(self, path):
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            return cls.from_dict(json.load(f))


def _names(values, tree, background):
    """Unique name per value: tree node name (+ value if the node is shared)."""
    l2n = tree.label_to_node
    names = []
    for v in values:
        node = l2n.get(v, 'background' if v == background else f'label_{v}')
        shared = node in tree.nodes and len(tree.nodes[node].labels) > 1
        names.append(f'{node}_{v}' if shared else node)
    return names
