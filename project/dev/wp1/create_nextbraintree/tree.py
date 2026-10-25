"""
Label tree: the ontology that drives the intensity model.

Copied from `generator_v2/tree_sampling.py` (classes `Node` and `LabelTree`,
unchanged). The sampling part of v2 (`TreeSampling`) is replaced by
`nxgen.online.tree_intensities`, which draws from an explicit random
generator instead of a private one.

JSON format
-----------
    {
      "version": "...",
      "colors":    ["GM", "WM", "CSF"],          # N principal colors; weights follow this order
      "contrasts": ["T1", "T2", ...],            # optional, validation of the r_max keys
      "root": "root",
      "nodes": {
        "<name>": {
          "parent":   "<name>" | null,           # null only for the root
          "children": ["<name>", ...],           # must agree with the children's "parent"
          "level":    0 | 1 | 2 | ...,           # root = 0, must equal parent level + 1
          "labels":   [int, ...],                # label-map values painted with this node's Gaussian
          "w_mu":     [w1, ..., wN],             # level 1 only: weights of the principal colors (sum = 1)
          "w_sigma":  [w1, ..., wN],             # level 1 only: weights of the principal sigmas (default = w_mu)
          "wr_mu":    float,                     # weight of the random component of the mean
          "wr_sigma": float,                     # weight of the random component of the sigma
          "r_max":    null | float | {"default": float, "T1": float, ...},
                                                 # coarsest voxel size (mm) at which the node is visible
          "null":     bool,                      # null color (mean = sigma = 0), inherited by descendants
          "description": "..."
        }
      }
    }

Visibility: a node of level >= 2 is visible at resolution r and contrast c
iff r <= r_max(c) and its parent is visible. Level-1 nodes are always
visible. A node that is not visible uses the Gaussian of its deepest visible
ancestor (its "representative"); all labels with the same representative
form one *generation class*.
"""

import json
import warnings
from collections import deque

import numpy as np

INF = float('inf')


# ------------------------------------------------------------------------------------------------ #
# Helpers                                                                                          #
# ------------------------------------------------------------------------------------------------ #

def _parse_r_max(r):
    """null -> always visible; number -> same for every contrast; dict -> per contrast (+ 'default')."""
    if r is None:
        return {'default': INF}
    if isinstance(r, (int, float)):
        return {'default': float(r)}
    out = {k: (INF if v is None else float(v)) for k, v in r.items()}
    out.setdefault('default', INF)
    return out


def _dump_r_max(r):
    clean = {k: (None if v == INF else v) for k, v in r.items()}
    return clean['default'] if list(clean) == ['default'] else clean


def _as_resolution(resolution):
    """Scalar voxel size in mm. Anisotropic spacings use the coarsest axis (conservative)."""
    if resolution is None:
        return 0.0
    return float(np.max(np.atleast_1d(resolution)))


# ------------------------------------------------------------------------------------------------ #
# Tree                                                                                             #
# ------------------------------------------------------------------------------------------------ #

class Node(object):
    FIELDS = ('parent', 'children', 'level', 'labels', 'w_mu', 'w_sigma',
              'wr_mu', 'wr_sigma', 'r_max', 'null', 'description')

    def __init__(self, name, parent=None, children=None, level=None, labels=None, w_mu=None,
                 w_sigma=None, wr_mu=0.0, wr_sigma=0.0, r_max=None, null=False, description=''):
        self.name = name
        self.parent = parent
        self.children = list(children or [])
        self.level = level
        self.labels = [int(l) for l in (labels or [])]
        self.w_mu = None if w_mu is None else [float(w) for w in w_mu]
        self.w_sigma = None if w_sigma is None else [float(w) for w in w_sigma]
        self.wr_mu = float(wr_mu)
        self.wr_sigma = float(wr_sigma)
        self.r_max = _parse_r_max(r_max)
        self.null = bool(null)
        self.description = description

    def r_max_for(self, contrast=None):
        return self.r_max.get(contrast, self.r_max['default'])

    def is_resolvable(self, resolution, contrast=None):
        return resolution <= self.r_max_for(contrast)

    def to_dict(self):
        d = {'parent': self.parent, 'children': self.children, 'level': self.level,
             'labels': self.labels, 'wr_mu': self.wr_mu, 'wr_sigma': self.wr_sigma,
             'r_max': _dump_r_max(self.r_max)}
        if self.w_mu is not None:
            d['w_mu'] = self.w_mu
        if self.w_sigma is not None:
            d['w_sigma'] = self.w_sigma
        if self.null:
            d['null'] = True
        if self.description:
            d['description'] = self.description
        return d

    def __repr__(self):
        return f'Node({self.name!r}, level={self.level}, parent={self.parent!r}, n_children={len(self.children)})'


class LabelTree(object):
    """Resolution-aware label ontology. Build it from JSON or programmatically with `add_node`."""

    def __init__(self, colors, nodes, root='root', contrasts=None, version=None, description='',
                 validate=True):
        self.colors = list(colors)
        self.n_colors = len(self.colors)
        self.contrasts = list(contrasts) if contrasts else None
        self.version = version
        self.description = description
        self.root = root
        self.nodes = {n.name: n for n in nodes} if not isinstance(nodes, dict) else dict(nodes)
        if validate:
            self.validate()

    # ---- I/O ----------------------------------------------------------------------------------- #

    @classmethod
    def from_dict(cls, d):
        unknown = {name: set(spec) - set(Node.FIELDS) for name, spec in d['nodes'].items()}
        unknown = {k: v for k, v in unknown.items() if v}
        if unknown:
            raise ValueError(f'Unknown node fields: {unknown}')
        nodes = [Node(name, **spec) for name, spec in d['nodes'].items()]
        return cls(colors=d['colors'], nodes=nodes, root=d.get('root', 'root'),
                   contrasts=d.get('contrasts'), version=d.get('version'),
                   description=d.get('description', ''))

    @classmethod
    def from_json(cls, path):
        with open(path) as f:
            return cls.from_dict(json.load(f))

    @classmethod
    def load(cls, tree):
        if isinstance(tree, LabelTree):
            return tree
        if isinstance(tree, dict):
            return cls.from_dict(tree)
        return cls.from_json(tree)

    def to_dict(self):
        return {'version': self.version, 'description': self.description, 'colors': self.colors,
                'contrasts': self.contrasts, 'root': self.root,
                'nodes': {name: self.nodes[name].to_dict() for name in self.order()}}

    def to_json(self, path):
        """Readable JSON: one field per line, lists kept on a single line."""
        d = self.to_dict()
        c = lambda v: json.dumps(v)
        lines = ['{']
        for key in ('version', 'description', 'colors', 'contrasts', 'root'):
            lines.append(f'  {c(key)}: {c(d[key])},')
        lines.append('  "nodes": {')
        names = list(d['nodes'])
        for i, name in enumerate(names):
            fields = [f'      {c(k)}: {c(v)}' for k, v in d['nodes'][name].items()]
            lines.append(f'    {c(name)}: {{\n' + ',\n'.join(fields) + '\n    }' + (',' if i < len(names) - 1 else ''))
        lines += ['  }', '}']
        with open(path, 'w') as f:
            f.write('\n'.join(lines) + '\n')

    # ---- Programmatic construction --------------------------------------------------------------- #

    @classmethod
    def new(cls, colors, root='root', **kwargs):
        return cls(colors, [Node(root, level=0)], root=root, validate=False, **kwargs)

    def add_node(self, name, parent, **kwargs):
        """Adds `name` under `parent`, setting level and the parent's children list. Returns the node."""
        if name in self.nodes:
            raise ValueError(f'Node {name!r} already exists')
        p = self.nodes[parent]
        node = Node(name, parent=parent, level=p.level + 1, **kwargs)
        self.nodes[name] = node
        p.children.append(name)
        return node

    # ---- Validation ------------------------------------------------------------------------------ #

    def validate(self, tol=1e-3):
        errors, warns = [], []

        if self.root not in self.nodes:
            raise ValueError(f'Root {self.root!r} not found')
        r = self.nodes[self.root]
        if r.parent is not None or r.level != 0:
            errors.append(f'Root {self.root!r} must have parent=null and level=0')
        if r.labels or r.w_mu is not None:
            errors.append('Root carries no labels or colors; put them on level-1 nodes')

        seen_labels = {}
        for name, n in self.nodes.items():
            # Parent / children consistency
            if name != self.root:
                if n.parent not in self.nodes:
                    errors.append(f'{name}: parent {n.parent!r} does not exist')
                    continue
                p = self.nodes[n.parent]
                if name not in p.children:
                    errors.append(f'{name}: not listed in children of its parent {n.parent!r}')
                if n.level != p.level + 1:
                    errors.append(f'{name}: level {n.level} but parent {n.parent!r} has level {p.level}')
            for c in n.children:
                if c not in self.nodes:
                    errors.append(f'{name}: child {c!r} does not exist')
                elif self.nodes[c].parent != name:
                    errors.append(f'{name}: child {c!r} declares parent {self.nodes[c].parent!r}')
            if len(set(n.children)) != len(n.children):
                errors.append(f'{name}: duplicated children')

            # Color weights
            if n.level == 1 and not n.null:
                for key in ('w_mu', 'w_sigma'):
                    w = getattr(n, key)
                    if w is None:
                        if key == 'w_mu':
                            errors.append(f'{name}: level-1 nodes need w_mu ({self.n_colors} weights)')
                        continue
                    if len(w) != self.n_colors:
                        errors.append(f'{name}: {key} has {len(w)} weights, expected {self.n_colors} {self.colors}')
                    elif min(w) < 0 or abs(sum(w) - 1) > tol:
                        errors.append(f'{name}: {key} must be non-negative and sum to 1 (sum={sum(w):.4f})')
            elif n.level is not None and n.level >= 2 and (n.w_mu is not None or n.w_sigma is not None):
                errors.append(f'{name}: level >= 2 nodes inherit from the parent; remove w_mu / w_sigma')

            if n.wr_mu < 0 or n.wr_sigma < 0:
                errors.append(f'{name}: wr_mu and wr_sigma must be >= 0')

            # Resolution
            if self.contrasts:
                bad = set(n.r_max) - set(self.contrasts) - {'default'}
                if bad:
                    errors.append(f'{name}: r_max uses unknown contrasts {sorted(bad)}')
            if n.level == 1 and any(v != INF for v in n.r_max.values()):
                warns.append(f'{name}: r_max on level-1 nodes is ignored (always visible). '
                             f'Group structures that merge at coarse resolution under a common parent.')
            if n.level is not None and n.level >= 2 and n.parent in self.nodes:
                p = self.nodes[n.parent]
                if p.level >= 2:
                    for c in set(n.r_max) | set(p.r_max):
                        if n.r_max_for(c) > p.r_max_for(c):
                            warns.append(f'{name}: r_max[{c}]={n.r_max_for(c)} > parent r_max '
                                         f'{p.r_max_for(c)}; it can never be seen beyond the parent limit')

            # Labels
            for lab in n.labels:
                if lab in seen_labels:
                    errors.append(f'Label {lab} assigned to both {seen_labels[lab]!r} and {name!r}')
                seen_labels[lab] = name

        # Reachability (no cycles, no orphan subtrees)
        if not errors:
            reached = set(self.order())
            orphans = set(self.nodes) - reached
            if orphans:
                errors.append(f'Nodes not reachable from root (cycle or orphan): {sorted(orphans)}')

        for w in warns:
            warnings.warn(w)
        if errors:
            raise ValueError('Invalid label tree:\n  ' + '\n  '.join(errors))
        return True

    # ---- Queries --------------------------------------------------------------------------------- #

    def order(self):
        """Breadth-first order: every parent comes before its children."""
        out, queue = [], deque([self.root])
        while queue:
            name = queue.popleft()
            out.append(name)
            queue.extend(self.nodes[name].children)
        return out

    def ancestors(self, name):
        out = []
        while self.nodes[name].parent is not None:
            name = self.nodes[name].parent
            out.append(name)
        return out

    def descendants(self, name):
        out, queue = [], deque(self.nodes[name].children)
        while queue:
            c = queue.popleft()
            out.append(c)
            queue.extend(self.nodes[c].children)
        return out

    def leaves(self):
        return [n for n in self.order() if not self.nodes[n].children]

    @property
    def label_to_node(self):
        return {lab: name for name, n in self.nodes.items() for lab in n.labels}

    def visibility(self, resolution=None, contrast=None):
        """{node: bool}. A node is visible iff it is resolvable and its parent is visible."""
        res = _as_resolution(resolution)
        vis = {}
        for name in self.order():
            n = self.nodes[name]
            if n.level <= 1:
                vis[name] = True
            else:
                vis[name] = vis[n.parent] and n.is_resolvable(res, contrast)
        return vis

    def representative(self, resolution=None, contrast=None):
        """{node: deepest visible node whose Gaussian it uses} (itself when visible)."""
        vis = self.visibility(resolution, contrast)
        rep = {}
        for name in self.order():
            n = self.nodes[name]
            rep[name] = name if vis[name] else rep[n.parent]
        return rep

    def label_groups(self, resolution=None, contrast=None):
        """{visible node: [labels]}: labels that share one Gaussian (and one loss target in WP4)."""
        rep = self.representative(resolution, contrast)
        groups = {}
        for lab, name in sorted(self.label_to_node.items()):
            groups.setdefault(rep[name], []).append(lab)
        return groups

    def print_tree(self, resolution=None, contrast=None):
        vis = self.visibility(resolution, contrast)
        rep = self.representative(resolution, contrast)

        def _rec(name, prefix=''):
            n = self.nodes[name]
            state = 'visible' if vis[name] else f'-> {rep[name]}'
            r = n.r_max_for(contrast)
            r = 'inf' if r == INF else f'{r:g}'
            labs = f' labels={n.labels}' if n.labels else ''
            print(f'{prefix}{name} [L{n.level}, r_max={r}, {state}]{labs}')
            for c in n.children:
                _rec(c, prefix + '    ')
        _rec(self.root)
