"""
Allen ontology tree: the nested structure graph of the Allen developmental human brain atlas.

Equivalent of `create_nextbraintree/tree.py` (LabelTree) for the Allen JSON format. There is no
resolution model here, only the hierarchy, so the queries are structural (find, ancestors, subtrees,
pruning).

JSON formats read by `AllenTree.from_json`
------------------------------------------
1. Allen structure graph (data/inputs/AllenBrainOntologyDev.json): one nested root node
    {
      "id": 10153, "acronym": "NP", "name": "neural plate", "color_hex_triplet": "D7D8D8",
      "graph_order": 0, "parent_structure_id": null, ...,
      "children": [ { ...same fields... }, ... ]
    }
2. NextBrain-Allen tree (data/outputs/nextbrain_allen_tree_<cortex>.json, from
   build_nextbrain_allen_tree.py): nested nodes with "node_type" = "allen" | "leaf",
    allen nodes:  {"allen_label", "allen_acronym", "allen_name", "allen_color", "children"}
    leaf nodes:   {"label", "name", "color", "allen_label", "allen_acronym", ..., "attributes"}

Node keys are strings: "A<allen id>" for Allen structures, "N<label>" for NextBrain leaves (the same
ids as the node_id column of nextbrain_allen_tree_<cortex>_nodes.csv).
"""

import json
from collections import deque


# ------------------------------------------------------------------------------------------------ #
# Tree                                                                                             #
# ------------------------------------------------------------------------------------------------ #

class AllenNode(object):
    def __init__(self, key, id, acronym='', name='', color=None, parent=None, children=None,
                 depth=None, kind='allen', extra=None):
        self.key = key
        self.id = id                      # Allen structure id, or NextBrain label for leaves
        self.acronym = acronym or ''
        self.name = name or ''
        self.color = color                # hex 'RRGGBB' or None
        self.parent = parent
        self.children = list(children or [])
        self.depth = depth
        self.kind = kind                  # 'allen' | 'leaf' (NextBrain leaf)
        self.extra = dict(extra or {})    # every other field of the JSON node

    @property
    def rgb(self):
        """Color as (r, g, b) in [0, 1], or None."""
        if not self.color:
            return None
        h = self.color.lstrip('#')
        return tuple(int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))

    def __repr__(self):
        return (f'AllenNode({self.key!r}, {self.acronym!r}, {self.name!r}, depth={self.depth}, '
                f'n_children={len(self.children)})')


class AllenTree(object):
    """Allen hierarchy. Build it with `from_json` / `load`, or derive one with `subtree` / `prune_to`."""

    def __init__(self, nodes, root, description=''):
        self.nodes = {n.key: n for n in nodes} if not isinstance(nodes, dict) else dict(nodes)
        self.root = root
        self.description = description
        self._set_depths()

    # ---- I/O ----------------------------------------------------------------------------------- #

    @classmethod
    def from_dict(cls, d):
        if isinstance(d, list):          # some Allen downloads wrap the root in a list / "msg"
            d = d[0]
        if 'msg' in d:
            d = d['msg'][0]
        nodes = []

        def _rec(spec, parent):
            node = cls._node_from_spec(spec, parent)
            nodes.append(node)
            for c in spec.get('children', []):
                node.children.append(_rec(c, node.key))
            return node.key

        root = _rec(d, None)
        return cls(nodes, root)

    @staticmethod
    def _node_from_spec(spec, parent):
        extra = {k: v for k, v in spec.items() if k != 'children'}
        if 'node_type' in spec:                                  # NextBrain-Allen tree
            if spec['node_type'] == 'leaf':
                return AllenNode(f'N{spec["label"]}', spec['label'], spec.get('allen_acronym') or '',
                                 spec['name'], spec.get('color'), parent, kind='leaf', extra=extra)
            return AllenNode(f'A{spec["allen_label"]}', spec['allen_label'], spec['allen_acronym'],
                             spec['allen_name'], spec.get('allen_color'), parent, extra=extra)
        return AllenNode(f'A{spec["id"]}', spec['id'], spec.get('acronym'), spec.get('name'),
                         spec.get('color_hex_triplet'), parent, extra=extra)

    @classmethod
    def from_json(cls, path):
        with open(path) as f:
            return cls.from_dict(json.load(f))

    @classmethod
    def load(cls, tree):
        if isinstance(tree, AllenTree):
            return tree
        if isinstance(tree, (dict, list)):
            return cls.from_dict(tree)
        return cls.from_json(tree)

    def to_dict(self, name=None):
        """Nested dict in the Allen structure-graph format (extra fields kept)."""
        name = self.root if name is None else name
        n = self.nodes[name]
        d = dict(n.extra)
        d['children'] = [self.to_dict(c) for c in n.children]
        return d

    def to_json(self, path):
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=1)

    # ---- Validation ------------------------------------------------------------------------------ #

    def validate(self):
        errors = []
        if self.root not in self.nodes:
            raise ValueError(f'Root {self.root!r} not found')
        if self.nodes[self.root].parent is not None:
            errors.append(f'Root {self.root!r} must have parent=None')
        for key, n in self.nodes.items():
            if key != self.root and n.parent not in self.nodes:
                errors.append(f'{key}: parent {n.parent!r} does not exist')
            for c in n.children:
                if c not in self.nodes:
                    errors.append(f'{key}: child {c!r} does not exist')
                elif self.nodes[c].parent != key:
                    errors.append(f'{key}: child {c!r} declares parent {self.nodes[c].parent!r}')
            if len(set(n.children)) != len(n.children):
                errors.append(f'{key}: duplicated children')
        if not errors:
            orphans = set(self.nodes) - set(self.order())
            if orphans:
                errors.append(f'Nodes not reachable from root (cycle or orphan): {sorted(orphans)}')
        if errors:
            raise ValueError('Invalid Allen tree:\n  ' + '\n  '.join(errors))
        return True

    # ---- Queries --------------------------------------------------------------------------------- #

    def _set_depths(self):
        queue = deque([(self.root, 0)])
        while queue:
            key, d = queue.popleft()
            self.nodes[key].depth = d
            queue.extend((c, d + 1) for c in self.nodes[key].children)

    def order(self, name=None):
        """Breadth-first order from `name` (default root): every parent comes before its children."""
        out, queue = [], deque([self.root if name is None else name])
        while queue:
            key = queue.popleft()
            out.append(key)
            queue.extend(self.nodes[key].children)
        return out

    def ancestors(self, name):
        out = []
        while self.nodes[name].parent is not None:
            name = self.nodes[name].parent
            out.append(name)
        return out

    def descendants(self, name):
        return self.order(name)[1:]

    def leaves(self, name=None):
        return [k for k in self.order(name) if not self.nodes[k].children]

    @property
    def max_depth(self):
        return max(n.depth for n in self.nodes.values())

    def find(self, query):
        """Node key from a key ('A10390'), Allen id (10390 / '10390'), acronym ('THM') or exact name
        ('thalamus'). Exact case first, then case-insensitive; Allen structures win over NextBrain leaves (which reuse the
        Allen acronym). Raises if 0 or > 1 nodes match."""
        q = str(query).strip()
        if q in self.nodes:
            return q
        if q.isdigit() and f'A{q}' in self.nodes:
            return f'A{q}'
        low = q.lower()
        hits = []
        for field in ('acronym', 'name'):     # Allen structures first (leaves reuse the acronym)
            for exact in (True, False):       # exact case first: 'CB' cerebellum vs 'cb' cingulum
                for kind in ('allen', 'leaf'):
                    hits = hits or [k for k, n in self.nodes.items() if n.kind == kind and
                                    (getattr(n, field) == q if exact else getattr(n, field).lower() == low)]
        if len(hits) != 1:
            what = 'no node' if not hits else f'{len(hits)} nodes ({", ".join(hits[:6])})'
            raise KeyError(f'{query!r} matches {what}')
        return hits[0]

    def search(self, text):
        """Keys of every node whose acronym or name contains `text` (case-insensitive)."""
        low = text.lower()
        return [k for k in self.order() if low in self.nodes[k].name.lower()
                or low in self.nodes[k].acronym.lower()]

    # ---- Derived trees ------------------------------------------------------------------------- #

    def _copy(self, keep, root):
        nodes = {}
        for key in self.order(root):
            if key not in keep:
                continue
            n = self.nodes[key]
            nodes[key] = AllenNode(key, n.id, n.acronym, n.name, n.color,
                                   parent=None if key == root else n.parent,
                                   children=[c for c in n.children if c in keep],
                                   kind=n.kind, extra=n.extra)
        return AllenTree(nodes, root, self.description)

    def subtree(self, name, max_depth=None):
        """Tree rooted at `name`, keeping at most `max_depth` levels below it (None = all)."""
        root = self.find(name)
        base = self.nodes[root].depth
        keep = {k for k in self.order(root)
                if max_depth is None or self.nodes[k].depth - base <= max_depth}
        return self._copy(keep, root)

    def prune_to(self, names):
        """Tree with only `names` and their ancestors (e.g. the structures used by a dataset)."""
        keys = [self.find(n) for n in names]
        keep = set(keys)
        for k in keys:
            keep.update(self.ancestors(k))
        return self._copy(keep, self.root)

    def print_tree(self, name=None, max_depth=None, file=None):
        name = self.root if name is None else self.find(name)
        base = self.nodes[name].depth

        def _rec(key, prefix=''):
            n = self.nodes[key]
            hidden = max_depth is not None and n.depth - base >= max_depth and n.children
            more = f'  (+{len(self.descendants(key))} below)' if hidden else ''
            tag = f'{n.id} {n.name}' if n.kind == 'leaf' else f'{n.acronym} ({n.id}) {n.name}'
            print(f'{prefix}{tag}{more}', file=file)
            if not hidden:
                for c in n.children:
                    _rec(c, prefix + '    ')
        _rec(name)
