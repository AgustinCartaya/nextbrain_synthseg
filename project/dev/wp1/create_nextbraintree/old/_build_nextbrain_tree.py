import os
import re
import sys
import csv

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(HERE, '..', '..', 'wp3', 'simple_tree_generator'))
from tree_sampling import LabelTree


ONTOLOGY_CSV = os.path.join(HERE, 'data', 'inputs', 'nextbrain_ontology_v1.csv')
OUTPUT_JSON = os.path.join(HERE, 'nextbrain_tree.json')

LEVEL_COLS = ['L1_tissue', 'L2_division', 'L3_structure', 'L4_subregion']

# ---- Level 1 (tissue classes): weights over the principal colors [GM, WM, CSF] -------------------
# Illustrative values (same spirit as _build_synthseg_tree.py); WP1 will estimate them.
L1_PARAMS = {
    'Background':       dict(null=True),
    'Cerebral GM':      dict(w_mu=[1, 0, 0], wr_mu=4, wr_sigma=0.5),
    'Cerebral WM':      dict(w_mu=[0, 1, 0], wr_mu=4, wr_sigma=0.5),
    'CSF / ventricles': dict(w_mu=[0, 0, 1], wr_mu=4, wr_sigma=0.5),
    'Deep GM':          dict(w_mu=[0.7, 0.3, 0], wr_mu=6, wr_sigma=1),
    'Brainstem':        dict(w_mu=[0.3, 0.7, 0], wr_mu=6, wr_sigma=1),
    'Cerebellar GM':    dict(w_mu=[0.9, 0.1, 0], wr_mu=6, wr_sigma=1),
    'Cerebellar WM':    dict(w_mu=[0.1, 0.9, 0], wr_mu=6, wr_sigma=1),
    'Other':            dict(w_mu=[0.34, 0.33, 0.33], wr_mu=10, wr_sigma=2),
}

# ---- Random jitter of the deeper nodes (illustrative) ---------------------------------------------
GROUP_WR = dict(wr_mu=5, wr_sigma=0.5)
LEAF_WR = dict(wr_mu=3, wr_sigma=0.5)

# ---- r_max of group nodes that the table does not give (the table only has L4 and leaves) ---------
# Illustrative, taken from the SynthSeg tree where the structure exists there. Nodes not listed here
# and without a table value get the coarsest r_max of their children.
GROUP_R_MAX = {
    'thalamus': 3.5,
    'hypothalamus': 1.5,
    'subthalamus': 1.5,
    'basal_ganglia': 3.5,
    'caudate': 2.0,
    'putamen_group': 2.0,
    'pallidum': 2.0,
    'accumbens_area': 1.2,
    'basal_forebrain': 1.0,
    'cerebral_cortex': 4.0,
    'amygdala': 2.0,
    'hippocampus': 2.0,
    'cerebral_white_matter': 4.0,
    'optic_pathway': 1.5,
    'substantia_nigra': {'default': 1.0, 'T1': 0.0},
    'cerebellum_cortex': 4.0,
    'deep_cerebellar_nuclei': {'default': 1.0, 'T1': 0.0},
}

# Table has a T1w r_max and a "best contrast" r_max. The best value is used as default for every
# contrast; T1 gets its own value. An empty T1w cell (with a best value) means "not resolvable in
# T1w" (prior-driven / contrast-specific structures), encoded as r_max 0 for T1.
T1_KEY = 'T1'
T1_MISSING = 0.0


def snake(s):
    return re.sub(r'[^0-9a-z]+', '_', s.lower()).strip('_')


def norm(s):
    return re.sub(r'[^0-9a-z]', '', s.lower())


def _float(s):
    s = (s or '').strip()
    return float(s) if s else None


def table_r_max(t1w, best):
    t1w, best = _float(t1w), _float(best)
    if best is None:
        return None if t1w is None else t1w
    t1 = T1_MISSING if t1w is None else t1w
    return best if t1 == best else {'default': best, T1_KEY: t1}


class _N(object):
    """Intermediate node, collapsed and converted to LabelTree nodes at the end."""

    def __init__(self, name, level, leaf=False):
        self.name, self.level, self.leaf = name, level, leaf
        self.children, self.labels = [], []
        self.r_max = None
        self.description = ''


def read_rows(path):
    with open(path, newline='') as f:
        return list(csv.DictReader(f))


def build_raw(rows):
    """Tree keyed by the path of level names. Repeated consecutive names (Thalamus/Thalamus) collapse.
    Labels of whole-structure rows ('unassigned') go on the group node instead of a leaf."""
    root = _N('root', 0)
    groups = {(): root}
    for row in rows:
        names = [row[c].strip() for c in LEVEL_COLS]
        l4_unassigned = 'unassigned' in names[3].lower()
        if l4_unassigned:
            names = names[:3]
        path, prev = [], None
        for i, name in enumerate(names):
            if not name or (prev is not None and norm(name) == norm(prev)):
                continue
            path.append(name)
            prev = name
            key = tuple(path)
            if key not in groups:
                node = _N(name, len(path))
                node.source_col = LEVEL_COLS[i]
                groups[tuple(path[:-1])].children.append(node)
                groups[key] = node
            if LEVEL_COLS[i] == 'L4_subregion' and not l4_unassigned:
                groups[key].r_max = table_r_max(row['L4_r_max_T1w_mm'], row['L4_r_max_best_mm'])
        parent = groups[tuple(path)]
        # L4 == L3 (no subregion level): the L4 r_max columns describe the merged node.
        if not l4_unassigned and norm(row['L4_subregion']) == norm(row['L3_structure']) and parent.r_max is None:
            parent.r_max = table_r_max(row['L4_r_max_T1w_mm'], row['L4_r_max_best_mm'])

        lab = int(row['allen_id'])
        if l4_unassigned or 'unassigned' in row['functional_class'].lower():
            parent.labels.append(lab)
        else:
            leaf = _N(row['leaf_name'], parent.level + 1, leaf=True)
            leaf.labels = [lab]
            leaf.r_max = _float(row['L5_r_max_best_mm'])
            parts = [row[c] for c in ('functional_class', 'layer', 'long_axis') if row[c]]
            leaf.description = ', '.join(parts)
            parent.children.append(leaf)
    return root


def collapse(node):
    """A node without labels and with a single child is redundant: merge it with the child.
    Level-1 nodes keep their own name (they carry the tissue colors); deeper ones take the child's
    name and keep their own r_max (it is the resolution at which they split from their siblings)."""
    node.children = [collapse(c) for c in node.children]
    if node.level >= 1 and not node.labels and len(node.children) == 1:
        c = node.children[0]
        if node.level == 1:
            node.labels, node.children = c.labels, c.children
            for g in node.children:
                _shift(g, -1)
            return node
        c.r_max = node.r_max if node.r_max is not None else c.r_max
        _shift(c, -1)
        return c
    return node


def _shift(node, d):
    node.level += d
    for c in node.children:
        _shift(c, d)


def _coarsest(rs):
    rs = [r for r in rs if r is not None]
    if not rs:
        return None
    keys = {k for r in rs if isinstance(r, dict) for k in r} | {'default'}
    get = lambda r, k: r if not isinstance(r, dict) else r.get(k, r['default'])
    out = {k: max(get(r, k) for r in rs) for k in keys}
    return out['default'] if len(set(out.values())) == 1 else out


def assign_names(root):
    """snake_case names; a group whose name is also a leaf name gets a '_group' suffix."""
    nodes = []
    stack = [root]
    while stack:
        n = stack.pop()
        nodes.append(n)
        stack.extend(n.children)
    leaf_names = {snake(n.name) for n in nodes if n.leaf}
    for n in nodes:
        n.key = snake(n.name) if n is not root else 'root'
        if not n.leaf and n is not root and n.key in leaf_names:
            n.key += '_group'
    keys = [n.key for n in nodes]
    dup = {k for k in keys if keys.count(k) > 1}
    if dup:
        raise ValueError(f'Duplicated node names: {sorted(dup)}')


def _finest(a, b):
    """Per-contrast min of two r_max (None = always visible)."""
    if a is None or b is None:
        return b if a is None else a
    keys = {k for r in (a, b) if isinstance(r, dict) for k in r} | {'default'}
    get = lambda r, k: r if not isinstance(r, dict) else r.get(k, r['default'])
    out = {k: min(get(a, k), get(b, k)) for k in keys}
    return out['default'] if len(set(out.values())) == 1 else out


def fill_group_r_max(node):
    for c in node.children:
        fill_group_r_max(c)
    if node.level >= 2 and not node.leaf:
        if node.key in GROUP_R_MAX:
            node.r_max = GROUP_R_MAX[node.key]
        elif node.r_max is None:
            node.r_max = _coarsest([c.r_max for c in node.children])


def clamp_leaf_r_max(node):
    """Leaves (L5 column) can not be seen beyond their parent's limit, e.g. T1 = 0 for prior-driven groups."""
    for c in node.children:
        if c.leaf and node.level >= 2:
            c.r_max = _finest(c.r_max, node.r_max)
        clamp_leaf_r_max(c)


def _all_keys(node):
    yield node.key
    for c in node.children:
        yield from _all_keys(c)


def to_label_tree(root):
    t = LabelTree.new(colors=['GM', 'WM', 'CSF'], contrasts=['T1', 'T2', 'FLAIR', 'PD'], version='nextbrain-0.1',
                      description='NextBrain labels (Allen ids) from nextbrain_ontology_v1.csv. Left and right '
                                  'share a node. Level-1 colors, jitters and group r_max are illustrative; '
                                  'L4 and leaf r_max come from the table (best contrast = default, T1 = T1w).')

    def add(n, parent):
        if n.level == 1:
            p = L1_PARAMS[n.name]
            kw = dict(p)
        else:
            kw = dict(LEAF_WR if n.leaf else GROUP_WR, r_max=n.r_max)
        t.add_node(n.key, parent, labels=sorted(n.labels), description=n.description, **kw)
        for c in n.children:
            add(c, n.key)

    for c in root.children:
        add(c, 'root')
    return t


rows = read_rows(ONTOLOGY_CSV)
root = build_raw(rows)
root.children = [collapse(c) for c in root.children]
assign_names(root)
fill_group_r_max(root)
clamp_leaf_r_max(root)
missing = set(GROUP_R_MAX) - {n for n in _all_keys(root)}
if missing:
    raise ValueError(f'GROUP_R_MAX names not in the tree: {sorted(missing)}')
t = to_label_tree(root)

t.validate()
t.to_json(OUTPUT_JSON)
print(len(t.nodes), 'nodes,', len(t.leaves()), 'leaves,', len(t.label_to_node), 'labels')
