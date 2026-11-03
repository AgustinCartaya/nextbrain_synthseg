#!/usr/bin/env python3
"""Convert the NextBrain-Allen tree (build_nextbrain_allen_tree.py output) to the
LabelTree JSON format used by the generator (same format as
data/inputs/nextbrain_tree_v1.json).

  level 1   L1_tissue of the leaves: one independent principal color per tissue
  level 2+  the Allen hierarchy, pruned to the leaves of that tissue
  leaves    NextBrain labels (labels = [NextBrain id]), parent = Allen parent

See convert_to_label_tree.md for a full explanation.

Usage
-----
    python convert_to_label_tree.py                     # gyral tree
    python convert_to_label_tree.py --cortex developmental
"""
import argparse
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_OUTDIR = HERE / "data" / "outputs"
LABEL_TREE_PY = HERE.parent / "create_nextbraintree"   # provides tree.LabelTree

NULL_TISSUES = {"Background"}
CONTRASTS = ["T1", "T2", "FLAIR", "PD"]

# Allen nodes above every brain leaf; they never separate labels, so they are skipped.
SKIP_ALLEN = {10153, 10154, 10155}  # NP neural plate, NT neural tube, Br brain

# ---- Random jitter (same illustrative values as _build_nextbrain_tree.py) -------------------------
L1_WR = dict(wr_mu=4.0, wr_sigma=0.5)
GROUP_WR = dict(wr_mu=5.0, wr_sigma=0.5)
LEAF_WR = dict(wr_mu=3.0, wr_sigma=0.5)

# ---- r_max of coarse structures (copied from _build_nextbrain_tree.py) ----------------------------
# Keys are the CSV groups (<snake(value)>_l<level>, level 2 = L2_division, 3 = L3_structure). Each one is
# placed on the Allen node that best matches the leaves of that group (see group_allen_nodes).
GROUP_R_MAX = {
    "thalamus_l2": 3.5,
    "hypothalamus_l2": 1.5,
    "subthalamus_l2": 1.5,
    "basal_ganglia_l2": 3.5,
    "basal_forebrain_l2": 1.0,
    "caudate_l3": 2.0,
    "putamen_l3": 2.0,
    "pallidum_l3": 2.0,
    "accumbens_area_l3": 1.2,
    "cerebral_cortex_l2": 4.0,
    "amygdala_l2": 2.0,
    "hippocampal_formation_l2": 2.0,
    "cerebral_white_matter_l2": 4.0,
    "optic_pathway_l2": 1.5,
    "brain_stem_l3": 4.0,
    "substantia_nigra_l3": {"default": 1.0, "T1": 1.0},
    "cerebellar_cortex_l2": 4.0,
    "cerebellar_nuclei_l2": {"default": 1.0, "T1": 1.0},
}
GROUP_COLS = {2: "L2_division", 3: "L3_structure"}
MIN_JACCARD = 0.5

T1_KEY = "T1"
T1_MISSING = 0.0


def snake(s):
    return re.sub(r"[^0-9a-z]+", "_", s.lower()).strip("_")


def _float(s):
    s = (s or "").strip()
    return float(s) if s else None


def table_r_max(t1w, best):
    """Same rule as _build_nextbrain_tree.py: best = default, empty T1w = 0 for T1."""
    t1w, best = _float(t1w), _float(best)
    if best is None:
        return None if t1w is None else t1w
    t1 = T1_MISSING if t1w is None else t1w
    return best if t1 == best else {"default": best, T1_KEY: t1}


def _combine(rs, op):
    """Per-contrast max / min of r_max values (None = always visible, ignored)."""
    rs = [r for r in rs if r is not None]
    if not rs:
        return None
    keys = {k for r in rs if isinstance(r, dict) for k in r} | {"default"}
    get = lambda r, k: r if not isinstance(r, dict) else r.get(k, r["default"])
    out = {k: op(get(r, k) for r in rs) for k in keys}
    return out["default"] if len(set(out.values())) == 1 else out


# ----------------------------------------------------------------------
#   Split the Allen tree by tissue
# ----------------------------------------------------------------------

class _N:
    def __init__(self, allen=None, leaf=None):
        self.allen, self.leaf = allen, leaf     # allen: internal node dict, leaf: leaf dict
        self.children = []
        self.key = None
        self.r_max = None
        self.up = None                          # value this node passes to its parent's r_max

    def leaves(self):
        if self.leaf is not None:
            return [self]
        return [l for c in self.children for l in c.leaves()]


def split_by_tissue(tree):
    """Return ({tissue: [top _N]}, tissues sorted as in nextbrain_tree_v1.json). Each tissue gets its own copy of the Allen nodes
    above its leaves; the leaf's Allen parent is unchanged."""
    order, tops, copies = [], {}, {}

    def rec(node, path):
        if node["node_type"] == "leaf":
            tissue = node["attributes"]["L1_tissue"]
            if tissue not in tops:
                order.append(tissue)
                tops[tissue] = []
            parent_list = tops[tissue]
            for depth in range(1, len(path) + 1):  # create/reuse the tissue's copies of the path
                k = (tissue,) + tuple(a["allen_label"] for a in path[:depth])
                if k not in copies:
                    copies[k] = _N(allen=path[depth - 1])
                    parent_list.append(copies[k])
                parent_list = copies[k].children
            parent_list.append(_N(leaf=node))
            return
        keep = node["allen_label"] not in SKIP_ALLEN
        for c in node["children"]:
            rec(c, path + [node] if keep else path)

    rec(tree, [])
    return tops, sorted(order)


def walk(n):
    yield n
    for c in n.children:
        yield from walk(c)


def name_nodes(tops):
    """Leaves: snake(leaf name) as in nextbrain_tree_v1.json. Allen nodes: snake(acronym), plus the
    tissue when the same Allen node appears under several tissues."""
    count = {}
    for tissue, nodes in tops.items():
        for top in nodes:
            for n in walk(top):
                if n.allen is not None:
                    count.setdefault(n.allen["allen_label"], set()).add(tissue)
    for tissue, nodes in tops.items():
        for top in nodes:
            for n in walk(top):
                if n.leaf is not None:
                    n.key = snake(n.leaf["name"])
                else:
                    a = n.allen
                    n.key = f"{snake(a['allen_acronym'])}_a{a['allen_label']}"
                    if len(count[a["allen_label"]]) > 1:
                        n.key += f"__{snake(tissue)}"


# ----------------------------------------------------------------------
#   r_max
# ----------------------------------------------------------------------

def group_allen_nodes(tops):
    """GROUP_R_MAX key -> (Allen copy, Jaccard). The CSV groups and the Allen hierarchy do not always
    agree, so each group goes to the Allen node whose leaves overlap it best (Jaccard index of the
    leaf sets, > MIN_JACCARD); on ties the highest node wins (where the structure splits from its
    siblings, as in _build_nextbrain_tree.py)."""
    leaves = [l for nodes in tops.values() for top in nodes for l in top.leaves()]
    out, missing = {}, []
    for gkey in GROUP_R_MAX:
        name, level = gkey.rsplit("_l", 1)
        col = GROUP_COLS[int(level)]
        group = {l.leaf["label"] for l in leaves if snake(l.leaf["attributes"][col]) == name}
        best = None
        for nodes in tops.values():
            for top in nodes:
                for n, depth in walk_depth(top):
                    if n.allen is None:
                        continue
                    labs = {l.leaf["label"] for l in n.leaves()}
                    score = (len(labs & group) / len(labs | group), -depth)
                    if best is None or score > best[0]:
                        best = (score, n)
        if best is None or best[0][0] <= MIN_JACCARD:
            missing.append((gkey, best[0][0] if best else 0.0))
        else:
            out[gkey] = (best[1], best[0][0])
    return out, missing


def walk_depth(n, depth=0):
    yield n, depth
    for c in n.children:
        yield from walk_depth(c, depth + 1)


def fill_r_max(n, overrides):
    """Leaves: L5 r_max; they pass their L4 r_max upwards (the coarse group they belong to).
    Allen nodes: GROUP_R_MAX if mapped, else the coarsest value passed by their children.
    Leaves are then clamped to their parent, as in _build_nextbrain_tree.py."""
    if n.leaf is not None:
        at = n.leaf["attributes"]
        n.r_max = _float(at["L5_r_max_best_mm"])
        l4 = table_r_max(at["L4_r_max_T1w_mm"], at["L4_r_max_best_mm"])
        n.up = l4 if l4 is not None else n.r_max
        return
    for c in n.children:
        fill_r_max(c, overrides)
    n.r_max = overrides.get(id(n), _combine([c.up for c in n.children], max))
    n.up = n.r_max
    for c in n.children:
        if c.leaf is not None and n.r_max is not None:
            c.r_max = _combine([c.r_max, n.r_max], min)


# ----------------------------------------------------------------------
#   LabelTree
# ----------------------------------------------------------------------

def describe(n):
    if n.allen is not None:
        a = n.allen
        return f"Allen {a['allen_label']} {a['allen_acronym']}: {a['allen_name']}"
    lf, at = n.leaf, n.leaf["attributes"]
    extra = ", ".join(at[c] for c in ("functional_class", "layer", "long_axis") if at.get(c))
    allen = (f"Allen {lf['allen_label']} {lf['allen_acronym']}: {lf['allen_name']}"
             if lf["allen_label"] is not None else "no Allen node")
    return f"{lf['name']} | {allen}" + (f" | {extra}" if extra else "")


def to_label_tree(tops, order, LabelTree, mode):
    colors = [t for t in order if t not in NULL_TISSUES]
    t = LabelTree.new(
        colors=colors, contrasts=CONTRASTS, version=f"nextbrain-allen-{mode}-0.1",
        description=(
            "NextBrain v1 leaves (labels = NextBrain id) with the Allen developmental ontology as hierarchy "
            f"({mode} cortex). Level 1 = L1_tissue (one independent color each); below it, the Allen nodes "
            "above that tissue's leaves (NP/NT/Br skipped). Allen node names are <acronym>_a<allen id>, "
            "with __<tissue> when the node is shared by several tissues; descriptions give the Allen id, "
            "acronym and name. Jitters are illustrative; leaf r_max = L5 table value, Allen node r_max = "
            "GROUP_R_MAX on the Allen node that best matches the CSV group, else the coarsest L4 r_max below."))

    def add(n, parent, level):
        kw = dict(LEAF_WR if n.leaf is not None else GROUP_WR, r_max=n.r_max)
        labels = [n.leaf["label"]] if n.leaf is not None else []
        t.add_node(n.key, parent, labels=labels, description=describe(n), **kw)
        for c in n.children:
            add(c, n.key, level + 1)

    for tissue in order:
        key = f"{snake(tissue)}_l1"
        if tissue in NULL_TISSUES:
            t.add_node(key, "root", null=True, description=tissue)
        else:
            t.add_node(key, "root", w_mu=[float(c == tissue) for c in colors],
                       description=tissue, **L1_WR)
        for top in tops[tissue]:
            add(top, key, 2)
    return t


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cortex", choices=["gyral", "developmental"], default="gyral")
    ap.add_argument("--input", type=Path, help="default: data/outputs/nextbrain_allen_tree_<cortex>.json")
    ap.add_argument("--output", type=Path, help="default: data/outputs/nextbrain_allen_label_tree_<cortex>.json")
    args = ap.parse_args()
    src = args.input or DEFAULT_OUTDIR / f"nextbrain_allen_tree_{args.cortex}.json"
    dst = args.output or DEFAULT_OUTDIR / f"nextbrain_allen_label_tree_{args.cortex}.json"

    sys.path.insert(0, str(LABEL_TREE_PY))
    from tree import LabelTree

    tree = json.loads(src.read_text())
    tops, order = split_by_tissue(tree)
    name_nodes(tops)
    keys = [n.key for nodes in tops.values() for top in nodes for n in walk(top)]
    dup = sorted({k for k in keys if keys.count(k) > 1})
    if dup:
        sys.exit(f"Duplicated node names: {dup}")

    groups, missing = group_allen_nodes(tops)
    overrides = {id(n): GROUP_R_MAX[g] for g, (n, _) in groups.items()}
    for nodes in tops.values():
        for top in nodes:
            fill_r_max(top, overrides)

    t = to_label_tree(tops, order, LabelTree, args.cortex)
    t.validate()
    dst.parent.mkdir(parents=True, exist_ok=True)
    t.to_json(dst)

    print(f"{src.name} -> {dst}")
    print(f"  {len(t.nodes)} nodes, {len(t.leaves())} leaves, {len(t.label_to_node)} labels, "
          f"max level {max(n.level for n in t.nodes.values())}")
    print(f"  colors: {t.colors}")
    for g, (n, jac) in groups.items():
        print(f"  GROUP_R_MAX {g:26s} -> {n.key}  (r_max {GROUP_R_MAX[g]}, Jaccard {jac:.2f})")
    for g, jac in missing:
        print(f"  GROUP_R_MAX {g:26s} -> not used (best Jaccard {jac:.2f} <= {MIN_JACCARD})")


if __name__ == "__main__":
    main()
