#!/usr/bin/env python3
"""Build a tree whose leaves are the NextBrain v1 labels and whose internal
nodes (and therefore each leaf's parent) come from the Allen developmental
ontology.

See build_nextbrain_allen_tree.md for a full explanation.

Usage
-----
    python build_nextbrain_allen_tree.py                    # default paths, gyral cortex
    python build_nextbrain_allen_tree.py --cortex developmental
"""
import argparse
import csv
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_LEAVES = Path("/home/agustin/phd/synthseg_nextbrain/data/basics/nextbrain_ontology_v1.csv")
DEFAULT_ONTOLOGY = HERE / "data" / "inputs" / "AllenBrainOntologyDev.json"
DEFAULT_LUT = HERE / "nextbrain-utils" / "nextbrain_utils" / "lut" / "NextBrainLUT.txt"
DEFAULT_OUTDIR = HERE / "data" / "outputs"

BACKGROUND_LABEL = 0

# ----------------------------------------------------------------------
#   Hand-made decisions
# ----------------------------------------------------------------------

# Leaf names that match more than one Allen node. The CSV context (L2/L4)
# says which one is meant.
AMBIGUOUS = {
    # CSV: Brainstem / Midbrain  -> midbrain DR, not the pontine DR-p (12472)
    "dorsal_raphe_nucleus": 12223,
    # CSV: Amygdala / Basolateral group -> amygdala La, not the pontine PNl (12410)
    "lateral_nucleus": 10367,
}

# Leaf names with no Allen node of the same (normalised) name.
# label -> (Allen id, reason)
MANUAL = {
    842: (12338, "same structure; Allen spells it 'fibers' instead of 'fibres'"),
    844: (12239, "label covers reticular formation + cuneiform area; Allen 'midbrain reticular "
                 "formation' is the node containing the cuneiform nucleus (CSV L4: Midbrain)"),
    827: (266441721, "Allen does not split pontine corticopontine fibres by cortical origin; "
                     "'cortico-pontine fibers, pontine part' (CSV L4: Pons)"),
    829: (266441721, "same as 827: frontopontine fibres in the pons belong to "
                     "'cortico-pontine fibers, pontine part' (CSV L4: Pons)"),
    828: (12774, "corticonuclear = corticobulbar; Allen hindbrain 'corticobulbar tract' (CSV L4: Pons)"),
    846: (10668, "Allen has no cerebellar white-matter node; 'white matter of hindbrain' is the "
                 "closest (it is also the old NextBrain name of this label)"),
    811: (10409, "NextBrain-specific pulvinar subdivision; attached to Allen 'pulvinar of thalamus'"),
    813: (10409, "NextBrain-specific pulvinar subdivision; attached to Allen 'pulvinar of thalamus'"),
}

# DK cortical parcels -> Allen node name. Copied from nextbrain-utils
# (to_allen._fscortex_to_allen_gyral / _dev), with "Heschl's" lowercased so
# that it matches the normalised Allen name (bug in nextbrain-utils).
CORTEX_GYRAL = {
    "unknown": "cerebral gyri and lobules",
    "bankssts": "cerebral gyri and lobules",
    "caudalanteriorcingulate": "cingulate gyrus caudal (posterior) part",
    "caudalmiddlefrontal": "middle frontal gyrus",
    "cuneus": "cuneus",
    "entorhinal": "anterior parahippocampal gyrus",
    "fusiform": "occipitotemporal (fusiform) gyrus occipital part",
    "inferiorparietal": "inferior parietal lobule",
    "inferiortemporal": "inferior temporal gyrus",
    "isthmuscingulate": "cingulate gyrus caudal (posterior) part",
    "lateraloccipital": "occipital lobe",
    "lateralorbitofrontal": "lateral orbital gyrus",
    "lingual": "lingual gyrus",
    "medialorbitofrontal": "gyrus rectus (straight gyrus)",
    "middletemporal": "middle temporal gyrus",
    "parahippocampal": "posterior parahippocampal gyrus",
    "paracentral": "paracentral lobule rostral part",
    "parsopercularis": "inferior frontal gyrus opercular part",
    "parsorbitalis": "inferior frontal gyrus orbital part",
    "parstriangularis": "inferior frontal gyrus triangular part",
    "pericalcarine": "occipital lobe",
    "postcentral": "postcentral gyrus",
    "posteriorcingulate": "cingulate gyrus caudal (posterior) part",
    "precentral": "precentral gyrus",
    "precuneus": "precuneus",
    "rostralanteriorcingulate": "cingulate gyrus rostral (anterior) part",
    "rostralmiddlefrontal": "middle frontal gyrus",
    "superiorfrontal": "superior frontal gyrus",
    "superiorparietal": "supraparietal lobule",
    "superiortemporal": "superior temporal gyrus",
    "supramarginal": "supramarginal gyrus",
    "frontalpole": "frontal pole",
    "temporalpole": "temporal pole",
    "transversetemporal": "transverse temporal gyrus (heschl's gyrus)",
    "insula": "insular lobe",
}
CORTEX_DEV = {
    "unknown": "neocortex (isocortex)",
    "bankssts": "temporal neocortex",
    "caudalanteriorcingulate": "cingulate neocortex",
    "caudalmiddlefrontal": "frontal neocortex",
    "cuneus": "occipital neocortex",
    "entorhinal": "periarchicortex",
    "fusiform": "temporal neocortex",
    "inferiorparietal": "parietal neocortex",
    "inferiortemporal": "temporal neocortex",
    "isthmuscingulate": "cingulate neocortex",
    "lateraloccipital": "occipital neocortex",
    "lateralorbitofrontal": "frontal neocortex",
    "lingual": "occipital neocortex",
    "medialorbitofrontal": "frontal neocortex",
    "middletemporal": "temporal neocortex",
    "parahippocampal": "periarchicortex",
    "paracentral": "frontal neocortex",
    "parsopercularis": "frontal neocortex",
    "parsorbitalis": "frontal neocortex",
    "parstriangularis": "frontal neocortex",
    "pericalcarine": "occipital neocortex",
    "postcentral": "parietal neocortex",
    "posteriorcingulate": "cingulate neocortex",
    "precentral": "frontal neocortex",
    "precuneus": "parietal neocortex",
    "rostralanteriorcingulate": "cingulate neocortex",
    "rostralmiddlefrontal": "frontal neocortex",
    "superiorfrontal": "frontal neocortex",
    "superiorparietal": "parietal neocortex",
    "superiortemporal": "temporal neocortex",
    "supramarginal": "parietal neocortex",
    "frontalpole": "frontal neocortex",
    "temporalpole": "temporal neocortex",
    "transversetemporal": "temporal neocortex",
    "insula": "insular neocortex",
}
CORTEX_TABLES = {"gyral": CORTEX_GYRAL, "developmental": CORTEX_DEV}


# ----------------------------------------------------------------------
#   Helpers
# ----------------------------------------------------------------------

def normalize_name(name):
    """Same normalisation as nextbrain_utils.to_allen.normalize_name."""
    name = name.lower().strip()
    if name.startswith("ctx-"):
        name = name[4:]
    if name.startswith(("lh-", "rh-")):
        name = name[3:]
    name = name.replace(",", "")
    name = name.replace("-", " ").replace("_", " ")
    return re.sub(r"\s+", " ", name).strip()


def is_cortex(label):
    return 1000 <= label < 3000


def load_leaves(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    # NOTE: the CSV column "allen_id" holds the NextBrain label ID.
    for r in rows:
        r["label"] = int(r["allen_id"])
    return rows


def load_lut_colors(path):
    colors = {}
    with open(path) as f:
        for line in f:
            line = line.split("#")[0].strip()
            if not line:
                continue
            label, _name, r, g, b, *_ = line.split()
            colors[int(label)] = f"{int(r):02X}{int(g):02X}{int(b):02X}"
    return colors


def index_ontology(root):
    """Return {id: node}, {id: parent_id} and {normalised name: [ids]}."""
    by_id, parent, by_name = {}, {}, {}

    def walk(node, par):
        by_id[node["id"]] = node
        parent[node["id"]] = par
        by_name.setdefault(normalize_name(node["name"]), []).append(node["id"])
        for child in node.get("children", []):
            walk(child, node["id"])

    walk(root, None)
    return by_id, parent, by_name


def ancestors(node_id, parent):
    """Allen ancestors of node_id, nearest first (node_id excluded)."""
    out = []
    node_id = parent[node_id]
    while node_id is not None:
        out.append(node_id)
        node_id = parent[node_id]
    return out


# ----------------------------------------------------------------------
#   Step 1: match every leaf to an Allen node
# ----------------------------------------------------------------------

def match_leaves(leaves, by_name, cortex_table):
    """Add 'allen_id' (int or None) and 'match' (how it was found) to each leaf."""
    problems = []
    for leaf in leaves:
        label, name = leaf["label"], leaf["leaf_name"]
        if label == BACKGROUND_LABEL:
            leaf["allen_match"], leaf["match"] = None, "none (background)"
            continue

        if is_cortex(label):
            key = normalize_name(name)
            target = cortex_table.get(key)
            ids = by_name.get(normalize_name(target), []) if target else []
            if len(ids) != 1:
                problems.append(f"cortex {label} {name}: target {target!r} -> {ids}")
                continue
            leaf["allen_match"], leaf["match"] = ids[0], "cortex table"
            continue

        if label in MANUAL:
            leaf["allen_match"], leaf["match"] = MANUAL[label][0], "manual: " + MANUAL[label][1]
            continue

        ids = by_name.get(normalize_name(name), [])
        if len(ids) == 1:
            leaf["allen_match"], leaf["match"] = ids[0], "exact name"
        elif len(ids) > 1 and name in AMBIGUOUS:
            assert AMBIGUOUS[name] in ids, (name, ids)
            leaf["allen_match"] = AMBIGUOUS[name]
            leaf["match"] = f"name (ambiguous {ids}, resolved with CSV context)"
        else:
            problems.append(f"{label} {name}: Allen candidates {ids}")

    if problems:
        raise SystemExit("Unresolved leaves - add them to AMBIGUOUS/MANUAL:\n  " + "\n  ".join(problems))


# ----------------------------------------------------------------------
#   Step 2: decide each leaf's parent and build the pruned tree
# ----------------------------------------------------------------------

def allen_fields(node):
    return {
        "allen_label": node["id"],
        "allen_name": node["name"],
        "allen_acronym": node["acronym"],
        "allen_color": node["color_hex_triplet"],
    }


def build_tree(leaves, by_id, parent, root_id, lut_colors):
    matched = [l for l in leaves if l["allen_match"] is not None]

    # Allen nodes that are a strict ancestor of some leaf's Allen node.
    # Such a node must stay an internal node of the new tree.
    internal_needed = set()
    for leaf in matched:
        internal_needed.update(ancestors(leaf["allen_match"], parent))

    # Parent rule (see the .md):
    #   - matched node X is itself an ancestor of another leaf -> leaf hangs under X
    #   - otherwise the leaf takes X's place                   -> leaf hangs under parent(X)
    for leaf in leaves:
        x = leaf["allen_match"]
        if x is None:
            leaf["parent_allen"], leaf["parent_rule"] = root_id, "attached to root (no Allen match)"
        elif x in internal_needed:
            leaf["parent_allen"], leaf["parent_rule"] = x, "under matched node (it has other leaves below)"
        else:
            leaf["parent_allen"], leaf["parent_rule"] = parent[x], "replaces matched node (Allen parent)"

    # Keep every Allen node on the path from the root to a leaf's parent.
    keep = {root_id}
    for leaf in leaves:
        p = leaf["parent_allen"]
        keep.add(p)
        keep.update(ancestors(p, parent))

    nodes = {aid: {"node_type": "allen", **allen_fields(by_id[aid]), "children": []} for aid in keep}

    # Attach Allen nodes in their original (Allen) order.
    def link(allen_node):
        for child in allen_node.get("children", []):
            if child["id"] in keep:
                nodes[allen_node["id"]]["children"].append(nodes[child["id"]])
                link(child)

    link(by_id[root_id])

    csv_cols = [c for c in leaves[0] if c not in
                ("allen_id", "label", "leaf_name", "allen_match", "match", "parent_allen", "parent_rule")]
    for leaf in leaves:
        x = leaf["allen_match"]
        node = {
            "node_type": "leaf",
            "label": leaf["label"],
            "name": leaf["leaf_name"],
            "color": lut_colors.get(leaf["label"]),
            **(allen_fields(by_id[x]) if x is not None else
               {"allen_label": None, "allen_name": None, "allen_acronym": None, "allen_color": None}),
            "match": leaf["match"],
            "parent_rule": leaf["parent_rule"],
            "attributes": {c: leaf[c] for c in csv_cols},
            "children": [],
        }
        nodes[leaf["parent_allen"]]["children"].append(node)

    return nodes[root_id]


# ----------------------------------------------------------------------
#   Step 3: checks and outputs
# ----------------------------------------------------------------------

def iter_nodes(node, depth=0, parent=None, path=()):
    path = path + (node.get("allen_acronym") or node.get("name") or "?",)
    yield node, depth, parent, path
    for child in node["children"]:
        yield from iter_nodes(child, depth + 1, node, path)


def node_id(node):
    return f"N{node['label']}" if node["node_type"] == "leaf" else f"A{node['allen_label']}"


def validate(tree, leaves):
    all_nodes = list(iter_nodes(tree))
    tree_leaves = [n for n, *_ in all_nodes if n["node_type"] == "leaf"]
    labels = sorted(n["label"] for n in tree_leaves)
    assert labels == sorted(l["label"] for l in leaves), "leaf set differs from the CSV"
    assert len(labels) == len(set(labels)), "a leaf appears twice"
    assert all(not n["children"] for n in tree_leaves), "a CSV leaf has children"
    ids = [node_id(n) for n, *_ in all_nodes]
    assert len(ids) == len(set(ids)), "duplicated node id"
    for n, *_ in all_nodes:      # every Allen node leads to at least one leaf
        if n["node_type"] == "allen":
            assert any(m["node_type"] == "leaf" for m, *_ in iter_nodes(n)), n["allen_name"]
    return all_nodes


def write_outputs(tree, all_nodes, outdir, stem):
    outdir.mkdir(parents=True, exist_ok=True)

    with open(outdir / f"{stem}.json", "w") as f:
        json.dump(tree, f, indent=2)

    cols = ["node_id", "parent_id", "depth", "node_type", "label", "name", "color",
            "allen_label", "allen_name", "allen_acronym", "allen_color",
            "n_children", "match", "parent_rule", "path"]
    with open(outdir / f"{stem}_nodes.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for n, depth, par, path in all_nodes:
            w.writerow({**n, "node_id": node_id(n), "parent_id": node_id(par) if par else "",
                        "depth": depth, "n_children": len(n["children"]), "path": "/".join(path)})

    with open(outdir / f"{stem}.txt", "w") as f:
        for n, depth, *_ in all_nodes:
            if n["node_type"] == "leaf":
                allen = f"  [allen {n['allen_label']} {n['allen_acronym']}]" if n["allen_label"] else ""
                f.write(f"{'  ' * depth}* {n['label']} {n['name']}{allen}\n")
            else:
                f.write(f"{'  ' * depth}{n['allen_acronym']} ({n['allen_label']}) {n['allen_name']}\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--leaves", type=Path, default=DEFAULT_LEAVES)
    ap.add_argument("--ontology", type=Path, default=DEFAULT_ONTOLOGY)
    ap.add_argument("--lut", type=Path, default=DEFAULT_LUT, help="NextBrain LUT (for leaf colours)")
    ap.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    ap.add_argument("--cortex", choices=sorted(CORTEX_TABLES), default="gyral",
                    help="Allen hierarchy used for the DK cortical leaves")
    args = ap.parse_args()

    leaves = load_leaves(args.leaves)
    ontology = json.load(open(args.ontology))
    by_id, parent, by_name = index_ontology(ontology)
    lut_colors = load_lut_colors(args.lut)

    match_leaves(leaves, by_name, CORTEX_TABLES[args.cortex])
    tree = build_tree(leaves, by_id, parent, ontology["id"], lut_colors)
    all_nodes = validate(tree, leaves)

    stem = f"nextbrain_allen_tree_{args.cortex}"
    write_outputs(tree, all_nodes, args.outdir, stem)

    n_leaf = sum(n["node_type"] == "leaf" for n, *_ in all_nodes)
    print(f"leaves: {n_leaf}   internal Allen nodes: {len(all_nodes) - n_leaf}   "
          f"max depth: {max(d for _, d, *_ in all_nodes)}")
    counts = {}
    for l in leaves:
        k = l["match"].split(":")[0].split(" (")[0]
        counts[k] = counts.get(k, 0) + 1
    print("matching:", counts)
    print("parent rule:", {r: sum(l["parent_rule"] == r for l in leaves)
                           for r in sorted({l["parent_rule"] for l in leaves})})
    print("written:", *(args.outdir / f"{stem}{s}" for s in (".json", "_nodes.csv", ".txt")), sep="\n  ")


if __name__ == "__main__":
    main()
