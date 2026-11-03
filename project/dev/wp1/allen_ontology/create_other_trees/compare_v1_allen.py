#!/usr/bin/env python3
"""Compare the groupings of nextbrain_tree_v1.json with the Allen tree over the same leaves
(nextbrain_allen_tree_<cortex>.json, from build_nextbrain_allen_tree.py).

Every internal node is reduced to the set of NextBrain labels below it. A group of one tree is
compared with the group of the other tree that overlaps it most (Jaccard index of the label sets).

See allen_ontology_explained/13_comparison_with_v1.md.

Usage
-----
    python compare_v1_allen.py [--cortex developmental]
"""
import argparse
import itertools
import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "plot_allan_ontology"))
from allen_tree import AllenTree  # noqa: E402

V1_TREE = HERE / "data" / "inputs" / "nextbrain_tree_v1.json"


def v1_groups(nodes):
    """{node: set of labels} for the internal nodes of the v1 LabelTree (root excluded)."""
    out = {}

    def rec(k):
        n = nodes[k]
        s = set(n["labels"]) if not n["children"] else set().union(*(rec(c) for c in n["children"]))
        if n["children"] and k != "root":
            out[k] = s
        return s

    rec("root")
    return out


def allen_groups(t):
    return {k: {t.nodes[l].id for l in t.leaves(k)} for k in t.nodes
            if t.nodes[k].children and k != t.root}


def best(s, pool):
    """(Jaccard, key) of the group in `pool` ({key: set}) that overlaps `s` most."""
    return max((len(s & p) / len(s | p), k) for k, p in pool.items())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cortex", choices=["gyral", "developmental"], default="gyral")
    args = ap.parse_args()

    v1 = json.loads(V1_TREE.read_text())["nodes"]
    al = AllenTree.load(HERE / "data" / "outputs" / f"nextbrain_allen_tree_{args.cortex}.json")
    V, A = v1_groups(v1), allen_groups(al)
    setsV, setsA = {frozenset(s) for s in V.values()}, {frozenset(s) for s in A.values()}

    print("== Shape")
    depths = [al.nodes[k].depth for k in al.leaves()]
    print(f"  v1    : {len(V)} internal nodes, {len(setsV)} distinct groupings, leaves at level "
          f"{sorted({v1[k]['level'] for k in v1 if not v1[k]['children']})}")
    print(f"  Allen : {len(A)} internal nodes, {len(setsA)} distinct groupings, leaf depth "
          f"{min(depths)}-{max(depths)} (mean {sum(depths) / len(depths):.1f})")
    print(f"  identical groupings in both: {len(setsA & setsV)}")

    print("\n== v1 groups -> best Allen node (Jaccard)")
    poolA = {k: s for k, s in A.items()}
    for lev in (1, 2, 3, 4):
        groups = {frozenset(V[k]) for k in V if v1[k]["level"] == lev}
        js = [best(g, poolA)[0] for g in groups]
        print(f"  L{lev}: {len(groups):3d} groups | exact {sum(j == 1 for j in js):3d} | >=0.8 "
              f"{sum(j >= 0.8 for j in js):3d} | <0.5 {sum(j < 0.5 for j in js):3d} | mean {sum(js) / len(js):.2f}")
    js = [best(set(g), dict(enumerate(setsV)))[0] for g in setsA]
    print(f"  Allen -> v1: {len(setsA)} groups | exact {sum(j == 1 for j in js)} | >=0.8 "
          f"{sum(j >= 0.8 for j in js)} | <0.5 {sum(j < 0.5 for j in js)} | mean {sum(js) / len(js):.2f}")

    print("\n== v1 level-2 divisions")
    for k in V:
        if v1[k]["level"] == 2 and k != "background_l2":
            g = V[k]
            j, n = best(g, poolA)
            print(f"  {k:28s} n={len(g):3d} -> {al.nodes[n].acronym:6s} {al.nodes[n].name[:32]:32s} "
                  f"J={j:.2f} missing={len(g - A[n])} extra={len(A[n] - g)}")

    print("\n== Sibling pairs (same parent)")
    pv = {v1[k]["labels"][0]: v1[k]["parent"] for k in v1 if not v1[k]["children"]}
    pa = {al.nodes[k].id: al.nodes[k].parent for k in al.leaves()}
    sv = {(x, y) for x, y in itertools.combinations(sorted(pv), 2) if pv[x] == pv[y]}
    sa = {(x, y) for x, y in itertools.combinations(sorted(pa), 2) if pa[x] == pa[y]}
    print(f"  v1 {len(sv)} | Allen {len(sa)} | both {len(sv & sa)}")

    print("\n== Where each v1 tissue lands in Allen")
    tissue = {l: k for k in V if v1[k]["level"] == 1 for l in V[k]}
    mixed = [k for k, s in A.items() if len({tissue.get(l) for l in s}) > 1]
    print(f"  Allen internal nodes mixing v1 tissues: {len(mixed)} of {len(A)}")
    for t in v1["root"]["children"]:
        c = Counter()
        for leaf in al.leaves():
            if tissue.get(al.nodes[leaf].id) == t:
                p = [al.nodes[q].acronym for q in al.ancestors(leaf)][::-1][3:5]
                c[" > ".join(p) or "(root)"] += 1
        print(f"  {t:20s} {dict(c)}")


if __name__ == "__main__":
    main()
