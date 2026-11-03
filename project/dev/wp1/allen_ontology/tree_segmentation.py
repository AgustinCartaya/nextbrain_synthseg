"""Turn a NextBrain segmentation into a segmentation of a few chosen tree nodes.

Given a NextBrain segmentation, an `AllenLabelTree` (e.g. data/outputs/nextbrain_tree_allen_gyral.json)
and a list of nodes, every voxel whose label belongs to the i-th node of the list (the node itself or
any descendant) gets the value i + 1. Everything else is 0 (background).

    seg = TreeSegmentation("data/outputs/nextbrain_tree_allen_gyral.json")
    seg.convert("case-1.nii.gz", ["FGM", "FWM", "FV"], "case-1_tissues.nii.gz")   # -> 0, 1, 2, 3

An item of the list can be any of
    - a tree node name            "white_matter_of_forebrain_l1", "putamen"
    - a NextBrain label (int)     79                    -> the node that has that label
    - an Allen id (int)           10338
    - an Allen acronym            "Pu"
    - an Allen name               "putamen"
An Allen acronym / name / id can sit on two nodes (an internal node and the leaf that stays its
child, e.g. THM -> thalamus_l3 and its leaf thalamus); the shallowest one is used, i.e. the whole
structure.

If two items overlap (e.g. ["FGM", "thalamus_l3"]), the deeper node wins: thalamus voxels get 2,
the rest of FGM gets 1. Labels of the segmentation that are not in the tree become 0.

Command line (input/output: one file each, or two folders -> every .nii/.nii.gz of the input folder):

    python tree_segmentation.py case-1.nii.gz case-1_tissues.nii.gz FGM FWM FV
    python tree_segmentation.py changed_recon_0.2mm/ tissues/ FGM FWM FV --tree my_tree.json
    python tree_segmentation.py case-1.nii.gz out.nii.gz FGM THM 79 "white matter of forebrain"
"""
import argparse
from pathlib import Path

import nibabel as nib
import numpy as np

from allen_tree import AllenLabelTree


class TreeSegmentation:

    def __init__(self, tree):
        """`tree`: an AllenLabelTree, or a path to its json."""
        self.tree = AllenLabelTree.load(tree)

    def find(self, item):
        """Name of the tree node that `item` refers to (see module docstring)."""
        nodes = self.tree.nodes
        if isinstance(item, str) and item in nodes:
            return item
        if isinstance(item, (int, np.integer)):
            if item in self.tree.label_to_node:
                return self.tree.label_to_node[item]
            found = [k for k, n in nodes.items() if n.allen_id == item]
        else:
            found = [k for k, n in nodes.items() if item in (n.allen_acronym, n.allen_name)]
        if not found:
            raise KeyError(f'{item!r} is not a node name, label, Allen id, acronym or name of the tree')
        return min(found, key=lambda k: nodes[k].level)

    def labels_of(self, node):
        """All NextBrain labels of `node` and its descendants."""
        return [lab for k in [node] + self.tree.descendants(node) for lab in self.tree.nodes[k].labels]

    def lookup_table(self, items):
        """{NextBrain label: new value}, new value = position in `items` + 1."""
        nodes = [self.find(item) for item in items]
        lut = {}
        # shallow nodes first, so deeper nodes overwrite them where they overlap
        for value, node in sorted(enumerate(nodes, 1), key=lambda x: self.tree.nodes[x[1]].level):
            for lab in self.labels_of(node):
                lut[lab] = value
        return lut

    def __call__(self, seg, items):
        """Remaps a segmentation array; returns an array of the same shape with values 0..len(items)."""
        seg = np.asarray(seg).astype(np.int64)
        table = np.zeros(max(seg.max(), max(self.tree.label_to_node)) + 1, dtype=np.uint16)
        for lab, value in self.lookup_table(items).items():
            table[lab] = value
        out = table[seg]
        return out.astype(np.uint8) if len(items) < 256 else out

    def convert(self, in_path, items, out_path):
        """Reads a segmentation .nii(.gz), remaps it and saves it with the same affine/header."""
        im = nib.load(in_path)
        out = self(np.asanyarray(im.dataobj), items)
        header = im.header.copy()
        header.set_data_dtype(out.dtype)
        nib.save(nib.Nifti1Image(out, im.affine, header), out_path)
        return out


def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--i', type=Path, help='segmentation .nii(.gz), or a folder of them')
    parser.add_argument('--o', type=Path, help='output .nii(.gz), or a folder (created if needed)')
    parser.add_argument('--v', nargs='+',
                        help='node names, labels, Allen ids, acronyms or names; value = position in the list')
    parser.add_argument('--tree', type=Path, default=here / 'data/outputs/nextbrain_tree_allen_gyral.json',
                        help='AllenLabelTree json (default: %(default)s)')
    args = parser.parse_args()

    items = [int(item) if item.isdigit() else item for item in args.v]
    seg = TreeSegmentation(args.tree)
    for i, item in enumerate(items, 1):
        print(f'{i}: {item!r} -> {seg.find(item)}')

    if args.i.is_dir():
        args.o.mkdir(parents=True, exist_ok=True)
        pairs = [(f, args.o / f.name) for f in sorted(args.i.glob('*.nii*'))]
    else:
        args.o.parent.mkdir(parents=True, exist_ok=True)
        pairs = [(args.i, args.o)]
    for in_path, out_path in pairs:
        seg.convert(in_path, items, out_path)
        print(f'{in_path} -> {out_path}')


if __name__ == '__main__':
    main()


# python tree_segmentation.py \
#     --i /home/agustin/phd/synthseg_nextbrain/data/basics/changed_recon_0.2mm/case-1.nii.gz \
#     --o /home/agustin/phd/synthseg_nextbrain/project/dev/wp1/allen_ontology/data/outputs/merged_sementations/case-1_tissues.nii.gz \
#     --v FGM FWM FV FSS MGM MWM MSS HGM HWM HSS