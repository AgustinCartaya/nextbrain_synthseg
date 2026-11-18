"""LabelTree whose nodes also carry Allen ontology fields.

`AllenLabelTree` is a subclass of `trees.tree.LabelTree`. Its nodes (`AllenNode`) accept four
more optional fields:

    allen_id       Allen structure id                  e.g. 10390
    allen_acronym  Allen abbreviation                  e.g. "THM"
    allen_name     Allen name                          e.g. "thalamus"
    allen_color    Allen color, hex RRGGBB             e.g. "E4CCE3"

Everything else (validation, visibility, label groups, to_json, ...) is inherited unchanged,
so an `AllenLabelTree` can be passed to any generator function that takes a `LabelTree`.

    tree = AllenLabelTree.from_json(path)    # read a tree with allen_* fields
    tree.to_json(path)                       # write it (allen_* included)
    tree.to_label_tree()                     # plain LabelTree, allen_* dropped

The base `LabelTree.from_json` rejects unknown node fields, so a JSON file with allen_*
fields must be read with `AllenLabelTree.from_json`.

To make another kind of tree, follow the same pattern: subclass `Node` declaring
`EXTRA_FIELDS = {name: converter}` and subclass `LabelTree` setting `node_cls`.
"""

from .tree import LabelTree, Node  # noqa: E402


class AllenNode(Node):
    """`Node` plus the optional allen_* fields (None = not set, not written)."""

    EXTRA_FIELDS = {'allen_id': int, 'allen_acronym': str, 'allen_name': str, 'allen_color': str}


class AllenLabelTree(LabelTree):
    """`LabelTree` built from `AllenNode`s."""

    node_cls = AllenNode
