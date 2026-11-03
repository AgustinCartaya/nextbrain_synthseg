"""LabelTree whose nodes also carry Allen ontology fields.

`AllenLabelTree` is a subclass of the generator's `nxgen.core.tree.LabelTree`
(wp3/generator_v2_modular/nxgen/core/tree.py). Its nodes (`AllenNode`) accept four
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
"""
import sys
from pathlib import Path

# nxgen lives in wp3/generator_v2_modular (this file: wp1/allen_ontology/allen_tree.py)
NXGEN = Path(__file__).resolve().parents[2] / "wp3" / "generator_v2_modular"
if str(NXGEN) not in sys.path:
    sys.path.insert(0, str(NXGEN))

from nxgen.core.tree import LabelTree, Node  # noqa: E402


class AllenNode(Node):
    """`Node` plus the optional allen_* fields (None = not set, not written)."""

    EXTRA_FIELDS = ('allen_id', 'allen_acronym', 'allen_name', 'allen_color')
    FIELDS = Node.FIELDS + EXTRA_FIELDS

    def __init__(self, name, allen_id=None, allen_acronym=None, allen_name=None,
                 allen_color=None, **kwargs):
        super().__init__(name, **kwargs)
        self.allen_id = None if allen_id is None else int(allen_id)
        self.allen_acronym = allen_acronym
        self.allen_name = allen_name
        self.allen_color = allen_color

    def to_dict(self, extra=True):
        d = super().to_dict()
        if extra:
            for key in self.EXTRA_FIELDS:
                if getattr(self, key) is not None:
                    d[key] = getattr(self, key)
        return d


class AllenLabelTree(LabelTree):
    """`LabelTree` built from `AllenNode`s."""

    # ---- I/O ----------------------------------------------------------------------------------- #

    @classmethod
    def from_dict(cls, d):
        unknown = {name: set(spec) - set(AllenNode.FIELDS) for name, spec in d['nodes'].items()}
        unknown = {k: v for k, v in unknown.items() if v}
        if unknown:
            raise ValueError(f'Unknown node fields: {unknown}')
        nodes = [AllenNode(name, **spec) for name, spec in d['nodes'].items()]
        return cls(colors=d['colors'], nodes=nodes, root=d.get('root', 'root'),
                   contrasts=d.get('contrasts'), version=d.get('version'),
                   description=d.get('description', ''))

    def to_dict(self, extra=True):
        d = super().to_dict()
        d['nodes'] = {name: self.nodes[name].to_dict(extra=extra) for name in self.order()}
        return d

    def to_label_tree(self):
        """The same tree as a plain generator `LabelTree` (allen_* fields dropped)."""
        return LabelTree.from_dict(self.to_dict(extra=False))

    # ---- Programmatic construction -------------------------------------------------------------- #
    # Same as LabelTree.new / add_node, but creating AllenNode instead of Node.

    @classmethod
    def new(cls, colors, root='root', **kwargs):
        return cls(colors, [AllenNode(root, level=0)], root=root, validate=False, **kwargs)

    def add_node(self, name, parent, **kwargs):
        """Adds `name` under `parent`, setting level and the parent's children list. Returns the node."""
        if name in self.nodes:
            raise ValueError(f'Node {name!r} already exists')
        p = self.nodes[parent]
        node = AllenNode(name, parent=parent, level=p.level + 1, **kwargs)
        self.nodes[name] = node
        p.children.append(name)
        return node
