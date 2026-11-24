"""
Label tree: the ontology that drives the intensity model.

The implementation lives in the shared tree library, `dev2/libraries/trees/tree.py`
(single source of truth for the generator and the ontology scripts). This module only
re-exports it, so `from nxgen.core.tree import LabelTree` keeps working. The generator
reads trees with `load_tree(path)`, which picks the class from the file's "tree_type"
(e.g. 'allen' -> AllenLabelTree; missing -> plain LabelTree).

Until the library is installed as a package (`pip install -e`), it is found through its
path relative to this file: dev2/libraries.
"""

import sys
from pathlib import Path

_LIBRARIES = Path(__file__).resolve().parents[4] / 'libraries'    # dev2/libraries
if not (_LIBRARIES / 'trees' / 'tree.py').is_file():
    raise ImportError(f'Tree library not found at {_LIBRARIES / "trees"}')
if str(_LIBRARIES) not in sys.path:
    sys.path.insert(0, str(_LIBRARIES))

from trees.tree import INF, TREE_TYPES, LabelTree, Node, load_tree, tree_class  # noqa: E402,F401

__all__ = ['INF', 'TREE_TYPES', 'LabelTree', 'Node', 'load_tree', 'tree_class']
