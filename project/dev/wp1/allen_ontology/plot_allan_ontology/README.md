# plot_allan_ontology

Tools to load and plot the Allen developmental ontology. They mirror `../../create_nextbraintree/tree.py`
and `tree_plot.py`, which do the same for the LabelTree format.

| File | Role |
|---|---|
| `allen_tree.py` | `AllenTree` / `AllenNode`: load, validate, query and cut the Allen hierarchy |
| `allen_tree_plot.py` | `plot_tree(...)` and a CLI to draw the whole tree or part of it |
| `outputs/` | example figures |

## Inputs

`AllenTree.from_json` reads two formats:

- **The Allen structure graph**, `../data/inputs/AllenBrainOntologyDev.json`: 3317 nodes, depth 14,
  2429 leaves.
- **The NextBrain-Allen tree**, `../data/outputs/nextbrain_allen_tree_<cortex>.json`, written by
  `build_nextbrain_allen_tree.py`. In this format, NextBrain leaves are marked `kind='leaf'`.

Node keys are `A<allen id>` for Allen nodes and `N<NextBrain label>` for NextBrain leaves. These are
the same as the `node_id` column of `nextbrain_allen_tree_<cortex>_nodes.csv`.

## AllenTree

```python
from allen_tree import AllenTree
t = AllenTree.load('../data/inputs/AllenBrainOntologyDev.json')
t.validate()
t.find('THM')            # 'A10390'  (key, Allen id, acronym or exact name)
t.search('pulvinar')     # every node whose name or acronym contains the text
t.ancestors('A10390'); t.descendants('A10390'); t.leaves()
t.subtree('THM', max_depth=2)       # new tree rooted at THM, at most 2 levels below it
t.prune_to(['Pu', 'Ca', 'THM'])     # these nodes and their ancestors only
t.print_tree('THM', max_depth=2)    # text view; cut nodes show "(+N below)"
t.to_json('thm.json')               # written back in the Allen nested format
```

## Plot

The Allen ontology has no resolution or `r_max`, so the plot shows the cuts instead of
visible/collapsed nodes:

- Each box is filled with the structure's Allen colour (`--palette` uses a matplotlib palette).
- A dashed box with **"+N below"** marks a node whose children were not drawn because of
  `--max-depth`.
- A thick border marks a NextBrain leaf. The box shows the NextBrain label and name, plus the
  acronym of the matched Allen node.

```bash
I=../data/inputs/AllenBrainOntologyDev.json
python allen_tree_plot.py --i $I --max-depth 5 --o outputs/allen_ontology_d5.png
python allen_tree_plot.py --i $I --root THM --max-depth 3 --print --o outputs/allen_THM_d3.png
python allen_tree_plot.py --i $I --keep Pu Ca GP THM --o outputs/allen_keep_basal_ganglia_thalamus.png
python allen_tree_plot.py --i ../data/outputs/nextbrain_allen_tree_gyral.json --root BN --o outputs/nextbrain_allen_BN.png
python allen_tree_plot.py --i $I --o outputs/allen_ontology_full.svg    # whole tree: use .svg / .pdf
```

The whole tree is 2429 leaves tall, about 2400 inches. A PNG is limited to 65 000 px per side, so
the DPI is lowered automatically (to about 26 dpi for the full tree) and a warning is printed. Use
`--root`, `--max-depth` or `--keep`, or save as `.svg` or `.pdf`, which are vector formats.
