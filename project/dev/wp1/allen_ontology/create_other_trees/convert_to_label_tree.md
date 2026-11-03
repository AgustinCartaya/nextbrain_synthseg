# `convert_to_label_tree.py`: NextBrain-Allen tree → LabelTree JSON

Converts the tree built by `build_nextbrain_allen_tree.py` (NextBrain v1 leaves, Allen parents) into
the **LabelTree** format used by the generator, the same format as
`data/inputs/nextbrain_tree_v1.json` (built by `../create_nextbraintree/_build_nextbrain_tree.py`).

```bash
python build_nextbrain_allen_tree.py [--cortex developmental]   # 1. Allen tree
python convert_to_label_tree.py      [--cortex developmental]   # 2. LabelTree
```

| Input | Output |
|---|---|
| `data/outputs/nextbrain_allen_tree_<cortex>.json` | `data/outputs/nextbrain_allen_label_tree_<cortex>.json` |

The script imports `LabelTree` from `../create_nextbraintree/tree.py`. It builds the tree with that
class, runs `LabelTree.validate()` and writes the file with `LabelTree.to_json()`. This guarantees the
generator can load the output. Both outputs validate with no errors and no warnings.

## Format reminder

Every node has `parent`, `children`, `level` (parent + 1), `labels`, `wr_mu`, `wr_sigma` and
`r_max`. Level-1 nodes also carry `w_mu`, the weights over the principal `colors`, or `null: true`.
The format does **not** allow extra fields: `LabelTree.from_dict` rejects unknown keys. The Allen
information is therefore stored in the node **name** and in its **`description`**.

## Structure

```
root
├── background_l1   (null)  → background [0]
├── brainstem_l1            → m_a10648 (midbrain) …, h_a10653__brainstem (hindbrain) …
├── cerebral_gm_l1          → f_a10156__cerebral_gm → fgm_a10157__cerebral_gm → tel_… → cx_… → …
├── deep_gm_l1              → f_a10156__deep_gm → … → thm_a10390__deep_gm → { dth_a10391, …, thalamus [218] }
└── …
```

- **Level 1 = `L1_tissue`.** In the LabelTree format, level 1 sets the principal colour (intensity
  class) and every node below inherits it. The Allen hierarchy has no tissue level: the forebrain,
  for example, contains cortex, white matter, deep GM and ventricles. The tissue therefore stays on
  level 1, exactly as in `nextbrain_tree_v1.json`:
  - the `colors` list is the same, in the same alphabetical order;
  - each tissue gets a one-hot `w_mu`;
  - `Background` is a `null` node.
- **Level ≥ 2 = Allen.** Under each tissue is the Allen tree pruned to that tissue's leaves. **Every
  leaf keeps the same Allen parent** as in `nextbrain_allen_tree_<cortex>.json`. The three Allen nodes
  above every brain leaf are skipped, because they never separate labels: NP neural plate, NT neural
  tube, Br brain.
- **Allen nodes shared by several tissues are duplicated**, one copy per tissue. Examples are F
  forebrain, THM thalamus and HWM white matter of hindbrain. In the gyral tree, 44 copies come from 18
  Allen nodes.
- **Leaves sit at different levels** (2 to 12), because Allen branches have different depths. The
  format allows this. Only the level-1 nodes are fixed.

## Naming

| Node | Name | Example |
|---|---|---|
| tissue | `<snake(tissue)>_l1` (same as v1) | `deep_gm_l1` |
| Allen node | `<snake(acronym)>_a<allen id>` | `pu_a10338` |
| Allen node in several tissues | `<snake(acronym)>_a<allen id>__<snake(tissue)>` | `thm_a10390__deep_gm` |
| leaf | `snake(leaf_name)` (same as v1) | `thalamus`, `inferior_olive_principal_nucleus` |

## Fields

| Field | Tissue (level 1) | Allen node | Leaf |
|---|---|---|---|
| `labels` | `[]` | `[]` | `[NextBrain id]` |
| `w_mu` | one-hot over `colors` | – | – |
| `wr_mu`, `wr_sigma` | 4.0, 0.5 | 5.0, 0.5 | 3.0, 0.5 |
| `r_max` | `null` | see below | see below |
| `description` | tissue name | `Allen <id> <acronym>: <name>` | `<leaf name> \| Allen <id> <acronym>: <name> \| <functional_class, layer, long_axis>` |

The jitter values (`wr_*`) are the illustrative ones from `_build_nextbrain_tree.py`.

## `r_max` (coarsest voxel size, in mm, at which a node is visible)

These rules follow `_build_nextbrain_tree.py`. The difference is that the internal nodes are now
Allen nodes rather than CSV groups.

1. **Leaves.** `r_max` is `L5_r_max_best_mm`, clamped (per contrast) to the parent's `r_max`. As in
   v1, this gives `{"default": 0.3, "T1": 0.0}` below nodes that are invisible in T1.
2. **Allen nodes: coarse structures.** The `GROUP_R_MAX` table is copied from v1 (e.g. thalamus 3.5,
   cerebral cortex 4.0, caudate 2.0). Each entry is a CSV L2/L3 group. It is placed on the Allen node
   whose leaf set has the highest Jaccard overlap with the group's leaf set; on a tie, the highest
   such node wins. An entry is placed only if the Jaccard overlap is > 0.5.
3. **Allen nodes: everything else.** `r_max` is the coarsest of the children's values. A leaf
   contributes its **L4** `r_max`: `table_r_max(L4_r_max_T1w_mm, L4_r_max_best_mm)`, the same rule as
   v1, or its L5 value if L4 is empty. So the Allen node directly above a group of leaves becomes
   visible at the resolution the table gives for their L4 subregion.

The script prints where each `GROUP_R_MAX` entry went. In the gyral tree:

| Group | Allen node | Jaccard |
|---|---|---|
| caudate, putamen, pallidum, amygdala, substantia nigra, cerebellar cortex/nuclei | Ca, Pu, GP, CN, SN, CbC, CbDN | 1.00 |
| hippocampal formation | ArCx (archicortex) | 0.96 |
| basal ganglia | BN | 0.90 |
| cerebral cortex | FSS (gyral) / NCx (developmental) | 0.89 / 0.84 |
| hypothalamus | HTH | 0.88 |
| thalamus | THM | 0.87 |
| optic pathway | fift | 0.67 |
| brain stem | H (hindbrain; the midbrain is outside it in Allen) | 0.62 |
| basal forebrain | BF | 0.60 |
| subthalamus, accumbens, cerebral white matter | **not placed** (0.50) | |

### Number of generation classes compared with v1

These are the counts from `LabelTree.label_groups`:

| r (mm) | 0.3 | 0.5 | 1.0 | 1.5 | 2.0 | 3.0 | 4.0 | 5.0 |
|---|---|---|---|---|---|---|---|---|
| v1, T1 | 284 | 53 | 24 | 21 | 19 | 15 | 13 | 9 |
| Allen (gyral), T1 | 299 | 114 | 63 | 25 | 23 | 17 | 12 | 9 |

The two trees agree from 1.5 mm up. At 0.5–1 mm the Allen tree keeps more classes. The leaves of one
CSV L4 subregion (e.g. the thalamic "ventral" nuclei) are spread over several Allen parents, and each
of those parents receives the L4 `r_max`. This follows from using the Allen hierarchy. If it matters,
add entries to `GROUP_R_MAX` (keys are CSV groups) or change rule 3.

## Caveats

- The `r_max` values on Allen nodes are **heuristic**. They are derived from the CSV's L4 values and
  the illustrative `GROUP_R_MAX`, not measured.
- Splitting by tissue means one Allen structure can appear in several places. For example, the
  pineal body (tissue `Other`) sits under `thm_a10390__other`, separate from the deep-GM thalamus.
  This is needed because the tissue sets the colour.
- Re-running the script overwrites the outputs.
