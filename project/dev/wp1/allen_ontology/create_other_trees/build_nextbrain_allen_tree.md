# NextBrain leaves on the Allen tree

`build_nextbrain_allen_tree.py` builds a new tree:

- its **leaves** are exactly the 335 labels of `nextbrain_ontology_v1.csv`;
- its **internal nodes**, and so the parent of every leaf, come from the Allen developmental ontology
  (`AllenBrainOntologyDev.json`).

Each leaf keeps its NextBrain label, name and colour. The matching Allen node's fields are stored
next to them with the `allen_` prefix: `allen_label`, `allen_name`, `allen_acronym` (the
abbreviation) and `allen_color`.

```bash
cd /home/agustin/phd/synthseg_nextbrain/project/dev/wp1/allen_ontology
python build_nextbrain_allen_tree.py                          # cortex on Allen's gyral hierarchy (default)
python build_nextbrain_allen_tree.py --cortex developmental   # cortex on Allen's neocortex lobes
```

The script only needs the Python standard library. It does **not** import `nextbrain-utils`. It
copies the two pieces it needs from that package (the name normalisation and the DK→Allen cortex
tables) and fixes one bug in them (see §3.4).


## 1. Inputs

| Input | Path | Used for |
|---|---|---|
| Leaves | `/home/agustin/phd/synthseg_nextbrain/data/basics/nextbrain_ontology_v1.csv` | the 335 leaves and all their attributes |
| Allen ontology | `data/inputs/AllenBrainOntologyDev.json` | internal nodes and parents (3317 nodes; root "neural plate", id 10153) |
| NextBrain LUT | `nextbrain-utils/nextbrain_utils/lut/NextBrainLUT.txt` | the leaves' NextBrain colours (the CSV has no colour column) |

Three things about the CSV are worth knowing:

- **Its column `allen_id` is the NextBrain label ID, not an Allen ID.** For example, 218 is the
  NextBrain thalamus; the Allen thalamus is 10390. In the output this value is called `label`, and
  `allen_label` always means a real Allen ID.
- Each row is one leaf. The columns `L1_tissue … L4_subregion` are the old NextBrain hierarchy.
  They are **not** used to build the tree, but they are kept as leaf attributes. They are also used
  to resolve the ambiguous matches in §3.2 and §3.3.
- NextBrain labels have **no abbreviation**. So leaves have `label`, `name` and `color`, but only
  the Allen side has an acronym (`allen_acronym`).


## 2. The idea in one picture

```
Allen ontology (3317 nodes)               New tree (pruned Allen + NextBrain leaves)

THM thalamus (10390)                       THM (10390)
├─ DTH dorsal thalamus                     ├─ DTH
│  └─ … Pul pulvinar (10409)               │  └─ … Pul (10409)
│        ├─ Pulr anterior n. (10410)       │        ├─ * 454 anterior_nucleus_of_pulvinar   [allen 10410 Pulr]
│        ├─ Pulm medial n.   (10411)       │        ├─ * 508 medial_nucleus_of_pulvinar     [allen 10411 Pulm]
│        └─ …                              │        ├─ * 578 pulvinar_of_thalamus           [allen 10409 Pul]
└─ …                                       │        └─ * 811 superficial_pulvinar_nucleus   [allen 10409 Pul]
                                           └─ * 218 thalamus                                 [allen 10390 THM]
```

`*` marks a NextBrain leaf. Allen nodes that lead to no leaf are removed.


## 3. Step 1: match every leaf to an Allen node

Names are compared after the same normalisation that `nextbrain-utils` uses (`normalize_name`):
lowercase, strip `ctx-` and `lh-`/`rh-`, remove commas, turn `-`/`_` into spaces, and collapse
repeated spaces. So `substantia_nigra__compact_part` matches `substantia nigra, compact part`.

| How the match was found | Leaves |
|---|---|
| exact (normalised) name | 290 |
| name with several Allen candidates, resolved with the CSV context (§3.2) | 2 |
| manual decision (§3.3) | 8 |
| DK cortex table (§3.4) | 34 |
| no match: `background` (§3.5) | 1 |
| **total** | **335** |

The script **stops with an error** if any leaf ends up with no match or with more than one
candidate. A future CSV change can therefore never be matched silently to the wrong node.

### 3.1 Exact names

290 leaves have exactly one Allen node with the same normalised name.

### 3.2 Ambiguous names

Two names appear twice in the Allen ontology. The CSV's L2/L4 columns say which node is meant:

| Leaf | Allen candidates | Chosen | Why |
|---|---|---|---|
| 521 `dorsal_raphe_nucleus` | 12223 DR (midbrain), 12472 DR-p (pons) | **12223 DR** | CSV L4 = Midbrain |
| 377 `lateral_nucleus` | 10367 La (amygdala), 12410 PNl (pontine nucleus) | **10367 La** | CSV L2 = Amygdala, L4 = Basolateral group |

(`nextbrain-utils` takes whichever node it visits last, so it gives these two leaves the pontine
nodes.)

### 3.3 Manual decisions

These leaves have no Allen node with the same name. The chosen node is the one closest in
anatomy, using the CSV's L4 column for location:

| Leaf | Allen node | Reason |
|---|---|---|
| 842 `decussation_of_trochlear_nerve_fibres` | 12338 x4 decussation of trochlear nerve fibers | same structure; Allen spells it "fibers" |
| 844 `reticular_formation_/_cuneiform_area` | 12239 MRF midbrain reticular formation | the label covers both; MRF contains the cuneiform nucleus (L4: Midbrain) |
| 827 `parietotemporopontine_tract` | 266441721 cpn cortico-pontine fibers, pontine part | Allen does not split pontine corticopontine fibres by cortical origin (L4: Pons) |
| 829 `frontopontine_tract` | 266441721 cpn (same as 827) | same reason |
| 828 `corticonuclear_tract` | 12774 cbu-h corticobulbar tract | corticonuclear = corticobulbar; hindbrain node (L4: Pons) |
| 846 `white_matter_of_the_cerebellum` | 10668 HWM white matter of hindbrain | Allen has no cerebellar white-matter node; this is also the old NextBrain name of label 846 |
| 811 `superficial_pulvinar_nucleus` | 10409 Pul pulvinar of thalamus | NextBrain-only pulvinar subdivision |
| 813 `diffuse_pulvinar_nucleus` | 10409 Pul | same reason |

These choices live in the `AMBIGUOUS` and `MANUAL` dictionaries at the top of the script. Edit
them there if you disagree with any of them.

### 3.4 DK cortical leaves (IDs 2001–2035)

NextBrain's cortex is FreeSurfer's Desikan-Killiany parcellation, which has no Allen equivalent.
The script uses the same hand-made tables as `nextbrain-utils` (`to_allen.py`). `--cortex` picks
one of them:

- `gyral` (default) puts the DK parcels on Allen's **gyral** hierarchy, under
  `FSS surface structures of forebrain → CeG cerebral gyri and lobules`, e.g.
  `ctx-rh-superiorfrontal → 12115 SFG superior frontal gyrus`.
- `developmental` puts them on Allen's main hierarchy under **neocortex lobes**, e.g.
  `ctx-rh-superiorfrontal → 10161 frontal neocortex`. This is coarser.

The full gyral table is in the script and in `nextbrain-utils allen -h`.

**Bug fixed while copying.** In `nextbrain-utils` the target for `transversetemporal` is written
`"... (Heschl's gyrus)"` with a capital H. Allen names are lowercased before comparing, so it never
matched. Here it is lowercased, and 2034 correctly maps to 12144 TTG.

In both modes, several DK parcels map to the same Allen node. They are kept as separate leaves that
share an `allen_label` (§6).

### 3.5 Background

Leaf 0 `background` has no Allen equivalent. It is attached directly to the root (`NP neural
plate`), and its `allen_*` fields are empty.


## 4. Step 2: choose each leaf's parent

Call X the Allen node that a leaf matched. There are two cases:

1. **X is not an ancestor of any other leaf's Allen node** (306 leaves in gyral mode). The leaf
   *takes X's place*: its parent is X's Allen parent, and X itself does not appear in the tree
   (it lives on as the leaf's `allen_label`).
   Example: `454 anterior_nucleus_of_pulvinar` matches `Pulr`, so its parent is `Pul`.

2. **X is also an ancestor of other leaves** (28 leaves). X has to stay in the tree as an internal
   node, because other leaves hang below it. A leaf cannot also be an internal node, so the leaf
   becomes a **child of X**. It stands for "X, not assigned to any of its sub-parts", which is
   exactly how the CSV describes such rows (e.g. 218 thalamus: "Whole thalamus (unassigned)").
   Examples: `218 thalamus` sits under `THM` next to the thalamic nuclei; `578
   pulvinar_of_thalamus` sits under `Pul`; `7 white_matter_of_forebrain` sits under `FWM`;
   `ctx-rh-inferiorparietal` sits under `IPL` next to `supramarginal`.

Each leaf records which case applied in the field `parent_rule`.

## 5. Step 3: prune and assemble

- Keep only the Allen nodes on the path from the root to some leaf's parent: **165** internal
  nodes in gyral mode, **157** in developmental mode (out of 3317).
- Keep the Allen order of children. Within an Allen node, the leaves are appended after its Allen
  children.
- The tree is not "compressed": chains of single-child Allen nodes (`NP → NT → Br → F …`) are kept
  as they are, so every internal node is a real Allen node with its real ID.

Checks run on every build:

- the leaf set equals the CSV, and each label appears once;
- no leaf has children;
- every node id is unique;
- every internal node leads to at least one leaf.

Result (gyral): 335 leaves, 165 internal nodes, maximum depth 13.


## 6. Outputs (`data/outputs/`)

Each `--cortex` mode writes three files, named `nextbrain_allen_tree_<mode>.*`:

| File | Content |
|---|---|
| `….json` | the nested tree. This is the main output |
| `…_nodes.csv` | the same tree as a flat table: one row per node with `node_id` / `parent_id` |
| `….txt` | an indented, human-readable view (leaves marked `*`) |

### Node fields (JSON and CSV)

| Field | Internal (Allen) node | Leaf |
|---|---|---|
| `node_type` | `"allen"` | `"leaf"` |
| `label` | — | NextBrain label ID (CSV column `allen_id`) |
| `name` | — | NextBrain name (CSV `leaf_name`) |
| `color` | — | NextBrain colour, hex `RRGGBB` (from `NextBrainLUT.txt`) |
| `allen_label` | Allen ID of the node | Allen ID of the matched node |
| `allen_name` | Allen name | Allen name of the matched node |
| `allen_acronym` | Allen abbreviation | Allen abbreviation of the matched node |
| `allen_color` | Allen colour (hex) | Allen colour of the matched node |
| `match` | — | how the Allen node was found (§3), including the reason for manual cases |
| `parent_rule` | — | which case of §4 placed the leaf |
| `attributes` (JSON only) | — | every other CSV column (`L1_tissue … v1_change`), unchanged |
| `children` (JSON only) | list of nodes | `[]` |

The flat CSV also has `node_id` (`A<allen id>` for Allen nodes, `N<label>` for leaves),
`parent_id`, `depth`, `n_children` and `path` (acronyms from the root).

### Leaves that share an Allen node

In gyral mode these leaves have the same `allen_label`. They are still separate leaves:

| Allen node | Leaves |
|---|---|
| 10409 Pul | 578 pulvinar_of_thalamus, 811 superficial_pulvinar, 813 diffuse_pulvinar |
| 10668 HWM | 611 white_matter_of_hindbrain, 846 white_matter_of_the_cerebellum |
| 12116 MFG | ctx caudalmiddlefrontal, rostralmiddlefrontal |
| 12148 OccL | ctx lateraloccipital, pericalcarine |
| 12158 CgGc | ctx caudalanteriorcingulate, isthmuscingulate, posteriorcingulate |
| 266441721 cpn | 827 parietotemporopontine, 829 frontopontine |

If you need a one-to-one leaf ↔ Allen mapping, these are the cases to deal with.


## 7. Things to keep in mind

- **In gyral mode the cortex sits in a different branch from the rest of the telencephalon.** The
  DK leaves hang under `FSS / CeG` (Allen's separate gyral hierarchy). Hippocampus, amygdala,
  piriform cortex and the other non-DK cortical leaves stay under `FGM / Tel / Cx`. This is how the
  Allen developmental ontology is organised. Use `--cortex developmental` to put the DK leaves
  under `Cx` as well, at the cost of resolution (lobes instead of gyri).
- **`white_matter_of_hindbrain` (HWM) is broad.** Leaves 611 and 846 hang under HWM next to the
  brainstem tracts (peduncles, corticobulbar tract, …), because Allen has no cerebellar white-matter
  node.
- **Hemisphere.** Leaves use NextBrain's right-hemisphere cortex IDs (`ctx-rh-*`, 20xx), as in the
  CSV. The Allen ontology is not lateralised.
- **Re-running** overwrites the files in `data/outputs/`. Neither input file is modified.
