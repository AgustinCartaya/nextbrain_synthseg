# NextBrain ontology: changes from v0 to v1

This document lists every change made to `data/inputs/nextbrain_ontology_v0.csv` to produce
`data/inputs/nextbrain_ontology_v1.csv`, and the reason for each one.
It also explains how `_build_nextbrain_tree.py` reads the table.

Every changed row is marked in a new column, `v1_change`, so each edit can be traced in the table.
v0 is not modified.


## 1. What "coherent" means here

The table is turned into a `LabelTree` (see `wp3/simple_tree_generator/tree_sampling.py`).
In that model:

- The columns `L1_tissue → L2_division → L3_structure → L4_subregion → leaf` are the levels of a tree.
- **Level 1** is a tissue class. It carries the weights over the principal colors (GM, WM, CSF).
  Level-1 nodes are always visible and never merge.
- **Deeper nodes** inherit the parent's mean and sigma, plus a random deviation.
  A node is visible at voxel size `r` only if `r <= r_max` and its parent is visible.
  Otherwise its labels merge into the parent's Gaussian.
- `r_max` is the coarsest voxel size (mm) at which a node can still be told apart from its siblings.

For this to work, the table must satisfy these rules:

1. **Every node has exactly one parent.** The same L2 name cannot appear under two L1s, and so on.
   If it does, the tree is ambiguous and that group cannot be built.
2. **Level 1 must be a tissue class**, because it defines the intensity color mix.
3. **All rows that share an L4 node must give it the same `r_max`.**
4. **`r_max` must be logically consistent:**
   - `L4_r_max_best_mm` is the value for the best contrast, so it can never be finer (smaller) than the T1w value.
   - A child cannot be visible beyond its parent's limit.
5. **Every label ID is unique and matches the NextBrain label map.**
   Names should match the NextBrain lookup table (LUT).

v0 was checked against these rules with a script, and against two NextBrain LUTs:

- the in-vivo output LUT `data/test/outputs/mrxfields/T1W/7T/001_GX/lut.txt` (265 labels, from a real segmentation);
- the full atlas LUT `.../ERC_bayesian_segmentation/data_full/AllenAtlasLUT` (918 labels).

Checks that v0 already passed:

- no duplicated `allen_id`;
- no duplicated `leaf_name`;
- no L3 with two L2 parents;
- the r_max values within each L4 were consistent.


## 2. Structural problems: nodes with more than one parent

### 2.1 L2 "Cerebellum" had two L1 parents
- **Problem:** `Cerebellum` was the L2 of both `Cerebellar GM` (granular and molecular layers)
  and `Cerebellar WM` (`white_matter_of_hindbrain`). A node cannot belong to two tissue classes.
- **Fix:** I split it into two divisions:
  - `Cerebellar GM / Cerebellar cortex` (rows 597, 595)
  - `Cerebellar WM / Cerebellar white matter` (row 611)
- **Why this option:** it keeps L1 as tissue classes (rule 2). The alternative was a mixed
  L1 "Cerebellum", as in the SynthSeg tree. That would have changed the tissue design of the table.

### 2.2 L2 "Epithalamus" had two L1 parents
- **Problem:** `Epithalamus` appeared under `Deep GM` (habenular nuclei 430, 444, 493)
  and under `Other` (`pineal_body`, 506).
- **Fix:** I gave the pineal body its own division: `Other / Pineal gland / Pineal-Body`.
  The habenula stays under `Deep GM / Epithalamus`.
- **Why:** the pineal is a gland, often calcified or cystic. Its intensity has nothing to do with
  deep grey matter, so it should not inherit the Deep GM color.

### 2.3 L4 "Mammillary / posterior" had two L3 parents
- **Problem:** the same L4 name was used under `Hypothalamus` (posterior hypothalamic nuclei)
  and under `Mammillary-Body` (mammillary nuclei). The two groups also had different r_max values:
  0.5 mm with no T1w value, versus 1.0/0.7 mm with T1w.
  As a single node it would have contradicted itself.
- **Fix:** I renamed the L4 depending on its parent:
  - under `Hypothalamus` → `Posterior hypothalamus` (256, 275, 297, 304)
  - under `Mammillary-Body` → `Mammillary nuclei` (305, 306, 307, 308)

### 2.4 The midbrain was split across two divisions
- **Problem:** midbrain structures appeared in two places.
  - L1 `Brainstem` / L2 `Brainstem` / L3 `Brain-Stem` / L4 `Midbrain` held 22 structures,
    with functional groups in the `functional_class` column.
  - A separate L2 `Midbrain` (L3 `VentralDC-Midbrain`, 12 rows, plus L3 `Substantia-Nigra`, 2 rows)
    held other midbrain structures.

  Also, in that second block the `L4_subregion` column held functional-class values
  (Colliculus, Nucleus, Red nucleus, Pretectum, ...) instead of subregions.
  So the same column meant different things in different rows.
- **Fix:**
  - The 12 `VentralDC-Midbrain` rows moved to `Brainstem / Brainstem / Brain-Stem / Midbrain`.
    Their old L4 value moved to `functional_class`, which is where the other brainstem rows keep it.
  - `midbrain_(mesencephalon)` (384) got `functional_class = Whole level (unassigned)`,
    like `myelencephalon` (662) and `pons` (580).
  - Substantia nigra (352, 310) moved from L2 `Midbrain` to L2 `Brainstem`.
    It keeps its own L3 `Substantia-Nigra` with the L4s pars compacta and pars reticulata,
    so its specific r_max (1.0 mm, T2*/QSM/NM) is not lost.
- **r_max side effect:** the 12 moved rows now belong to the L4 node `Midbrain`.
  Under rule 3 they must share its r_max, so they take the brainstem midbrain values
  (`T1w 1.0, best 1.0, T1w, image-driven`).
  Their v0 values (0.7 / 0.5, T2w/qMRI) were the same generic block found on many unrelated rows,
  such as background, ventricles and WM. That suggests a default fill rather than a measurement.


## 3. Tissue-class problems (L1)

### 3.1 L1 "Other" was not a tissue class
- **Problem:** `Other` contained the pineal body and the whole optic pathway:
  optic chiasm (161), optic nerve (114), optic tract (208) and supraoptic decussation (209).
  The optic pathway is white matter. Under `Other` it would be sampled with a mixed color
  that has nothing to do with WM.
- **Fix:** I moved the `Optic pathway` division (all 4 rows) to `Cerebral WM`.
  This matches the SynthSeg tree, where `optic_chiasm` hangs from `cerebral_wm`.
  `Other` now contains only the pineal gland.

### 3.2 Zona incerta and fields of Forel were placed under Thalamus
- **Problem:** `zona_incerta` (226) and `nucleus_of_the_field_of_forel` (400) had L4
  `Subthalamic region (ZI / Forel)` under L2 `Thalamus`.
  Anatomically they are subthalamic, and the table already has a `Subthalamus` division
  for the subthalamic nucleus. The table contradicted its own L4 label.
- **Fix:** I moved them to `Deep GM / Subthalamus / Zona-Incerta / Subthalamic region (ZI / Forel)`.
  Their r_max values were kept.


## 4. Misidentified and missing labels

### 4.1 The four "UNNAMED" brainstem labels are the deep cerebellar nuclei
- **Problem:** `UNNAMED_715`, `UNNAMED_721`, `UNNAMED_751` and `UNNAMED_752` were in
  `Brainstem / ... / Medulla` with functional class `UNNAMED - to confirm`.
  Both NextBrain LUTs identify them as:
  - 715 `fastigial_(medial)_nucleus`
  - 721 `dentate_(lateral)_nucleus`
  - 751 `medial_interpositus_(globose)_nucleus`
  - 752 `lateral_interpositus_(emboliform)_nucleus`

  These are cerebellar, not part of the medulla.
- **Fix:**
  - Names taken from the LUT.
  - Hierarchy set to `Cerebellar GM / Cerebellar nuclei / Cerebellar-Nuclei / Deep cerebellar nuclei`.
  - `functional_class` cleared.
  - `supersynth_label` and `fs_block` set to `Cerebellum-White-Matter`, because they are embedded in it.
- **r_max (estimate, to be confirmed):** T1w left empty, best 1.0 mm, `T2* / QSM`, `contrast-specific`.
  The dentate nucleus is iron-rich and visible on T2*/QSM at about 1 mm, but poorly on T1w.
  I used the same pattern as the substantia nigra and subthalamic nucleus rows.
  The v0 value (brainstem 1.0 T1w) only applied because the rows had been put in the wrong place.

### 4.2 `UNNAMED_326` is the uncal subiculum
- **Problem:** row 326 was `UNNAMED_326` with L4 `Hippocampus (other)`.
- **Fix:** the LUT names it `uncal_subiculum`. I set L4 = `Subiculum` and `long_axis = uncal`,
  next to `rostral_subiculum` and `caudal_subiculum`.

### 4.3 Missing label: 846 `white_matter_of_the_cerebellum`
- **Problem:** label 846 appears in the in-vivo NextBrain LUT, and therefore in real in-vivo
  segmentations, but it was not in v0. The tree sampler fails on labels missing from the tree
  (`unassigned='error'`).
- **Fix:** I added a row under `Cerebellar WM / Cerebellar white matter`, copying the attributes of
  `white_matter_of_hindbrain` (611). The sheet note says 611 is the label used for cerebellar WM.
  - Note: 846 is not in the full atlas LUT. It seems to exist only in the in-vivo label set.


## 5. Resolution (r_max) problems

### 5.1 T1w r_max larger than the best-contrast r_max (152 rows)
- **Problem:** `L4_r_max_best_mm` is the r_max obtained with the *best* contrast.
  By definition it cannot be smaller than the T1w value: if T1w already resolves a structure
  at 0.8 mm, the best contrast resolves it at 0.8 mm or coarser.
  In v0 many rows broke this rule:

  | Block (L3)                                        | T1w | best | best contrast |
  |---------------------------------------------------|-----|------|---------------|
  | Thalamus nuclei                                   | 0.8 | 0.5  | WMn / T1 / DWI |
  | Hippocampal subfields                             | 0.8 | 0.5  | T2w |
  | Caudate, putamen, accumbens, claustrum, habenula, WM, fornix, ventricles, paleocortex, optic pathway, pineal | 0.7 | 0.5 | T2w / qMRI |
  | Mammillary body, medial septal nucleus            | 1.0 | 0.7  | **T1w** (best contrast is T1w itself, yet the two values differ) |

- **Fix:** `L4_r_max_T1w_mm := L4_r_max_best_mm` (the T1w value is capped at the best value).
- **Why cap T1w instead of raising best:** in these blocks the T1w value is the implausible one.
  Thalamic nuclei and hippocampal subfields are not separable on standard T1w at 0.8 mm.
  Sub-millimetre (about 0.5 mm) data with a dedicated contrast is needed.
  Capping T1w is also the conservative option: it only makes structures merge sooner and never
  invents visibility. Rows where the values were already consistent (pallidum: T1w 0.8 < best 1.0
  with T2*/QSM; brainstem: 1.0 = 1.0) were not touched.

### 5.2 Background had r_max values
- **Problem:** `background` (0) had `0.7 / 0.5 / T2w / qMRI / image-driven` and `L5 = 0.3`.
  Background is the null node (mean = sigma = 0), so a resolution limit means nothing.
- **Fix:** I cleared `L4_r_max_T1w_mm`, `L4_r_max_best_mm`, `L4_best_contrast`, `L4_evidence`
  and `L5_r_max_best_mm`.


## 6. Conventions and naming

### 6.1 One convention for "whole structure" labels
Some labels stand for the whole structure, or the part of it not given to any subregion:

- `white_matter_of_forebrain` (7)
- `amygdaloid_complex` (214)
- `thalamus` (218)
- `myelencephalon` (662), `pons` (580) and `midbrain` (384)

v0 used different wordings for them: `Whole thalamus (unassigned)`, `Whole amygdala (unassigned)`,
`Cerebral white matter (unparcellated)` and `Whole level (unassigned)`.
I renamed `Cerebral white matter (unparcellated)` to `Cerebral white matter (unassigned)`.
Now every such row contains the word **unassigned**, which the build script uses:

- `unassigned` in `L4_subregion` → the label goes on the **L3** node itself
  (for example, label 218 on `thalamus`);
- `unassigned` in `functional_class` → the label goes on the **L4** node
  (for example, label 580 on `pons`).

This matches the SynthSeg tree, where label 2 is painted by the `cerebral_wm` node that also has children.

### 6.2 Names aligned with the NextBrain LUT
`leaf_name` mixed styles in v0: snake_case, "Title Case With Spaces", shortened names
("Inferior Vestibular") and typos ("inferiro").
To keep the table consistent with the label map:

- **48 rows** were renamed to the name in the in-vivo NextBrain LUT.
  This includes `ctx-rh-*` for the cortical parcels 2001–2035,
  e.g. `caudalanteriorcingulate` → `ctx-rh-caudalanteriorcingulate`.
- **15 rows** were renamed to the name in the full atlas LUT, e.g.:
  - `inferior_vestibular` → `spinal_(inferior)_vestibular_nucleus`
  - `facial_nerve` → `root_of_facial_nerve`
  - `transverse_pontocerebellar_tracts` → `pontocerebellar_tract`
- **32 rows** whose v0 name already matched the full LUT once written in snake_case were only
  normalised, e.g. `Ventral tegmental area` → `ventral_tegmental_area`.
  This also fixes the typo `commissure_of_inferiro_colliculus`.
- `background` (0) keeps its name; the LUT calls it `Unknown`.

After v1, every `allen_id` except 846 is in the full atlas LUT. All 259 labels of the test in-vivo
segmentation (`001_GX/seg.left.nii.gz`) are in the table.


## 7. Summary of the v1 hierarchy

| L1 tissue | L2 divisions |
|-----------|--------------|
| Background | Background |
| Brainstem | Brainstem (→ Medulla, Pons, Midbrain; Substantia-Nigra) |
| CSF / ventricles | Ventricular system |
| Cerebellar GM | Cerebellar cortex, **Cerebellar nuclei** |
| Cerebellar WM | **Cerebellar white matter** |
| Cerebral GM | Amygdala, Cerebral cortex, Hippocampal formation |
| Cerebral WM | Cerebral white matter, **Optic pathway** |
| Deep GM | Basal forebrain, Basal ganglia, Claustrum, Epithalamus, Hypothalamus, Subthalamus (+ **Zona-Incerta**), Thalamus |
| Other | **Pineal gland** |

The table now has 335 rows (334 + label 846). 
- Every L2, L3 and L4 has a single parent.
- Every L4 node has a single set of r_max values.
- `T1w <= best` holds on every row.
- IDs and names are unique.


## 8. How `_build_nextbrain_tree.py` uses the table

This is not a change to the table, but it sets how the columns are read.

- **Path:** `L1 → L2 → L3 → L4 → leaf`.
  Consecutive levels with the same name (ignoring case and punctuation: `Thalamus/Thalamus`,
  `Brainstem/Brain-Stem`) collapse into one node.
- **Redundant groups:** a group with no labels and a single child is merged with that child.
  It keeps its own r_max, because that is the resolution at which it splits from its siblings.
  Level-1 nodes keep their tissue name, e.g. `other` carries label 506.
- **r_max per contrast:**
  - The best-contrast value is the default for every contrast; T1 gets the T1w value.
  - An empty T1w cell next to a filled best cell means "not resolvable in T1w".
    It is encoded as `T1: 0.0` (hypothalamic and amygdala nuclei, SN, STN, deep cerebellar nuclei,
    cerebellar cortical layers).
- **Group r_max:** the table only gives r_max for L4 nodes (`L4_*`) and leaves (`L5_r_max_best_mm`).
  - L2/L3 groups take an illustrative value from `GROUP_R_MAX`, borrowed from the SynthSeg tree
    where the structure exists (thalamus 3.5, caudate 2.0, ...).
  - Groups not listed there use the coarsest r_max of their children.
  - Leaves are clamped to their parent's r_max.
- **Illustrative values:** the level-1 color weights (`L1_PARAMS`) and the jitters
  (`wr_mu`, `wr_sigma`), as in `_build_synthseg_tree.py`. WP1 will estimate them.


## 9. Open points (not changed, to be decided)

- **Cortical parcels:** both r_max cells are empty (`geometry / atlas-driven`).
  So the lobes take their leaves' 0.3 mm, and above 0.3 mm all DK parcels merge into one cortex group,
  including in `label_groups` for the WP4 loss. The SynthSeg tree used 1.0 mm for the parcels.
  This needs a decision: either give the cortex rows an explicit r_max, or treat parcels as
  always visible.
- **`L5_r_max_best_mm = 0.3` on every row** is a blanket value, not a per-structure estimate.
- **Mixed tissues kept in place** (fine for merging, not for intensity):
  - `septum_pellucidum` sits under the lateral ventricle (CSF);
  - `alveus` and `fimbria` sit under hippocampus (GM);
  - thalamic and hypothalamic WM tracts sit under Deep GM;
  - `lateral_olfactory_stria` sits under paleocortex.
- **Mapping columns** (`supersynth_label`, `fs_block`) were not reviewed in depth, except for the
  moved cerebellar nuclei. Some still disagree with the hierarchy.
  For example, `amygdalohippocampal_transition_area` has fs_block `Hippocampus` but L3 `Amygdala`.
- **Coverage:** the full atlas LUT has 918 labels; v1 covers 335.
  Missing examples: cerebellar lobules, 3rd and 4th ventricles, internal capsule, corpus callosum,
  and many cortical areas. v1 seems to be the in-vivo subset, so nothing was added beyond label 846.
