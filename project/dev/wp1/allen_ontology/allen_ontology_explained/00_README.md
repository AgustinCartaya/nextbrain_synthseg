# The Allen brain ontology, explained step by step

`data/inputs/AllenBrainOntologyDev.json` is the ontology of the **Allen Developing Human Brain
Atlas** (BrainSpan). It is **one tree** with **3317 structures**, up to **14 levels** deep and with
2429 leaves. It is big, but it is built from a few simple ideas that repeat everywhere.

Read the files in order. Each one zooms in on the previous one.

| # | File | What you learn |
|---|---|---|
| 01 | [01_big_picture.md](01_big_picture.md) | The first 3 levels: brain = forebrain + midbrain + hindbrain |
| 02 | [02_the_repeating_pattern.md](02_the_repeating_pattern.md) | **The key idea:** every part is split into grey matter / white matter / ventricles / surface |
| 03 | [03_forebrain_overview.md](03_forebrain_overview.md) | Forebrain: telencephalon and diencephalon |
| 04 | [04_cerebral_cortex.md](04_cerebral_cortex.md) | The neocortex, its lobes and its areas |
| 05 | [05_hippocampus_and_old_cortex.md](05_hippocampus_and_old_cortex.md) | Hippocampus, entorhinal cortex and olfactory cortex |
| 06 | [06_cerebral_nuclei.md](06_cerebral_nuclei.md) | Basal ganglia, amygdala, claustrum and basal forebrain |
| 07 | [07_diencephalon.md](07_diencephalon.md) | Thalamus, hypothalamus, epithalamus and subthalamus |
| 08 | [08_midbrain.md](08_midbrain.md) | Tectum, tegmentum, substantia nigra and red nucleus |
| 09 | [09_hindbrain.md](09_hindbrain.md) | Cerebellum, pons and medulla |
| 10 | [10_white_matter.md](10_white_matter.md) | Fibre tracts and how they are grouped |
| 11 | [11_surface_ventricles_transient.md](11_surface_ventricles_transient.md) | Why the same cortex appears **three times** in the tree |
| 12 | [12_link_to_nextbrain.md](12_link_to_nextbrain.md) | Which branches the NextBrain labels use |
| 13 | [13_comparison_with_v1.md](13_comparison_with_v1.md) | How different Allen is from `nextbrain_tree_v1.json`, measured |

## How to read a node

Each node has the following fields:

| Field | Meaning | Example |
|---|---|---|
| `id` | unique number | `10390` |
| `acronym` | short name | `THM` |
| `name` | full name | `thalamus` |
| `color_hex_triplet` | colour of the structure | `E4CCE3` |
| `children` | the parts this structure is divided into | |

A parent is the **sum of its children**. For example, the thalamus is made of the dorsal thalamus,
the epithalamus, the ventral thalamus and the reticular nucleus.

**Careful with acronyms.** They are case-sensitive: `CB` is the cerebellum, but `cb` is the cingulum
bundle, a fibre tract. As a rule, **lowercase acronyms are white-matter tracts**.

## Explore it yourself

```bash
cd ../plot_allan_ontology
python allen_tree_plot.py --i ../data/inputs/AllenBrainOntologyDev.json --max-depth 4 --print
python allen_tree_plot.py --i ../data/inputs/AllenBrainOntologyDev.json --root THM --max-depth 2
```

The functions given for each region are a **simplified textbook summary**. Real brain functions
involve networks of regions, not single boxes.
