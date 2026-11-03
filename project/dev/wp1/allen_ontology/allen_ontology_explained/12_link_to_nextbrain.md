# 12 · Which parts of Allen the NextBrain labels use

`build_nextbrain_allen_tree.py` hangs the **335 NextBrain labels** under their Allen parents (gyral
mode). Only **165 of the 3317** Allen nodes are needed. This is where the labels land:

| Allen branch | NextBrain leaves | What they are |
|---|---|---|
| `F → FGM` forebrain grey matter | **179** | hippocampal fields and layers, amygdala nuclei, thalamic and hypothalamic nuclei, basal ganglia, basal forebrain |
| `F → FSS` forebrain surface | 36 | the 34 Desikan-Killiany cortical parcels (via `CeG` gyri) + 2 others |
| `F → FWM` forebrain white matter | 20 | fornix, internal capsule, optic tract, ... |
| `F → FV` forebrain ventricles | 1 | |
| `H → HGM` hindbrain grey matter | 43 | cerebellar cortex and nuclei, pontine and medullary nuclei |
| `H → HWM` hindbrain white matter | 18 | cerebellar peduncles, pyramidal tract, lemnisci, ... |
| `H → HSS` hindbrain surface | 2 | |
| `M → MGM` midbrain grey matter | 22 | substantia nigra, red nucleus, colliculi, raphe, ... |
| `M → MWM` midbrain white matter | 10 | cerebral peduncle, decussations, ... |
| `M → MSS` midbrain surface | 2 | |
| directly under `M` | 1 | the whole midbrain label |
| root (no Allen node) | 1 | background |

## What this tells you

- **NextBrain is detailed in the deep structures**: thalamus, hippocampus, amygdala and brainstem
  nuclei. It is **coarse in the cortex**, with only 34 gyral parcels and no Brodmann areas or layers.
- **Branches NextBrain never uses:** the transient structures (FTS, MTS, HTS), the spinal cord,
  most of the neocortex areas under `NCx`, the cortical layers and the sulci (`CeS`). That is about
  95% of the Allen tree.
- **The cortex can hang in two places** (see [11](11_surface_ventricles_transient.md)):
  - under the gyri `FSS → CeG` in gyral mode;
  - under the neocortex lobes `NCx` in developmental mode (`--cortex developmental`).

## See it

```bash
cd ../plot_allan_ontology
python allen_tree_plot.py --i ../data/outputs/nextbrain_allen_tree_gyral.json --max-depth 5
python allen_tree_plot.py --i ../data/outputs/nextbrain_allen_tree_gyral.json --root THM
```

Back to the index: [00_README.md](00_README.md)
