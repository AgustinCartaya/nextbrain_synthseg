# 11 · Surface, ventricles and transient structures

## Why the same cortex appears **three times**

This is the biggest source of confusion. The same piece of cortex can be described in three
branches:

| Branch | Point of view | Example for the auditory cortex |
|---|---|---|
| `FGM → Tel → Cx → NCx` | **cytoarchitecture**: what the cells look like, the Brodmann areas | `TCx → DLTC → A1C` primary auditory cortex |
| `FSS → CeG` (gyri) and `FSS → CeS` (sulci) | **macroscopic shape**: what you see on the surface (and on MRI) | `TemL → TTG` transverse temporal gyrus (Heschl's) |
| `FGM → FTS` (CP, SP, VZ, ...) | **development**: fetal layers under that cortex | `tCP` cortical plate of temporal cortex |

These are **not different structures**; they are different maps of the same tissue. A gyrus
(e.g. the precentral gyrus) roughly matches an area (the motor cortex, `PoFC`), but the borders
are not the same. This is why NextBrain's Desikan-Killiany labels, which are defined by gyri, can
attach either to `CeG` (gyral mode) or to `NCx` (developmental mode).

## FSS: surface structures of the forebrain

```
FSS
├── CeS   cerebral sulci  (PriS primary: central, lateral, calcarine, ...; SecS secondary)
├── CeG   cerebral gyri and lobules
│   ├── FroL frontal lobe   (PrCG precentral, SFG, MFG, IFG, orbital gyri, frontal pole, ...)
│   ├── ParL parietal lobe  (PoCG postcentral, SPL, IPL, precuneus, ...)
│   ├── TemL temporal lobe  (STG, MTG, ITG, fusiform, TTG Heschl, temporal pole, ...)
│   ├── OccL occipital lobe (cuneus, lingual, occipital pole, ...)
│   ├── InL  insular lobe   (long and short insular gyri)
│   └── LimL limbic lobe    (CgG cingulate, PHG parahippocampal, HiF hippocampal gyrus, ...)
├── ASFV  structures next to the ventricles (choroid plexus, septum pellucidum, lamina terminalis, ...)
└── small landmarks: mammillary body, tuber cinereum, infundibular stalk, limen insulae, ...
```

## Ventricles (CSF cavities)

| Node | Cavity |
|---|---|
| `FV → LV` lateral ventricles | anterior (frontal) horn, body, atrium, posterior (occipital) horn, inferior (temporal) horn |
| `FV → 3V` third ventricle | the midline slit inside the diencephalon |
| `FV → IVF` interventricular foramen | the hole that connects each lateral ventricle with the third ventricle |
| `MV → Aq` cerebral aqueduct | the canal through the midbrain |
| `HV` fourth ventricle | between the pons/medulla and the cerebellum |

CSF is made by the **choroid plexus** (under `ASFV`) and flows from the lateral ventricles to the
third ventricle, then the aqueduct, then the fourth ventricle.

## Transient structures (FTS, MTS, HTS): only in development

The cortex is built from the inside out. Neurons are born next to the ventricle and migrate outwards.
`FTS` lists these fetal layers, **from deep to surface**:

| Zone | What happens there |
|---|---|
| **VZ** ventricular zone | stem cells divide and make neurons (`GE` ganglionic eminence = the VZ that makes interneurons and basal-ganglia neurons) |
| **SZ** subventricular zone | a second proliferative layer, very large in humans |
| **IZ** intermediate zone | migrating neurons; it becomes the white matter |
| **SP** subplate | a temporary layer that guides incoming axons; it disappears around birth |
| **CP** cortical plate | the future cortex (layers II–VI) |
| **MZ** marginal zone | the future layer I |
| **SG** subpial granular zone | a transient layer of migrating cells under the surface |

Each zone is again split by lobe (e.g. `fCP`, `pCP`, `tCP`, `oCP`), which is why `FTS` has 651
nodes. **For adult MRI segmentation you can ignore FTS, MTS and HTS completely.**

Next: [12 · Link to NextBrain](12_link_to_nextbrain.md)
