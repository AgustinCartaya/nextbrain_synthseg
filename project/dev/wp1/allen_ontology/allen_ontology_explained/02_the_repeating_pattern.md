# 02 · The repeating pattern (level 4)

This is the most important idea for not getting lost. **Each of the three vesicles is split the same
way, by tissue type**, before the anatomy starts:

| Kind | Forebrain (F) | Midbrain (M) | Hindbrain (H) | What it is |
|---|---|---|---|---|
| **Grey matter** | `FGM` (2111) | `MGM` (151) | `HGM` (449) | neuron cell bodies: nuclei and cortex |
| **White matter** | `FWM` (139) | `MWM` (46) | `HWM` (81) | fibre tracts (axons) that connect regions |
| **Ventricles** | `FV` (14) | `MV` (1) | `HV` (5) | cavities filled with cerebrospinal fluid |
| **Surface structures** | `FSS` (186) | `MSS` (9) | `HSS` (82) | what you see from outside: gyri, sulci, lobes, nerve roots |
| Blood vessels | `fbv` | `mbv` | `hbv` | empty (no children) |

The numbers in brackets are how many structures sit below each node. The spinal cord uses the same
pattern (`SGM`, `SWM`, `SV`, `SSS`).

```
F forebrain
├── FGM  grey matter      → the anatomy: cortex, nuclei   (03–07)
├── FWM  white matter     → fibre tracts                  (10)
├── FV   ventricles       → lateral and third ventricle   (11)
└── FSS  surface          → gyri, sulci, lobes            (11)
```

## A second pattern: "transient structures"

Inside the grey matter there is a branch for **transient structures**: `FTS` in the forebrain,
`MTS` in the midbrain and `HTS` in the hindbrain. These are layers and zones that exist **only
during development**, where neurons are born and migrate, and then disappear. Examples are the
ventricular zone, the subplate and the cortical plate. They are large: `FTS` alone has 651
structures. You can ignore them for adult brains; see [11](11_surface_ventricles_transient.md).

## Consequences for you

- **The same structure name can appear in several branches.** For example, cortex areas appear
  under the grey matter (FGM), as gyri under the surface (FSS), and as fetal zones under FTS.
- **A tract is never under the region it serves.** The fornix belongs to the hippocampal system,
  but it sits under `FWM`, not under the hippocampus.
- **Most of the real anatomy is under the grey-matter branches:** `FGM`, `MGM` and `HGM`.

Next: [03 · Forebrain overview](03_forebrain_overview.md)
