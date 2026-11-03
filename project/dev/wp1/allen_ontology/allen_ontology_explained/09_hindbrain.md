# 09 · Hindbrain (H)

The hindbrain grey matter follows the embryo again. The back vesicle splits into two:
- **metencephalon** = cerebellum + pons;
- **myelencephalon** = medulla.

```
H hindbrain (622)
├── HGM grey matter (449)
│   ├── Met  metencephalon (205)
│   │   ├── CB  cerebellum (67)
│   │   └── Pn  pons       (136)
│   ├── Mo   myelencephalon = medulla oblongata (137)
│   └── HTS  transient (fetal) structures (104)
├── HWM white matter (81)   cerebellar peduncles, pyramidal tract, lemnisci, ...
├── HV  ventricles          4th ventricle
└── HSS surface (82)        cerebellar lobes and lobules, nerve roots, ...
```

## Cerebellum (CB)

```
CB cerebellum
├── CBC   cerebellar cortex
│   ├── CBV  vermis (midline)        → anterior / posterior / flocculonodular lobe
│   └── CBH  hemisphere
│       ├── CBPV paravermis (intermediate) → anterior / posterior / flocculonodular
│       └── CBL  lateral hemisphere        → anterior / posterior / flocculonodular
└── CbDN  deep nuclei: DT dentate, InP interpositus, Fas fastigial
```

The cerebellar cortex is cut in two directions:
- **side to side:** vermis → paravermis → lateral hemisphere;
- **front to back:** anterior → posterior → flocculonodular lobe.

Each strip sends its output to its own deep nucleus.

| Part | Output nucleus | Function |
|---|---|---|
| Vermis | fastigial | posture, balance, walking, eye movements |
| Paravermis | interpositus | adjusting limb movements while they happen |
| Lateral hemisphere | dentate | planning and timing skilled movements; also thinking and language (with the prefrontal cortex) |
| Flocculonodular lobe | vestibular nuclei | balance and keeping your gaze stable |

The cerebellum's surface lobules (I–X, culmen, declive, ...) are listed separately under
`HSS → CbSS → CBLL`.

## Pons (Pn)

| Part | Contents | Function |
|---|---|---|
| `PnBa` basilar part (front) | PN pontine nuclei | relays the cortex to the cerebellum (via the middle cerebellar peduncle) |
| `PnTg` tegmentum (back) | cranial nerve nuclei V, VI, VII and VIII; auditory relays (superior olive); raphe; **PnRF** reticular formation; **nucleus coeruleus** (`NC`, in `PnNA`) | chewing and facial sensation (V), eye abduction (VI), facial expression (VII), hearing and balance (VIII), sound localisation, REM sleep, **noradrenaline**: alertness and stress (coeruleus; the `LC` node "locus coeruleus" is the surface landmark under `HSS → AS4V`) |

## Medulla (Mo)

| Group | Main nuclei | Function |
|---|---|---|
| `MoEN` motor cranial nuclei | Amb ambiguus, 12N hypoglossal, 10N dorsal vagal | swallowing and voice (ambiguus), tongue (XII), heart and gut control (vagus) |
| `MoAN` sensory cranial nuclei | Sol solitary, Sp5 spinal trigeminal, 8Ve vestibular | taste and the body's internal signals (solitary), face pain and temperature, balance |
| `MoSR` sensory relays | Gr gracile, Cu cuneate | fine touch and body position, on the way to the thalamus |
| `PrCbN` precerebellar | **IO inferior olive** | "error signals" to the cerebellum for motor learning |
| `MoRF` reticular formation | gigantocellular, intermediate, lateral reticular | **breathing rhythm** (pre-Bötzinger complex, under XMoTg), blood pressure, vomiting, posture |
| `MoRa` raphe | raphe magnus, obscurus, pallidus | serotonin to the spinal cord: pain control, motor tone |

The medulla controls **vital functions**: breathing, heart rate and blood pressure. Damage here can
be fatal.

Next: [10 · White matter](10_white_matter.md)
