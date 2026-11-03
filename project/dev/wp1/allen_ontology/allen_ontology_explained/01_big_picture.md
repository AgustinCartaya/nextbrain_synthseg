# 01 · The big picture (levels 0–3)

```
NP  neural plate                       level 0  (root)
└── NT  neural tube                    level 1
    ├── Br  brain                      level 2  (3292 structures below it)
    │   ├── F  forebrain               level 3  (2455)  ← 75% of the tree
    │   ├── M  midbrain                level 3  ( 212)
    │   └── H  hindbrain               level 3  ( 622)
    └── SpC spinal cord                level 2  (  21)
```

## Why does it start with a "neural plate"?

This is a **developmental** ontology, because the atlas follows the brain from embryo to adult. So
the tree is organised the way the brain **grows**:

1. In the embryo, the nervous system starts as a flat sheet: the **neural plate**.
2. The sheet folds into a tube: the **neural tube**.
3. The front end of the tube swells into three bubbles ("vesicles"). These become the **forebrain**,
   the **midbrain** and the **hindbrain**. The rest of the tube becomes the **spinal cord**.

The top of the tree is therefore an embryology lesson. NP and NT contain everything, so they carry
no useful information when segmenting an adult brain.

## The three vesicles, in one line each

| Part | Contains (main structures) | Main role |
|---|---|---|
| **F forebrain** (prosencephalon) | cerebral cortex, basal ganglia, amygdala, hippocampus, thalamus, hypothalamus | thinking, memory, emotion, perception, voluntary movement, hormones |
| **M midbrain** (mesencephalon) | colliculi, substantia nigra, red nucleus, periaqueductal grey | eye movements, visual and auditory reflexes, dopamine, motor control, pain control |
| **H hindbrain** (rhombencephalon) | cerebellum, pons, medulla | coordination and balance, breathing, heart rate, swallowing, sleep |
| **SpC spinal cord** | spinal grey and white matter | connects the brain with the body |

The midbrain, pons and medulla together are what clinicians call the **brainstem**. Allen does
**not** have a "brainstem" node. The midbrain sits under M, and the pons and medulla sit under H.
This matters when you look for the "brainstem" in the tree.

Next: [02 · The repeating pattern](02_the_repeating_pattern.md)
