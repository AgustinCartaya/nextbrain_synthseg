# 13 · How different is Allen from `nextbrain_tree_v1.json`?

**In short:** the two trees agree on the main structures (thalamus, hippocampus, basal ganglia,
hypothalamus), but they disagree on how to group them above that.
- **v1** groups the labels by **tissue type**, which is what the intensity model needs.
- **Allen** groups them by **development** (forebrain / midbrain / hindbrain), splits by grey and
  white matter only below that, and puts every fibre tract in a separate white-matter branch.

## How it was measured

Both trees have the same **335 NextBrain leaves**. On the Allen side we use the Allen tree cut
down to those leaves (`nextbrain_allen_tree_gyral.json`, see 12).

Every internal node is reduced to **the set of labels below it**. Each group is then compared with
the group of the other tree that overlaps it most. The overlap is the **Jaccard index**: shared
labels divided by all labels. 1.0 means exactly the same leaves; 0 means none in common.

```bash
python ../compare_v1_allen.py                     # reproduces every number on this page
python ../compare_v1_allen.py --cortex developmental
```

## Shape

| | v1 | Allen (same leaves) | Allen (full) |
|---|---|---|---|
| Internal nodes | 131 | 164 | 3317 nodes in total |
| Distinct groupings | 99 | 151 | |
| Leaf depth | always 5 | 1 to 13 (mean 8.7) | up to 14 |

- **v1 has a fixed depth:** tissue → division → structure → subregion → leaf.
- **Allen has variable depth:** a hippocampal layer is 13 levels down, the whole midbrain label
  only 4.

## How much the groupings match

| v1 level | Groups | Same leaves in Allen | Overlap ≥ 0.8 | Overlap < 0.5 | Mean |
|---|---|---|---|---|---|
| L1 tissue | 9 | 1 | 1 | 4 | 0.46 |
| L2 division | 19 | 3 | 8 | 5 | 0.63 |
| L3 structure | 30 | 10 | 12 | 9 | 0.63 |
| L4 subregion | 73 | 28 | 32 | 26 | 0.61 |
| *Allen → v1* | *151* | *33* | *44* | *77* | *0.51* |

- **Identical groupings:** only **33** groupings appear identically in both trees.
- **Sibling pairs:** v1 makes **2078** pairs of leaves siblings (same parent), and Allen makes
  **549**. Only **345** pairs are siblings in both.
- **The tissue level matches worst** (mean 0.46), because Allen has no tissue level.

## Where they agree

| v1 division | Best Allen node | Overlap |
|---|---|---|
| claustrum, cerebellar nuclei, cerebellar cortex | Cla, CbDN, PVA | 1.00 |
| hippocampal formation | HIP | 0.96 |
| basal ganglia | BN | 0.90 |
| cerebral cortex | CeG (gyri) | 0.89 |
| hypothalamus | HTH | 0.88 |
| thalamus | DTH | 0.85 |

## The five big differences

1. **The top of the tree.** v1 starts with tissue (Cerebral GM, Deep GM, Brainstem, ...). Allen
   starts with neural plate → neural tube → brain, then forebrain, midbrain and hindbrain (see 01).
   **20 of Allen's 164 groups mix leaves of different v1 tissues.**
2. **The brainstem.** v1 has one Brainstem group with 90 leaves. Allen has no brainstem node: 35 of
   those leaves are under the **midbrain** and 55 under the **hindbrain**, where they share the
   branch with the cerebellum. The best single match is 0.56.
3. **White-matter tracts.** v1 puts a tract with the structure it serves. Allen puts every tract
   under white matter (see 10):
   - 9 Deep-GM leaves of v1 are tracts under `FWM` in Allen: mammillothalamic tract, stria
     medullaris, thalamic fasciculus, inferior thalamic peduncle, lenticular fasciculus, habenular
     commissure, and others.
   - The alveus, fimbria and lateral olfactory stria are Cerebral GM in v1 but `FWM` in Allen.
   - v1's *thalamus* group (62 leaves) loses 9 leaves in Allen. Four are these tracts. The others
     are the reticular nucleus and its two divisions, the paraventricular nucleus (in the
     epithalamus) and the whole-thalamus label, which sit in other branches of `THM`.
   - v1's *cerebral white matter* (0.20) and *optic pathway* (0.24) groups are spread across
     Allen's long list of ~75 tracts.
4. **The cerebellum.** v1 has separate Cerebellar GM and Cerebellar WM tissues. Allen has **no
   cerebellar white-matter node**, so it shares the hindbrain white-matter node with the brainstem
   tracts (overlap 0.11).
5. **The amygdala and the cortex.**
   - Allen keeps the three amygdala transition areas (`ATA`) outside the amygdala node (overlap
     0.79).
   - The amygdala is a **cerebral nucleus** in Allen (next to the basal ganglia), whereas v1 puts it
     in Cerebral GM.
   - The Desikan-Killiany cortex parcels go to the surface gyri branch (`FSS → CeG`) in gyral mode.
     This is separate from the grey-matter branch that holds the hippocampus (see 11).

## Where each v1 tissue lands in Allen

| v1 tissue | Leaves | Allen branches |
|---|---|---|
| Cerebral GM | 105 | F > FGM 68, F > FSS 34, F > FWM 3 |
| Deep GM | 120 | F > FGM 110, F > FWM 9, F > FSS 1 |
| Brainstem | 90 | M > MGM 22, M > MWM 10, M > MSS 2, M 1 · H > HGM 37, H > HWM 16, H > HSS 2 |
| Cerebral WM | 8 | F > FWM 8 |
| Cerebellar GM | 6 | H > HGM 6 |
| Cerebellar WM | 2 | H > HWM 2 |
| CSF / ventricles | 2 | F > FV 1, F > FSS 1 |
| Other (pineal) | 1 | F > FGM 1 |

## Practical takeaway

- **Allen is a good source of anatomical detail and relationships**, but it mixes tissues within one
  group, so as a hierarchy it is a poor fit for the intensity model.
- This is why the converted label tree (`nextbrain_allen_label_tree_*.json`,
  `convert_to_label_tree.md`) keeps **v1's tissue level on top** and uses **Allen only below it**.

Back to the index: [00_README.md](00_README.md)
