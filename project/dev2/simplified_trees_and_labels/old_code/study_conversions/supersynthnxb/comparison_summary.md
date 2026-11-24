# SuperSynth labels (from NextBrain) vs SuperSynth segmentation: main problems

Summary of `study_conversions.ipynb`. All numbers come from `data/metrics/`.

**The three maps**

* **REF** is the simplified NextBrain labels converted to SuperSynth classes (`allennxb_to_supersynthnxb.py`,
  `NextBrainToSuperSynth_corrected.yaml`): `data/labels/case-N.nii.gz`.
* **SEG** is `mri_super_synth` run on the synthetic image:
  `.../deterministic/segmentation_nextbrain/case-N.nii.gz/SuperSynth/segmentation_resampled.nii.gz`.
* **FINE** is the simplified NextBrain map that REF was converted from (`../simplifiednxb/data/labels/case-N.nii.gz`).
  It is used to trace each disagreement back to the NextBrain label that causes it.

**How to read the tables**

* SuperSynth ids follow the FreeSurfer LUT. NextBrain ids are written as `NXB <id>`. 0 is background.
* SEG is harmonised to `Right-*` ids before comparing (see problem 5).
* `REF x -> SEG` lists where the REF voxels of class x go in SEG.
* `SEG x <- REF` lists what the SEG voxels of class x were in REF.
* Volumes are means over the 5 cases. Percentages are voxel fractions over all cases.

SuperSynth classes are large, so here (unlike in `simplifiednxb`) a low dice is not explained by structure size.

---

## 1. Mapping problems: NextBrain labels put in a class that SuperSynth defines differently

This is the main finding. For every NextBrain label, the notebook measures the fraction of its voxels that SEG puts
in the class the conversion gave it (`frac_ok`). The labels below have `frac_ok` ≈ 0 in all 5 cases. They explain
every class with dice < 0.6.

### 866 `Right-Basal-Forebrain`: dice 0.00 in all cases, centroid shift 19 mm

REF 805 mm³, SEG 291 mm³. **No** NextBrain label of this REF class lands in SEG 866:

| NextBrain label in REF 866 | vol (mm³) | SEG puts it in |
|---|---|---|
| NXB 310 `substantia_nigra__reticular_part` | 289 | 41 Right-Cerebral-White-Matter (64 %) |
| NXB 385 `red_nucleus` | 213 | 41 (70 %) |
| NXB 352 `substantia_nigra__compact_part` | 155 | 16 Brain-Stem (68 %) |
| NXB 321 / 316 / 315 `lateroventral / laterodorsal / medial_portion_of_sth` | 70 / 41 / 37 | 41 |

* `SEG 866 <- REF`: 74 % were REF 41.
* In the NextBrain labels these SEG voxels are NXB 120 `ventral_pallidus` (68 % of it goes to SEG 866) and
  NXB 108 `basal_nucleus_of_meynert` (63 % goes to SEG 866). The conversion puts both of them in REF 41
  white matter.
* **Fix:** move NXB 310, 352, 385, 315, 316 and 321 out of Basal-Forebrain (to Brain-Stem 16 or to ventral DC /
  white matter). Move NXB 120, 108 and probably 128 `substantia_innominata` into Basal-Forebrain.

### 58 `Right-Accumbens-area`: dice 0.26, centroid shift 22 mm

REF 1 748 mm³, SEG 603 mm³.

| NextBrain label in REF 58 | vol (mm³) | frac_ok | SEG puts it in |
|---|---|---|---|
| NXB 384 `midbrain_(mesencephalon)` | 935 | 0.00 | 16 Brain-Stem |
| NXB 101 `core_of_nucleus_accumbens` | 467 | 0.65 | 51 Putamen (rest) |
| NXB 208 `optic_tract` | 232 | 0.00 | 41 White-Matter |
| NXB 435 `ventral_tegmental_area` | 40 | 0.00 | 16 Brain-Stem |
| NXB 111 `diagonal_band` | 33 | 0.02 | 866 Basal-Forebrain |
| NXB 843 `infundibular_stalk` | 21 | 0.00 | 24 CSF |
| NXB 412 `fasciculus_retroflexus`, NXB 498 `peripeduncular_nucleus` | 10 / 10 | 0.00 | 49 Thalamus / 41 |

* Only NXB 101 belongs in Accumbens. The other labels (more than half of the volume) are midbrain, optic and
  hypothalamic structures, which is why the REF centroid lies 22 mm from SEG's.
* **Fix:** NXB 384, 435 -> 16 Brain-Stem. NXB 208 -> 41 (or ventral DC). NXB 111 -> 866 Basal-Forebrain.
  NXB 843 -> 820 HypoThal-noMB.

### 844 `Right-MammillaryBody`: dice 0.21

REF 297 mm³, SEG 54 mm³ (-81 %).

* The real mammillary nuclei agree reasonably: NXB 305, 306 and 307 have `frac_ok` 0.50-0.60.
* REF also contains posterior hypothalamic structures that SEG never puts in 844:
  * NXB 275 `posterior_hypothalamic_nucleus` (124 mm³, SEG puts it in 49 Thalamus);
  * NXB 297 `lateral__hypothalamic_area__posterior_part` (79 mm³, 41);
  * NXB 256 `tuberomammillary_nucleus` (30 mm³, 41).
* **Fix:** NXB 275, 297, 256 -> 820 HypoThal-noMB.

### 870 `Right-SeptalNuc` (dice 0.23), 822 `Right-Fornix` (0.40), 85 `Optic-Chiasm` (0.34)

* NXB 103 `medial_septal_nucleus` (= REF 870, 232 mm³): SEG puts 37 % in 43 Lateral-Ventricle and 22 % in 870.
* NXB 199 `fornix` (= REF 822, 273 mm³): 47 % in 822, and the rest mainly in 43 Lateral-Ventricle.
* NXB 161 `optic_chiasm` (106 mm³, 35 % in 85) and NXB 114 `optic_nerve` (56 mm³, 95 % in 24 CSF).
* These are not definition problems. They are thin structures next to the ventricles or CSF, and SuperSynth labels
  them as fluid. NXB 114 `optic_nerve` lies outside the brain for SuperSynth and could go to background (0).

### Other NextBrain labels in the wrong class

| NextBrain label | REF class | vol (mm³) | frac_ok | SEG puts it in |
|---|---|---|---|---|
| NXB 354 `alveus` | 53 Hippocampus | 918 | 0.27 | 41 White-Matter / 44 Inf-Lat-Vent |
| NXB 184 `stria_terminalis` | 50 Caudate | 315 | 0.05 | 41 White-Matter |
| NXB 393 `tail_of_caudate` | 50 Caudate | 200 | 0.00 | 41 White-Matter |
| NXB 68 `septum_pellucidum` | 41 White-Matter | 249 | 0.07 | 43 Lateral-Ventricle (75 %) |
| NXB 117 `bed_nucleus_of_stria_terminalis` | 41 White-Matter | 228 | 0.18 | 50 Caudate (31 %) |
| NXB 309 `lenticular_fasciculus` | 820 HypoThal-noMB | 43 | 0.01 | 49 Thalamus |
| NXB 506 `pineal_body` | 49 Thalamus | 30 | 0.00 | 24 CSF |

* NXB 393 `tail_of_caudate` is the clearest case: SuperSynth does not label the thin tail of the caudate.
* NXB 354 `alveus` and NXB 184 `stria_terminalis` are white-matter sheets/tracts, which fits SEG labelling them
  white matter.

## 2. Outer cortex and CSF (the largest disagreement in volume)

SEG labels CSF (24), a class REF never contains, for ~88 cm³ per case:

* `SEG 24 <- REF`: 56 % REF background and **38 % REF 42 Cerebral-Cortex**. That is ~34 cm³ per case: the single
  largest confusion, 21 % of all disagreeing voxels.
* The NextBrain labels involved are the outer cortex parcels (NXB 2028 `ctx-rh-superiorfrontal`, 2027, 2024, 2029,
  2022...), each losing 15-25 % of its volume to CSF.
* As a result, REF 42 Cerebral-Cortex is 12 % smaller in SEG (258 -> 228 cm³; dice 0.84).
* This is the same problem as in `simplifiednxb` (there, NextBrain labelled the outer cortex as background). The
  generator's outer cortical border looks like CSF to both tools.
* Cortex and white matter also exchange ~15-17 cm³ per case in each direction (REF 42 -> SEG 41 and REF 41 -> SEG 42):
  a normal gray/white boundary error.

## 3. Ventricles: SEG labels them, REF has background (or tissue) there

| SEG class | SEG vol per case | `SEG x <- REF` |
|---|---|---|
| 43 `Right-Lateral-Ventricle` | 9.2 cm³ | 59 % REF 0 (background inside the brain), 31 % REF 41 (NXB 7 `white_matter_of_forebrain`), plus NXB 68, 48, 354, 118, 103, 184 |
| 44 `Right-Inf-Lat-Vent` | 0.75 cm³ | 44 % REF 53 Hippocampus (mostly NXB 354 `alveus`), REF 41 |
| 14 `3rd-Ventricle` / 15 `4th-Ventricle` | 0.25 / 0.16 cm³ | mostly REF 0 |
| 77 `WM-hypointensities` | 0.49 cm³ | 85 % REF 41 |
| 915 `Bone-Cortical`, 907 `Other-Tissues` | < 0.1 cm³ | REF 0 |

* The generator's label maps contain no ventricle label: the ventricles are background (0), and NXB 50
  `anterior_horn_of_lateral_ventricle` is not used. SuperSynth also extends the ventricles into the periventricular
  structures (septum, fornix, septal nucleus, caudate border).
* For SuperSynth-style training labels, the ventricles would need a class in REF. For example, REF 0 voxels inside
  the brain mask could become 43.

## 4. Cerebellum and brainstem: border errors, no definition problem

| class | REF vol | SEG vol | dice |
|---|---|---|---|
| 47 `Right-Cerebellum-Cortex` | 45 046 | 47 420 | 0.86 |
| 46 `Right-Cerebellum-White-Matter` | 12 581 | 11 714 | 0.72 |
| 16 `Brain-Stem` | 6 659 | 6 578 | 0.69 |

* NXB 611 `white_matter_of_hindbrain` (REF 46): 8.5 cm³ go to SEG 46, 2.2 cm³ to 47 and 1.1 cm³ to 16. The
  part that NextBrain calls brainstem white matter is labelled Brain-Stem by SuperSynth, which is consistent with the
  611/846 problem found in `simplifiednxb`.
* NXB 414 `pontine_nucleus` (REF 16): 2.3 cm³ go to SEG 16 and 0.8 cm³ to 24 CSF (the ventral surface of the pons).
* Cerebellar cortex and white matter exchange ~2.5 cm³ in each direction (REF 47 -> SEG 46, REF 46 -> SEG 47): a
  border error. The molecular/granular confusion of NextBrain does not matter here, because both layers are in 47.

## 5. Hemisphere ids (handled, but keep in mind)

* `data/labels/sides.json` uses side **R** for every case, so REF has `Right-*` ids, also for cases 4-5.
* SuperSynth found `seg_side` = right for cases 1-3 and **left** for cases 4-5 (`Left-*` ids, e.g. 2/3/10). This
  matches the NextBrain run.
* For the comparison, every SEG `Left-*` id is mapped to its `Right-*` id. Without that mapping, cases 4-5 would have
  dice 0 for every lateralised class.
* SEG voxels with the ids of the other side are negligible: 0-320 mm³ per case.

## What is fine

* 49 Thalamus 0.89, 41 Cerebral-White-Matter 0.89, 51 Putamen 0.87, 47 Cerebellum-Cortex 0.86, 42 Cerebral-Cortex
  0.84, 52 Pallidum 0.83, 54 Amygdala 0.81, 50 Caudate 0.81, 53 Hippocampus 0.79 (mean dice over cases).
  * Thalamus is fine even though REF 49 groups 68 NextBrain nuclei: 90 % of its voxels agree.
* Total brain volume without CSF: SEG is 2-7 % smaller than REF.
* No mutual swaps. Misplaced classes are only 866, and sometimes 844, 870 and 58, all explained by problem 1.

## Priorities

1. Correct `NextBrainToSuperSynth_corrected.yaml` for Basal-Forebrain 866, Accumbens 58 and MammillaryBody 844
   (problem 1). This fixes the three worst classes.
2. Decide what to do with the classes that REF lacks: ventricles 43/44/14/15 (problem 3) and CSF 24 (problem 2).
3. The outer cortex looking like CSF (problem 2) is a generator problem shared with the NextBrain comparison.
4. If REF and real SuperSynth labels are mixed, convert REF with the real side (`--side auto`) or harmonise the ids
   (problem 5).
