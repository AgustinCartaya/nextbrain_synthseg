# Simplified NextBrain labels vs NextBrain segmentation: main problems

Summary of `study_conversions.ipynb`. Every number comes from `data/outputs/simplifiednxb/metrics/`.

**The two maps**

* **REF** is the reference, the simplified label map: `data/outputs/simplifiednxb/labels/case-N.nii.gz`. The generator's
  labels contain none of the 71 removed labels (`removed_voxels = 0` in every case), so REF is exactly the label map
  the synthetic image was generated from.
* **SEG** is the NextBrain segmentation of that synthetic image:
  `.../deterministic/segmentation_nextbrain/case-N.nii.gz/seg.<side>_resampled.nii.gz`.

**How to read the tables**

* Label ids are the same in both maps: the NextBrain LUT. Label 0 is background.
* `REF x -> SEG` lists where the REF voxels of label x end up in SEG.
* `SEG x <- REF` lists what the SEG voxels of label x were in REF.
* Percentages are voxel fractions summed over the 5 cases.
* Volumes are means over the 5 cases.

Small structures (< ~100 mm³) are expected to have low dice at 0.4 mm and are not counted as problems here unless they
fail in a systematic way. The problems are sorted by severity.

---

## 1. Cerebellar white matter: REF and SEG use different label ids (611 vs 846)

This is a naming convention, not a segmentation error.

| | REF | SEG |
|---|---|---|
| cerebellar white matter | **611** `white_matter_of_hindbrain` | **846** `white_matter_of_the_cerebellum` |
| brainstem white matter | **611** `white_matter_of_hindbrain` | **611** `white_matter_of_hindbrain` |
| label 846 | never used (0 voxels) | ~9 906 mm³ |
| label 611 volume | ~12 185 mm³ | ~1 912 mm³ |

* `REF 611 -> SEG`: 77 % become **846**, 6 % become 580 `pons`, 5 % become 597 `granular_cell_layer_of_pva`.
* `SEG 846 <- REF`: 95 % were **611**.
* `SEG 611 <- REF`: 22 % were 414 `pontine_nucleus`, 18 % were 580 `pons`, 11 % were 7 `white_matter_of_forebrain`.
  So SEG 611 is only the brainstem white matter.
* **Effects:**
  * Label 611 gets dice 0.03 and a centroid shift of 24.5 mm. It looks like the most misplaced label, but it is not misplaced.
  * 846 shows up as "only in SEG".
  * This difference causes ~7 % of all disagreeing voxels.
* **Fix:** the simplified tree already holds both ids in one leaf (`white_matter_of_hindbrain: [846, 611]`). After merging
  the two ids, the dice is ~0.8. Any direct comparison of label maps, or training on them, must map
  SEG 846 -> 611, or split REF 611 into 846 (cerebellum) and 611 (brainstem).

## 2. Cerebellar cortex layers: SEG does not separate 595 and 597

| label | REF vol (mm³) | SEG vol (mm³) | dice (min-max) |
|---|---|---|---|
| **595** `molecular_layer_of_pva` | 24 620 | 23 132 | 0.50 (0.47-0.53) |
| **597** `granular_cell_layer_of_pva` | 20 425 | 23 378 | 0.47 (0.44-0.50) |

* `REF 595 -> SEG`: 48 % stay **595**, 44 % become **597**, 7 % become 0.
* `REF 597 -> SEG`: 50 % stay **597**, 42 % become **595**, 7 % become 0.
* `SEG 597 <- REF`: 46 % were **595** and 44 % were **597**.
* These are two of the largest structures, and they are mixed almost 50/50. The labels are not swapped. SEG simply
  cannot tell the two layers apart in the synthetic image. Either the generator gives them similar intensities, or the
  layers are too thin and interleaved at 0.4 mm.
* The confusion between 595 and 597 is the largest source of disagreement: **~14 % of all disagreeing voxels**.
* **To check:** the intensity distributions the generator gives to 595 and 597. Another option is to merge the two
  layers into one cerebellar cortex label.

## 3. SEG puts background on the outer cortex

| case | REF voxels that become SEG 0 (cm³) | of which REF cortex 2001-2035 | of which REF 7 `white_matter_of_forebrain` | total volume change (SEG vs REF) |
|---|---|---|---|---|
| 1 | 27.0 | 17.1 | 1.6 | +0 % |
| 2 | 37.6 | 26.1 | 3.5 | -5 % |
| 3 | 53.6 | 40.9 | 5.2 | -9 % |
| 4 | 44.9 | 32.4 | 3.8 | -6 % |
| 5 | 42.2 | 31.6 | 3.1 | -6 % |

* Most of the loss is REF cortex, mainly:
  * 2028 `ctx-rh-superiorfrontal` (~2.8 cm³ per case);
  * 2024 `ctx-rh-precentral` (~2.3);
  * 2027 `ctx-rh-rostralmiddlefrontal` (~2.2);
  * 2022 `ctx-rh-postcentral` (~1.8);
  * 2029 `ctx-rh-superiorparietal` (~1.7).
* REF 595 `molecular_layer_of_pva` loses ~1.7 cm³ per case.
* SEG puts the outer edge of the cortex further inside than REF. In case 1 the loss is compensated: SEG also adds
  27 cm³ where REF is 0. In cases 2-5 it is not.
* The cortex parcels still have dice 0.75-0.87, so this is a boundary shift and not a parcellation error.
* **To check:** how the generator renders the outer border of the brain and the image artifacts seen there (stripes in
  the slices of section 7.4 of the notebook).

## 4. Large and medium REF labels that SEG gives a different extent or position (all 5 cases)

These structures are too large for the "small structure" explanation, and the pattern repeats in every case. In most
of them, a REF label and a SEG label with the same id cover different regions. That points to a difference in
definition between the atlas labels (REF) and NextBrain's segmentation (SEG), rather than random errors.

### 4a. General or parent labels (pons, thalamus, midbrain white matter)

**580 `pons`**: REF 1 097 mm³, SEG 2 067 mm³, dice 0.07.
* `REF 580 -> SEG`: 31 % become 611, 27 % become 0, only 10 % stay 580.
* `SEG 580 <- REF`: 55 % were 414 `pontine_nucleus`, 33 % were 611.
* In SEG, `pons` is mostly the region that REF calls `pontine_nucleus`.

**218 `thalamus`** (general label): REF 823 mm³, SEG 410 mm³, dice 0.15.
* `REF 218 -> SEG`: only 12 % stay 218, and 11 % become 222 `parvocellular_division_of_va`.
* `SEG 218 <- REF`: 24 % were 218, 13 % were 254 `reticular_nucleus_of_thalamus`, 11 % were 384 `midbrain_(mesencephalon)`.
* REF and SEG use this general label for different leftover regions.

**531 `superior_cerebellar_peduncle_(brachium_conjunctivum)`**: REF 444 mm³, SEG 95 mm³, dice 0.23 (0.00 in case 1).
* `REF 531 -> SEG`: 40 % become 611, 15 % stay 531, 13 % become 580.
* Because of this, the level-1 tree node `white_matter_of_midbrain` has dice 0.00-0.36. Its REF volume is 340-640 mm³;
  its SEG volume is 80-135 mm³.

### 4b. Specific structures that SEG absorbs into a neighbour or into background

| REF label | REF vol | SEG vol | dice | `REF x -> SEG` | `SEG x <- REF` |
|---|---|---|---|---|---|
| **354** `alveus` | 918 | 566 | 0.08 | 7 `white_matter_of_forebrain` 46 %, **0** 41 %, 354 only 6 % | 345 `stratum_pyramidale_of_rostral_ca1` 22 %, 347 `rostral_subiculum` 11 % |
| **128** `substantia_innominata` | 485 | 44 | 0.08 | 7 `white_matter_of_forebrain` 29 %, 79 `putamen` 12 %, 0 9 % | 128 49 %, 125 `lateral_olfactory_area` 12 % |
| **184** `stria_terminalis` | 315 | 82 | 0.02 | **0** 49 %, 7 `white_matter_of_forebrain` 43 %, 184 only 1 % | 118 `body_of_caudate` 22 %, 295 `amygdalostriatal_transition_area` 10 % |
| **68** `septum_pellucidum` | 249 | 7 | 0.03 | **0** 95 % | 7 51 %, 68 37 % |
| **654** `inferior_olive__principal_nucleus` | 160 | 4 | 0.01 | 662 `myelencephalon_(medulla_oblongata)` 59 %, **0** 38 % | 662 35 %, 580 34 % |
| **381** `ventral_subdivision_of_vlc` | 120 | 97 | 0.01 | 441 `lateral_posterior_nucleus_of_thalamus` 89 % | 222 `parvocellular_division_of_va` 41 %, 218 20 %, 314 `rostral_division_of_vl` 15 % |
| **382** `medial_subdivision_of_vlc_(area_x)` | 71 | 86 | 0.00 | 441 43 %, 454 `anterior_nucleus_of_pulvinar` 41 % | 222 62 %, 283 `magnocellular_division_of_va` 26 % |
| **666** `facial_nucleus` | 76 | 1 | 0.00 | 611 52 %, 662 34 % | (almost no SEG voxels) |
| **435** `ventral_tegmental_area` | 40 | 98 | 0.04 | 384 `midbrain_(mesencephalon)` 32 %, **0** 25 %, 414 15 % | 384 30 %, 385 `red_nucleus` 15 % |

Volumes are in mm³; dice is the mean over cases.

**What the patterns show**

* **REF structures that SEG labels background (0)**: 68 `septum_pellucidum` (95 %), 184 `stria_terminalis` (49 %),
  354 `alveus` (41 %) and 654 `inferior_olive` (38 %). SEG labels their REF location as background (CSF or ventricle),
  so the synthetic image probably renders these REF voxels with CSF-like intensity. 68 and 184 border the lateral
  ventricle. Label 50 `anterior_horn_of_lateral_ventricle` was removed from the tree, and the generator's maps contain
  no voxels of it.
* **Ventral lateral thalamus is shifted.** REF 381/382 (VLc) is labelled 441/454 in SEG. SEG's 381/382 sits on the REF
  ventral anterior nuclei 222/283. The thalamic parcellation is offset towards anterior in SEG, and as a result
  SEG 441 `lateral_posterior_nucleus_of_thalamus` (+87 %) and SEG 454 `anterior_nucleus_of_pulvinar` (+240 %) grow.
* **Brainstem nuclei** (REF 654, 666) are absorbed by the general labels 662 `myelencephalon_(medulla_oblongata)`,
  611 and 580.

## 5. Small REF labels that SEG never predicts (minor, but systematic)

| REF label | REF vol (mm³) | cases | `REF x -> SEG` |
|---|---|---|---|
| **451** `interstitial_nucleus_of_cajal` | ~3 | 5/5 | 384 `midbrain_(mesencephalon)` 52 % |
| **569** `stratum_oriens_of_caudal_ca2` | ~1.4 | 4/5 | 7 `white_matter_of_forebrain` 34 %, 354 `alveus` 19 % |
| **114** `optic_nerve` | 29-65 | 3/5 (low dice in the other 2) | **0** 100 % |
| 160 `suprachiasmatic_nucleus`, 411 `stratum_oriens_of_rostral_ca2`, 506 `pineal_body`, 567 `stratum_radiatum_of_caudal_ca2`, 572 `stratum_lucidum__of_caudal_ca3` | < 12 | 2/5 | neighbours or 0 |

* 451, 569 and 114 are candidates for `data/inptus/simplifiednxb/labels_to_remove.json`. So are 409/410/411 (the
  rostral CA2 layers), whose dice is ~0 in every case.
* The labels that appear only in SEG are the large 846 (problem 1) and a few tiny ones (< 5 mm³: 209, 375, 443, 578, 765).
* Label 50 `anterior_horn_of_lateral_ventricle` is in the NextBrain LUT but not in the simplified tree. SEG never
  predicts it in these cases.

## What is fine

* Brain outline: dice 0.94-0.96. There is no global registration offset: the centroids of the labels > 1000 mm³ move
  < 0.6 mm per axis on average.
* 7 `white_matter_of_forebrain`: dice ~0.97.
* Cortex 2001-2035: dice 0.75-0.87.
* 79 `putamen`, 119/206 globus pallidus, 48 `head_of_caudate`: dice 0.87-0.93.
* Tree level 1: dice > 0.77 for every class except `gray_matter_of_midbrain` (0.64-0.72) and `white_matter_of_midbrain`.

## Priorities

1. Harmonise 611/846 (problem 1). It only takes a label mapping, and it removes the largest false error.
2. Find out why SEG cannot separate 595/597 (problem 2), using the generator's intensities for these two labels.
3. Find out why the REF voxels of 68, 184, 354 and 654 look like background in the synthetic image, and why the
   ventral lateral thalamus is shifted (problem 4). Compare the definitions of the general labels 580, 218 and 662 in
   REF and in NextBrain.
4. Check the outer cortical border in the generator (problem 3).
5. Add the labels that are systematically missing (problem 5) to the simplification.
