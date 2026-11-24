# nxb16 vs nxbsupersynth: common label space

Both ontologies come from the same simplified NextBrain labels (`data/outputs/nxbsimplified`):

* `nxbsimplified_to_nxb16.py` → 16 classes + background (`GMM_label_grouping_corrected.yaml`)
* `nxbsimplified_to_nxbsupersynth.py` → 18 SuperSynth classes + background (`NextBrainToSuperSynth_corrected.yaml`, side R, claustrum merged into WM)

So you can check every NextBrain label: you know which nxb16 class and which nxbsupersynth class it goes to.
Two classes can be compared only if they hold the same NextBrain labels. The full table is `crosswalk.csv`.
It has every (nxb16 class, nxbsupersynth class) pair, the NextBrain labels in it and their mean volume over the 5 cases.

If you take this strictly (connected components of the crosswalk), almost every cerebral class ends up in one blob.
A few small labels link them together. For example, SuperSynth `accumbens_area` pulls in white matter, gray matter,
caudate and both brainstem classes. So the common space below:

* merges classes where needed,
* **ignores** two nxbsupersynth classes that mix unrelated structures,
* accepts the leftover differences only where they are small. The *best possible dice* column measures them.

**Best possible dice**: the dice between the two perfect conversions of the same nxbsimplified map, with ignored voxels left out.
It is the ceiling a pair of segmentations can reach in that class. Comparing the ground-truth maps
`test_conversion/data/nxb16` vs `test_conversion/data/nxbsupersynth` reproduces it (test 1, `test_conversion/results/gt_vs_gt_dice.csv`).

## Files

| file | what |
|---|---|
| `common_space.yaml` | Definition of the common space (single source of truth), one column per ontology: `nxb16`, `nxbsupersynth`, `supersynth` (real `mri_super_synth`) |
| `to_common_space.py` | Conversion code: `load_common_space`, `to_common`, `comparable_mask` + CLI |
| `test_conversion/crosswalk.py` | Theoretical check from the two conversion scripts → `crosswalk.csv`, `common_space_check.csv` (best possible dice) |
| `test_conversion/test_conversion.py` | Tests on `test_conversion/data` → `test_conversion/results/` (`results.md` + csv) |

```bash
conda activate nxb
# convert (file or directory; output keeps the input grid, uint8)
python to_common_space.py --ontology nxb16         -i seg16.nii.gz -o seg16_common.nii.gz
python to_common_space.py --ontology nxbsupersynth -i ss_dir/      -o ss_common_dir/
python to_common_space.py --ontology supersynth    -i mri_super_synth_seg.nii.gz -o out.nii.gz

# tests
cd test_conversion && python crosswalk.py && python test_conversion.py
```

```python
from to_common_space import load_common_space, to_common, comparable_mask
cs = load_common_space()
a = to_common(seg16, 'nxb16', cs)
b = to_common(segss, 'nxbsupersynth', cs)
valid = comparable_mask(a, b, 'nxb16', 'nxbsupersynth', cs)   # compute metrics on a[valid], b[valid]
```

Both SuperSynth ontologies accept Right- and Left- ids. A label that is not in `common_space.yaml` raises an
error (`--unknown ignore` → ignore label). **Always compare with `comparable_mask`.** It leaves out voxels that are
ignore (255) in either map, and voxels in a class that the other ontology does not have. For example, SN/STN/RN does not exist in real SuperSynth.
The maps must be on the same grid. Resample with nearest neighbour first (as `test_conversion.py` does for the 0.4 mm SuperSynth maps).

### Real SuperSynth (`supersynth` column)

The real SuperSynth outputs in `test_conversion/data/supersynth` have **more labels** than nxbsupersynth. Two labels also **change meaning**.
The cross-tabulation against the ground truth (5 cases) gave this mapping:

| SuperSynth label | where it falls in the NextBrain GT | common class |
|---|---|---|
| 4/43 lateral ventricle, 5/44 inf-lat-vent, 14 3rd ventricle, 15 4th ventricle | background 88 / 72 / 82 / 67 % (ventricles are 0 in NextBrain) | background |
| 77 WM hypointensities | white matter 90 % | cerebral_white_matter |
| 26/58 accumbens (real) | NXB 101 accumbens core 54 %, caudate 33 % (nxb16 caudate_and_accumbens 87 %) | caudate |
| 865/866 basal forebrain (real) | NXB 128, 120, 108, ... (GM in nxb16, WM in nxbsupersynth) | ignore |
| 85 optic chiasm | (same as nxbsupersynth) | ignore |
| SN / STN / red nucleus | no SuperSynth class (they fall in WM and brain stem) | class 9 not available |

All other labels map as in nxbsupersynth. No label of nxbsupersynth is missing from the real SuperSynth maps.
Labels never seen in these 5 cases (CSF 24, ventral DC 28/60, vessels, choroid plexus, ...) are not in the yaml. Add them if they show up.

### Test results (`test_conversion/results/results.md`)

* **Test 0, coverage**: every label of every map converts with `unknown=error`. PASS.
* **Test 1, nxb16 GT vs nxbsupersynth GT**: every class reaches its best possible dice (|diff| ≤ 0.0001).
  All are ≥ 0.995 except putamen (0.907, the claustrum, see below). PASS.
* **Test 2, real SuperSynth scored against each GT** (same voxels): the dice is the same with both
  references (|diff| ≤ 0.003). The exception is putamen (0.881 vs nxbsupersynth, 0.814 vs nxb16), which is expected. So for every class except
  putamen, a segmentation scores the same against the nxb16 or the nxbsupersynth GT in the common space. PASS.
* **Test 3, native SuperSynth classes** (reference): the classes that are merged or ignored in the common space are the ones
  that disagree natively. Accumbens 0.26, optic chiasm 0.36, fornix 0.43, mammillary body 0.21, basal forebrain 0.00,
  septal nucleus 0.25.

## Common space

Volumes are means over the 5 cases (mm³). The ignore label is 255.

| id | common class | nxb16 classes | nxbsupersynth classes | status | best dice |
|---|---|---|---|---|---|
| 0 | background | 0 background | 0 background | direct | 1.000 |
| 1 | cerebral_white_matter | 1 white_matter | 41 cerebral_white_matter + 822 fornix | merged | 0.995 |
| 2 | cerebral_gray_matter | 2 gray_matter + 16 hippocampal_white_matter | 42 cortex + 53 hippocampus + 54 amygdala + 870 septalnuc | merged | 0.999 |
| 3 | cerebellum_white_matter | 3 cerebellum_white_matter + 15 cerebellar_other | 46 cerebellum_white_matter | merged | 1.000 |
| 4 | cerebellum_cortex | 4 cerebellum_gray_matter | 47 cerebellum_cortex | direct | 1.000 |
| 5 | thalamus | 7 white_thalamus + 8 gray_thalamus | 49 thalamus | merged | 0.997 |
| 6 | caudate | 5 caudate_and_accumbens | 50 caudate | partial | 1.000* |
| 7 | putamen | 6 putamen | 51 putamen | approximate | 0.907 |
| 8 | pallidum | 9 pallidum | 52 pallidum | direct | 1.000 |
| 9 | substantia_nigra_stn_red_nucleus | 10 SN + STN + red nucleus | 866 "basal_forebrain" | direct | 1.000 |
| 10 | brainstem | 11 gray_brainstem + 12 white_brainstem | 16 brain_stem | partial | 0.997* |
| 11 | hypothalamus | 13 gray_hypothalamus + 14 white_hypothalamus | 820 hypothal_nomb + 844 mammillarybody | merged | 0.998 |
| 255 | ignore | – | 58 accumbens_area, 85 optic_chiasm (supersynth: 85, 866) | – | – |

\* After the ignored voxels are left out. Part of the structure is then not compared at all (see "partial" below).

### 1. Directly comparable (1 class ↔ 1 class, same NextBrain labels)

* **Background**
* **Pallidum**: NXB 119, 206
* **Cerebellum cortex**: NXB 595, 597
* **Substantia nigra + STN + red nucleus** (nxb16 10 ↔ SuperSynth 866). ⚠ Same labels, different names.
  In `NextBrainToSuperSynth_corrected.yaml`, "Basal-Forebrain" holds exactly NXB 310, 315, 316, 321, 352 and 385.
  This holds for a model trained on the nxbsupersynth labels made here. For a real `mri_super_synth` segmentation
  (`supersynth` column), 866 is the real basal forebrain: it is ignored, and this class is not available.

### 2. Comparable after merging

* **Cerebellum white matter**: nxb16 `cerebellum_white_matter` + `cerebellar_other` (deep nuclei NXB 715, 721, 751, 752)
  ↔ SuperSynth `cerebellum_white_matter`. Exact.
* **Thalamus**: nxb16 `white_thalamus` + `gray_thalamus` ↔ SuperSynth `thalamus`. SuperSynth also includes the
  pineal body (NXB 506, white brainstem in nxb16, 44 mm³) and NXB 433 parasubthalamic nucleus (white hypothalamus, 4 mm³). Dice ≤ 0.997.
* **Hypothalamus**: nxb16 `gray_` + `white_hypothalamus` ↔ SuperSynth `hypothal_nomb` + `mammillarybody`. Only NXB 433 differs (it goes to
  thalamus). Dice ≤ 0.998.
* **Cerebral white matter**: nxb16 `white_matter` ↔ SuperSynth `cerebral_white_matter` + `fornix` (NXB 199).
  SuperSynth WM also has claustrum NXB 102/174 (putamen in nxb16, 1 022 mm³) and NXB 108, 117, 128. Those three are the basal nucleus of Meynert,
  the BNST and the substantia innominata, all gray matter in nxb16 (738 mm³). Dice ≤ 0.995.
* **Cerebral gray matter**: nxb16 `gray_matter` + `hippocampal_white_matter` ↔ SuperSynth `cerebral_cortex` +
  `hippocampus` + `amygdala` + `septalnuc`. You have to merge them because nxb16 `gray_matter` already contains the hippocampal gray matter,
  the amygdala and the septal nucleus. Only NXB 108/117/128 differ. Dice ≤ 0.999.

### 3. Partial (exact where compared, but part of the structure is ignored)

SuperSynth `accumbens_area` (58) is not the accumbens in this conversion. Of its 1 866 mm³:

* 53 % is midbrain (NXB 384, white brainstem in nxb16)
* 25 % is the accumbens core (NXB 101, caudate_and_accumbens in nxb16)
* 14 % is the optic tract (NXB 208, white matter)
* the rest is VTA, fasciculus retroflexus, peripeduncular nucleus and infundibulum (gray brainstem), plus the diagonal band (gray matter)

It cannot be matched to any union of nxb16 classes, so it is **ignored**. Voxels labelled 58 in the nxbsupersynth map
are left out of both maps. Because of that:

* **Caudate**: nxb16 `caudate_and_accumbens` ↔ SuperSynth `caudate`. Exact on the compared voxels. The accumbens core
  (474 mm³, 9 % of the nxb16 class) is not compared.
* **Brainstem**: nxb16 `gray_` + `white_brainstem_nuclei` ↔ SuperSynth `brain_stem`. Exact apart from the pineal body.
  The midbrain NXB 384 and the other ignored labels (~1 170 mm³, 12 % of the nxb16 brainstem) are not compared.

### 4. Approximate

* **Putamen**: nxb16 `putamen` includes the claustrum (NXB 102, 174 = 17 % of the class). SuperSynth puts the claustrum in
  cerebral white matter. Best possible dice 0.907. You can't fix this by merging: the only other class you could merge
  with is the whole cerebral WM. `--claustrum` in `nxbsimplified_to_nxbsupersynth.py` would separate it, but it
  currently fails, because the SuperSynth LUT in nextbrain-utils has no "Claustrum" entry.

### 5. Not comparable (exists only in one ontology)

**Only in nxb16** (the distinction disappears in the common space):

* white vs gray thalamus
* gray vs white brainstem nuclei
* gray vs white hypothalamus
* hippocampal white matter (NXB 339, 343, 354, 461, ... → part of SuperSynth hippocampus)
* `cerebellar_other`: deep cerebellar nuclei, inside SuperSynth cerebellum WM
* accumbens core inside `caudate_and_accumbens`
* claustrum inside `putamen`

**Only in nxbsupersynth**:

* **hippocampus**, **amygdala** and **septal nuclei**: inside nxb16 `gray_matter`, so they can't be told apart from the cortex
* **fornix**: inside nxb16 `white_matter`
* **mammillary body** vs rest of the hypothalamus: the split doesn't follow nxb16 gray/white. Both nxb16 classes feed both SuperSynth classes.
* **accumbens_area** (58) and **optic_chiasm** (85): mixtures, ignored. Optic chiasm is 60 % NXB 161 optic chiasm (white matter in nxb16)
  and 40 % NXB 114 optic nerve (gray brainstem nuclei in nxb16).

Ignored NextBrain labels in total: NXB 101, 111, 114, 161, 208, 384, 412, 435, 498, 843. That is ~2 060 mm³, 0.1 % of the volume.
