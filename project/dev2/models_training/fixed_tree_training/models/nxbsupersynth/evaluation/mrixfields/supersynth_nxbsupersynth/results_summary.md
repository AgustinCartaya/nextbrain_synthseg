# nxbsupersynth on MRIxFields vs SuperSynth — evaluation summary (3T and 7T)

Date: 2026-11-24. Command: `python evaluate.py --workers 48` (log: `eval_3T_7T.log`, tables: `results/`).
Same evaluation as `models/nxb16/evaluation/mrixfields/nextbrain_nxb16`, with SuperSynth as the reference.

**Figures.** `results_visual.ipynb` (next to this file) shows every finding below on real images
(image | SuperSynth | nxbsupersynth | disagreement), with 3T and 7T confusion matrices
(`results/confusion_3T.npz`, `results/confusion_7T.npz`). `make_results_visual.py` regenerates it.

**Reference.** SuperSynth is an automatic method, not manual ground truth, run separately on each contrast.
Low Dice means "disagrees with SuperSynth", not necessarily "wrong".

**How SuperSynth is converted to the network's 18 labels** (`evaluate.py`; the network predicts the SuperSynth ids
of the right side for both hemispheres, `nnUNet_raw/Dataset103_nxbsupersynth/label_space.json`):
- left ids → right ids (2→41, 3→42, …, 869→870);
- WM-hypointensities (77) → cerebral WM (41);
- ventricles (4, 5, 14, 15, 43, 44), CSF (24) and extra-cerebral classes (9xx) → background. The network's training labels
  come from NextBrain, where all of these are background.

Cases: T1w 163 / 255, FLAIR 163 / 104, T2w 162 / 101 (3T / 7T). Every reference has its prediction and both are on
the same 0.5 mm grid (no resampling).

## 1. Overall: mean Dice over the 18 structures

| Contrast | Field | n | Dice mean ± std | median | min | without accumbens, basal forebrain, mammillary b. | 11 large structures |
|---|---|---|---|---|---|---|---|
| T1w   | 3T | 163 | **0.632 ± 0.030** | 0.639 | 0.501 | 0.723 | 0.841 |
| T1w   | 7T | 255 | **0.632 ± 0.015** | 0.633 | 0.548 | 0.727 | 0.852 |
| FLAIR | 3T | 163 | 0.613 ± 0.034 | 0.623 | 0.438 | 0.702 | 0.837 |
| FLAIR | 7T | 104 | 0.587 ± 0.024 | 0.593 | 0.513 | 0.671 | 0.821 |
| T2w   | 3T | 162 | 0.499 ± 0.062 | 0.515 | 0.212 | 0.572 | 0.720 |
| T2w   | 7T | 101 | 0.512 ± 0.042 | 0.521 | 0.372 | 0.585 | 0.750 |

("11 large structures" = labels 1–11: brainstem to amygdala.)

- **T1w ≈ FLAIR > T2w**, at both field strengths. The headline numbers are very close to nxb16 vs NextBrain
  (T1w 0.628 / 0.641, FLAIR 0.611 / 0.594, T2w 0.505 / 0.508), but the two are not directly comparable (different
  reference and label sets).
- The 18-structure mean is pulled down by three classes that are **different structures in the two label sets**
  (section 3); without them T1w is ~0.72, and the 11 large structures reach 0.84–0.85 on T1w.
- T2w is worse and more variable: the worst cases are all T2w `sub-R*` (`sub-R0411` 0.21, `sub-R0370` 0.30, `sub-R0345` 0.32),
  while the 20 `sub-P*` T2w cases reach 0.558 (vs 0.491 for `R`).
- **Same subjects at 3T and 7T** (20 `sub-P*`): 7T − 3T = −0.001 (T1w), −0.002 (FLAIR), −0.013 (T2w, lower in 17/20).

## 2. Per-structure Dice (mean over cases)

| # | Structure | T1w 3T | T1w 7T | FLAIR 3T | FLAIR 7T | T2w 3T | T2w 7T | SuperSynth vol. (mL, T1w 3T) | median vol. diff (T1w 3T) |
|---|---|---|---|---|---|---|---|---|---|
| 1 | Brainstem | 0.841 | 0.827 | 0.820 | 0.770 | 0.832 | 0.808 | 19.4 | +5 % |
| 2 | Cerebral WM | **0.914** | **0.919** | 0.899 | 0.896 | 0.665 | 0.770 | 418 | +2 % |
| 3 | Cerebral cortex | 0.814 | 0.838 | 0.841 | 0.822 | 0.697 | 0.708 | 423 | **+26 %** |
| 4 | Cerebellum WM | 0.821 | 0.815 | 0.839 | 0.778 | 0.732 | 0.785 | 28.2 | +9 % |
| 5 | Cerebellum cortex | 0.871 | 0.870 | **0.910** | 0.877 | 0.833 | 0.838 | 93.9 | +10 % |
| 6 | Thalamus | 0.874 | 0.880 | 0.882 | 0.860 | 0.773 | 0.820 | 13.6 | +2 % |
| 7 | Caudate | 0.754 | 0.816 | 0.718 | 0.724 | **0.333** | **0.373** | 6.9 | +25 % (T2w **+178 %**) |
| 8 | Putamen | 0.879 | 0.870 | 0.837 | 0.823 | 0.744 | 0.738 | 10.2 | −9 % |
| 9 | Pallidum | 0.827 | 0.838 | 0.836 | 0.872 | 0.756 | 0.825 | 3.1 | −17 % |
| 10 | Hippocampus | 0.831 | 0.868 | 0.815 | 0.803 | 0.759 | 0.771 | 7.9 | +12 % |
| 11 | Amygdala | 0.824 | 0.835 | 0.812 | 0.809 | 0.797 | 0.812 | 3.4 | +6 % |
| 12 | Accumbens ⚑ | 0.252 | 0.232 | 0.237 | 0.228 | 0.230 | 0.251 | 1.0 | **+312 %** |
| 13 | Optic chiasm | 0.418 | 0.311 | 0.150 | 0.051 | 0.068 | 0.015 | 0.2 | −14 % |
| 14 | Hypothalamus (no MB) | 0.615 | 0.612 | 0.605 | 0.549 | 0.483 | 0.409 | 0.8 | +32 % |
| 15 | Fornix | 0.301 | 0.327 | 0.309 | 0.238 | 0.041 | 0.031 | 1.0 | −70 % |
| 16 | Mammillary body ⚑ | 0.273 | 0.233 | 0.272 | 0.266 | 0.186 | 0.184 | 0.1 | **+482 %** |
| 17 | Basal forebrain ⚑ | **0.000** | **0.000** | **0.000** | **0.000** | **0.000** | **0.000** | 0.6 | +227 % |
| 18 | Septal nuclei | 0.268 | 0.281 | 0.260 | 0.194 | 0.059 | 0.072 | 0.2 | +41 % |

⚑ = defined differently in the training labels (section 3).

- **Large structures (1–11) agree well on T1w/FLAIR (0.72–0.92).** The pallidum and putamen are slightly smaller
  than SuperSynth's; the confusion matrices show 8–21 % of every deep nucleus going to WM (border differences).
- **T2w:** WM drops to 0.67–0.77 (28 % of SuperSynth WM is labeled cortex at 3T), and the caudate collapses (section 4b).
- **Volumes still track SuperSynth across subjects** for the large structures (Pearson r 0.88–0.98 for WM, cortex,
  cerebellum, thalamus on T1w/FLAIR), but not for the small ones (|r| < 0.5) nor for the caudate (r 0.2–0.75).

## 3. Three classes are different structures in the two label sets (accumbens, basal forebrain, mammillary body)

The training labels come from NextBrain through `libraries/new_nextbrain_utils/mapping/nextbrain_to_nxbsupersynth.yaml`
(a copy of `nextbrain-utils/.../NextBrainToSuperSynth.yaml`; only the cortex group and the optic nerve in Optic-Chiasm differ, not these classes). Some SuperSynth names there contain
other anatomy:

| class | NextBrain labels put in it for training | what the network draws | what SuperSynth draws |
|---|---|---|---|
| Accumbens-area (58) | nucleus accumbens core **+ diagonal band, optic tract, fasciculus retroflexus, VTA, midbrain (384), peripeduncular nucleus, infundibular stalk** ("looks more like ventral DC" in the yaml) | accumbens **plus a large region in the upper midbrain / ventral diencephalon** (5.3 mL vs 1.3 mL; 44 % of it is SuperSynth brain stem) | the nucleus accumbens only |
| Basal-Forebrain (866) | **substantia nigra, subthalamic nucleus, red nucleus** | SN / STN / RN in the midbrain (SuperSynth calls this WM) | the basal forebrain band below the striatum (the network calls this WM) |
| MammillaryBody (844) | mammillary nuclei **+ posterior and lateral-posterior hypothalamus, tuberomammillary, supramammillary** | ×5–7 larger than SuperSynth's | the mammillary bodies only |

The basal forebrain has **zero overlap in every one of the 948 cases**. These three Dice values measure the mapping, not the
network. ⚠ Checked visually (notebook section 6).

**To fix it**, either leave them out when comparing with SuperSynth (as in section 1), or retrain with a mapping that follows
SuperSynth's definitions. For example: a ventral-DC class (midbrain, SN, STN, RN, VTA, …), and NextBrain's basal forebrain nuclei
(nucleus basalis, diagonal band, substantia innominata, …) in Basal-Forebrain.

## 4. Volume biases

### 4a. The cortex (and so the brain) is larger than SuperSynth's: sulcal CSF

The sum of the 18 structures is **+14 % (T1w), +9 % (FLAIR), +24 % (T2w)** at 3T, and +9 / +9 / +24 % at 7T. Almost all of it is
cortex (median +11 to +54 %).
⚠ Checked visually: **86–94 % of the extra tissue is SuperSynth CSF (24)**, labeled cortex by the network. SuperSynth labels the CSF
in the sulci and around the brain. The network, trained on NextBrain labels (no CSF label inside the brain), draws the cortex out to
the outer pial surface and fills the sulci. This is a definition difference, the same one seen with nxb16 vs NextBrain.
In numbers: 32–43 % of SuperSynth CSF is labeled cortex on T1w/FLAIR, 58–70 % on T2w.

### 4b. T2w caudate +178 %: the ventricles

On T1w/FLAIR the SuperSynth ventricles are 68–80 % background in the prediction. On T2w **only 4–9 %**: the network fills the
bright lateral ventricles with tissue labels, mostly caudate (39–40 % of SuperSynth ventricle voxels). In the representative
subject, 92 % of the extra T2w caudate (17 mL) is SuperSynth lateral ventricle. This is the same T2w failure as nxb16.
The likely cause: in training, ventricles are always background. Background on skull-stripped images is dark, but ventricles
on T2w are bright.

### 4c. Other

- Fornix (−66 to −96 %) and septal nuclei (on T2w −74 to −78 %): mostly labeled WM by the network.
- Hippocampus +3 to +32 %, amygdala −2 to +22 % (larger on FLAIR/T2w).
- Hypothalamus +17 to +88 % (largest on T2w 7T).

## 5. Worst cases

- **3T T2w** (`sub-R0411` 0.21, `sub-R0370` 0.30, `sub-R0345` 0.32): **large ventricles** (42–148 mL in SuperSynth) and bright WM
  lesions. **93–94 % of their ventricle volume is labeled as tissue**, and the deep gray structures around them break down.
  Their T1w is segmented normally. These are the same worst cases as nxb16.
- **7T T2w**: `sub-R0757` (0.38) and `sub-R0789` (0.43) are the same ventricle failure. `sub-R0821` (0.37) is an
  **image-quality failure**: the lower half of the image is dark (B1 inhomogeneity), the network loses the brainstem and cerebellum
  and oversegments the cortex rim at the top. Its T1w is normal. Same as with nxb16.

## 6. 7T

Same patterns as 3T on the 7T representative subject (`sub-P0007`): 85–89 % of the extra tissue is SuperSynth CSF; on T2w the
ventricles are labeled caudate (12 mL). T1w 7T is the most consistent condition (std 0.015). FLAIR 7T is lower than FLAIR 3T
mostly for the `R` subjects (0.582 vs 0.614); for the paired `P` subjects 3T and 7T are equal.

## Take-aways

1. **On the large structures the network agrees well with SuperSynth on T1w/FLAIR** (0.84–0.85 mean Dice over 11 structures,
   volumes correlate r ≈ 0.9+), at 3T and 7T alike.
2. **Accumbens, basal forebrain and mammillary body are not comparable** with SuperSynth: the training mapping puts the midbrain /
   ventral DC in "accumbens" and SN/STN/RN in "basal forebrain". Report Dice without them, or fix the mapping and retrain.
3. **Cortex +11–26 % on T1w/FLAIR, +38–54 % on T2w** = SuperSynth sulcal CSF labeled as cortex: a definition difference inherited from NextBrain labels.
4. **T2w ventricles are labeled as tissue** (caudate ×2.8 median), and this causes the worst cases (large ventricles). Same
   problem as nxb16; it comes from the training data (ventricles = background), not from the label set.
5. Fornix, septal nuclei and optic chiasm are mostly not found (Dice ≤ 0.42, ~0 on T2w).
