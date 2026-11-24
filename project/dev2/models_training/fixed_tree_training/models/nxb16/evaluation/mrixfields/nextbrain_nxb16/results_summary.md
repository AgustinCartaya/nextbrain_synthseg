# nxb16 on MRIxFields vs NextBrain — evaluation summary (3T and 7T)

Date: 2026-11-23. Command: `python evaluate.py --workers 48 --cache_gt` (log: `eval_3T_7T.log`,
tables: `results/`, covering 3T and 7T). Sections 1–4 describe **3T**. **7T is in section 5.**

**Figures.** `results_visual.ipynb` (next to this file) shows every finding below on real images (3T, plus a 7T section)
(image | NextBrain | nxb16 | disagreement) and adds confusion matrices over all 3T and 7T cases
(`results/confusion_3T.npz`, `results/confusion_7T.npz`). The visual check corrected two of my first guesses; see the ⚠ notes.

**Reference.** The NextBrain segmentations are an automatic method, not manual ground truth.
NextBrain was run separately on each contrast, so the reference itself changes between contrasts (e.g.
mean reference gray matter: 446 mL on T1w, 491 mL on FLAIR, 481 mL on T2w). Low Dice therefore means
"disagrees with NextBrain", not necessarily "wrong".

Cases: FLAIR 163, T1w 163, T2w 162 (every reference has its prediction; prediction and reference
are on the same grid, no resampling; 162 subjects have all three contrasts).

## 1. Overall: mean Dice over the 16 structures

| Contrast | n | Dice mean ± std | median | min | Mean \|rel. vol err\| over structures (median over cases) |
|---|---|---|---|---|---|
| T1w   | 163 | **0.628 ± 0.027** | 0.637 | 0.523 | 37.0 % |
| FLAIR | 163 | 0.611 ± 0.032 | 0.615 | 0.448 | 36.9 % |
| T2w   | 162 | 0.505 ± 0.076 | 0.523 | 0.152 | 50.0 % |

T1w ≈ FLAIR > T2w. T2w is worse on average and much more variable: all of the 12 worst cases
are T2w (e.g. `sub-R0411` 0.15, `sub-R0345` 0.24, `sub-R0370` 0.27). The T2w drop is concentrated in the
`R` subjects (mean 0.494, n=142); the 20 `P` subjects reach 0.585 on T2w, close to their T1w (0.640) and FLAIR (0.606).
⚠ Checked visually: same voxel size and grid as the `P` scans. The worst `R` cases have **enlarged
ventricles** and a lot of bright CSF, and nxb16 fills the ventricles with tissue labels on T2w (point 1b below).
The same subjects are segmented normally on T1w.

The mean over the 16 structures is pulled down by 4–5 tiny or badly defined structures (section 3).
On the large structures the agreement is reasonable.

## 2. Per-structure Dice (mean over cases)

| # | Structure | FLAIR | T1w | T2w | Mean reference volume (mL, T1w) |
|---|---|---|---|---|---|
| 1 | White Matter | 0.853 | **0.918** | 0.672 | 396 |
| 2 | Gray Matter | 0.814 | 0.814 | 0.716 | 446 |
| 3 | Cerebellum WM | 0.802 | 0.815 | 0.741 | 29 |
| 4 | Cerebellum GM | **0.901** | 0.862 | 0.830 | 88 |
| 5 | Caudate + Accumbens | 0.738 | 0.766 | 0.366 | 8.5 |
| 6 | Putamen | 0.821 | 0.858 | 0.754 | 11.4 |
| 7 | White Thalamus | 0.689 | 0.749 | 0.606 | 8.4 |
| 8 | Gray Thalamus | 0.655 | 0.724 | 0.452 | 5.8 |
| 9 | Pallidum | 0.844 | 0.805 | 0.758 | 3.2 |
| 10 | SN + STN + Red Nucleus | 0.652 | 0.528 | 0.542 | 1.5 |
| 11 | Gray Brainstem nuclei | 0.227 | 0.269 | 0.175 | 3.0 |
| 12 | White Brainstem nuclei | 0.544 | 0.596 | 0.557 | 18.3 |
| 13 | Gray Hypothalamus | 0.588 | 0.589 | 0.419 | 1.0 |
| 14 | White Hypothalamus | 0.211 | 0.287 | 0.053 | 0.16 |
| 15 | Cerebellar Other | 0.201 | 0.084 | 0.172 | 1.3 |
| 16 | Hippocampal WM | 0.229 | 0.387 | 0.267 | 2.0 |

- **Good agreement (Dice ≥ 0.8 on T1w/FLAIR):** cerebral WM/GM, cerebellum WM/GM, putamen, pallidum.
- **Moderate (0.5–0.8):** caudate+accumbens, both thalamus parts, SN/STN/RN, white brainstem, gray hypothalamus.
- **Poor (< 0.4):** gray brainstem nuclei, white hypothalamus, cerebellar other, hippocampal WM.

## 3. Volumes: systematic biases

Median relative volume difference (prediction − reference)/reference:

| Structure | FLAIR | T1w | T2w |
|---|---|---|---|
| White Matter | +10 % | +6 % | −8 % |
| Gray Matter | +11 % | +24 % | **+54 %** |
| Cerebellum GM | +11 % | +19 % | +18 % |
| Caudate + Accumbens | +4 % | +10 % | **+125 %** |
| Gray Thalamus | −35 % | −25 % | −52 % |
| Pallidum | −10 % | −21 % | −10 % |
| SN + STN + RN | +31 % | +31 % | +33 % |
| **Gray Brainstem nuclei** | **+201 %** | **+189 %** | **+189 %** |
| White Brainstem nuclei | −29 % | −23 % | −28 % |
| Gray / White hypothalamus | +21 / +12 % | +43 / +47 % | +41 / −70 % |
| Hippocampal WM | +22 % | +33 % | +37 % |

What stands out:

1. **The predicted brain is larger than NextBrain's.** Summed over the 16 structures it is larger by
   +10.6 ± 2.5 % (FLAIR), +15.3 ± 3.1 % (T1w) and +24.8 ± 6.8 % (T2w), almost all of it gray matter
   (cortex +112 mL on T1w, +258 mL on T2w) and cerebellar GM. All of the extra tissue lies on NextBrain
   label 0, which NextBrain also uses for CSF (ventricles and sulci have no label). Checked visually, it
   has two sources:
   - **(a) all contrasts:** a thin rim along the whole pial surface and inside the sulci. nxb16 draws the cortex
     1–2 voxels further out. That is small per voxel but adds up to 10–15 % of the brain at 0.5 mm, and it is very
     consistent across subjects (small std): a boundary-definition difference.
   - **(b) ⚠ T2w only: the lateral ventricles are filled with tissue labels** (mostly GM and caudate;
     putamen/thalamus colors in the enlarged ventricles of the worst cases). On T1w/FLAIR the
     ventricles are correctly empty. CSF is bright on T2w. A likely explanation, which I have not verified: the
     nxb16 label set has no ventricle/CSF class (ventricles are merged into background, labels 0 and 50),
     so during training the ventricles always looked like background, and on these skull-stripped scans background
     is dark. Then bright ventricles are never "background" for the network.
   This is also why the T2w cerebral WM Dice is low (28 % of NextBrain WM becomes GM on T2w).
2. **Gray vs white brainstem nuclei are swapped in proportion.** In all three contrasts the gray
   brainstem nuclei are ~3× too large and the white brainstem nuclei ~25 % too small. The volume the
   network adds to the gray nuclei (~+5 mL) matches what it removes from the white ones (~−5 mL), so it
   assigns a large part of the brainstem to "gray nuclei". Checked visually: it is the **ventral pons
   (basis pontis)**. NextBrain labels it mostly `pons` (580, in the *white* group of
   `nextbrain_to_nxb16.yaml`) with scattered `pontine_nucleus` (414, *gray*), while nxb16 labels the
   whole basis pontis gray (3.5–5.5 mL of `pons` per subject). Same in T1w, T2w and FLAIR, so it is about how
   the label is defined, not about contrast. Check how the basis pontis is labeled in the training label maps and whether
   `pons` belongs in the white group.
3. **Gray thalamus under-segmented (−25 to −50 %) while white thalamus is over-segmented**: the same
   kind of gray/white split disagreement, smaller.
4. **Caudate + accumbens on T2w (+125 %, Dice 0.37)**: specific to T2w (only +4/+10 % on FLAIR/T1w).
   ⚠ Checked visually: ~96 % of the extra "caudate" voxels lie in the **lateral-ventricle lumen** (NextBrain
   label 0), so this is the ventricle problem of 1b, not a caudate border problem.
5. **Tiny structures** (white hypothalamus 0.15 mL, cerebellar other 0.6–1.3 mL, hippocampal WM 2 mL):
   a few voxels of boundary disagreement already give Dice < 0.3, and their predicted vs reference
   volumes are barely correlated (Pearson r ≤ 0.5, cerebellar other r < 0). The confusion matrices show
   that the disagreement is about *where* each method puts these structures: NextBrain's cerebellar other
   (scattered pieces inside the cerebellar WM) → nxb16 cerebellar WM 72–95 %; hippocampal WM → GM 43–71 %;
   white hypothalamus → WM 36–74 %. These scores mostly
   reflect noise and differences in how the labels are defined. They pull the 16-structure mean down
   noticeably, so also report the mean without them.

## 4. Volume correlation (does the network rank subjects like NextBrain?)

Pearson r between predicted and reference volumes across subjects:

- Large structures: high on T1w (WM 0.97, GM 0.94, cerebellum GM 0.93, white thalamus 0.89,
  gray thalamus 0.86, white brainstem 0.83), slightly lower on FLAIR, clearly lower on T2w (GM 0.71,
  gray thalamus 0.17).
- Caudate+accumbens (0.19–0.43), putamen on T2w (0.25), SN/STN/RN, gray brainstem nuclei and the
  hypothalamus parts: weak. Even when the mean volume is close, the per-subject volumes don't track the reference.
- So for volumetry the nxb16 network is usable relative to NextBrain on T1w (and to a lesser extent
  FLAIR) for the large and medium structures, but not yet for deep-gray-nucleus volumetry on T2w.

## 5. 7T

Cases: T1w 255, T2w 101, FLAIR 104 (every reference has its prediction, 0 errors). 7T images are on the
same 0.5 mm grid (364×436×364) as 3T, and no reference needed resampling.

| Contrast | 3T Dice (n) | 7T Dice (n) | Brain volume vs NextBrain 3T → 7T |
|---|---|---|---|
| T1w   | 0.628 ± 0.027 (163) | **0.641 ± 0.019** (255) | +15.3 % → +10.8 % |
| FLAIR | 0.611 ± 0.032 (163) | 0.594 ± 0.035 (104) | +10.6 % → +10.1 % |
| T2w   | 0.505 ± 0.076 (162) | 0.508 ± 0.053 (101) | +24.8 % → +24.1 % |

**Only the 20 `sub-P*` subjects have both 3T and 7T.** The `sub-R*` subjects are different people at 3T
(R03xx–R04xx) and 7T (R07xx–R08xx), so only the `P` subjects give a same-subject comparison. Paired, 7T − 3T Dice for the same subject and contrast:
T1w +0.008 ± 0.013, FLAIR +0.014 ± 0.027, T2w **−0.033 ± 0.025** (T2w is lower at 7T for all 20 subjects).

What changes at 7T:
- **Overall the picture is the same as 3T.** Every 3T finding is still there: T1w ≥ FLAIR ≫ T2w; the brain is
  ~10 % larger than NextBrain's on T1w/FLAIR and ~24 % larger on T2w; gray brainstem nuclei +130 to +185 % and white −27 to −42 %;
  gray thalamus −20 to −46 %; T2w caudate+accumbens +97 % (Dice 0.38). This fits these being label-definition
  issues plus the T2w ventricle problem, none of which depend on field strength. Checked visually on
  `sub-P0018` 7T (notebook section 9): same pial rim, same ventricles filled on T2w, same pons → gray nuclei (~5 mL).
- **T1w 7T is the best and most consistent condition** (0.641 ± 0.019, min 0.56): better WM (0.925),
  GM (0.840), caudate (0.812), white thalamus (0.788). Its volumes also track NextBrain better: Pearson r for
  caudate 0.19 → 0.66, putamen 0.67 → 0.90, SN/STN/RN 0.30 → 0.68, gray brainstem 0.37 → 0.61.
- **FLAIR 7T is slightly worse than 3T** (cerebellum WM 0.80 → 0.72, gray thalamus 0.66 → 0.55, white
  hypothalamus 0.21 → 0.06; white brainstem volume r 0.46 → −0.22). The `P` subjects do not drop
  (0.606 → 0.620), so the drop comes from the 7T `R` subjects (0.588).
- **T2w 7T:** the spread is smaller (std 0.053 vs 0.076, min 0.29 vs 0.15) and WM improves (0.67 → 0.77),
  but the same-subject comparison is lower (−0.03 for every `P` subject). The worst cases are again `R` subjects
  (`sub-R0821` 0.29, `sub-R0757` 0.36, `sub-R0789` 0.39). Checked visually: `sub-R0757` (and `sub-R0789`) are
  the ventricle problem again (enlarged ventricles filled with tissue labels). **`sub-R0821` is a different failure: image
  quality.** The lower part of the image is dark and the signal is uneven (typical 7T B1 inhomogeneity), and nxb16 loses the brainstem/thalamus;
  its T1w is fine.
- **Small structures** stay unreliable and vary a lot between conditions (cerebellar other 0.04–0.42,
  white hypothalamus 0.02–0.27), with their volumes uncorrelated with NextBrain.

## 6. Take-aways / next steps

- T1w ≥ FLAIR ≫ T2w. **The T2w gap is a real failure: nxb16 labels the bright ventricles as tissue.** It hurts
  most in subjects with large ventricles (`sub-R0411`, `sub-R0345`, `sub-R0370`). This is the only finding
  that looks like a network error rather than a definition difference. A fix probably needs a
  ventricle/CSF class (or random intensities for the ventricles, separate from the background) in the generator.
- Contrast-independent differences that come from how the labels are defined: (a) cortex drawn
  1–2 voxels further out than NextBrain, (b) basis pontis gray (nxb16) vs white `pons` (NextBrain mapping),
  (c) gray/white thalamus border, (d) small structures placed differently. Decide per item whether the
  training labels or the NextBrain → nxb16 mapping should change before comparing Dice further.
- 7T confirms all of this. T1w 7T is the best condition, and nothing new appears at 7T.
- At 7T, T2w also fails on some images with strong B1 inhomogeneity/dropout (`sub-R0821`). Stronger bias-field /
  intensity augmentation in the generator should help there.

Files: `results/per_case.csv` (case × structure), `results/per_case_mean.csv`,
`results/summary_contrast_field.csv`, `summary_contrast.csv`, `summary_field.csv`, `summary_overall.csv`,
`results/missing.csv` (empty for 3T).
