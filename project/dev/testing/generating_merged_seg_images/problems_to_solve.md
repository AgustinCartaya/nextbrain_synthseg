# Problems to solve

Possible flaws found by reading `generate_images.py` and `src/tree_generator.py`
(template, margin and random-field code). Not fixed yet. Found by code reading,
not by new tests. Line numbers are as of 2026-10-20.

Priority: 1-4 first (they can silently produce a dataset whose files, labels
and metadata do not match), then 6-7 (the docs overstate what the automatic
margin guarantees).

---

## A. Can produce wrong or mixed datasets without an error

### 1. Dataset config and tree overwritten on every run
`generate_images.py:372-373`

`generator_config.json` and `tree.json` in the dataset are rewritten by each
run. Adding images to an existing dataset with another `--config` or `--tree`
leaves copies that only describe the last run, while every spec says
`"config": "generator_config.json"` (`generate_images.py:433`), so older images
point to a config that is not theirs.

### 2. Label indices can change between runs into the same dataset
`src/tree_generator.py:1061-1062`, `generate_images.py` (`write_dataset_json`)

With `output_labels: "templates"`, the index of each label depends on the
labels found in the current templates. A second run with other templates can
give the same index to a different label, and `dataset.json` is rewritten with
the last run's names: earlier `labelsTr` files are then read wrongly.

### 3. Template cache keyed by file name, not by source
`src/tree_generator.py:1661` (`build_templates`), check at `:1699`

The build record stores `source`, but only `settings` is compared. If
`--seg-dir` points to another folder with the same file names (e.g.
`case-1.nii.gz`), the cached template of the old map is reused.

### 4. Specs do not identify the template version used
`generate_images.py:429`

With `--seg-dir`, `template` in the spec is the original label map, not the
cached HR template. If the template is later rebuilt (settings changed), no
spec says which version an image used, and regenerating an old image uses the
new one.

### 5. `--save-hr-template-dir` can erase a valid build record
`generate_images.py:391`

If `--hr-template-dir` is the `templates/` folder of the same dataset, saving
rewrites each record as `{"hr_template_dir": true}`. The record of the
`--seg-dir` build (with its margin) is lost: the next `--seg-dir` run rebuilds
the template, and `load_hr_template` no longer recognises it as built with a
margin.

---

## B. Automatic margin weaker than documented

### 6. `"auto"` is not an exact upper bound
`src/tree_generator.py:359` (`auto_margin_mm`), rotation grid at `:394`

Documented as exact, but rotations are only evaluated at 5 values per axis;
the maximum extent can fall between them (irregular shapes, large
`rotations`). Zoom extremes are safe; with non-zero `shears` the same gap
exists.

### 7. `{"auto": q}` quantile taken per axis independently
`src/tree_generator.py:490`

Documented as "enough for 99% of the deformations", but the quantile is per
axis; with three axes the probability that at least one is exceeded can
approach 3%.

### 8. Simulation field of view differs from the real template
`src/tree_generator.py:464` (`sampled_margin_mm`)

The simulated grid is crop + worst-case padding; the real template is crop +
chosen margin. The elastic control points span the whole field of view, so
the control spacing (smoothness) differs slightly. Same displacement
magnitude: an approximation, not a large error.

### 9. `sampled_margin_mm` ignores the chosen device
`src/tree_generator.py:456`

Always uses `cuda` (GPU 0) when a GPU exists, ignoring `--device` /
`--template-device`; on a shared machine it may run on the busiest GPU.

### 10. Recorded templates not checked against the current deformation
`generate_images.py:270` (`load_hr_template`)

For templates with a build record only the recorded margin is printed. A
template built with 16 mm reused with a stronger deformation (e.g.
`rotations: 30`) gets no warning.

### 11. Morphology slack approximate
`src/tree_generator.py:419`

The growth into background is added after the affine part, so it is not
scaled by the zoom (~+10%), and several background pairs could stack at the
same edge. Below a millimetre.

---

## C. Robustness

### 12. Empty label map crashes with an unclear error
`src/tree_generator.py:595` (`prepare_template`), `auto_margin_mm`,
`generate_images.py` (`load_hr_template`)

A map with no foreground fails on `idx.min` of an empty array (numpy error,
no clear message).

### 13. `affine_linear` copies cornucopia 0.5.1
`src/tree_generator.py:301`

Replicates cornucopia's matrix layout, including its unusual shear filling.
If cornucopia changes, the margin calculation silently stops matching the
generator.

### 14. Uneven rounding of the field factors
`src/tree_generator.py:1127`, `:1130`

Python's `round` rounds halves to even (`round(1.5) = 2`, `round(2.5) = 2`),
so `field_voxel_mm` rounding is uneven.

### 15. Resume trusts the spec only
`generate_images.py:415`

A case is skipped if its `specs/*.json` exists, even if the image or label is
missing or truncated.

### 16. CUDA random state not restored
`src/tree_generator.py:470`

`sampled_margin_mm` restores the CPU random states but not the CUDA ones.
Harmless now (each image reseeds), fragile if reused elsewhere.

### 17. Case names use 4 digits
`generate_images.py:411`

Beyond 9999 images, names get 5 digits and sort out of order.

---

## D. Documentation

### 18. Docstring examples use a missing config
`generate_images.py:109`

The examples added to the docstring use `generator_config3.json`, which no
longer exists.

### 19. Reproducibility overstated in the docstring
`generate_images.py:44`

Says an image "can be regenerated exactly". Only the segmentation is exact;
intensities vary slightly from GPU nondeterminism (`usage.md` section 9
already says so).
