# NextBrain fails when the brain touches the +Y or +Z image border

Date: 2026-11-14

## Summary

`mri_histo_atlas_segment_fireants` crashes on some of our synthetic T1w images with

```
Exception: Could not initialize label group 9 (this should not happen unless SynthSeg did really poorly)
```

The message is misleading. SuperSynth segments these images correctly. The crash comes from a
bug in NextBrain's cropping function, which cuts away part of the brain whenever the tissue is
within 5 voxels of the upper end of the Y or Z axis of the image.

Workaround: zero-pad the input before running NextBrain. This is now done in
`nextbrain_for_changed_restored.sh` (same folder as this note).

## Symptoms

- The run stops right after `Computing initial values for means and variances` with the
  exception above. The group number can differ from case to case.
- The output folder contains `SuperSynth/`, `temp_reference.nii.gz` and
  `temp_fake_linear.nii.gz`, but no `seg.right.nii.gz`.
- `temp_reference.nii.gz` is much shorter than the brain along one axis. For case-1 it is
  204 x 195 x 259 voxels, while the input is 224 x 433 x 294.

## Cause

The bug is in `cropLabelVolTorch`, in

```
/home/code/freesurfer/freesurfer-nextbrain/python/packages/ERC_bayesian_segmentation/ext/my_functions.py
```

lines 123-124:

```python
i2 = i2 if i2 < V.shape[0] else torch.tensor(V.shape[0] - 1, ...)
j2 = j2 if j2 < V.shape[1] else torch.tensor(V.shape[0] - 1, ...)   # should be V.shape[1] - 1
k2 = k2 if k2 < V.shape[2] else torch.tensor(V.shape[0] - 1, ...)   # should be V.shape[2] - 1
```

`segment_fireants.py` calls this function with `margin=5` to crop the image around the tissue
mask of the hemisphere being segmented. If the mask plus the margin goes past the end of the
Y or Z axis, the upper limit is clamped to the size of the X axis instead of the size of that
axis.

Our images are single hemispheres, so X is the shortest axis (about 90 mm) and Y the longest
(about 175 mm). Clamping Y to the X size therefore discards a large part of the brain.

Worked example, case-1 (224 x 433 x 294 voxels at 0.4 mm):

1. The tissue mask spans Y slices 33 to 432, so it reaches the last slice.
2. The upper Y limit should be clamped to 432. It is clamped to 223 (X size minus 1).
3. The crop keeps Y slices 28 to 223: 195 slices instead of about 405.
4. The right pallidum (label 52, group 9) is entirely outside the kept region, so the
   intensity initialisation finds no voxels for it and raises the exception.

Only the upper ends of Y and Z are affected. The lower ends and the X axis are clamped
correctly.

## Which of our cases are affected

Empty voxels between the last non-zero image voxel and the upper end of each axis:

| Case | Shape (voxels) | Gap at +X, +Y, +Z | Result without padding |
|------|----------------|-------------------|------------------------|
| case-1 | 224 x 433 x 294 | 5, 0, 25 | fails |
| case-2 | 209 x 447 x 293 | 0, 8, 0 | works |
| case-3 | 224 x 433 x 294 | 5, 0, 25 | fails |
| case-4 | 234 x 489 x 365 | 17, 24, 32 | not affected by this bug, but fails for another reason (see below) |
| case-5 | 223 x 449 x 325 | 21, 11, 10 | not affected by this bug, but fails for another reason (see below) |

Notes:

- What matters is the tissue mask of the segmented hemisphere (SuperSynth labels without
  ventricles and CSF), not the non-zero voxels of the image. Case-2 has non-zero voxels up to
  the last Z slice, but its right-hemisphere tissue ends about 8 mm before the border, so it is
  not affected. Its `seg.right.nii.gz` was checked and covers the whole hemisphere.
- Case-4 and case-5 never ran in the original script because `set -e` stopped the loop at the
  first failing case.

## Things that were checked and ruled out

- NIfTI headers: sform and qform agree, and the SuperSynth outputs are in the same world
  space as the inputs.
- SuperSynth quality: the segmentation of case-1 is complete, with all right-hemisphere
  labels present and in one connected component.
- Hemisphere side for cases 1-3: they are right hemispheres and are detected as such.

## Workaround in our script

`nextbrain_for_changed_restored.sh` now does the following for each case:

1. Writes a copy of the input with 10 voxels of zeros on every side (`PAD=10`) to
   `<output>/input_padded.nii.gz`. The affine is shifted so that world coordinates do not
   change.
2. Runs `mri_histo_atlas_segment_fireants` on the padded copy.
3. Resamples `seg.right.nii.gz` and `SuperSynth/ribbon.mgz` onto the original, unpadded image
   with `mri_vol2vol --regheader`, as before. The `*_resampled.nii.gz` files are therefore on
   the original grid.
4. Deletes the padded copy.

Tested on case-3, which failed before: the run completes, the resampled segmentation has the
same shape and affine as the input, contains 259 labels, and spans Y slices 33 to 431 of 433.

## Second problem: case-4 and case-5 are left hemispheres

The script used `--side right` for every case. SuperSynth labels case-4 and case-5 as left
hemispheres (case-4: 207 cm3 of left cerebral white matter against 0.02 cm3 on the right).
With `--side right` NextBrain discards all left labels, almost nothing is left to segment
(`temp_reference.nii.gz` is 13 x 40 x 29 voxels), and the run fails. This is unrelated to the
cropping bug and padding does not help.

The script now detects the side per case:

1. It runs `mri_super_synth` on the padded image first. NextBrain reuses that output, so no
   time is lost.
2. It reads `SuperSynth/volumes.csv` and picks the hemisphere with the larger cerebral white
   matter plus cortex volume.
3. It passes that side to NextBrain.

Output names follow the side: `seg.left.nii.gz`, `seg.left_resampled.nii.gz` and
`vols.left.csv` for case-4 and case-5, `seg.right...` for cases 1-3. Both sides use the same
label numbers and the same `lut.txt`, so only the file names change for downstream code.

Tested on case-4 and case-5: both are detected as left, both runs complete, and the resampled
segmentations match the input grid and cover the hemisphere (257 and 256 labels).

## Proper fix

Change `V.shape[0]` to `V.shape[1]` on line 123 and to `V.shape[2]` on line 124 of
`my_functions.py`. The file is part of the shared FreeSurfer/NextBrain install and has not
been modified. The bug should also be reported upstream.

## Advice for other datasets

- Any image where the brain is within 5 voxels of the +Y or +Z border can trigger this, not
  only our synthetic images. Tightly cropped or hemisphere-only volumes are the likely cases.
- The run does not necessarily crash. From the code, if the wrong crop removes part of the
  brain without emptying a whole label group, NextBrain would finish and return a truncated
  segmentation. We have not observed this, but it is worth checking that `seg.<side>.nii.gz`
  covers the whole hemisphere when the brain is close to those borders.
- Padding the input by a few millimetres avoids the problem in all cases.
