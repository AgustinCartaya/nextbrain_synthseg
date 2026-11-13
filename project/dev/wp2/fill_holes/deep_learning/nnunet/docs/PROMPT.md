# Task: train a model that restores corrupted NextBrain segmentations

## Goal
Build a self-contained project in this folder
(`/home/agustin/phd/synthseg_nextbrain/project/dev/wp2/fill_holes/deep_learning/model_training`)
that trains a 3D network to map a **corrupted label map → clean label map**. It must remove
holes, missing sections and shifted sections/blocks, and output a complete, anatomically
plausible segmentation.

The final target is the 5 real reconstructions in
`/home/agustin/phd/synthseg_nextbrain/project/dev/wp2/fill_holes/deep_learning/data/test/recon_holes_0.2mm/case-*.nii.gz`.
These are ex-vivo NextBrain hemispheres at 0.2 mm, uint16, RAS, ~667×1024×869. They were
reconstructed from oblique histological sections, so they contain exactly these artefacts.
They have **no ground truth**.

## Training data (clean labels)
- Source: `seg.left.nii.gz` and `seg.right.nii.gz` under `/home/Data/FOMO-300K/derivatives/nextbrain/<dataset>/<sub>/<ses>/anat/`.
  - Each hemisphere is a separate training sample.
  - Format: float32, ~0.4 mm, RAS, ~257 NextBrain labels. The label names are in `lut.txt` in the same folder.
- Use `/home/Data/FOMO-300K/derivatives/scans_to_process.csv` (~31k rows) to find each scan:
  - `nextbrain_dir` is the segmentation folder.
  - `input_file` is the original T1w (under `/mnt/cafetera/...`).
- Keep only scans whose original T1w has voxel size **≤ 1 mm in all three axes** (read the
  NIfTI header). Prefer higher-resolution scans: make it easy to raise that threshold, and to
  weight or oversample sub-millimetre scans, since the target is 0.2 mm. Cache the filtered list
  to a CSV.
- Hold out whole subjects for validation and test.
- Also convert the 5 test cases to L1 (see Decisions 1).

## Corruption generator (do not reimplement)
Use `../corrupt_nifti.py`. Read `../corrupt_nifti_guide.ipynb` for every parameter.
- Generate corrupted inputs **on the fly** with `corrupt_array(seg, affine, angle_range, axis, rng, brain=..., fg_box=..., **kw)`.
  - It returns `(corrupted, params, sidx)`. The clean `seg` is the target.
  - Precompute `brain_mask(seg)` and `bbox(seg > 0)` once per volume.
- Randomise all geometry and severity parameters per sample, covering at least what the test
  data shows:
  - coronal-ish planes tilted ~9–14° (also try `random_sign` and other tilts);
  - varied `thickness`;
  - all five corruption types, from mild to severe.
- Run the corruption in DataLoader workers on CPU, on a crop with its own affine (see Decisions 3).
  Precompute the brain mask for the crop, or let `corrupt_array` compute it.

## Decisions (already taken — implement them)
1. **Label space: L1 (9 tissues) first.** Convert both the FOMO training labels and the test
   cases with the existing merger, either as a library or via its CLI:
   `/home/agustin/phd/synthseg_nextbrain/project/dev/wp1/create_merged_segmentations/merge_labels.py`
    USING CLI

   ```bash
      python .../merge_labels.py --level L1 \
       --mapping /home/agustin/phd/synthseg_nextbrain/project/dev/wp1/create_merged_segmentations/data/inputs/mapping_l1.json \
       --ontology /home/agustin/phd/synthseg_nextbrain/data/basics/nextbrain_ontology_v1.csv \
       --input <folder> --output <folder>_L1
    ```
    USING THE LIBRARIE
    ```bash       
        import /home/agustin/phd/synthseg_nextbrain/project/dev/wp1/create_merged_segmentations/merge_labels.py as ml

        ontology_csv = "/home/agustin/phd/synthseg_nextbrain/data/basics/nextbrain_ontology_v1.csv"
        img = nib.load("/home/agustin/phd/synthseg_nextbrain/project/dev/testing/segment_mrixfields/data/outputs/T1W/3T/R_T1W_3T_0329/seg_registered.nii.gz")

        mapping = {
            "Background": 0,
            "Brainstem": 1,
            "CSF / ventricles": 2,
            "Cerebellar GM": 3,
            "Cerebellar WM": 4,
            "Cerebral GM": 5,
            "Cerebral WM": 6,
            "Deep GM": 7,
            "Other": 8
          }
        ontology = ml.load_ontology(ontology_csv)
        label_map = ml.build_label_map(ontology, 'L1', mapping)

        merged_data, unknown_labels = ml.merge_segmentation(img.get_fdata(), ontology, label_map)

    ```
   - In Python: `load_ontology`, `build_label_map(ontology, "L1", mapping)`, then
     `merge_segmentation(seg, ontology, label_map)`.
   - All FOMO label IDs exist in the ontology.
   - L1 classes: 0 Background, 1 Brainstem, 2 CSF/ventricles, 3 Cerebellar GM, 4 Cerebellar WM,
     5 Cerebral GM, 6 Cerebral WM, 7 Deep GM, 8 Other.
   - Keep the code label-space agnostic (number of classes from the config), so finer levels can
     be trained later.
2. **Resolution: train at 0.2 mm** to match the target.
   - Upsample the ~0.4 mm FOMO labels by **linear interpolation of the one-hot channels, then argmax**
     (not nearest neighbour). Do this after the L1 merge, so only 9 channels are resampled.
   - Upsample on the fly or cache. Full 0.2 mm hemispheres are ~300 M voxels, so check disk
     space before caching all of them.
3. **Model: residual 3D U-Net** (nnU-Net ResEnc style; MONAI or nnU-Net building blocks).
   - Patches 192³ at 0.2 mm (~38 mm, larger than the biggest hole), ~6 resolution levels,
     instance norm, batch 2–4 per GPU, mixed precision.
   - **Residual output:** `logits = net(x) + α·onehot(x)`, so unchanged voxels are kept by default.
   - **Inputs:** the one-hot corrupted L1 map, plus 1 channel = background voxels inside the closed brain
     mask ("suspicious holes"). Do **not** use `sidx`: it is unknown for the real test cases.
   - **Corrupt crops, not whole volumes** (a full 0.2 mm hemisphere takes ~10 s):
     1. crop a region larger than the patch (e.g. 256³);
     2. shift the affine to the crop;
     3. call `corrupt_array`;
     4. take the patch from the centre.

     The generator works in world coordinates, so the section geometry stays consistent.
   - Keep ~10–20% of patches uncorrupted, so the model learns not to change correct regions.
   - Inference: sliding window with overlap and Gaussian weighting over the full test volumes.
   - Possible v2 if large holes remain: a cascade with a coarse 0.8 mm whole-hemisphere stage.
4. **Loss: cross-entropy + soft Dice** (Dice over the batch, background excluded).
   - Up-weight CE ×5–10 on voxels where corrupted ≠ clean, dilated by 1–2 voxels; otherwise the
     identity mapping dominates.
   - Track Dice on corrupted voxels as the main validation metric.
   - Note in the README that block shifts are ambiguous: the model can only make boundaries
     consistent, not recover the absolute position.

## Compute
4× NVIDIA H200 (141 GB each), 128 CPU cores, ~1 TB RAM. Use mixed precision and multi-GPU
(DDP) if useful.

## Deliverables
- `README.md`: setup, how the decisions above were implemented, how to run each step.
- Scripts or modules, each with a config file (YAML or argparse):
  - data indexing and filtering;
  - dataset with on-the-fly corruption;
  - training (checkpoints, TensorBoard or CSV logs);
  - inference;
  - evaluation.
- **Evaluation:**
  - On held-out FOMO hemispheres with synthetic corruption: Dice per label (overall and on
    corrupted voxels only), and the fraction of holes filled.
  - On the 5 real test cases: predictions saved as `.nii.gz`, plus QC figures (sagittal, coronal
    and axial slices of input / output / changed voxels), with notes on the remaining artefacts.
- Start with a small smoke test: a few volumes, a few iterations, end to end. Only then launch
  the full training. Do not modify `../corrupt_nifti.py`, the test data or FOMO.
