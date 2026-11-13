# Restoring corrupted NextBrain segmentations with a 3D residual U-Net

Trains a network that maps a **corrupted label map → clean label map**: it fills holes and
missing sections and makes shifted sections/blocks consistent. The target is the 5 real ex-vivo
reconstructions in `../../data/inputs/recon_holes_0.2mm/` (0.2 mm, no ground truth).

Training pairs are made on the fly: clean FOMO-300K NextBrain hemispheres are merged to L1
(9 tissues), upsampled to 0.2 mm and corrupted with the existing generator
`../../image_corruption/corrupt_nifti.py`, which is used unchanged.

> **Status.** The code works end to end (`launchers/smoke_test.sh` passes). One full training was
> run (`results/l1_resenc_2h`). It does well on synthetic corruption but is **not usable on the real
> cases yet**: see [Results so far](#results-so-far).

## Folder layout

```
nnunet/
├── README.md                    this file
├── configs/
│   ├── default.yaml             every setting, commented (the real run)
│   └── smoke.yaml               tiny settings for the smoke test
│
├── 01_build_index.py            the pipeline: run these in order
├── 02_prepare_test_cases.py
├── 03_preprocess_cache.py
├── 04_train.py
├── 05_infer.py
├── 06_evaluate.py
│
├── launchers/
│   ├── smoke_test.sh            runs all six steps on a few volumes (a few minutes)
│   └── run_2h.sh                the exact command line of the run in results/l1_resenc_2h
│
├── lib/                         the code the numbered scripts import (not run directly)
│   ├── config.py                reads the YAML files
│   ├── external.py              imports corrupt_nifti.py and merge_labels.py
│   ├── labels.py                label merge (L1), upsampling to 0.2 mm, label cache
│   ├── corruption.py            random corruption settings + "suspicious holes" channel
│   ├── data.py                  scan index, subject split, training patches
│   ├── model.py                 the network
│   ├── losses.py                loss and evaluation metrics
│   ├── sliding.py               sliding-window restoration of a whole volume
│   └── qc.py                    QC figures and statistics
│
├── intermediate/                generated inputs of the training; can be rebuilt, not results
│   ├── scan_headers.csv         T1w voxel size of all 31k scans   (step 1, ~10 min)
│   ├── index.csv                scans kept + train/val/test split (step 1)
│   └── label_cache/             FOMO labels merged to L1, 41 GB   (step 3 or during training)
│
├── results/                     one folder per training run
│   └── l1_resenc_2h/
│       ├── best.pt, last.pt     trained model (identical here: the last epoch was the best)
│       ├── config.yaml          settings used
│       ├── train_log.csv, val_log.csv, full_run.log
│       ├── eval_patch/          metrics on held-out FOMO patches (step 6)
│       └── predictions/         the 5 restored real cases, QC figures, report.md (step 5)
│
└── docs/PROMPT.md               the original task description
```

| step | script | reads | writes |
|---|---|---|---|
| 1 | `01_build_index.py` | FOMO scan list + T1w headers | `intermediate/scan_headers.csv`, `intermediate/index.csv` |
| 2 | `02_prepare_test_cases.py` | `../../data/inputs/recon_holes_0.2mm` | `../../data/inputs/recon_holes_0.2mm_L1` (skips files that exist) |
| 3 | `03_preprocess_cache.py` (optional) | FOMO `seg.left/right.nii.gz` | `intermediate/label_cache/` |
| 4 | `04_train.py` | index + label cache | `results/<run>/` |
| 5 | `05_infer.py` | `best.pt` + the L1 test cases | `results/<run>/predictions/` |
| 6 | `06_evaluate.py` | `best.pt` + held-out FOMO subjects | `results/<run>/eval_patch/` or `eval_volume/` |

Every script has a docstring (`python <script> --help`) and takes
`--config <yaml>` plus `--set section.key=value ...` to override single values.
All paths are in the `paths` section of `configs/default.yaml`.

## Setup

Use the existing `nnunet` conda environment. It has everything needed: PyTorch,
`dynamic_network_architectures` (the nnU-Net building blocks), nibabel, scipy, pandas, pyyaml,
matplotlib.

```bash
conda activate nnunet
cd /home/agustin/phd/synthseg_nextbrain/project/dev/wp2/fill_holes/deep_learning/nnunet
pip install tensorboard     # optional; without it only the CSV logs are written
```

MONAI is not installed in any environment, so the model uses nnU-Net's `ResidualEncoderUNet`.

**Read-only data.** Nothing is written under `/home/Data`: the FOMO files are only read, and every
writer goes through `labels.assert_writable`, which refuses paths inside `/home/Data`. The raw test
cases and `corrupt_nifti.py` are not modified either. All outputs go to `intermediate/`, `results/` and
`../../data/inputs/recon_holes_0.2mm_L1/`.

## How to run

```bash
# 0. smoke test first (40 scans, tiny network, a few minutes, GPU 0)
bash launchers/smoke_test.sh

# 1. index: T1w voxel sizes of the ~31k scans (parallel header reads, cached), filter, split
python 01_build_index.py

# 2. real test cases -> L1          (already done: ../../data/inputs/recon_holes_0.2mm_L1)
python 02_prepare_test_cases.py

# 3. optional: cache the merged training labels (otherwise done lazily by the data workers)
python 03_preprocess_cache.py --workers 64

# 4. training on the 4 GPUs (batch size and workers are per GPU)
torchrun --nproc_per_node=4 04_train.py --config configs/default.yaml
#    one GPU:   python 04_train.py --config configs/default.yaml
#    continue:  add --resume

# 5. evaluation with synthetic corruption on held-out subjects
python 06_evaluate.py --mode patch
python 06_evaluate.py --mode volume --save-volumes 2

# 6. restore the 5 real cases + QC
python 05_infer.py
```

Outputs of a run are in `results/<train.run_name>/`: `config.yaml`, `train_log.csv`, `val_log.csv`,
`tb/`, `last.pt`, `best.pt`, `epoch_XXXX.pt`, `eval_patch/`, `eval_volume/`, `predictions/`.

## How the decisions were implemented

### Training data
- `01_build_index.py` reads the header of every original T1w (`input_file`) and stores the voxel
  size of **all** scans in `intermediate/scan_headers.csv` (the slow part, done once). It then keeps scans
  with voxel size ≤ `data.max_voxel_mm` (1.0) in all three axes and writes `intermediate/index.csv`.
- **Raising the bar** is a config change: `--set data.max_voxel_mm=0.8`. The dataset applies the
  threshold again when it loads the index, so no re-indexing is needed.
- **Oversampling of sub-millimetre scans:** a scan is drawn with probability ∝
  `(1 / largest voxel size) ** data.weight_gamma`. With the default 2, a 0.7 mm scan is drawn 2×
  and a 0.5 mm scan 4× as often as a 1 mm scan; 0 gives uniform sampling.
- **Split by subject:** a hash of (seed, dataset, subject) decides train / val / test
  (2 % / 2 % by default), so all sessions and both hemispheres of a subject stay together, and a
  subject keeps its split if the index is rebuilt.
- Each hemisphere (`seg.left`, `seg.right`) is one sample. In the smoke subset about half of the
  scans passed the 1 mm filter (many T1w are 1 × 1 × 1.2 mm).

### 1. Label space: L1
- `labels.LabelMerger` uses `merge_labels.py` **as a library** (`load_ontology`,
  `build_label_map`, `build_lut`) and applies the lookup table. Labels missing from the ontology
  go to background and are reported.
- Merged FOMO hemispheres are cropped to their bounding box and cached as uint8 NIfTI in
  `intermediate/label_cache/L1/` (about 1.3 MB each, 41 GB in total).
- The code is label-space agnostic: the number of classes comes from `labels.num_classes`. For a
  finer level change `labels.level`, `paths.mapping_json`, `labels.num_classes` and `labels.names`.
- **Test cases.** The raw test cases contain labels 1, 2 and 3, which are not in the ontology.
  `labels.test_extra_label_map: {1: 1, 2: 2, 3: 5}` maps raw 3 to Cerebral GM and keeps 1 and 2.
  This reproduces the existing L1 test cases exactly (checked on cases 1 and 2). **Please confirm that 1 → Brainstem and 2 → CSF is intended**, since in that
  folder they look like labels that were simply left unchanged.

### 2. Resolution: 0.2 mm
- Upsampling is on the fly and only for the crop that is needed: linear interpolation of each
  one-hot channel, then argmax (`labels.upsample_labels`), after the L1 merge. Nothing is cached
  at 0.2 mm, so there is no disk problem.
- The factor is `round(voxel size / data.target_spacing)` (2 for the 0.4 mm FOMO labels) and the
  affine is updated so that world coordinates stay correct.

### 3. Model and samples
- **Network:** nnU-Net `ResidualEncoderUNet`, 6 levels, features 32-64-128-256-320-320, residual
  blocks 1-3-4-6-6-6, instance norm, LeakyReLU; 102 M parameters.
- **Residual output:** `logits = net(x) + α · onehot(x)` with `model.alpha = 5`. The output layer
  is zero-initialised, so the untrained model is exactly the identity. Options: `learn_alpha`,
  and `alpha_skip_holes` (no prior on hole-channel voxels).
- **Inputs:** one-hot corrupted labels (9 channels) + 1 "suspicious holes" channel = background
  voxels inside the closed brain mask (`brain_mask(x, fill_holes=True) & (x == 0)`). `sidx` is not
  used.
- **One sample** (`data.RestorationPatches`, in DataLoader workers):
  1. pick a hemisphere and a patch centre (90 % on labelled voxels);
  2. crop 128³ at 0.4 mm and upsample to a 256³ context crop at 0.2 mm;
  3. call `corrupt_array` with the affine of that crop (world-coordinate geometry);
  4. compute the hole channel on the corrupted context crop;
  5. keep the central 192³ patch; random left-right mirror.
- 15 % of the patches are left uncorrupted (`corruption.p_clean`).
- **Corruption parameters** (`corruption` section, `lib/corruption.py`), drawn per sample:
  - geometry: 55 % coronal tilted 9–14° around LR (the test data), the rest wider tilts, tilt
    around SI, axial and sagittal sections; `random_sign` on;
  - thickness: one voxel (30 %) or log-uniform 0.2–2 mm;
  - severity: a level in [0, 1] interpolates every parameter between a `mild` and a `severe`
    preset, with jitter per parameter; each of the four families (section shifts, block shift,
    holes, missing sections) is switched off with probability 0.2, so types also appear alone.
- **Inference** (`lib/sliding.py`): windows of 192³ with 50 % overlap inside the labelled
  bounding box, Gaussian weighting, argmax. The hole channel of every window is computed on its
  256³ context crop, exactly as in training. Options: left-right flip averaging and several passes.

### 4. Loss and metrics
- Cross-entropy + soft Dice (Dice over the batch, background excluded).
- CE weight ×8 (`loss.corrupted_weight`) on voxels where corrupted ≠ clean, dilated by 2 voxels.
- Validation uses a fixed, seeded set of corrupted patches of the validation subjects.
  `best.pt` is chosen on **Dice on corrupted voxels** (`dice_corrupted_mean`).
- Reported by `06_evaluate.py` and in `val_log.csv`:

| metric | meaning |
|---|---|
| `dice_all` | Dice per label over all voxels |
| `dice_corrupted` | Dice per label on voxels where input ≠ clean (main metric) |
| `dice_input` | Dice of the corrupted input itself, i.e. the "do nothing" baseline |
| `hole_filled` / `hole_correct` | fraction of hole voxels that got a label / the right label |
| `acc_corrupted` | fraction of corrupted voxels predicted correctly |
| `damage` | fraction of voxels that were correct in the input and wrong in the output |

### Evaluation on the real cases
`05_infer.py` writes, per case, the restored labels, a mask of changed voxels and a QC figure
(sagittal, coronal and axial slices: input | output | changed). `report.md` gives the number of
changed voxels, the suspicious holes before and after, and the volume per label before and after,
with a line per case to write the notes on remaining artefacts after looking at the figures.

## Results so far

### Run `l1_resenc_2h` (12 Nov 2026)
3 GPUs, batch 4 per GPU, 40 epochs x 200 iterations (8,000 steps). It was planned for under
2 hours but took **4 h 10 min**: the average was 1.8 s per step, of which 0.84 s waiting for data
(the label cache was being filled during training). The validation Dice was still rising at the
end, so the model had not converged.

Held-out FOMO subjects, synthetic corruption, 256 patches (`eval_patch/metrics.md`):

| metric | value |
|---|---|
| Dice on corrupted voxels, mean over labels | 0.553 (CSF/ventricles and Other score 0; large tissues 0.78-0.88) |
| Dice over all voxels, mean over labels | 0.954 (corrupted input: 0.936) |
| hole voxels filled / filled with the right label | 95.3 % / 90.8 % |
| accuracy on corrupted voxels | 83.4 % |
| correct voxels made wrong | 0.20 % |

Real test cases (`predictions/report.md`, `predictions/case-*_qc.png`): **not usable**.
- The model changes 34-40 % of the labelled voxels of every case.
- It fills the sulci with cortex (Cerebral GM volume +25 % in case 1) and smooths fine structure,
  most visibly in the cerebellum (Cerebellar WM -24 %).
- Likely cause, not yet verified: the FOMO training labels are in-vivo segmentations at 0.4 mm
  with much narrower or closed sulci, so open sulci in the ex-vivo cases look like holes.

Next steps to consider: open the sulci in the training labels before corruption (e.g. random
erosion of the cortex at the pial surface); fill `intermediate/label_cache/` before training
(mostly done: 30,503 hemispheres are already cached) so data loading is not the bottleneck.

### Timings
| what | result |
|---|---|
| one training sample (256³ context), one CPU core | 1-2 s |
| full model, 192³, mixed precision, one H200 | batch 2: 0.30 s/it, 19 GB; batch 4: 0.51 s/it, 37 GB |
| restoring one real case | about 1 minute |

## Limitations to keep in mind

- **Block shifts are ambiguous.** When a block of sections is displaced, nothing in the input says
  which side is the correct one. The model can only make the boundaries consistent; it cannot
  recover the absolute position. The Dice against the clean volume therefore under-rates plausible
  results in shifted regions.
- **The hole channel is a hint, not a detector.** In the test samples it marked roughly half of
  the true hole voxels: holes that open to the outside or into a sulcus are not "inside" the closed
  mask, and sulci are marked although they are correct. The network also sees the labels themselves.
- **Corruption density differs between crops and whole volumes.** `n_blobs` and the block shift
  are per image, so a 51 mm crop is corrupted more densely than a whole hemisphere with the same
  arguments. The whole-volume evaluation multiplies `n_blobs` by `evaluation.volume_blob_scale`.
- **Shifts near the crop border** bring in background from outside the crop. The 32-voxel margin
  (6.4 mm) absorbs most of it, but very large shifts can reach the patch.
- **Domain gap.** Training labels are in-vivo NextBrain segmentations upsampled from 0.4 mm;
  the targets are ex-vivo reconstructions at a native 0.2 mm.
- Training samples are drawn from OS entropy (not reproducible run to run); validation and
  evaluation sets are seeded and fixed.
- Possible v2 if large holes remain: a coarse 0.8 mm whole-hemisphere stage before this one
  (not implemented), or `inference.passes > 1`.
