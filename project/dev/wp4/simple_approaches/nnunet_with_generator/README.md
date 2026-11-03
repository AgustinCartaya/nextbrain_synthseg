# nnU-Net v2 trained on images generated on the fly (nxgen, NextBrain L1)

This is the same nnU-Net as `../nn_unet` (nnU-Net v2.8.1, ResEnc M, `3d_fullres`, 0.6 mm), but
training images are not read from a folder. A custom trainer, `nnUNetTrainerNxGen`, asks the
modular generator (`wp3/generator_v2_modular`, package `nxgen`) for a new batch at every iteration,
on the GPU, inside the training process. No training image is written to disk and none is seen twice.

First approach: level 1 of the NextBrain ontology (9 labels) only, from the 5 ground-truth L1 label maps
in `wp3/generator_v2_modular/__data_test__/inputs/L1/changed_recon_0.2mm_level1/` (case-1 … case-5, 0.2 mm).

```
                    offline, once                                      every training iteration
 L1 label maps ──► 00_build_templates ──► HR templates (0.2 mm) ──► nxgen Generator (patch mode) ──► batch ──► network
 (case-1..5, 0.2 mm)                          │                                                     ▲
                                              └──► 01: 4 full volumes per template ──► nnU-Net plan + preprocess
                                                   (fixed validation set)              (architecture, patch size,
                                                                                        validation data)
```

## Contents

```
nnunet_with_generator/
├── README.md                    this file
├── env.sh                       paths (nnU-Net folders, generator, config, templates), dataset id 101
├── configs/nextbrain_l1_0.6mm.json   generator config (= the one of the 1k-image L1 dataset)
├── 00_build_templates.sh        L1 label maps -> HR templates          (nxgen offline step)
├── 01_prepare_and_preprocess.sh validation set + nnU-Net plan/preprocess + splits
├── prepare_dataset.py           used by 01 (stages `raw` and `splits`)
├── 02_train.sh                  nnUNetv2_train ... -tr nnUNetTrainerNxGen
├── 03_predict.sh                nnUNetv2_predict ... -tr nnUNetTrainerNxGen
├── check_batches.py             makes training batches without training: speed, shapes, NIfTI files
├── trainers/                    $nnUNet_extTrainer (nnU-Net finds the trainer here)
│   ├── nnUNetTrainerNxGen.py    the trainer (subclass of nnUNetTrainer)
│   └── nxgen_batches.py         generator -> nnU-Net batches (axes, mirroring, z-score, deep supervision)
├── templates/                   created by 00
├── nnUNet_raw/ nnUNet_preprocessed/ nnUNet_results/   created by 01 and 02
└── logs/
```

## Quick start

```bash
conda activate nnunet          # the nnU-Net env of ../nn_unet; nxgen only needs torch, numpy, nibabel
cd wp4/simple_approaches/nnunet_with_generator

bash 00_build_templates.sh cuda:0          # ~3.5 min, once
bash 01_prepare_and_preprocess.sh          # ~4 min: 20 validation volumes, plans, preprocessing, splits
python check_batches.py --fold 0           # optional: look at what the network will see (source env.sh first)

bash 02_train.sh 0 1                       # fold 0 on GPU 1
EXTRA="--c" bash 02_train.sh 0 1           # resume
bash 03_predict.sh <in_dir> <out_dir> 0 1  # inference, fold 0, GPU 1
```

Short smoke test (2 epochs × 10 iterations, ~3 min including the final validation):
`NXGEN_NUM_EPOCHS=2 NXGEN_ITERS_PER_EPOCH=10 bash 02_train.sh 0 1`. It writes into
`nnUNet_results/.../fold_0`: delete that folder before the real run, or the real run will be
mixed up with the test (`--c` would resume it).

## How it works

### 1. Why nnU-Net still needs a dataset on disk

nnU-Net is built around a fixed dataset: `nnUNetv2_plan_and_preprocess` reads a dataset fingerprint
(spacings, shapes, intensities) and derives everything else from it: target spacing, patch size,
batch size, network topology (number of stages, strides, features), normalisation. The trainer
then validates on preprocessed cases.

So a small dataset is still generated once, with the same generator and the same config as the
training stream (`prepare_dataset.py raw`): **4 full synthetic volumes per template**
(`N_VAL=4`, 20 volumes), named `case-<t>_<j>` (e.g. `case-1_000`). They play two roles:

- **planning**: they look like the training images (same generator, same 0.6 mm grid, same
  field of view), so nnU-Net plans the same network as for the 1000-image dataset of
  `../nn_unet`: ResEnc M, patch 128×192×96, batch 2, spacing 0.6 mm, z-score normalisation,
  6 stages. This was checked: the plan is identical;
- **validation**: the per-epoch pseudo Dice and the final validation (`fold_x/validation/`) are
  computed on them.

They are never used for training. Their seeds (`image_seed(1000, t*1000 + j)`) are independent from
the training stream's.

### 2. Splits: leave one template out

As in `../nn_unet`, a fold holds out one anatomy. `prepare_dataset.py splits` writes:

- `nnUNet_preprocessed/Dataset101_NextBrain_L1_nxgen/splits_final.json`: fold *i* validates on the
  4 volumes of template *i* (case-*(i+1)*);
- `nxgen.json` in the same folder: the template files, their names, the template held out by each
  fold (`fold_holdout`), the generator config and the label space.

The trainer builds its generator **without the held-out template**: in fold 0, the network never
sees case-1 (not even deformed), and is validated on synthetic images of case-1. Fold `all` uses the
5 templates and validates on all 20 volumes (that validation is not on unseen anatomy).

The `train` lists of `splits_final.json` hold the other templates' validation volumes only to keep
nnU-Net's format; the trainer does not load them.

### 3. The trainer (`trainers/nnUNetTrainerNxGen.py`)

nnU-Net finds trainers that are not part of its package through `$nnUNet_extTrainer`, set in
`env.sh` to `trainers/`. Training (`nnUNetv2_train ... -tr nnUNetTrainerNxGen`) and inference
(`nnUNetv2_predict ... -tr nnUNetTrainerNxGen`) both need it, so always `source env.sh`.

`nnUNetTrainerNxGen` is `nnUNetTrainer` with a different source of training batches. Unchanged:
network construction from the plan, loss (Dice + CE with deep supervision), SGD + poly LR,
1000 epochs × 250 iterations, mixed precision, checkpoints, EMA pseudo-Dice, final validation,
inference. What it changes:

| method | what it does |
|---|---|
| `get_dataloaders` | validation loader: nnU-Net's own (`nnUNetDataLoader` + validation transforms, in worker processes) on the fold's validation cases. Training "loader": a `GeneratorBatches` object (below), made by `build_from_plans` |
| `on_train_start` | nnU-Net's, then copies `nxgen.json`, `nxgen_config.json`, `label_space.json`, `tree.json` into `fold_x/` and writes `nxgen_runtime.json` (seed, device, prefetch, epochs) |
| `on_train_epoch_end` | logs, per epoch, the time spent generating and the time the training loop waited for batches |
| `on_train_end` | stops the background generation thread |

`train_step` is not touched: nnU-Net's training loop calls `next(self.dataloader_train)` and gets a
dict `{'data': ..., 'target': [...]}` in exactly the format its own loader produces.

### 4. From a generator sample to an nnU-Net batch (`trainers/nxgen_batches.py`)

`build_from_plans` (called by the trainer, and by `check_batches.py`) reads the plan and
`nxgen.json` and builds an `nxgen.Generator`:

- **patch mode**: the generator config is loaded with `patch.size` = the nnU-Net patch size. The
  generator then only processes the part of each template the patch needs (a crop of the HR template
  around a random position, morphology, deformation, partial volume, synthesis, bias, motion,
  resolution, noise on the patch only). The patch centre is drawn uniformly in the template's
  foreground bounding box, so nearly every patch contains brain (nnU-Net's 33 % foreground
  oversampling is not needed);
- **templates**: every template but the one held out by the fold, all kept on the generator GPU
  as int16 label indices (~0.8 GB each);
- **label space**: the one saved with the validation set (`label_space.json`), so index *k* is the
  same label in the generated batches, the validation set and `dataset.json`;
- **checks**: the plan spacing must equal `grid.target_voxel_mm` (0.6 mm); the normalisation must be
  z-score; each template must be larger than the patch.

Then for each of the `batch_size` samples of a batch, `GeneratorBatches.make_sample`:

1. **generates** the sample: `generator(image_seed(run_seed, i))` → image (minmax-normalised,
   float) and segmentation (label indices) on the 0.6 mm grid;
2. **reorders the axes**. nnU-Net readers (SimpleITK, and its NibabelIO too) return the NIfTI array
   with reversed axes (z, y, x), then apply the plan's `transpose_forward`. Axis *i* of a
   preprocessed nnU-Net array is NIfTI axis `2 - transpose_forward[i]` (`nnunet_axes`). Here
   `transpose_forward` is `[0, 1, 2]`, so the generator works with patch `[96, 192, 128]` in NIfTI
   axes and the result is permuted `[2, 1, 0]` to `[128, 192, 96]`. This was checked: a validation
   label map read with nibabel and permuted this way is identical, voxel for voxel, to nnU-Net's
   preprocessed array;
3. **mirrors** each axis with probability 0.5. This is the only nnU-Net augmentation kept: nnU-Net
   uses the same axes for test-time mirroring, so the network should see flipped images. The flips
   are drawn from `module_rng(seed, 'nnunet_mirror')`, so they are part of the reproducible stream;
4. **z-score** normalisation (nnU-Net's `ZScoreNormalization`, `use_mask_for_norm = False` in this
   plan): `(x - mean) / std`, computed on the patch;
5. collates the batch, moves it to the training GPU and builds the **deep-supervision targets**:
   the segmentation downsampled to each output scale of the network with `'nearest-exact'`, as
   nnU-Net's `DownsampleSegForDSTransform` (here 5 scales: 128×192×96 … 8×12×6).

nnU-Net's other training augmentations (rotation, scaling, elastic, gamma, brightness, contrast,
noise, blur, low resolution) are **not** applied: the generator does its own versions of them
(spatial deformation, random contrasts from the label tree, bias, noise, motion, low-resolution
acquisition), with the ranges of the generator config.

### 5. Seeds, reproducibility, resuming

Sample *i* of the training stream has seed `image_seed(NXGEN_SEED, i)`; batch *b* holds samples
`b·B … b·B + B − 1` (with DDP, ranks take interleaved slices). Every module of the generator draws
from its own stream derived from that seed (see the generator's `docs/pipeline.md`), so:

- two runs with the same `NXGEN_SEED`, fold and plan see the same images in the same order (up to GPU
  floating-point non-determinism);
- any sample can be regenerated alone: `generator(image_seed(seed, i))`, or re-rendered with changes
  from its plan (`generator.sample_plan(image_seed(seed, i))`);
- resuming at epoch *e* (`--c`) starts the stream at batch `e × iterations_per_epoch`: it continues
  where it stopped instead of replaying the first images.

### 6. Speed

The generator runs on the GPU inside the training process (CUDA cannot run in nnU-Net's forked
data-augmentation workers). To overlap it with the network, `GeneratorBatches` generates in a
background thread that keeps `NXGEN_PREFETCH` (default 2) batches ready. The log shows, every epoch,
`nxgen: X s generating, Y s waited by the training loop`: if *Y* is close to 0, generation is
hidden behind training; if *Y* ≈ *X*, generation sets the pace.

Measured on an H200 (shared with other jobs), patch 128×192×96, 0.2 mm templates → 0.6 mm:
**~0.4 s per sample, ~0.8 s per batch of 2**, peak ~6.6 GB. Most of it is morphology (0.16 s) and
partial volume (0.12 s), which run on the HR grid (3×3×3 HR voxels per output voxel). With 250
iterations per epoch that is ~200 s of generation per epoch, probably more than the network step,
so for now **the generator is likely the bottleneck**. Options, from simplest:

- run the generator on a second GPU so it does not compete with the network:
  `NXGEN_DEVICE=cuda:1 bash 02_train.sh 0 "1,2"` (training on GPU 1, generation on GPU 2);
- make generation cheaper in the config: lower `morphology.prob` or `n_iter`, or use coarser HR
  templates (`template.hr_voxel_mm: 0.3`, factor 2);
- fewer iterations per epoch with more epochs (same number of images, more frequent validation);
- several generator threads / GPUs (not implemented: one thread per device in `GeneratorBatches`).

### 7. Inference

`03_predict.sh` is `nnUNetv2_predict` with `-tr nnUNetTrainerNxGen` (the trainer class is only
used to build the network; nxgen is not needed for inference). Inputs must be named
`<case>_0000.nii.gz`. Images at another resolution are resampled by nnU-Net to 0.6 mm and the
prediction is written back on the input grid. Folds: `0`, `"0 1 2 3 4"` (ensemble) or `all`.
`CHK=checkpoint_best.pth bash 03_predict.sh ...` uses the best-EMA checkpoint instead of the final one.

## Things to know / known differences with `../nn_unet`

- **Normalisation of patches vs. volumes.** Training patches are z-scored with the statistics of
  the patch; at inference nnU-Net z-scores the whole (cropped) volume. A patch with much background
  has a different mean/std than the whole head. The intensity augmentation of the generator is
  much wider than that difference, but it is a real train/test mismatch. Possible fixes later:
  z-score with statistics of the full template-sized field of view, or with a foreground mask.
- **The generator's own normalisation** (`normalization.method: minmax` in the config) runs before
  the z-score; it does not matter for training (z-score removes it) but the validation volumes on
  disk are min-max normalised in [0, 1].
- **Validation set size.** 4 volumes per template (fixed seeds). The pseudo-Dice of each epoch uses
  50 random patches from them; the final validation segments the 4 full volumes. Change with
  `N_VAL=8 bash 01_prepare_and_preprocess.sh` (then rerun the preprocessing).
- **`other` label (index 8)** is almost absent from the templates (~0 % of the patch voxels in
  `check_batches.py`), so its Dice will be noisy or undefined.
- **Comparability with `../nn_unet`**: same labels, same resolution, same config of the generator
  (`configs/nextbrain_l1_0.6mm.json` = `__data_test__/inputs/l1_1k_dataset/nextbrain_l1.json`),
  same templates, same planner and network, same leave-one-template-out folds. The difference is
  1000 fixed images (with nnU-Net augmentation) vs. 250 000 new images (with generator augmentation
  only). The validation sets differ (here: 4 generated volumes of the held-out template).
- **Changing the generator config** (ranges, probabilities): edit `configs/nextbrain_l1_0.6mm.json`
  and rerun `01` (with `--overwrite` in `prepare_dataset.py raw`, or delete `nnUNet_raw/Dataset101_*`)
  so that the validation set and the training stream come from the same config. Changing
  `grid.target_voxel_mm` changes the plan: redo 01. Changing `template.*` needs 00 again.
- **Other levels of the ontology**: point `NXGEN_SEG_DIR` and the config's `labels.tree` to the label
  maps and tree of that level, use another `DATASET_ID`/`DATASET_NAME`, and run 00–02 again.

## Environment variables

| variable | default | used by |
|---|---|---|
| `nnUNet_raw`, `nnUNet_preprocessed`, `nnUNet_results` | folders in this directory | nnU-Net |
| `nnUNet_extTrainer` | `trainers/` | nnU-Net trainer lookup (train and predict) |
| `nnUNet_n_proc_DA` | 8 | validation loader workers only |
| `DATASET_ID`, `DATASET_NAME` | 101, `Dataset101_NextBrain_L1_nxgen` | all scripts |
| `NXGEN_ROOT` | `wp3/generator_v2_modular` (also put on `PYTHONPATH`) | all |
| `NXGEN_CONFIG` | `configs/nextbrain_l1_0.6mm.json` | 00, 01 |
| `NXGEN_SEG_DIR` | the L1 ground-truth folder | 00 |
| `NXGEN_TEMPLATES` | `templates/` | 00, 01 |
| `NXGEN_DEVICE` | training device | trainer: GPU of the generator, as seen in the process |
| `NXGEN_SEED` | 0 | trainer: run seed of the training stream |
| `NXGEN_PREFETCH` | 2 | trainer: batches generated ahead (0 = synchronous) |
| `NXGEN_NUM_EPOCHS`, `NXGEN_ITERS_PER_EPOCH` | 1000, 250 | trainer |
| `N_VAL` | 4 | 01: validation volumes per template |
