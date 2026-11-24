# Three nnU-Nets on the same synthetic images (fixed tree)

The 1000 synthetic images made by `generate_data/` have three ground truths. One nnU-Net v2
(3d_fullres, ResEnc M) is trained per GT:

| net             | dataset                      | labels |
|-----------------|------------------------------|--------|
| `nxbsimplified` | `Dataset101_nxbsimplified`   | 231    |
| `nxb16`         | `Dataset102_nxb16`           | 17     |
| `nxbsupersynth` | `Dataset103_nxbsupersynth`   | 19     |

## Layout
```
fixed_tree_training/
  env.sh                         paths of one net: source env.sh <net>
  01_preprocess.sh .. 06_predict.sh   the steps below, in order
  make_splits.py, share_images.py, map_labels.py   helpers called by the steps
  generate_data/                 generator script, config and logs (see generate_data/README.md)
  data_shared/
    generation/                  generator output, shared by the three nets
      images/                    synth_XXXXX.nii.gz
      labels/<net>/              synth_XXXXX.nii.gz, GT of each net (label indices)
      plans/  templates/  trees/  config.json
    shared_preprocessed/
      nnUNetPlans_3d_fullres/    preprocessed images, once for the three nets
  models/<net>/                  one folder per net = its nnU-Net roots
    nnUNet_raw/Dataset<ID>_<net>/
      imagesTr/                  hard links to data_shared/generation/images/
      labelsTr -> ../../../../data_shared/generation/labels/<net>
      dataset.json, label_space.json
    nnUNet_preprocessed/Dataset<ID>_<net>/
      nnUNetPlans_3d_fullres/    <case>.b2nd -> data_shared/shared_preprocessed/... (symlink),
                                 <case>_seg.b2nd and <case>.pkl of this net
      gt_segmentations/, plans, splits_final.json, splits_templates.json
    nnUNet_results/Dataset<ID>_<net>/
      inference_information.json, inference_instructions.txt   (step 7)
      nnUNetTrainer__nnUNetResEncUNetMPlans__3d_fullres/
        fold_0..4/               checkpoint_final.pth, checkpoint_best.pth, progress.png,
                                 validation/ (predictions *.nii.gz + summary.json)
        crossval_results_folds_0_1_2_3_4/   1000 validation predictions, summary.json,
                                            postprocessing.pkl (step 7)
        plans.json, dataset.json, dataset_fingerprint.json
    logs/                        preprocess_*, train_*_fold*, crossval_* logs
```
`source env.sh <net>` points `nnUNet_raw`, `nnUNet_preprocessed` and `nnUNet_results` to
`models/<net>/`.

## Run everything, step by step
All commands are run from this folder:
```bash
cd /home/agustin/phd/synthseg_nextbrain/project/dev2/models_training/fixed_tree_training
conda activate nnunet
```

**0. Generate the images** (`nxb` env, ~15 min on 4 GPUs): `generate_data/launch.sh`, see
`generate_data/README.md`.

**1. Check that the generator has finished.** Each dataset must have a `dataset.json` with
`numTraining` 1000 (the generator writes it at the end; if it is missing, run
`python generate_data/generate.py --finalize` with the `nxb` env).
```bash
grep -H numTraining models/*/nnUNet_raw/*/dataset.json      # 3 lines, each "numTraining": 1000
```

**2. Preprocess the three nets, one after the other** (plan + preprocess + shared images +
template split). It takes a few hours, so run it in the background:
```bash
nohup bash -c 'for n in nxb16 nxbsupersynth nxbsimplified; do bash 01_preprocess.sh $n || exit 1; done' \
    > preprocess_all.log 2>&1 &
tail -f preprocess_all.log                          # per-net logs: models/<net>/logs/preprocess_*.log
```

**3. Check the preprocessing.** Each net must show 5 folds of ~200 validation cases, and
`data_shared/shared_preprocessed/` must hold 1000 images:
```bash
grep -h "fold\|linked" models/*/logs/preprocess_*.log
ls data_shared/shared_preprocessed/nnUNetPlans_3d_fullres/*.b2nd | wc -l     # 1000
```

**4. Launch the 15 trainings** (3 nets x 5 folds) in the background. The GPU lists are rotated
so that the 4 GPUs get 4, 4, 4 and 3 runs. With 8 data-augmentation workers per run, the
15 runs use ~120 of the 128 cores.
```bash
export nnUNet_n_proc_DA=8
bash 03_launch_folds.sh nxb16         "0 1 2 3"
bash 03_launch_folds.sh nxbsupersynth "1 2 3 0"
bash 03_launch_folds.sh nxbsimplified "2 3 0 1"
```
To use fewer resources, launch one net at a time (or only some folds, e.g.
`bash 03_launch_folds.sh nxb16 "0 1" "0 1"`) and start the next when it finishes.

**5. Follow the trainings.**
```bash
nvidia-smi
tail -f models/nxb16/logs/train_*_fold0.log                     # one log per net and fold
ls models/*/nnUNet_results/*/*/fold_*/progress.png              # nnU-Net training curves
```
Stop a net: `pkill -f "nnUNetv2_train 102"` (101 = nxbsimplified, 102 = nxb16, 103 = nxbsupersynth).

**6. Resume after a crash or a stop** (same GPUs as in step 4):
```bash
export nnUNet_n_proc_DA=8
EXTRA="--c" bash 03_launch_folds.sh nxb16         "0 1 2 3"
EXTRA="--c" bash 03_launch_folds.sh nxbsupersynth "1 2 3 0"
EXTRA="--c" bash 03_launch_folds.sh nxbsimplified "2 3 0 1"
```

**7. Cross-validation results and final network** (when the 5 folds of a net have finished):
```bash
bash 04_crossval.sh nxb16                 # same for nxbsupersynth and nxbsimplified
```
It gathers the 1000 validation predictions (each predicted by the fold that did not see its
template), computes the cross-validation Dice and checks whether removing all but the largest
component of each label improves it (`postprocessing.pkl`). The CV Dice is in
`models/<net>/logs/crossval_*.log` and `.../crossval_results_folds_0_1_2_3_4/summary.json`
(per label in `mean`). The **final network** of a net is the ensemble of its 5 folds
(`checkpoint_final.pth`) followed by that postprocessing; `06_predict.sh` uses it.

Per-label Dice of each fold:
```bash
python -c "import json,sys; s=json.load(open(sys.argv[1]))['mean']; [print(k, round(v['Dice'],3)) for k,v in s.items()]" \
    models/nxb16/nnUNet_results/Dataset102_nxb16/*/crossval_results_folds_0_1_2_3_4/summary.json
```

**8. Free the disk** (after step 7):
```bash
bash 05_cleanup.sh nxb16                  # same for the other nets
```
Removes `fold_*/validation/*.npz` and `*.pkl`, the softmax probabilities only needed to ensemble
several configurations. They are only written with `--npz` (no longer the default, see below).
Everything needed for the next steps is kept.

**9. Segment new images** (e.g. real scans) with the final network:
```bash
bash 06_predict.sh nxb16 <input_dir> <output_dir> [gpu]
```
- `<input_dir>`: one image per case, named `<case>_0000.nii.gz`. Any resolution and orientation:
  nnU-Net resamples to the training spacing (0.4 mm) and returns the segmentation on the input
  grid. Intensities are normalised per image (z-score), like the training images.
- `<output_dir>/*.nii.gz`: final segmentation in **raw label values** (`map_labels.py` maps the
  network's label indices with `nnUNet_raw/<dataset>/label_space.json`; it matters for
  nxbsimplified, where indices and raw values differ). `indices/` and `pp/` keep the network
  output before and after postprocessing.
- One 3D image of ~250 mm at 0.4 mm takes a few minutes per fold on a GPU; the 5-fold ensemble
  takes 5x that. For a quicker look use only one fold: `bash 06_predict.sh nxb16 in out 0 "0"`.

**10. Optional, a single network trained on all 1000 images** (no validation, ~16 h on one GPU):
```bash
bash 02_train.sh nxb16 all 0
bash 06_predict.sh nxb16 <input_dir> <output_dir> 0 all     # uses it instead of the ensemble
```
It sees all five templates, so it may generalise slightly better than each fold, but it cannot
be validated on synthetic data: compare it with the 5-fold ensemble on real images.

Other options:
- One fold in the foreground: `bash 02_train.sh nxb16 0 1` (net, fold, GPU).
- Train on all 1000 images (no validation): fold `all`, e.g. `bash 02_train.sh nxb16 all 0`.
- `--npz` is not passed to the training: with it, each validation case also saves its
  probabilities (~2.4 GB per case for 17 labels, ~13x more for the 231 labels of nxbsimplified,
  more than the free disk). Use `EXTRA="--npz"` only to ensemble several configurations.
- Planner, plans, trainer and data-augmentation workers can be changed with `PLANNER`, `PLANS`,
  `TRAINER` and `nnUNet_n_proc_DA` (see `env.sh`). `PLANNER`/`PLANS` must be the same in
  preprocessing and training.

Logs go to `models/<net>/logs/`, results to `models/<net>/nnUNet_results/`.
- To use nnU-Net commands by hand (e.g. `nnUNetv2_predict -d 102 ...`), first `source env.sh <net>`.

## Results (cross-validation on synthetic images, step 7)
| net | CV Dice (mean over labels) | per fold (held-out template case-1..5) | postprocessing |
|---|---|---|---|
| `nxb16` | 0.829 | 0.814 0.832 0.844 0.823 0.831 | largest component of pallidum and gray hypothalamus (+0.00004) |
| `nxbsupersynth` | | | |
| `nxbsimplified` | | | |

nxb16 per label: ≥ 0.90 white/gray matter, caudate, putamen, pallidum, cerebellar gray matter;
0.82–0.86 cerebellar white matter, thalamus, brainstem nuclei, gray hypothalamus; weakest
SN/STN/red nucleus 0.75, white hypothalamus 0.67, cerebellar_other 0.67, hippocampal white
matter 0.60.

## Disk use
| what | size |
|---|---|
| `data_shared/generation` | 35 GB |
| `data_shared/shared_preprocessed` | 55 GB |
| `models/<net>/nnUNet_preprocessed` | 10 GB (20 GB for nxbsimplified) |
| `models/<net>/nnUNet_results`, after step 8 | ~10 GB per net (5 x 2 checkpoints of 0.8 GB + predictions) |
| validation `*.npz` with `--npz` (removed by step 8) | 2.4 TB for nxb16 |

Everything else (`nnUNet_raw`, logs) is links or small files.

## Splits
Image `i` is synthesised from template `case-(i % 5 + 1)`. `make_splits.py` reads the template of
every image from the generator plans and writes `splits_final.json`: fold k validates on every
image of template k and trains on the other four, so validation measures generalisation to unseen
anatomy. The three nets get identical splits.

## Shared preprocessed images
The three nets have the same images, so `share_images.py` (called by `01_preprocess.sh`) keeps a
single copy of the preprocessed images in `data_shared/shared_preprocessed/` and replaces each net's copy
with a symlink. Each net keeps its own GT (`*_seg.b2nd`) and properties (`*.pkl`). Before linking,
the script checks that the image-related plan fields match, that all shapes match, and compares the
first 3 images voxel by voxel. About 54 GB of shared images, plus about 8 GB per net (more for the
231-label net), instead of about 180 GB.
- Run `01_preprocess.sh` for one net at a time: preprocessing writes the full images before they
  are deduplicated, and two `share_images.py` running at once could collide.
- Do not delete or move `data_shared/`: the nets' links (`labelsTr`, preprocessed images) would
  break. The links are relative, so the whole `fixed_tree_training/` folder can be moved as one.
- `imagesTr` holds hard links: deleting `data_shared/generation/images/` does not free the space
  until the `imagesTr` copies are deleted too.
