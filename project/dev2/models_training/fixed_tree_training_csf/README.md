# nxb18 nnU-Net on synthetic images with ventricles and CSF (fixed tree)

Same pipeline as `../fixed_tree_training/` (nxb16), with two labels more: the 1000 synthetic
images of `generate_data/` are made from label maps with ventricles and CSF, and synthesised from
**nxb18** = the 16 nxb16 classes + ventricles (17) + CSF (18). One nnU-Net v2 (3d_fullres,
ResEnc M) is trained on nxb18. The generator also writes the GT of `nxbsimplified_csf`, which can
be trained later with the same scripts.

| net                 | dataset                          | labels | status |
|---------------------|----------------------------------|--------|--------|
| `nxb18`             | `Dataset102_nxb18`               | 19     | trained |
| `nxbsimplified_csf` | `Dataset101_nxbsimplified_csf`   | 233    | raw dataset only |

Progress, decisions and warnings of the run: `DONE.txt`. Plan: `TODO.txt`.

## Layout
```
fixed_tree_training_csf/
  env.sh                         paths of one net: source env.sh <net>
  01_preprocess.sh .. 06_predict.sh   the steps below, in order
  make_splits.py, share_images.py, map_labels.py   helpers called by the steps
  generate_data/                 generator script, config and logs (see generate_data/README.md)
  data_shared/
    generation/                  generator output, shared by the nets
      images/  labels/<net>/  plans/  templates/  trees/  config.json
    shared_preprocessed/nnUNetPlans_3d_fullres/   preprocessed images, once for all the nets
    test -> ../../fixed_tree_training/data_shared/test   MRIxFields test set (symlink, not copied)
  models/<net>/                  one folder per net = its nnU-Net roots
    nnUNet_raw/  nnUNet_preprocessed/  nnUNet_results/  logs/
    predictions/mrixfields/      predict.sh + predictions <contrast>/<field>/<case>.nii.gz
    evaluation/mrixfields/       nextbrain_nxb18/ (16 structures vs NextBrain),
                                 supersynth_ventricles_csf/ (ventricles, CSF vs SuperSynth)
```
`source env.sh <net>` points `nnUNet_raw`, `nnUNet_preprocessed` and `nnUNet_results` to
`models/<net>/`. Details of the shared images and the splits: see `../fixed_tree_training/README.md`
(identical mechanism).

## Run everything, step by step
From this folder, env `nnunet` for nnU-Net commands, `nxb` for the generator and the evaluation.

**0. Generate the images** (env `nxb`): `generate_data/launch.sh --gpus 1 2 3`.

**1. Check that the generator has finished:**
```bash
grep -H numTraining models/*/nnUNet_raw/*/dataset.json      # "numTraining": 1000
```
**2. Preprocess** (plan + preprocess + shared images + template split, a few hours):
```bash
nohup bash 01_preprocess.sh nxb18 > preprocess_nxb18.log 2>&1 &
```
**3. Check:** `grep -h "fold\|linked" models/nxb18/logs/preprocess_*.log` (5 folds of 200
validation cases), `ls data_shared/shared_preprocessed/nnUNetPlans_3d_fullres/*.b2nd | wc -l` (1000).

**4. Train the 5 folds** in the background (resume with `EXTRA="--c"`, same command):
```bash
export nnUNet_n_proc_DA=12
bash 03_launch_folds.sh nxb18 "1 2 3"
```
Follow: `tail -f models/nxb18/logs/train_*_fold0.log`. Stop: `pkill -f "nnUNetv2_train 102"`.

**5. Cross-validation, final network and cleanup:**
```bash
bash 04_crossval.sh nxb18
bash 05_cleanup.sh nxb18
```
**6. Segment new images:** `bash 06_predict.sh nxb18 <input_dir> <output_dir> [gpu] [folds]`
(input `<case>_0000.nii.gz`; output in raw label values 0..18). MRIxFields:
`bash models/nxb18/predictions/mrixfields/predict.sh "1 2 3"`.

**7. Evaluate on MRIxFields** (env `nxb`): see `models/nxb18/evaluation/mrixfields/*/`.

## Results
Cross-validation on synthetic images (step 5):

| net | CV Dice (mean over labels) | per fold (held-out template case-1..5) | postprocessing |
|---|---|---|---|
| `nxb18` | | | |
| `nxb16` (reference, `../fixed_tree_training`) | 0.829 | 0.814 0.832 0.844 0.823 0.831 | largest component of pallidum and gray hypothalamus |

MRIxFields (step 7): see `models/nxb18/evaluation/mrixfields/nextbrain_nxb18/results_summary.md`.
