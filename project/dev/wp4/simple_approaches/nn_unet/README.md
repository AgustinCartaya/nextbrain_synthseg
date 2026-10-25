# nnU-Net v2 on Dataset001_NextBrain_L1

Data: 1000 synthetic 0.6 mm isotropic volumes (~190x340x260) with 9 L1 labels, from
`testing/generating_merged_seg_images/data/outputs/Dataset001_NextBrain_L1` (symlinked, not copied).

## Setup (once)
```bash
bash setup_env.sh            # conda env "nnunet": torch (cu128) + nnunetv2
conda activate nnunet
```

## Run
```bash
bash 01_prepare_and_preprocess.sh                 # link data, template-grouped splits, plan + preprocess (3d_fullres, ResEnc M)
bash 02_train.sh 0 1                              # fold 0 on GPU 1
EXTRA="--c" bash 02_train.sh 0 1                  # resume
bash 03_predict.sh <in_dir> <out_dir> 0 1         # inference
```
For other planners: `bash 01_prepare_and_preprocess.sh nnUNetPlannerResEncL` and pass
`nnUNetResEncUNetLPlans` as the third arg to `02_train.sh`.

## Splits
`prepare_dataset.py` writes `nnUNet_preprocessed/Dataset001_NextBrain_L1/splits_final.json` with one
fold per template (fold i validates on template case-(i+1), ~200 cases), so validation measures
generalisation to unseen anatomy. `python prepare_dataset.py --random_split` keeps nnU-Net's random split.

Paths (`nnUNet_raw`, `nnUNet_preprocessed`, `nnUNet_results`) are set in `env.sh`.
