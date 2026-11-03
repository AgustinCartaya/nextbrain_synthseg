# nnU-Net v2 step-by-step guide (for beginners)

This guide covers every command used to train and run nnU-Net on
`Dataset001_NextBrain_L1`: what each one does, why we use it, and what it writes to disk.
The `.sh` scripts in this folder are thin wrappers around these commands.

---

## 0. What is nnU-Net, in one paragraph

nnU-Net ("no-new-U-Net") is a framework that **configures a 3D U-Net for you**. You give it
images plus labels in a fixed folder format. It then:

1. **Inspects** your data: voxel spacing, image sizes, intensities, number of labels.
   This is called the *fingerprint*.
2. **Plans** a network from that fingerprint: patch size, batch size, number of layers,
   and normalisation.
3. **Preprocesses** every case (resampling, normalisation) and saves them in a fast format.
4. **Trains** with 5-fold cross-validation (5 networks, each validated on a different 1/5 of the data).
5. **Predicts** new images, optionally averaging the 5 networks (an *ensemble*).

You run all of this from the terminal with commands that start with `nnUNetv2_...`.

---

## 1. One-time setup: the Python environment

```bash
bash setup_env.sh          # creates the conda env "nnunet"
conda activate nnunet      # run this in every new terminal
```

What `setup_env.sh` does:

| Command | Meaning |
|---|---|
| `conda create -y -n nnunet python=3.12` | Creates an isolated Python 3.12 environment named `nnunet`, so it can't break your other projects. |
| `conda activate nnunet` | Switches to that environment. |
| `pip install torch torchvision --index-url .../cu129` | Installs PyTorch built for CUDA 12.x so it can use the GPUs. |
| `pip install nnunetv2` | Installs nnU-Net v2, which provides all the `nnUNetv2_*` commands. |
| `python -c "import torch, nnunetv2; ... torch.cuda.is_available()"` | Sanity check. It must print `cuda True`. If it prints `False`, training will run on the CPU and be extremely slow. |

---

## 2. Environment variables: `env.sh`

nnU-Net finds its data through **three environment variables**. If any of them is missing,
every `nnUNetv2_*` command fails. Every script runs `source env.sh` first, which sets them.

```bash
source env.sh
```

| Variable | Folder | Contents |
|---|---|---|
| `nnUNet_raw` | `nnUNet_raw/` | Original data in nnU-Net format (images + labels + `dataset.json`) |
| `nnUNet_preprocessed` | `nnUNet_preprocessed/` | Fingerprint, plans, preprocessed data, CV splits |
| `nnUNet_results` | `nnUNet_results/` | Trained models, logs, validation predictions |
| `nnUNet_n_proc_DA` | – | Number of CPU workers for data augmentation during training (16 here) |

It also sets project variables that the scripts use:

- `DATASET_ID=1`: nnU-Net refers to datasets by number. `1` means `Dataset001_...`.
- `DATASET_NAME=Dataset001_NextBrain_L1`
- `SOURCE_DATASET=/home/Data/Synthetic/NextBrain/Dataset001_NextBrain_L1`: where the generated data really lives.

> **Tip:** If you run an `nnUNetv2_*` command by hand (outside the scripts), run
> `conda activate nnunet && source env.sh` in that terminal first.

---

## 3. Step 1: prepare and preprocess the data

```bash
bash 01_prepare_and_preprocess.sh                       # default: ResEnc M planner
bash 01_prepare_and_preprocess.sh nnUNetPlannerResEncL  # bigger network variant
```

This script runs two things.

### 3a. `python prepare_dataset.py`: put the data where nnU-Net expects it

nnU-Net requires this exact layout:

```
nnUNet_raw/Dataset001_NextBrain_L1/
├── dataset.json        # label names, channel names, number of cases, file ending
├── imagesTr/           # training images: <case>_0000.nii.gz   (_0000 = channel/modality 0)
└── labelsTr/           # training labels: <case>.nii.gz        (same name, without _0000)
```

The script:

1. **Symlinks** `imagesTr`, `labelsTr`, and `dataset.json` from `SOURCE_DATASET` into
   `nnUNet_raw/`. A symlink is a shortcut, so nothing gets copied. The images are about 55 GB.
2. **Checks** that the number of images, the number of labels, and `numTraining` in `dataset.json` all match.
3. **Writes custom cross-validation splits** to
   `nnUNet_preprocessed/Dataset001_NextBrain_L1/splits_final.json`.

**Why custom splits?** The 1000 synthetic cases come from only 5 anatomical templates
(`case-1` … `case-5`, about 200 each). nnU-Net's default random split would put cases from the same
template in both training and validation. Validation would then look much better than it really is.
We use **leave-one-template-out** instead:

| fold | validation set | training set |
|---|---|---|
| 0 | template case-1 (~200) | the other 4 templates (~800) |
| 1 | template case-2 | … |
| 2 | template case-3 | … |
| 3 | template case-4 | … |
| 4 | template case-5 | … |

(`splits_templates.json` records the fold-to-template mapping.)
If `splits_final.json` exists, nnU-Net uses it. If not, nnU-Net creates a random split.
`python prepare_dataset.py --random_split` skips writing it, so you get the random split.

### 3b. `nnUNetv2_plan_and_preprocess`: fingerprint, plan, preprocess

```bash
nnUNetv2_plan_and_preprocess -d 1 -c 3d_fullres -pl nnUNetPlannerResEncM -np 16 -npfp 16 --verify_dataset_integrity
```

| Flag | Meaning |
|---|---|
| `-d 1` | Dataset ID (→ `Dataset001_...`). |
| `-c 3d_fullres` | Only preprocess the **3D full-resolution** configuration. nnU-Net also has `2d`, `3d_lowres`, and `3d_cascade_fullres`. Our data is isotropic 0.6 mm and moderately sized, so `3d_fullres` is the right choice and the others would waste time and disk. |
| `-pl nnUNetPlannerResEncM` | Which *planner* designs the network. `ResEnc` = U-Net with a residual encoder, which the nnU-Net authors now recommend over the classic one. `M` targets ~9–11 GB of GPU memory; `L` targets ~24 GB (bigger, often a bit better, slower). `ExperimentPlanner` = the original classic U-Net. |
| `-np 16` | Number of CPU processes for preprocessing. |
| `-npfp 16` | Number of CPU processes for fingerprint extraction. |
| `--verify_dataset_integrity` | Checks every image/label pair: matching shapes and spacing, label values that match `dataset.json`, no NaNs. Run it at least once, because it catches many silent bugs. |

Under the hood this runs three sub-steps (you can also run them separately):
`nnUNetv2_extract_fingerprint` → `nnUNetv2_plan_experiment` → `nnUNetv2_preprocess`.

**Output** (`nnUNet_preprocessed/Dataset001_NextBrain_L1/`):

- `dataset_fingerprint.json`: statistics of your data.
- `nnUNetResEncUNetMPlans.json`: **the plans**: patch size, batch size, network architecture, and so on.
  Open it to see what nnU-Net chose.
- `nnUNetPlans_3d_fullres/`: preprocessed cases (`.b2nd` = compressed arrays, `.pkl` = metadata).
- `gt_segmentations/`: copies of the labels used for evaluation.

> ⚠️ **Planner name vs plans name:** The planner you choose here decides the *plans name* you
> must pass to training and prediction later:
>
> | planner (`-pl`, step 1) | plans name (`-p`, steps 2–3) |
> |---|---|
> | `nnUNetPlannerResEncM` | `nnUNetResEncUNetMPlans` |
> | `nnUNetPlannerResEncL` | `nnUNetResEncUNetLPlans` |
> | `ExperimentPlanner` | `nnUNetPlans` |

---

## 4. Step 2: training

### 4a. One fold at a time: `02_train.sh`

```bash
bash 02_train.sh 0 1                           # fold 0 on GPU 1
bash 02_train.sh 0 1 nnUNetResEncUNetLPlans    # if you planned with ResEncL
EXTRA="--c" bash 02_train.sh 0 1               # resume an interrupted run
```

Arguments: `<fold> [gpu] [plans]`. The underlying command is:

```bash
CUDA_VISIBLE_DEVICES=1 nnUNetv2_train 1 3d_fullres 0 -p nnUNetResEncUNetMPlans -tr nnUNetTrainer --npz \
    2>&1 | tee logs/train_nnUNetResEncUNetMPlans_nnUNetTrainer_fold0.log
```

| Piece | Meaning |
|---|---|
| `CUDA_VISIBLE_DEVICES=1` | Lets the process see **only GPU 1**. Inside the process that GPU is called `cuda:0`. This is how you pick a GPU on a shared machine. Check which GPUs are free with `nvidia-smi`. |
| `nnUNetv2_train` | The training command. |
| `1` | Dataset ID. |
| `3d_fullres` | Configuration (must have been preprocessed in step 1). |
| `0` | **Fold** to train (0–4). Use `all` to train on 100% of the data with no validation; that is useful for a final model but gives you no validation score. |
| `-p nnUNetResEncUNetMPlans` | Plans to use (see the table in step 1). |
| `-tr nnUNetTrainer` | Trainer class, i.e. the training recipe. The default is 1000 epochs × 250 iterations, SGD, poly LR, heavy data augmentation. Variants exist, for example `nnUNetTrainer_250epochs` for shorter runs. Override it with `TRAINER=... bash 02_train.sh ...`. |
| `--npz` | Also saves the validation **softmax probabilities** (not just the hard labels). These are **needed** if you later want `nnUNetv2_find_best_configuration` to evaluate ensembles. |
| `--c` | *Continue*: resumes from the latest checkpoint instead of starting over. Pass it with `EXTRA="--c"`. |
| `2>&1 \| tee logs/...log` | Prints the output to the terminal **and** saves it to a log file. |

### 4b. Several folds in parallel in the background: `02_train_v2.sh`

```bash
bash 02_train_v2.sh "2 3"             # folds 1-4, spread round-robin over GPUs 2 and 3
EXTRA="--c" bash 02_train_v2.sh "2 3" # resume them
```

This launches folds **1, 2, 3, 4** at once (fold 0 is launched separately with `02_train.sh`).
GPUs are assigned in turn: fold 1 → GPU 2, fold 2 → GPU 3, fold 3 → GPU 2, fold 4 → GPU 3.
So two trainings share each GPU. Make sure the GPU has enough memory for two (ResEnc M uses ~10 GB each).

Differences from `02_train.sh`:

| Piece | Meaning |
|---|---|
| `nohup ... &` | `&` runs the job in the background. `nohup` keeps it running **after you close the terminal or SSH session**. |
| `> $LOG 2>&1` | All output goes only to the log file (nothing is printed on screen). |
| `echo "... pid $!"` | Prints the process ID so you can stop it later with `kill <pid>`. |

**Monitoring a background run:**

```bash
tail -f logs/train_nnUNetResEncUNetMPlans_nnUNetTrainer_fold1.log   # live log (Ctrl+C to stop watching, the training keeps running)
nvidia-smi                                                           # GPU usage / memory
ps aux | grep nnUNetv2_train                                         # is it still running?
```

### 4c. What training writes

`nnUNet_results/Dataset001_NextBrain_L1/nnUNetTrainer__nnUNetResEncUNetMPlans__3d_fullres/fold_X/`:

| File | Meaning |
|---|---|
| `progress.png` | **Look at this first.** It plots training/validation loss and pseudo-Dice per epoch. |
| `training_log_*.txt` | Per-epoch text log (loss, pseudo-Dice, LR, epoch time). |
| `checkpoint_latest.pth` | Saved regularly. `--c` resumes from it. |
| `checkpoint_best.pth` | The checkpoint with the best *pseudo*-Dice on validation. |
| `checkpoint_final.pth` | The checkpoint after the last epoch. **This is the one nnU-Net uses for prediction by default.** |
| `validation/` | Predictions on the held-out fold + `summary.json` with the **real** Dice per label and per case. This is computed once, at the end. |

> **Pseudo-Dice vs real Dice:** During training, nnU-Net reports a quick "pseudo Dice" computed
> on random patches. The real metric is in `validation/summary.json`, computed on whole volumes
> after training finishes.

---

## 5. Step 3: choose the best configuration (after all 5 folds finish)

```bash
source env.sh
nnUNetv2_find_best_configuration 1 -c 3d_fullres -p nnUNetResEncUNetMPlans -tr nnUNetTrainer
```

| Flag | Meaning |
|---|---|
| `1` | Dataset ID. |
| `-c 3d_fullres` | Configuration(s) to compare. We only have one. |
| `-p` / `-tr` | Plans and trainer, the same as in training. |

It collects the 5 fold validations into one cross-validation score, tests whether
**post-processing** helps, and writes the results to `nnUNet_results/Dataset001_NextBrain_L1/`.
The post-processing it tests is "keep only the largest connected component" for each label.
It writes:

- `crossval_results_folds_0_1_2_3_4/`: merged validation results, `summary.json`, `postprocessing.pkl`.
- `inference_instructions.txt`: **the exact commands** to use for prediction and post-processing.
- `inference_information.json`: the scores.

This step needs all 5 folds trained **with `--npz`**.

---

## 6. Step 4: prediction on new images

```bash
bash 03_predict.sh <input_dir> <output_dir> [folds] [gpu] [plans]

bash 03_predict.sh data/inputs data/outputs 0 1              # only fold-0 model, GPU 1
bash 03_predict.sh data/inputs data/outputs "0 1 2 3 4" 1    # ensemble of all 5 folds (recommended once all are trained)
```

The underlying command is:

```bash
CUDA_VISIBLE_DEVICES=1 nnUNetv2_predict -i data/inputs -o data/outputs -d 1 -c 3d_fullres -p nnUNetResEncUNetMPlans -f 0 1 2 3 4
```

| Flag | Meaning |
|---|---|
| `-i` | Input folder. **Files must be named `<case>_0000.nii.gz`.** The `_0000` suffix is the channel index; nnU-Net ignores files without it. |
| `-o` | Output folder. You get one `<case>.nii.gz` segmentation per input. |
| `-d 1` | Dataset ID (a name like `Dataset001_NextBrain_L1` also works). |
| `-c 3d_fullres` | Configuration. |
| `-p` | Plans (the same as in training). |
| `-tr` | Trainer (defaults to `nnUNetTrainer`, so we omit it). |
| `-f 0 1 2 3 4` | Which fold models to use. With several folds, nnU-Net **averages their probabilities** (an ensemble), which is usually more accurate and more robust. With one fold, you get that model only. |

Useful optional flags:

| Flag | Meaning |
|---|---|
| `-chk checkpoint_best.pth` | Use the best checkpoint instead of `checkpoint_final.pth`. |
| `--save_probabilities` | Also save the softmax probabilities (`.npz`). |
| `--disable_tta` | Turns off test-time augmentation (mirroring). This is about 8× faster in 3D and slightly less accurate. |
| `-npp 4 -nps 4` | Number of CPU processes for pre- and post-processing (lower these if you run out of RAM). |

### Optional: apply post-processing

If `find_best_configuration` decided that post-processing helps, apply it to the predictions
using the paths written in `inference_instructions.txt`:

```bash
nnUNetv2_apply_postprocessing -i data/outputs -o data/outputs_pp \
  -pp_pkl_file nnUNet_results/Dataset001_NextBrain_L1/nnUNetTrainer__nnUNetResEncUNetMPlans__3d_fullres/crossval_results_folds_0_1_2_3_4/postprocessing.pkl \
  -plans_json  nnUNet_results/Dataset001_NextBrain_L1/nnUNetTrainer__nnUNetResEncUNetMPlans__3d_fullres/crossval_results_folds_0_1_2_3_4/plans.json \
  -np 8
```

### Optional: evaluate predictions against ground truth

```bash
nnUNetv2_evaluate_folder <gt_folder> <pred_folder> \
  -djfile nnUNet_preprocessed/Dataset001_NextBrain_L1/dataset.json \
  -pfile  nnUNet_preprocessed/Dataset001_NextBrain_L1/nnUNetResEncUNetMPlans.json
```

This writes a `summary.json` in `<pred_folder>` with the Dice and IoU for each label.

---

## 7. The full pipeline at a glance

```bash
# once
bash setup_env.sh
conda activate nnunet

# 1. data + plans + preprocessing
bash 01_prepare_and_preprocess.sh

# 2. train the 5 folds
bash 02_train.sh 0 1                 # fold 0 on GPU 1 (foreground)
bash 02_train_v2.sh "2 3"            # folds 1-4 on GPUs 2,3 (background)

# 3. after all folds finish
source env.sh
nnUNetv2_find_best_configuration 1 -c 3d_fullres -p nnUNetResEncUNetMPlans -tr nnUNetTrainer

# 4. predict with the 5-fold ensemble
bash 03_predict.sh data/inputs data/outputs "0 1 2 3 4" 1
```

---

## 8. Common problems

| Symptom | Cause / fix |
|---|---|
| `nnUNet_raw is not defined` or similar | You didn't `source env.sh` in this terminal. |
| `nnUNetv2_train: command not found` | The conda env isn't active: `conda activate nnunet`. |
| `CUDA out of memory` | Another job is on that GPU (check `nvidia-smi`), or two folds share a GPU that is too small. Use another GPU or the `M` planner instead of `L`. |
| Training interrupted (crash, reboot) | Re-run the same command with `EXTRA="--c"`. |
| `Could not find ... plans` | The `-p` plans name doesn't match the planner used in step 1 (see the table in step 1). |
| Prediction outputs nothing | Input files are not named `*_0000.nii.gz`. |
| Training is very slow / GPU utilisation is low | Data augmentation is CPU-bound. Raise `nnUNet_n_proc_DA` in `env.sh`, as long as the shared machine has free CPUs. |
| `find_best_configuration` complains about missing files | A fold was trained without `--npz`, or a fold isn't finished. |
