# 1000 synthetic images with ventricles and CSF for two nnU-Nets

One generator run makes two nnU-Net v2 raw datasets that share the same
images (training: see `../README.md`). Each dataset has its own labels:
`nxb_all` (233) and `nxb18` (19). The images are synthesised from **nxb18**,
using the nxb18 tree as the intensity tree.

Differences with `fixed_tree_training/`:
- Label maps: `simplified_trees_and_labels/data/outputs/nxbsimplified_csf/labels`,
  the nxbsimplified maps with ventricles (label 50) and CSF (label 24) added
  (`other_codes/add_csf_to_labels`, combined approach).
- Morphology tree (MT): `nextbrain_tree_allen_gyral_simplified_csf.json`, the
  nxbsimplified tree with `ventricles_of_forebrain_l1` (colour FV) added back,
  with the leaves `ventricles` and `csf` (`nxbsimplified_to_nxbsimplified_csf.py`).
- Levels: `nxb_all` and `nxb18` (`nxbsimplified_csf_to_nxb18.py`: nxb16 +
  ventricles (17) + CSF (18), with CSF as an 8th principal colour). There is no
  nxbsupersynth level.
- Morphology pairs: the CSF and the ventricles take the place of the background
  in the pairs (see below).

```bash
conda activate nxb
cd generate_data
./launch.sh                 # runs in the background on every GPU; log in logs/
./launch.sh --gpus 2 3      # only GPUs 2 and 3
tail -f logs/generate_*.log
```

It takes about 3 s per image per GPU, so roughly 15 min on 4 GPUs. If a run
stops, run the same command again: finished cases are skipped. Image `i`
always uses the seed `image_seed(0, i)` and template `i % 5`, whatever the
number of GPUs.

## Output

Shared data in `../data_shared/generation/` (`--out`):
```
templates/                     L re-cropped with a 12 mm margin (cached)
images/synth_XXXXX.nii.gz      images in [0, 1], uint8
labels/<net>/synth_XXXXX.nii.gz  GT of each net: label INDICES 0..N-1
plans/synth_XXXXX.json         every random parameter (replay with Generator.render)
config.json, trees/            exactly what was used
```
One nnU-Net raw dataset per net in `../models/<net>/nnUNet_raw/` (`--models`):
```
nxbsimplified_csf/nnUNet_raw/Dataset101_nxbsimplified_csf/   level nxb_all
nxb18/nnUNet_raw/Dataset102_nxb18/                           level nxb18 (the principal level)
    imagesTr/                  hard links to images/ (no extra disk space)
    labelsTr -> data_shared/generation/labels/<net>   (relative symlink)
    label_space.json           index -> raw value and name (map predictions back)
    dataset.json
```
The level `nxb_all` of `config.json` is the net `nxbsimplified_csf`.

## Augmentation (`config.json`)

These settings are stronger than in `fast_tests/` because only 5 label maps
are available.

| | fast_tests | here |
|---|---|---|
| rotation | ±5° | ±15° |
| zoom | 0.95–1.05 | 0.85–1.15 |
| shear | 0 | ±0.03 |
| elastic | 7.5 mm | 10 mm |
| low-resolution probability | 0.5 | 0.8 |
| largest voxel, isotropic / slice spacing | 1 mm / 1 mm | 1.5 mm / 3 mm |
| noise σ | 0.01–0.07 | 0–0.1 |
| template margin | 4 mm | 12 mm (keeps the anatomy inside the field of view after rotation) |

Morphology: 11 pairs, probability 0.5, 1–3 layers (except where a pair sets
its own values):
- background ↔ csf, csf ↔ cortex (`surface_structures_of_forebrain_l1`),
  csf ↔ cerebellar cortex (`gray_matter_of_the_hindbrain_l1`)
- ventricles ↔ forebrain gray matter, forebrain white matter and allocortex
- cortex ↔ white matter, cerebellar cortex ↔ cerebellar white matter,
  basal ganglia ↔ white matter, thalamus ↔ white matter
- background ↔ pons (probability 0.3)

The other settings are unchanged:
- contrasts T1, T2, FLAIR and PD, with tree-based intensities
- bias field (always applied)
- motion with probability 0.3

nnU-Net then applies its own augmentation on top: rotation, scaling,
mirroring, gamma, blur and low-resolution simulation.
