# Notebooks

A guided tour of every part of the generator. Each notebook runs on its own, and they are saved
with their outputs, so you can read them without running them.

| Notebook | What it explains |
|---|---|
| [00_overview](00_overview.ipynb) | the big picture: grids, plan → render, one image end to end, patches |
| [01_config_and_randomness](01_config_and_randomness.ipynb) | config sections, defaults, validation, distribution specs, seeds, per-module random streams |
| [02_labels_tree_label_space](02_labels_tree_label_space.ipynb) | label tree format, visibility, generation classes, label space (raw values ↔ indices), label groups |
| [03_offline_templates](03_offline_templates.ipynb) | HR templates: label restriction, crop + margin, sinc super-resolution, build record, cache |
| [04_acquisition](04_acquisition.ipynb) | step 0: SynthSeg resolution priors, slice thickness and profile, contrast |
| [05_morphology](05_morphology.ipynb) | step 1: growth between label groups, sampling, halo and crops |
| [06_spatial](06_spatial.ipynb) | step 2: affine + elastic (B-splines), source box, patch exactness |
| [07_partial_volume](07_partial_volume.ipynb) | step 3: downsample / smooth, sparse top-k, histogram trick, truncation |
| [08_intensities_synthesis](08_intensities_synthesis.ipynb) | step 4: Gaussians from the tree, collapse with resolution, synthesis, Gaussian check on pure voxels |
| [09_artefacts_normalization](09_artefacts_normalization.ipynb) | steps 5-6: bias field, motion, resolution, noise (low-res vs high-res), normalisation |
| [10_pipeline_plan_recorder](10_pipeline_plan_recorder.ipynb) | the Generator: plan fields, render steps, editing plans, Recorder, patch mode, speed and memory |
| [11_training_cli_datasets](11_training_cli_datasets.ipynb) | launchers, dataset layout, resume/regenerate, `SyntheticStream` for training, reproducibility |
| [12_tests_and_checks](12_tests_and_checks.ipynb) | the test suite, diagnostics (vanished labels, adjacency), benchmark, figures |

Suggested order: 00, then 10 (how everything fits), then the step notebooks 03-09 as needed.

## Running them

```bash
conda activate nxb
cd generator_v2_modular/notebooks
jupyter lab            # or open them in VS Code with the nxb kernel
```

* Every notebook starts with `from _setup import *` (`_setup.py`: paths, device, plotting
  helpers). It adds the package root to `sys.path`, so nothing needs to be installed.
* Device: the GPU with the most free memory, or the CPU. To choose, set `NXGEN_DEVICE`
  (e.g. `NXGEN_DEVICE=cuda:1 jupyter lab`, or `NXGEN_DEVICE=cpu`).
* Data: the NextBrain `case-1` label map from `../generator_v2/data/inputs/`. The HR template
  `../data/templates/case-1_hr0.2mm.nii.gz` is built automatically if missing (~30 s).
* Files written by the notebooks go to `../data/outputs/notebooks/` (safe to delete).
* Times and memory depend on the GPU and on how busy it is (the GPUs are shared).
