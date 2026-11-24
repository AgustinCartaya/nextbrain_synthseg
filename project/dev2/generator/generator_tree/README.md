# nxgen: modular synthetic brain MRI generator (WP3)

A rewrite of `../generator_v2` as small, independent modules. Same model
(SynthSeg-style generation with a NextBrain label tree), but every step can
be run, inspected and tested on its own, every random choice is recorded,
and it is fast enough to generate images during training.

![steps](data/outputs/steps_full.png)

| | v2 | nxgen |
|---|---|---|
| Full 0.4 mm volume (`case-1`, 300 labels) | ~5 s | ~0.37 s, peak 7 GB |
| 160³ patch at 0.4 mm | — | ~0.07 s, peak 2.4 GB |
| Parameters returned | some (affine, elastic, bias, motion hidden in cornucopia) | all, in one JSON *plan* |
| Replay / edit one image | regenerate with the same seed | `render(plan)`, any field editable |
| Tests | one script | 46 unit tests on phantoms + benchmark, run on request |

Timings: H200 GPU, `configs/nextbrain.json`, `tests/benchmark.py`.

## Contents

```
generator_v2_modular/
├── README.md                  this file
├── generator_structure.md     design (why it is built this way)
├── docs/
│   ├── pipeline.md            the steps, grids, plan, randomness, patch mode
│   ├── config.md              every config key
│   ├── modules.md             one section per module: inputs, outputs, params
│   └── testing.md             tests, benchmark, figures
├── nxgen/                     the package
│   ├── core/                  config, rng, io, tree, label_space, geometry,
│   │                          bspline, recorder
│   ├── offline/template.py    label map -> HR template
│   ├── online/                one file per step: acquisition, morphology,
│   │                          spatial, partial_volume, tree_intensities,
│   │                          synthesis, bias, motion, resolution, noise,
│   │                          normalization
│   ├── pipeline.py            Generator: sample_plan + render
│   ├── training.py            SyntheticStream (torch IterableDataset)
│   ├── checks.py              diagnostics used by the tests (never by the generator)
│   └── cli/                   build_templates.py, generate.py
├── configs/                   default.json, nextbrain.json, nextbrain_patch.json
├── examples/plot_steps.py     figure of every step
├── notebooks/                 13 executed notebooks explaining every part (start with notebooks/README.md)
├── tests/                     run_tests.py, test_*.py, phantoms.py, benchmark.py
└── data/
    ├── inputs/                nextbrain_complete_tree.json
    ├── templates/             HR templates (built by the offline launcher)
    └── outputs/               figures, datasets
```

Environment: `conda activate nxb` (torch, nibabel, numpy; scipy optional).
nxgen does not use cornucopia.

## Quick start

All commands run from `generator_v2_modular/`.

```bash
conda activate nxb

# 1. offline, once per label map: HR template (0.2 mm, cropped, smoothed)
python -m nxgen.cli.build_templates --config configs/nextbrain.json \
    --seg ../generator_v2/data/inputs/case-1.nii.gz --out data/templates

# 2. online: 10 images as an nnU-Net v2 raw dataset
python -m nxgen.cli.generate --templates data/templates \
    --config configs/nextbrain.json -n 10 --out data/outputs \
    --dataset-id 1 --dataset-name NextBrain --device cuda:0

# 3. look at every step
python examples/plot_steps.py --templates data/templates \
    --config configs/nextbrain.json --device cuda:0 --out data/outputs/steps.png

# 4. tests (only when you want them)
python tests/run_tests.py                       # CPU, ~10 s
python tests/run_tests.py --device cuda:0
```

## Python

```python
from nxgen import Generator, Recorder, load_config

gen = Generator(['data/templates/case-1_hr0.2mm.nii.gz'],
                'configs/nextbrain.json', device='cuda:0')
s = gen(seed=42)          # s.image (X,Y,Z) float, s.seg (X,Y,Z) int16 indices,
                          # s.affine, s.plan (every parameter), s.class_of

# two phases: draw the parameters, then render them
plan = gen.sample_plan(42)
plan['motion'] = None                            # same image without motion
plan['noise'] = None                             # ... without noise
plan['acquisition'] = None                       # ... and without low resolution
# every augmentation section can be None (only 'tree' is required)
rec = Recorder(steps='all', out_dir='debug/', profile=True)
s = gen.render(plan, rec)                        # rec.steps, rec.timings

# change the config in code
cfg = load_config('configs/nextbrain.json', patch={'size': [160, 160, 160]})

# raw label values instead of indices
raw = gen.labels.to_raw(s.seg)
```

During training (patches, generated on the GPU in the training process):

```python
from torch.utils.data import DataLoader
from nxgen.training import SyntheticStream

gen = Generator(templates, 'configs/nextbrain_patch.json', device='cuda:0')
loader = DataLoader(SyntheticStream(gen, run_seed=0), batch_size=2,
                    num_workers=0)       # CUDA generation: num_workers=0
for image, seg in loader:                # (B,1,160,160,160) float / long
    ...
```

## The pipeline in one table

| # | Step | Module | Grid | Random parameters (in the plan) |
|---|---|---|---|---|
| off | HR template | `offline/template.py` | input -> HR | — (build record JSON) |
| 0 | acquisition | `online/acquisition.py` | — | spacing, thickness, alpha, contrast (mode derived) |
| 1 | morphology | `online/morphology.py` | HR | grow/shrink groups, layers |
| 2 | spatial | `online/spatial.py` | HR | affine matrix, translation, elastic control points |
| 3 | partial volume | `online/partial_volume.py` | HR -> output | — |
| 4a | tree intensities | `online/tree_intensities.py` | — | C, V, per-label mu/sigma, generation classes, noise seed |
| 4b | synthesis | `online/synthesis.py` | output | — (uses 4a) |
| 5a | bias | `online/bias.py` | output | control values |
| 5b | motion | `online/motion.py` | output | axis, per-shot rotation/translation |
| 5c | resolution | `online/resolution.py` | output -> low-res -> output | — (uses 0) |
| 5d | noise | `online/noise.py` | low-res or output | sigma, seed |
| 6 | normalisation | `online/normalization.py` | output | — |

Details: [docs/pipeline.md](docs/pipeline.md) and [docs/modules.md](docs/modules.md).

## Main differences with v2

- Every random parameter is drawn by the module's `sample()` from its own
  random stream and stored in the plan; `apply()` is deterministic.
- Patch mode: only the part of the template a patch needs is processed.
- Partial volumes are kept as the top-k labels per voxel (memory does not
  grow with the number of labels).
- The label space (index <-> raw value) is frozen per dataset
  (`label_space.json`), configs are kept by hash, template caches check the
  source file hash.
- Config: sections per module, unknown keys and invalid values raise,
  explicit ranges (`"rotation_deg": [-10, 10]`, not `10`).
- Removed: cornucopia, the automatic template margins (`template_margin:
  "auto"`), `field_voxel_mm` (fields are evaluated directly where needed),
  `with_config`.

A guided tour with figures, one notebook per part: [notebooks/README.md](notebooks/README.md).

Known limits are listed in [docs/pipeline.md](docs/pipeline.md#known-limits).
