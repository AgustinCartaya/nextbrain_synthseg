# Tree generator v1 (WP3)

`generator_v0` plus two changes:

1. **Labels are saved as uint16** (int32 if a value is above 65535), never uint8. HR templates and
   output segmentations both use it, so NextBrain (~333 labels, values > 255) is safe. Inside the
   generator, label indices are int16 (up to 32767 labels).
2. **SynthSeg resolution augmentation** (Billot et al. 2023, Sec. 3.1.4, eqs. 20–26), built from
   cornucopia's low-res transforms, as in
   `../siam_generator/understanding/resolution_variability.ipynb` (Sec. 8). The tree uses the sampled
   acquisition resolution to decide which nodes are visible, so labels the image cannot resolve share
   one Gaussian.

![steps](data/outputs/steps.png)

## Files

```
generator_v1/
├── tree_generator.py     library + CLI (TreeGenerator, TreeGeneratorConfig, SynthSegLowResTransform, ...)
├── tree_sampling.py      label tree + TreeSampling (unchanged from v0)
├── run_example.py        step-by-step figure + samples; every config field is a CLI flag
├── test_generator.py     checks (see below)
├── generator_explained.ipynb   guided tour of every parameter, with comparison figures (English)
├── generador_explicado.ipynb   same notebook in Spanish
├── configs/freesurfer.json
└── data/inputs/          seg.nii.gz, T1w.nii.gz, synthseg_ctx_tree.json
```

Environment: `nxb` (torch + cornucopia). It uses the local cornucopia fork, like v0, and falls back
to the installed `cornucopia`.

## Pipeline

0. **Acquisition** (`SynthSegLowResTransform.sample`): with probability `1 - lowres_prob`, the native
   0.75 mm. Otherwise it is isotropic with probability `lowres_iso_prob`
   (`r_spac ~ U(0.75, lowres_max_iso)` on all axes), or else one random axis with
   `r_spac ~ U(0.75, lowres_max_aniso)`. On each degraded axis: `r_thick ~ U(0.75, r_spac)`,
   `alpha ~ U(1/1.03, 1.03)` and `sigma = 2·alpha·log(10)/(2π) · r_thick / 0.75` voxels.
1–5. Same as v0: super-resolution (offline), erosion/dilation, affine/elastic, partial volume.
6. **Tree intensities** at `tree_resolution='acquisition'`: the coarsest sampled spacing
   (`'config'` brings back v0 behaviour with the `resolution` field).
7. Bias field, motion.
8. **Low resolution** (`LowResAcquisitionFinalTransform`): Gaussian slice profile (cornucopia
   `smoothnd`), downsampling to `ceil(n / r_spac)` slices, noise in low-res space
   (`noise_space='lowres'`, as in a real scan), then upsampling back to 0.75 mm.
   `noise_space='highres'` adds the noise before this step, which is SynthSeg's order.
9. Min-max to [0, 1].

The segmentation target is never degraded: it stays at 0.75 mm with sharp boundaries.

`LowResAcquisitionFinalTransform` is the per-axis version of `cc.LowResSliceFinalTransform`. It is
needed so isotropic acquisitions can also have `thickness ≠ spacing`. On a single axis it matches
cornucopia exactly (tested).

## Usage

```bash
conda activate nxb
python run_example.py -n 6                                  # figures + volumes in data/outputs/
python run_example.py --help                                # every parameter, with its meaning
python run_example.py --print-config --lowres-prob 1        # show the final config
python run_example.py --lowres-prob 1 --lowres-iso-prob 0 --lowres-max-aniso 6 \
    --noise-space highres --noise-sigma 0.02 0.05 --contrasts T1 T2 \
    --morpho-pairs '[["cerebral_wm", "cortex"]]'
python test_generator.py
```

Precedence: dataclass defaults < `--config` JSON < field flags. Flag values are parsed as JSON
(`null`, `true`, lists). Several values become a tuple (a range).

```python
import tree_generator as tg
cfg = tg.TreeGeneratorConfig.from_json('configs/freesurfer.json', lowres_prob=1.0)
templates = tg.build_templates(['seg.nii.gz'], tree_json, cfg, 'cache/', device='cuda')
gen = tg.TreeGenerator(templates, tree_json, cfg, device='cuda')
img, seg, affine, info = gen()                  # info['acquisition'], info['resolution'], ...
acq = gen.lowres.sample()                       # or build a fixed acquisition dict
img, seg, affine, info = gen(acquisition=acq, motion=False, return_steps=True)
g2 = gen.with_config(lowres_prob=0., noise_sigma=(0.05, 0.05))   # variant sharing the templates
```

## New config fields (relative to v0)

| field | default | meaning |
|---|---|---|
| `lowres_prob` | 0.95 | probability of simulating a lower resolution (SynthSeg `1 - prob_min`) |
| `lowres_iso_prob` | 0.1 | probability of an isotropic acquisition (SynthSeg `prob_iso`) |
| `lowres_max_iso` | 4.0 | max isotropic voxel (mm) |
| `lowres_max_aniso` | 8.0 | max slice spacing (mm); the notebook uses `b_res = 9` |
| `lowres_alpha` | (1/1.03, 1.03) | range of alpha |
| `lowres_random_thickness` | true | `r_thick ~ U(r_HR, r_spac)`; false → `r_thick = r_spac` |
| `noise_space` | `'lowres'` | `'lowres'` or `'highres'` (see step 8) |
| `tree_resolution` | `'acquisition'` | `'acquisition'` or `'config'` (v0: uses `resolution`) |

## Checks (`test_generator.py`, all pass)

- A single-axis low-res run gives the same output as `cc.LowResSliceFinalTransform` (max diff 0).
- Sampled modes and ranges follow the priors (4000 draws).
- uint16 on disk, and values > 255 survive the save/load round trip.
- With the WM split into 300 parcels (values 5000–5299, 397 labels in total), the whole generator runs.
  Indices go up to 396 with no overflow, all labels are present, and raw values are saved exactly.
  351 generation classes at 0.75 mm become 8 at 6 mm.
- Shapes are correct, output is in [0, 1] with no NaN, and both noise spaces work. With the same seed,
  the segmentation is identical whatever the acquisition. The low-res grid has `ceil(n / 8)` slices
  at 6 mm.

## Notes

- With SynthSeg's priors, most images are well below 0.75 mm in resolution. Because the tree follows
  the acquisition, the fine nodes (e.g. cortical parcels, `r_max = 1`) get their own Gaussian in only
  a small fraction of images. To see them more often, lower `lowres_prob` or `lowres_max_aniso`.
- An anisotropic scan uses its thick-slice spacing for visibility, even though the in-plane
  resolution is fine. This is the conservative choice of `tree_sampling._as_resolution`.
