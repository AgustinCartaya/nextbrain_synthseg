# Pipeline

How one image is made, and the rules every module follows. Per-module
details are in [modules.md](modules.md); config keys in [config.md](config.md).

## 1. Grids

Three grids are involved:

| Grid | Voxel | What lives there |
|---|---|---|
| input | e.g. 0.2 mm | the label maps you give to the offline launcher |
| HR (template) | `template.hr_voxel_mm` | HR templates; morphology and deformation |
| output | `grid.target_voxel_mm` | images and segmentations |

`factor = target_voxel_mm / hr_voxel_mm` must be an integer. An output voxel
covers `factor³` HR voxels: output voxel `i` covers HR voxels
`factor*i .. factor*i + factor - 1`, so its centre is HR voxel
`factor*i + (factor - 1)/2` (this is how the output affine is computed,
`core/geometry.pooled_affine`). The generator pads each template at the end
of each axis (with background) so its shape is a multiple of `factor`.

The **low-resolution grid** of step 5c only exists inside that step (and in
the recorder, step `lowres`).

All spatial transforms use *voxel-axis millimetres*: array axes scaled by
the voxel size, centred on the volume centre. They do not depend on the
orientation stored in the NIfTI affine.

## 2. Labels

Inside the generator, label maps hold **indices** `0..K-1` (int16), not raw
values. Index 0 is the background. `Generator.labels` (a `LabelSpace`) maps
indices to raw values and names:

- `labels.output_labels = 'templates'`: the labels present in the templates
  (and in the tree); `'tree'`: every label of the tree.
- `labels.unassigned = 'error'`: a template label missing from the tree
  raises; `'background'`: it becomes background.

The CLI saves the label space as `label_space.json` in the dataset the first
time and loads it in later runs, so an index always means the same label.

## 3. Plan, then render

```
seed ──► sample_plan ──► plan (JSON) ──► render ──► image, seg, affine
                           ▲   │
                 edit / load   └── saved as plans/<case>.json
```

`sample_plan(seed)` draws everything random for one image:

```
plan = {
  "version": 2, "seed": 1234, "config_hash": "3f2a...",
  "template": 0, "template_path": "data/templates/case-1_hr0.2mm.nii.gz",
  "roi":         {"start": [x, y, z], "size": [X, Y, Z]},   # output grid
  "acquisition": {...} | null,   # step 0: spacing_mm, thickness_mm, alpha, contrast
  "morphology":  {...} | null,
  "spatial":     {...} | null,
  "tree":        {...},   # per-label mu, sigma, class_of, class_names, ...
  "bias":        {...} | null,
  "motion":      {...} | null,
  "noise":       {...} | null
}
```

`null` means the module is not applied. `sample_plan` returns `null` only for
modules applied with a probability (morphology, bias, motion), but every
section except `tree` can be set to `null` (or removed) by hand to skip that
step: `morphology`, `spatial`, `acquisition`, `bias`, `motion`, `noise`.
`tree` holds the image intensities, so `render` raises an error if it is
missing. `render(plan)` only reads the plan,
the templates and the deterministic parts of the config (partial volume
method, normalisation). Rendering the same plan twice gives the same
segmentation, and the same image except for tiny floating-point
differences on the GPU.

What you can do with a plan:

- save it (`nxgen.core.io.save_json`) and render it later (exact replay);
- set a module to `None` to remove its effect from that image (e.g.
  `plan['noise'] = None`, `plan['acquisition'] = None`);
- change any value (e.g. `plan['noise']['sigma'] = 0.05`,
  `plan['acquisition']['spacing_mm'] = [0.4, 0.4, 4.0]`). The acquisition
  stores only physical values; its mode and slice-profile std are derived
  from them (`acquisition.describe`, `acquisition.sigma_vox`), so an edit
  always means what it says.

Setting `acquisition` to `None` skips the degradation only. The tree
Gaussians were drawn for the original acquisition (nodes it could not
resolve still share one Gaussian); to draw them for a sharp scan, sample a
new plan with `acquisition.lowres_prob = 0`.

Plan version 1 (with `mode`, `axis`, `sigma_vox` in `acquisition`) still
renders the same: those keys are ignored and `sigma_vox` was computed from
the same values.
- render the same plan with a `Recorder` to look at every step.

`plan['config_hash']` identifies the config the plan was drawn with; the CLI
saves that config in `configs/<hash>/`.

## 4. Randomness

- Image seed: `image_seed(run_seed, i)` (`core/rng.py`), independent for
  every `(run_seed, i)`.
- Each module draws from `module_rng(image_seed, module_name)`, its own
  numpy generator. Turning a module off (or changing its settings) does not
  change what the other modules draw: two configs that only differ in motion
  give the same deformation, colours, bias and noise.
- Modules always draw their numbers in the same order, even when a module
  ends up not applied (e.g. bias with probability < 1 draws its control
  values anyway). This keeps the streams aligned.
- Voxel-wise noise (tree noise, acquisition noise) is not in the plan; the
  plan stores a seed and a `torch.Generator` rebuilds the noise.
- Nothing uses the global random state of python, numpy or torch.

## 5. Render steps

```
template (HR, indices, on the device)
  │  spatial.source_box: the box of the template the patch reads (+ halo of
  │  the morphology)
  ▼
crop ──► morphology ──► spatial ──► partial_volume ──► synthesis
                                         │
                                         └──► seg (output grid, never degraded)
synthesis ──► bias ──► motion ──► resolution (+ noise on the low-res grid)
          ──► [noise, if added on the output grid] ──► normalisation ──► image
```

The order of the artefacts is fixed (bias, motion, resolution, noise), as
in v2.

Noise placement (`noise.space`):

| space | acquisition degraded | acquisition 'none' |
|---|---|---|
| `lowres` | added on the low-res grid, inside step 5c | added on the output grid |
| `highres` | added on the output grid, before step 5c | added on the output grid |

## 6. Patch mode

With `patch.size = [X, Y, Z]`, each plan picks a patch of the output grid:
its centre is drawn uniformly inside the foreground bounding box of the
template, and the patch is shifted to stay inside the volume. Then:

1. `spatial.source_box` computes the box of the template that the patch
   reads. The affine maps the patch box to a parallelepiped, bounded by its
   8 corners; the elastic displacement is at most the largest control
   value. The bound is exact, not an estimate.
2. The template is cropped to that box plus a halo of `morphology.halo`
   voxels (the total number of layers grown). Morphology on the crop gives,
   inside the halo, exactly the full-volume result.
3. Deformation, partial volume, synthesis: on the patch only.
4. Bias field and elastic field are evaluated at the patch's global
   positions: exactly the crop of the full-volume fields.

So a patch equals the same crop of a full-volume render (tested), except
for:

- tree noise and acquisition noise: drawn per patch (same distribution,
  different values);
- motion: simulated on the patch (ghosts wrap inside the patch);
- resolution: blur and resampling see the patch borders;
- normalisation: min/max of the patch, not of the whole image.

## 7. Speed and memory

Measured with `tests/benchmark.py` on `case-1` (HR 0.2 mm, output 0.4 mm,
300 labels, `configs/nextbrain.json`), H200, with motion and degraded
acquisition forced (`--worst-case`):

| step | full volume 239×439×289 | patch 160³ |
|---|---|---|
| morphology | 0.10 s, 7.0 GB | 0.027 s, 2.4 GB |
| spatial | 0.05 s, 2.9 GB | 0.007 s, 2.0 GB |
| partial_volume | 0.18 s, 3.2 GB | 0.024 s, 1.2 GB |
| synthesis | 0.02 s, 4.4 GB | 0.004 s, 1.1 GB |
| motion | 0.015 s, 2.2 GB | 0.002 s, 0.8 GB |
| others | < 0.01 s | < 0.01 s |
| **total** | **~0.39 s** | **~0.08 s** |

Peak memory includes the template held on the GPU (~0.5 GB as int16).

What keeps it fast: templates stay on the GPU as int16 indices; no
per-label loops in the online steps; the partial-volume histogram does not
depend on the number of labels; patch mode processes only what the patch
needs; parameters are drawn on the CPU with numpy (no GPU syncs) and are
small.

## Known limits

- Motion and resolution in patch mode are approximations (section 6).
- `template_margin: "auto"` of v2 is not available: use a fixed
  `template.margin_mm`. Anything a deformation moves outside the template
  becomes background.
- With `top_k` < factor³, a few voxels with many labels lose a small
  fraction (reported in the `lost` map of step 3).
- GPU reproducibility: segmentations are exact; images can differ by
  floating-point rounding between runs/devices.
- `SyntheticStream` with a CUDA generator needs `num_workers=0`.
