# Modules

Every online module is one file in `nxgen/online/` with plain functions:

- `sample(cfg_section, rng, ...) -> dict | None` draws its random
  parameters (no voxels; `None` = not applied);
- `apply(volume, params, ...) -> volume` is deterministic.

Modules without random parameters (partial volume, resolution,
normalisation) only have `apply`. Parameters are plain dicts of JSON types.
Each section below lists inputs, outputs, the parameter dict and the tests
(`tests/`).

---

## Offline: `offline/template.py`

`build_template(seg_path, out_dir, hr_voxel_mm, sigma_mm, margin_mm,
allowed, background, device, overwrite) -> (path, record)`

1. `restrict_labels`: labels not in `allowed` -> background.
2. `crop_to_foreground`: bounding box + `margin_mm`, padded with background
   where needed.
3. `super_resolve`: per label, in its bounding box: one-hot -> sinc
   upsampling (zero-padded spectrum) -> Gaussian (`sigma_mm`) -> argmax;
   background = 1 - sum of the others. Factor 1 and sigma 0 = unchanged.

Output: `<name>_hr<voxel>mm.nii.gz` (uint16) + `<name>_hr<voxel>mm.json`:
source path and SHA-256, settings, crop box, shape, removed labels, label
volumes before/after, vanished labels. A template is reused only if the
source hash and settings match.

Tests: identity, no new labels, volumes within 15 %, crop/padding, label
restriction, record and cache, **bit-exact equivalence with v2
`super_resolve_labels`**.

---

## 0. `acquisition.py`

`sample(cfg, rng, voxel_mm)`:

```
{"spacing_mm": [3], "thickness_mm": [3], "alpha": [3],
 "contrast": str | null}
```

SynthSeg priors (eqs. 20-23), see the module docstring. The physical values
are the only source of truth: an axis is degraded when its spacing is
larger than the output voxel (thickness and alpha of the other axes are
ignored). Everything else is derived from them:

- `degraded_axes(acq, voxel)`: the degraded axes;
- `describe(acq, voxel) -> (mode, axes)`: `'none'`, `'iso'`, `'aniso'`, or
  `'custom'` for hand-edited combinations `sample` never draws (e.g. two
  degraded axes; they still render);
- `sigma_vox(acq, voxel)`: slice-profile std per axis in output voxels
  (`slice_sigma_vox(thickness, alpha, voxel)`, eq. 23; 0 when not degraded).

`acq = None` means not degraded. `none_params(voxel, contrast)` gives the same
thing as a dict (all spacings = voxel).

Tests: frequencies of the modes, ranges, one axis for aniso, contrasts,
only physical keys stored; derived mode and `sigma_vox` (non-degraded
thickness ignored, `'custom'`).

---

## 1. `morphology.py`

`sample(cfg, rng, label_space, tree)`:

```
{"ops": [{"grow": group, "shrink": group, "n_iter": int,
          "grow_idx": [indices], "shrink_idx": [indices]}, ...]}   # or null
```

`apply(lab, params, n_labels)`: for each op, `n_iter` times: voxels of the
shrinking group that touch (26-neighbourhood) the growing group take the
largest neighbouring growing index. `halo(params)` = total layers (voxels
needed around a crop for an exact result).

Tests: only the shrinking group loses voxels, others untouched; crop + halo
== full volume.

---

## 2. `spatial.py`

`sample(cfg, rng)`:

```
{"rotation_deg": [3], "zoom": [3], "shear": [3], "translation_mm": [3],
 "matrix": 3x3,                      # M = R S Z (core/geometry.affine_matrix)
 "elastic_dmax_mm": float,
 "elastic_ctrl_mm": n x n x n x 3 | null}
```

Output HR voxel `v` reads the template at
`round((M (v - c) h + t + e(v)) / h + c)` (nearest; outside = background),
with `e` the cubic B-spline of the control vectors over the whole template
(`core/bspline.py`).

- `source_box(params, start, size, shape, h, halo)`: exact box of the
  template read by a region.
- `apply(crop, crop_lo, params, start, size, shape, h)`: labels of the
  region, computed in slabs.
- `identity_params()`.

Tests: identity, patch == crop of full volume, 180-degree rotation == flip.

---

## 3. `partial_volume.py`

`apply(lab, method, factor, top_k, sigma_vox)` returns

```
index  (k, X, Y, Z) int16    label indices, largest fraction first (0 where weight 0)
weight (k, X, Y, Z) float32  fractions, sum to 1
seg    (X, Y, Z)    int16    hard segmentation
lost   (X, Y, Z)    float32  fraction dropped by the top-k truncation
```

`downsample`: box of `factor³` HR voxels, stride `factor`; seg = largest
fraction, ties -> smallest index. `smooth` (factor 1): Gaussian window,
stride 1; seg = the HR labels. Weighted histogram per window without
K-sized arrays (see docstring), in slabs.

Tests: factor 1 = hard labels; mass of every label conserved; seg ==
brute-force majority; truncation reported in `lost`; smooth keeps seg and
mixes boundaries.

---

## 4a. `tree_intensities.py`

`sample(cfg, rng, tree, label_space, acquisition)`:

```
{"colors_mu": [n_colors], "colors_sigma": [n_colors],
 "resolution_mm": float, "contrast": str | null,
 "mu": [K], "sigma": [K],            # per label index, tree units
 "class_of": [K], "class_names": [G],   # generation classes
 "intensity_scale": float, "noise_seed": int}
```

Sampling rules in the module docstring (same model as v2 `TreeSampling`).
`class_of` tells which labels share a Gaussian at this resolution (for the
WP4 loss).

Tests: collapsed node == parent (and same class), per-contrast `r_max`,
visible nodes identical at any resolution, null nodes, `min_sigma`.

## 4b. `synthesis.py`

`apply(pv, tree_params) -> image (X, Y, Z)`:
`sum_i w_i (mu[l_i] + sigma[l_i] eps_i) / intensity_scale`, one eps per voxel
and per generation class (seeded by `noise_seed`).

Tests: on pure voxels, mean/std/KS match the Gaussians; two labels of the
same class in a voxel share one noise draw.

---

## 5a. `bias.py`

`sample(cfg, rng)` -> `{"ctrl_values": n x n x n, "vmax": float}` or null.
`apply(img, params, start, shape)`: `img * (1 + B)`, B the B-spline of the
control values, evaluated at the region's global positions; in
[1 - vmax, 1 + vmax]. `field(...)` returns the field itself.

Tests: range, patch == crop, prob 0.

## 5b. `motion.py`

`sample(cfg, rng, shape)` -> `{"axis", "shots", "reference_shot",
"rotation_deg": [shots][3], "translation_mm": [shots][3]}` or null.
`apply(img, params, voxel_mm)`: k-space lines along `axis` in `shots`
blocks, each from the image moved by its shot's pose (`rigid`); the shot at
the k-space centre stays still. Only a 1D FFT along `axis` is needed.

Tests: zero motion = identity; motion changes the image; `rigid`
translation.

## 5c. `resolution.py`

`apply(img, acquisition, voxel_mm, noise_fn=None) -> (img, lowres)`: blur
(Gaussian, `acquisition.sigma_vox`, computed from thickness and alpha) ->
downsample to `ceil(n / ratio)` -> `noise_fn` -> upsample (trilinear,
align_corners). `acquisition` None or no degraded axis: unchanged, `lowres`
None. `lowres_shape(...)`, `gaussian_blur_axis(...)`.

Tests: not degraded (None or all spacings = voxel) unchanged; low-res shape; noise applied on the low-res
grid; **close to v2 `LowResAcquisitionFinalTransform`** (max difference
< 0.02 away from the borders; only the kernel truncation differs).

## 5d. `noise.py`

`sample(cfg, rng)` -> `{"sigma", "seed", "space"}`;
`apply(img, params)` -> `img + sigma N(0,1)` (seeded).

Tests: std, same seed same noise, sigma 0.

## 6. `normalization.py`

`apply(img, cfg) -> (img, {"method", "lo", "hi"})`: `minmax`,
`percentile`, `none`.

---

## Core (`nxgen/core/`)

| file | content |
|---|---|
| `config.py` | `DEFAULTS`, `load_config`, `validate`, `config_hash` |
| `rng.py` | `image_seed`, `module_rng`, `draw`, `check_spec`, `torch_generator` |
| `io.py` | NIfTI (uint16 labels, float32 images), JSON, `file_sha256` |
| `tree.py` | `LabelTree` (from v2 `tree_sampling.py`, unchanged) |
| `label_space.py` | `LabelSpace`: indices <-> raw values/names, `resolve(group)`, `lut` |
| `geometry.py` | `rotation_matrix`, `affine_matrix`, `pooled_affine`, `shifted_affine` |
| `bspline.py` | smooth fields from control points (`axis_weights`, `evaluate`) |
| `recorder.py` | `Recorder`: keep/save intermediate steps, time each step |

## Diagnostics (`nxgen/checks.py`)

Never called by the generator. `labels_present`, `label_volumes`,
`new_labels`, `vanished_labels`, `contacts`, `adjacency_violations` (e.g.
WM touching CSF), `gaussian_fit_pure`.
