# Configuration

One JSON file, one section per module. The offline launcher and the
generator read the same file; each uses its own sections. The defaults are
in `nxgen/core/config.py` (`DEFAULTS`, every key commented) and in
`configs/default.json`.

Rules:

- A file only needs the values that differ from the defaults.
- Unknown sections or keys raise (`Unknown config key noise.sgma`).
- Keys with a fixed set of options are checked
  (`labels.unassigned must be one of ('error', 'background')`).
- Relative paths (`labels.tree`) are relative to the config file.
- Precedence: defaults < JSON file < `load_config(path, section={...})`.

## Distribution specs

Random parameters accept:

| JSON | Meaning |
|---|---|
| `5.0` | fixed value (nothing is drawn) |
| `[lo, hi]` | uniform in [lo, hi] |
| `{"uniform": [lo, hi]}` | same, explicit |
| `{"randint": [lo, hi]}` | integer in [lo, hi], both included |
| `{"choice": [a, b, ...]}` | one of the values, equally likely |

A fixed value pins the parameter, which is how a config gives predictable
results. Note the change from v2: `"rotations": 10` meant U(-10, 10) in v2;
here write `"rotation_deg": [-10, 10]` (`"rotation_deg": 10` means exactly
10 degrees).

Parameters drawn "per axis" (rotation, zoom, translation, shear) draw one
value per axis from the same spec.

## Sections

### `grid`
| key | default | meaning |
|---|---|---|
| `target_voxel_mm` | `0.6` | Output voxel size (mm). Must be an integer multiple of the HR voxel of the templates. |

### `labels`
| key | default | meaning |
|---|---|---|
| `tree` | `null` | Label tree JSON (required by the generator). |
| `background` | `0` | Raw value of the background. |
| `unassigned` | `"error"` | Template labels not in the tree: `"error"` or `"background"`. With `"background"`, the offline launcher also removes them from the templates. |
| `output_labels` | `"templates"` | `"templates"`: labels present in the templates; `"tree"`: every tree label. |

### `template` (offline only)
| key | default | meaning |
|---|---|---|
| `hr_voxel_mm` | `0.2` | Voxel of the HR templates. Input maps must have a voxel that is an integer multiple of it. |
| `sigma_mm` | `0.0` | Gaussian smoothing (mm) of the one-hot labels before the argmax. |
| `margin_mm` | `16.0` | Background kept around the foreground: a number or `[x, y, z]` (mm). |

### `patch`
| key | default | meaning |
|---|---|---|
| `size` | `null` | `[X, Y, Z]` output voxels, or `null` for the full volume. |

### `acquisition` (step 0)
| key | default | meaning |
|---|---|---|
| `lowres_prob` | `0.95` | Probability of a lower resolution than `target_voxel_mm`. |
| `iso_prob` | `0.1` | Probability that a degraded acquisition is isotropic. |
| `max_iso_mm` | `4.0` | Largest isotropic voxel. |
| `max_aniso_mm` | `8.0` | Largest slice spacing of an anisotropic acquisition. |
| `alpha` | `[1/1.03, 1.03]` | Range of the slice-profile deviation. |
| `random_thickness` | `true` | Thickness ~ U(target voxel, spacing); `false`: thickness = spacing. |
| `contrasts` | `null` | List of contrasts for the per-contrast `r_max` of the tree (one drawn per image). |

### `morphology` (step 1)
| key | default | meaning |
|---|---|---|
| `pairs` | `[]` | Pairs `[a, b]` of label groups: tree node name (its whole subtree), raw value, or a list of those. |
| `n_iter` | `{"randint": [1, 4]}` | HR voxel layers grown. |
| `prob` | `0.5` | Probability of applying each pair. |

### `spatial` (step 2)
| key | default | meaning |
|---|---|---|
| `rotation_deg` | `[-10, 10]` | Rotation per axis (degrees). |
| `zoom` | `[0.9, 1.1]` | Scale per axis. |
| `shear` | `0.0` | Shear per pair of axes (xy, xz, yz). |
| `translation_mm` | `0.0` | Translation per axis (mm). |
| `elastic_dmax_mm` | `7.5` | Control vectors drawn in U(-dmax, dmax) mm (0 = no elastic). |
| `elastic_ctrl` | `{"randint": [2, 7]}` | Control points per axis (>= 2; fewer = smoother). |

### `partial_volume` (step 3)
| key | default | meaning |
|---|---|---|
| `method` | `"downsample"` | `"downsample"` (HR finer than output) or `"smooth"` (factor 1). |
| `top_k` | `8` | Labels kept per voxel. `factor³` is exact for `downsample`. |
| `smooth_sigma_vox` | `0.5` | `smooth` only: window std (voxels), truncated at 2 std. |

### `tree` (step 4)
| key | default | meaning |
|---|---|---|
| `colors_mu` | `[0, 255]` | Range of the principal colour means C. |
| `colors_sigma` | `[0.255, 2.55]` | Range of the principal colour stds V. |
| `min_sigma` | `0.0` | Lower bound of every label std. |
| `intensity_scale` | `255.0` | Image divided by this after the synthesis. |
| `resolution` | `"acquisition"` | Resolution for node visibility: `"acquisition"` (coarsest spacing of step 0) or a spec (mm). |

### `bias` (step 5a)
| key | default | meaning |
|---|---|---|
| `prob` | `1.0` | Probability of a bias field. |
| `vmax` | `0.5` | Field in [1 - vmax, 1 + vmax]. |
| `ctrl` | `5` | Control points per axis (>= 2). |

### `motion` (step 5b)
| key | default | meaning |
|---|---|---|
| `prob` | `0.3` | Probability of motion. |
| `shots` | `3` | k-space shots (>= 2 for any effect). |
| `rotation_deg` | `[-10, 10]` | Rotation of each shot, per axis. |
| `translation_mm` | `[-10, 10]` | Translation of each shot, per axis. |

### `noise` (step 5d)
| key | default | meaning |
|---|---|---|
| `sigma` | `[0.01, 0.1]` | Noise std (on the image divided by `intensity_scale`). |
| `space` | `"lowres"` | `"lowres"` (on the low-res grid) or `"highres"` (output grid, before step 5c). |

### `normalization` (step 6)
| key | default | meaning |
|---|---|---|
| `method` | `"minmax"` | `"minmax"`, `"percentile"` or `"none"`. |
| `percentiles` | `[0.5, 99.5]` | `percentile` only. |

## Converting a v2 config

| v2 | nxgen |
|---|---|
| `target_voxel`, `factor` | `grid.target_voxel_mm`; HR voxel = `template.hr_voxel_mm` |
| `template_sigma_mm`, `template_margin` (mm) | `template.sigma_mm`, `template.margin_mm` |
| `morpho_pairs`, `morpho_iter: [a, b]`, `morpho_prob` | `morphology.pairs`, `morphology.n_iter: {"randint": [a, b]}`, `morphology.prob` |
| `rotations: r`, `zooms: z`, `shears: s`, `translations_mm: t` | `spatial.rotation_deg: [-r, r]`, `zoom: [1-z, 1+z]`, `shear: [-s, s]`, `translation_mm: [-t, t]` |
| `elastic_dmax_mm`, `elastic_ctrl: n` | `spatial.elastic_dmax_mm`, `elastic_ctrl: {"randint": [2, n]}` |
| `colors_*`, `min_sigma`, `intensity_scale` | `tree.*` |
| `tree_resolution: "config"` + `resolution` | `tree.resolution: <spec>` |
| `contrasts` | `acquisition.contrasts` |
| `bias_*`, `motion_*` | `bias.*`, `motion.*` (motion ranges explicit) |
| `noise_sigma`, `noise_space` | `noise.sigma`, `noise.space` |
| `lowres_*` | `acquisition.*` |
| `field_voxel_mm` | removed (not needed) |

`configs/nextbrain.json` is `../generator_v2/data/inputs/nextbrain_configs.json`
converted this way (with `unassigned: "background"`: the v2 file had
`"background_l1"`, which v2 silently treated as background).
