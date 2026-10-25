# Synthetic image generation: usage

`generate_images.py` generates `n` synthetic (image, label) pairs from brain
label maps and writes them as an nnU-Net v2 raw dataset. Every image gets a
random shape (morphology + deformation), random intensities drawn from a
label tree, and random scanner artefacts (bias, motion, low resolution,
noise). Every drawn parameter is saved with the image.

Contents

1. [Quick start](#1-quick-start)
2. [Files](#2-files)
3. [Pipeline](#3-pipeline)
4. [Command line](#4-command-line)
5. [Templates](#5-templates)
6. [Generator config](#6-generator-config---config)
7. [Label tree](#7-label-tree---tree)
8. [Outputs](#8-outputs)
9. [Reproducibility](#9-reproducibility)
10. [GPU memory and speed](#10-gpu-memory-and-speed)
11. [Troubleshooting](#11-troubleshooting)

---

## 1. Quick start

```bash
conda activate nxb            # needs torch, nibabel, cornucopia, torch-interpol
cd project/dev/testing/generating_merged_seg_images

# 0.6 mm images from the 0.2 mm L1 label maps (default inputs)
python generate_images.py -n 100 --device cuda:0

# 0.4 mm images from one 0.2 mm label map, margin chosen automatically
#   (set "template_margin": {"auto": 0.99} in the config, see 6.2)
python generate_images.py -n 100 \
    --seg-dir data/inputs/changed_recon_0.2mm_level1_small \
    --config data/inputs/generator_config2.json \
    --dataset-id 2 --device cuda:0
```

The dataset is written to `data/outputs/Dataset<ID>_<NAME>/` (section 8).

Use the `nxb` environment. Other environments (e.g. `base`) may lack
`torch-interpol`, which cornucopia needs; the script then stops at the
import with `No module named 'interpol'`.

---

## 2. Files

| Path | What |
|---|---|
| `generate_images.py` | The script described here (CLI, output layout, per-image specs). |
| `src/tree_generator.py` | The generator: template building, every pipeline step, config (`TreeGeneratorConfig`). |
| `src/tree_sampling.py` | Label tree (`LabelTree`) and intensity sampling (`TreeSampling`). |
| `src/merge_labels.py` | Collapses NextBrain labels to an ontology level (e.g. L1), to make the input label maps. |
| `build_l1_tree.py` | Builds the L1 label tree JSON. |
| `data/inputs/changed_recon_0.2mm_level1/` | 5 L1 label maps at 0.2 mm (default `--seg-dir`). |
| `data/inputs/changed_recon_0.2mm_level1_small/` | 1 of them (`case-1`), for quick tests. |
| `data/inputs/l1_nextbrain_tree.json` | L1 label tree (default `--tree`). |
| `data/inputs/generator_config.json` | 0.6 mm images from 0.2 mm HR templates (default `--config`). |
| `data/inputs/generator_config2.json` | 0.4 mm images from 0.2 mm HR templates. |

---

## 3. Pipeline

Three grids are involved:

- **input grid**: the label maps you give (e.g. 0.2 mm);
- **HR grid** (high resolution): `target_voxel / factor` mm, where shapes are
  edited and deformed (e.g. 0.2 mm);
- **training grid**: `target_voxel` mm, the grid of the output image and
  labels (e.g. 0.4 or 0.6 mm).

### 3.1 Offline, once per label map: HR template (`prepare_template`)

Only with `--seg-dir`; cached and reused afterwards (section 5).

1. **Label space.** Raw label values are kept. Labels missing from the tree
   go to background (`unassigned: "background"`) or raise
   (`unassigned: "error"`).
2. **Crop.** The map is cropped around the brain with a background margin
   (`template_margin`, section 6.2), padding with background where the
   margin goes past the image. The shape is made a multiple of `factor`.
3. **Super-resolution** to the HR voxel: one-hot -> sinc upsampling ->
   Gaussian smoothing (`template_sigma_mm`) -> argmax, label by label in
   its own bounding box. If the map is already at the HR voxel and
   `template_sigma_mm` is 0, the labels are unchanged.

With `--hr-template-dir`, this step is skipped: the files are used as they
are, only padded to a multiple of `factor` (section 5).

### 3.2 Online, per image (`TreeGenerator.__call__`)

| # | Step | Grid | Config fields |
|---|---|---|---|
| 0 | **Acquisition**: draw the simulated scan resolution (none, isotropic, or one thick-slice axis). | | `lowres_*` |
| 0b | **Contrast and colors**: principal colors `C`, `V`; tree resolution; contrast. | | `colors_*`, `tree_resolution`, `resolution`, `contrasts` |
| 1 | **Morphology**: random pairs of label groups, one grows into the other by a few voxel layers. | HR | `morpho_*` |
| 2 | **Deformation**: random affine (rotation, zoom, shear, translation) + elastic, nearest-neighbour resampling of the labels. | HR | `rotations`, `zooms`, `shears`, `translations_mm`, `elastic_*`, `field_voxel_mm` |
| 3 | **Partial volume**: each training voxel gets the fraction of every label among its `factor`^3 HR voxels. The output segmentation is the majority label. | HR -> training | `factor` |
| 4 | **Tree intensities**: one Gaussian per label from the tree; image = sum over classes of fraction x (mu + sigma x noise). | training | tree, `min_sigma`, `intensity_scale` |
| 5 | **Bias field**: smooth multiplicative field. | training | `bias_*`, `field_voxel_mm` |
| 6 | **Motion**: k-space ghosting/blurring from motion between shots. | training | `motion_*` |
| 7 | **Resolution**: slice-profile blur -> downsample to the acquisition spacing -> (noise) -> upsample back. | training | `lowres_*`, `noise_space` |
| 8 | **Noise** (if not added in 7), then min-max to [0, 1]. | training | `noise_sigma`, `noise_space` |

The segmentation is taken at step 3 and never degraded: it stays on the
training grid with sharp boundaries, whatever the simulated acquisition
(SynthSeg, Billot et al. 2023, Sec. 3.1.5).

**Resolution and the tree.** The acquisition drawn at step 0 also decides
which tree nodes are visible (`tree_resolution: "acquisition"`): a node is
visible if the coarsest acquisition spacing is `<= r_max`. Invisible nodes
take exactly their parent's Gaussian, so their labels become
indistinguishable in that image. Labels sharing a Gaussian form a
*generation class*; the classes of each image are saved in its spec (for a
loss that merges labels at that resolution).

---

## 4. Command line

```bash
python generate_images.py \
    -n 100 \
    --seg-dir data/inputs/changed_recon_0.2mm_level1 \
    --tree data/inputs/l1_nextbrain_tree.json \
    --config data/inputs/generator_config.json \
    --out data/outputs \
    --dataset-id 1 \
    --dataset-name NextBrain_L1 \
    --prefix synth \
    --device cuda:0 \
    --template-device cpu \
    --seed 0 \
    --start 0 \
    --overwrite
```

| Flag | Default | Description |
|---|---|---|
| `-n` | **required** | Number of images to generate. |
| `--seg-dir` | `data/inputs/changed_recon_0.2mm_level1` | Folder of label maps (`*.nii.gz`) to build the HR templates from (section 5). |
| `--hr-template-dir` | none | Folder of ready HR templates (`*.nii.gz`, at `target_voxel / factor` mm). Skips template building; `--seg-dir` is ignored (section 5). |
| `--save-hr-template-dir` | off | With `--hr-template-dir`: save the templates as used (padded) in the dataset's `templates/`. |
| `--tree` | `data/inputs/l1_nextbrain_tree.json` | Label tree JSON (section 7). |
| `--config` | `data/inputs/generator_config.json` | Generator config JSON (section 6). |
| `--out` | `data/outputs` | Output root; the dataset goes to `<out>/Dataset<ID>_<NAME>/`. |
| `--dataset-id` | `1` | nnU-Net dataset ID -> `Dataset001_...`. |
| `--dataset-name` | `NextBrain_L1` | nnU-Net dataset name -> `Dataset001_NextBrain_L1`. |
| `--prefix` | `synth` | Case name prefix -> `synth_0000`, `synth_0001`, ... |
| `--device` | `cuda` if available, else `cpu` | Device that generates the images. |
| `--template-device` | `cpu` | Device of the template super-resolution (~70 GB peak per 0.1 mm template). Unused with `--hr-template-dir`. |
| `--seed` | `0` | Run seed. Image `i` uses its own seed derived from `(seed, i)`. |
| `--start` | `0` | Index of the first image. |
| `--overwrite` | off | Regenerate cases that already exist (by default they are skipped). |

### Recipes

```bash
# 500 images on the second GPU
python generate_images.py -n 500 --device cuda:1

# Resume an interrupted run: same command, existing cases are skipped
python generate_images.py -n 500 --device cuda:1

# Add 500 more images to the same dataset (indices 500..999)
python generate_images.py -n 500 --start 500

# Regenerate image 42 (same inputs and seed as the original run)
python generate_images.py -n 1 --start 42 --overwrite

# A separate dataset with another config
python generate_images.py -n 200 --config my_config.json \
    --dataset-id 2 --dataset-name NextBrain_L1_noMotion

# 0.4 mm images from 0.2 mm label maps (crop + margin + padding by the code,
# cached in Dataset002_NextBrain_L1/templates/)
python generate_images.py -n 100 \
    --seg-dir data/inputs/changed_recon_0.2mm_level1_small \
    --config data/inputs/generator_config2.json --dataset-id 2

# A new dataset reusing those templates (no template building; the config
# must keep the same grid and label-space fields)
python generate_images.py -n 100 \
    --hr-template-dir data/outputs/Dataset002_NextBrain_L1/templates \
    --config data/inputs/generator_config2.json --dataset-id 3

# 0.2 mm label maps used directly as HR templates (only padded, no margin),
# keeping the padded templates with the dataset
python generate_images.py -n 100 \
    --hr-template-dir data/inputs/changed_recon_0.2mm_level1_small \
    --config data/inputs/generator_config2.json --dataset-id 4 \
    --save-hr-template-dir
```

---

## 5. Templates

The only hard requirement of the code is that an HR template is at
`target_voxel / factor` mm and its shape is a multiple of `factor`. Both
sources take care of the shape, so you never pad or crop a label map
yourself.

| | `--seg-dir` | `--hr-template-dir` |
|---|---|---|
| Input | label maps, voxel size an integer multiple of the HR voxel | HR templates (or label maps) already at the HR voxel |
| Labels missing from the tree | per `unassigned`, removed from the template | per `unassigned`, mapped to background when generating |
| Crop + margin (`template_margin`) | yes | no |
| Padding to a multiple of `factor` | yes | yes (at the end of each axis, affine unchanged) |
| Super-resolution + smoothing | yes | no |
| Voxel size check | yes | yes (error if not `target_voxel / factor`) |
| Saved in `templates/` | always (cache) | only with `--save-hr-template-dir` |
| Cost | slow once, then cached | immediate |

**Cache.** With `--seg-dir`, each template is saved as
`templates/<case>_hr<voxel>mm.nii.gz` with a build record
`<case>_hr<voxel>mm.json` (the settings it was built with: grid,
`template_sigma_mm`, `template_margin`, label space, and the deformation
fields if the margin is automatic). Later runs on the same dataset reuse it
if the settings are unchanged and rebuild it otherwise, printing which
settings changed. Templates cached before records existed are reused with a
note; delete them to rebuild. A new dataset ID builds its own templates.

**Margin check.** With `--hr-template-dir`, templates that have a build
record print the margin they were built with. Others (e.g. raw label maps)
get a warning if the background around the brain is smaller than the
worst-case margin of the deformation:

```
WARNING case-1.nii.gz: background margin [0.0, 6.0, 0.0] mm, the deformation
can need up to [29.2, 29.1, 28.2] mm: the brain may be cut at the border.
```

**Which to use.** If the brain touches the border of the label map (e.g.
`changed_recon_0.2mm_level1_small/case-1.nii.gz`), use `--seg-dir`, even if
the map is already at the HR voxel: the margin keeps the deformation from
pushing parts of the brain out of the image. Then reuse the cached
templates in later datasets with `--hr-template-dir <dataset>/templates`.
With `--hr-template-dir`, the labels must match `--tree` and the label-space
fields of the config; this is not checked.

**Example**, `case-1` (467 x 923 x 589 at 0.2 mm), `generator_config2.json`
(0.4 mm, `factor` 2):

| | HR template | Output images |
|---|---|---|
| `--seg-dir`, `template_margin: 16` | 598 x 998 x 698 | 299 x 499 x 349 |
| `--hr-template-dir` (raw map, padded by 1) | 468 x 924 x 590 | 234 x 462 x 295 |

---

## 6. Generator config (`--config`)

A JSON file with the fields of `TreeGeneratorConfig`
(`src/tree_generator.py`). Unknown keys raise an error; missing keys take
the default. Ranges `[lo, hi]` are sampled uniformly once per image. A copy
of the config used is saved in the dataset (`generator_config.json`).

### Full example (every field)

```json
{
  "target_voxel": 0.6,
  "factor": 3,
  "background": 0,
  "template_sigma_mm": 0.0,
  "template_margin": 16,

  "output_labels": "templates",
  "unassigned": "background",

  "morpho_pairs": [
    ["background", "cerebral_gm"],
    ["background", "cerebellar_gm"],
    ["cerebral_gm", "cerebral_wm"],
    ["cerebellar_gm", "cerebellar_wm"],
    ["cerebral_wm", "csf_ventricles"],
    ["cerebral_wm", "deep_gm"]
  ],
  "morpho_iter": [1, 4],
  "morpho_prob": 0.5,

  "rotations": 10.0,
  "zooms": 0.1,
  "shears": 0.0,
  "translations_mm": 0.0,
  "elastic_dmax_mm": 7.5,
  "elastic_ctrl": 7,

  "colors_mu": [0.0, 255.0],
  "colors_sigma": [0.255, 2.55],
  "min_sigma": 0.0,
  "intensity_scale": 255.0,
  "tree_resolution": "acquisition",
  "resolution": [0.75, 3.0],
  "contrasts": null,

  "bias_prob": 1.0,
  "bias_vmax": 0.5,
  "bias_ctrl": 5,
  "motion_prob": 0.3,
  "motion_shots": 3,
  "motion_rotations": 10.0,
  "motion_translations_mm": 10.0,
  "noise_sigma": [0.01, 0.1],
  "field_voxel_mm": null,

  "lowres_prob": 0.95,
  "lowres_iso_prob": 0.1,
  "lowres_max_iso": 4.0,
  "lowres_max_aniso": 4.0,
  "lowres_alpha": [0.970873786407767, 1.03],
  "lowres_random_thickness": true,
  "noise_space": "lowres"
}
```

### 6.1 Grid

| Field | Default | Description |
|---|---|---|
| `target_voxel` | `0.75` | Voxel size (mm) of the output images and labels. |
| `factor` | `3` | HR voxel = `target_voxel / factor`. Each output voxel is the partial volume of `factor`^3 HR voxels (`1` = no partial volume). |
| `background` | `0` | Raw label value of the background. |
| `template_sigma_mm` | `0.5` | Gaussian smoothing (mm) of the labels when building the HR template (`0` = none). |
| `template_margin` | `16` | Background margin around the brain when building the HR template (section 6.2). |

Choosing the grid for 0.2 mm label maps (HR voxel = 0.2 mm):

| Output images | `target_voxel` | `factor` | `field_voxel_mm` | Config |
|---|---|---|---|---|
| 0.6 mm | `0.6` | `3` | `null` | `generator_config.json` |
| 0.4 mm | `0.4` | `2` | `0.8` or `null` | `generator_config2.json` |
| 0.2 mm | `0.2` | `1` | `0.6` (required, memory) | |

These fields, `output_labels` and `unassigned` define the HR templates.
With `--seg-dir`, changing them rebuilds the cached templates; with
`--hr-template-dir`, they must match how the templates were built.

### 6.2 Template margin (`template_margin`)

Background kept on each side of the brain when cropping the label map
(`--seg-dir` only), so that the deformation does not push the brain out of
the image.

| Value | Meaning |
|---|---|
| `16` | 16 mm on every axis (default). |
| `[16, 10, 12]` | mm per axis (x, y, z). |
| `{"mm": 16}`, `{"mm": [16, 10, 12]}` | Same, with the unit explicit. |
| `{"vox": 80}`, `{"vox": [80, 50, 60]}` | Voxels of the input label map (80 voxels = 16 mm at 0.2 mm). |
| `"auto"` | Worst case of the deformation: the brain is never cut. |
| `{"auto": 0.99}` | Enough for 99% of the deformations (any fraction in (0, 1]). |

The margin used is printed per template
(`case-1.nii.gz: margin [15.6, 17.6, 17.6] mm ([78, 88, 88] voxels)`).

- **`"auto"`** (`auto_margin_mm`): exact upper bound. The generator samples
  the labels at `A x + t + e`, so a brain point `p` appears at
  `A^-1 (p - t - e)`. The extent of the brain is maximised over its convex
  hull, over the extreme zooms and shears and a grid of 5 rotations per
  axis; the elastic displacement is at most `elastic_dmax_mm` per component
  (cubic B-spline of control points drawn in `[-dmax, dmax]`). Added: the
  translation, the morphology growth (`morpho_iter[1]` HR voxels) and about
  one voxel of slack. A few seconds.
- **`{"auto": q}`** (`sampled_margin_mm`): draws 300 deformations with the
  generator's own transform on a 2 mm grid and takes the `q`-quantile of the
  margin each one needs, plus the same slack. ~10 s per template on a GPU
  (minutes on CPU). Images whose deformation needs more lose a few mm of
  brain at the border, in the image and labels alike (consistent pairs).

`"auto"` assumes every maximum at once, which practically never happens.
For `case-1` with `generator_config2.json`:

| `template_margin` | Margin x, y, z |
|---|---|
| `"auto"` | 29.2, 29.1, 28.2 mm |
| `{"auto": 0.99}` | 15.6, 17.6, 17.6 mm |
| `{"auto": 0.9}` | 13.6, 13.6, 13.6 mm |
| `{"auto": 0.5}` | 7.6, 5.6, 7.6 mm |
| largest of 300 draws | 19.6, 19.6, 17.6 mm |

A larger margin costs memory and time in proportion to the number of voxels
(16 mm instead of none on `case-1`: 1.6x).

### 6.3 Label space

| Field | Default | Description |
|---|---|---|
| `output_labels` | `"templates"` | `"templates"`: output labels = labels present in the templates; `"tree"`: every label of the tree (some may never appear). |
| `unassigned` | `"error"` | Template labels missing from the tree: `"error"` (raise) or `"background"` (map to background). |

Output labels are indices `0..K-1` (0 = background), as nnU-Net requires;
`dataset.json` keeps the raw value of each index.

### 6.4 Shape augmentation: morphology (step 1)

| Field | Default | Description |
|---|---|---|
| `morpho_pairs` | `[]` | Pairs `[a, b]` of label groups. A group is a tree node name (all labels of its subtree), a raw label value, or a list of those. |
| `morpho_iter` | `[1, 4]` | Range of HR voxel layers grown. |
| `morpho_prob` | `0.5` | Probability of applying each pair. |

For each pair, with probability `morpho_prob`, one group (chosen at random)
grows into the other by `n ~ U{morpho_iter}` layers (26-neighbourhood);
the other group shrinks by the same amount. Pairs with `background` move
the brain surface. Logged per image in the spec (`morphology`).

### 6.5 Spatial deformation (step 2)

| Field | Default | Description |
|---|---|---|
| `rotations` | `10.0` | Max rotation (deg) about each axis, `U(-r, r)`. |
| `zooms` | `0.1` | Max zoom per axis, `1 + U(-z, z)`. |
| `shears` | `0.0` | Max shear. |
| `translations_mm` | `0.0` | Max translation (mm) per axis. |
| `elastic_dmax_mm` | `7.5` | Max elastic displacement (mm) per component. |
| `elastic_ctrl` | `7` | Max number of elastic control points per axis (drawn in `[2, elastic_ctrl]`; fewer = smoother). |
| `field_voxel_mm` | `null` | Voxel size (mm) of the grid the deformation (and bias) field is drawn on, then linearly upsampled (`null` = `target_voxel`). |

The affine is about the centre of the image (cornucopia
`RandomAffineElasticTransform`). Labels are resampled with nearest
neighbour, out-of-image = background. These fields set the margin needed
(section 6.2).

**`field_voxel_mm`.** A random field costs ~200 bytes of GPU memory per
voxel of its grid; on a 0.2 mm training grid that is >100 GB. Drawing it
coarser (e.g. 0.6 mm) changes nothing visible, as the fields are smooth.
It is rounded to a multiple of the HR voxel (deformation) and of
`target_voxel` (bias field); a multiple of both (e.g. `0.8` for 0.4 / 0.2
mm) avoids the rounding.

### 6.6 Tree intensities (step 4)

| Field | Default | Description |
|---|---|---|
| `colors_mu` | `[0, 255]` | Range of the principal color means `C` (one per tree color, e.g. GM, WM, CSF), tree units. |
| `colors_sigma` | `[0.255, 2.55]` | Range of the principal color stds `V`. |
| `min_sigma` | `0.0` | Lower bound of every label std. |
| `intensity_scale` | `255.0` | The image is divided by this before the artefacts (so noise and bias act on ~[0, 1]). |
| `tree_resolution` | `"acquisition"` | Resolution deciding node visibility: `"acquisition"` (coarsest spacing drawn at step 0; `target_voxel` if not degraded) or `"config"` (field `resolution`). |
| `resolution` | `null` | With `tree_resolution: "config"`: `null` (= `target_voxel`), a number, or a `[lo, hi]` range in mm. |
| `contrasts` | `null` | Contrasts for the per-contrast `r_max`, e.g. `["T1", "T2"]`; one drawn per image (`null` = `r_max` `"default"`). |

How the tree turns `C`, `V` into one Gaussian per label: section 7.

### 6.7 Artefacts (steps 5, 6, 8)

| Field | Default | Description |
|---|---|---|
| `bias_prob` | `1.0` | Probability of a bias field. |
| `bias_vmax` | `0.5` | Bias field values in `(1 - v, 1 + v)`. |
| `bias_ctrl` | `5` | Bias field control points per axis (smoothness). |
| `motion_prob` | `0.3` | Probability of motion artefacts. |
| `motion_shots` | `3` | Number of k-space shots (the head moves between shots). |
| `motion_rotations` | `10.0` | Max motion rotation (deg). |
| `motion_translations_mm` | `10.0` | Max motion translation (mm). |
| `noise_sigma` | `[0.01, 0.1]` | Range of the Gaussian noise std (image scaled by `intensity_scale`). |

Motion is simulated along a random axis; the shot holding the k-space
centre stays at the reference pose, so the image remains aligned with the
labels and the other shots produce ghosting/blurring.

### 6.8 Resolution augmentation (step 7, SynthSeg Sec. 3.1.4)

| Field | Default | Description |
|---|---|---|
| `lowres_prob` | `0.95` | Probability of simulating a lower acquisition resolution. |
| `lowres_iso_prob` | `0.1` | Probability that a degraded acquisition is isotropic. |
| `lowres_max_iso` | `4.0` | Max isotropic voxel size (mm). |
| `lowres_max_aniso` | `8.0` | Max slice spacing (mm) of anisotropic acquisitions. |
| `lowres_alpha` | `[1/1.03, 1.03]` | Range of alpha (deviation of the slice profile). |
| `lowres_random_thickness` | `true` | Slice thickness `~ U(target_voxel, spacing)` (slice gap); `false`: thickness = spacing. |
| `noise_space` | `"lowres"` | `"lowres"`: noise added on the low-resolution grid, before upsampling (as in a real scan); `"highres"`: on the training grid before the resolution step (SynthSeg). |

Per image: with probability `1 - lowres_prob` no degradation. Otherwise
isotropic (spacing `~ U(target_voxel, max_iso)` on all axes) or anisotropic
(one random axis, spacing `~ U(target_voxel, max_aniso)`). Per degraded
axis: thickness, alpha, and a Gaussian slice profile with
`sigma = 2 alpha log(10) / (2 pi) * thickness / target_voxel` voxels. The
image is blurred, downsampled to the spacing, (noised), and upsampled back
to the training grid. Values above `target_voxel` only.

### 6.9 Example configs

Clean images (no artefacts, no degradation):

```json
{ "target_voxel": 0.6, "factor": 3, "template_sigma_mm": 0.0,
  "output_labels": "templates", "unassigned": "background",
  "bias_prob": 0.0, "motion_prob": 0.0, "noise_sigma": [0.0, 0.0],
  "lowres_prob": 0.0 }
```

Contrast-dependent visibility with a fixed resolution range:

```json
{ "target_voxel": 0.6, "factor": 3, "unassigned": "background",
  "tree_resolution": "config", "resolution": [0.5, 2.0],
  "contrasts": ["T1", "T2", "FLAIR"] }
```

0.2 mm output (needs `field_voxel_mm`, ~47 GB of free GPU memory):

```json
{ "target_voxel": 0.2, "factor": 1, "template_sigma_mm": 0.0,
  "field_voxel_mm": 0.6, "unassigned": "background",
  "lowres_max_iso": 2.0, "lowres_max_aniso": 2.0 }
```

---

## 7. Label tree (`--tree`)

Each node of the tree sets the intensity of the labels it paints. Build it
with `build_l1_tree.py` or edit the JSON directly.

```json
{
  "version": "synthseg-0.1",
  "description": "NextBrain L1 segmentation tree",
  "colors": ["GM", "WM", "CSF"],
  "contrasts": ["T1", "T2", "FLAIR", "PD"],
  "root": "root",
  "nodes": {
    "root": {
      "parent": null, "children": ["background", "cerebral_gm"],
      "level": 0, "labels": [], "wr_mu": 0.0, "wr_sigma": 0.0, "r_max": null
    },
    "background": {
      "parent": "root", "children": [], "level": 1, "labels": [0],
      "wr_mu": 0.0, "wr_sigma": 0.0, "r_max": null, "null": true
    },
    "cerebral_gm": {
      "parent": "root", "children": ["hippocampus_ca1"], "level": 1, "labels": [5],
      "w_mu": [1.0, 0.0, 0.0],
      "w_sigma": [1.0, 0.0, 0.0],
      "wr_mu": 4.0, "wr_sigma": 0.5, "r_max": null,
      "description": "cortical grey matter"
    },
    "hippocampus_ca1": {
      "parent": "cerebral_gm", "children": [], "level": 2, "labels": [50, 51],
      "wr_mu": 2.0, "wr_sigma": 0.2,
      "r_max": {"default": 1.0, "T2": 1.5}
    }
  }
}
```

| Node field | Description |
|---|---|
| `parent` | Parent node name (`null` only for the root). |
| `children` | Child node names (must agree with the children's `parent`). |
| `level` | Root = 0; must equal parent level + 1. |
| `labels` | Label-map values painted with this node's Gaussian. |
| `w_mu` | **Level 1 only.** Weights of the principal colors (same order as `colors`, sum = 1). |
| `w_sigma` | **Level 1 only.** Weights of the principal sigmas (default = `w_mu`). |
| `wr_mu` | Weight of the random component of the mean. |
| `wr_sigma` | Weight of the random component of the sigma. |
| `r_max` | Coarsest voxel size (mm) at which the node is visible: `null` (always), a number, or a per-contrast dict `{"default": x, "T1": y, ...}`. Ignored on level 1. |
| `null` | Null color (mean = sigma = 0), inherited by descendants (e.g. background). |
| `description` | Free text. |

Sampling, for an image at resolution `r` and contrast `c` (step 0b):

```
level 1   : mu = sum_i w_mu[i]    * C[i] + wr_mu    * N(0,1)
            sd = sum_i w_sigma[i] * V[i] + wr_sigma * N(0,1)
level >= 2: visible   -> mu = mu_parent + wr_mu * N(0,1), sd = sd_parent + wr_sigma * N(0,1)
            otherwise -> (mu_parent, sd_parent)
sd = max(sd, min_sigma)
```

`C[i]` / `V[i]` are drawn per image from `colors_mu` / `colors_sigma`. A
node is visible iff `r <= r_max(c)` and its parent is visible. Two random
numbers are drawn for every node even when it collapses, so with a fixed
seed the visible nodes get the same values at any resolution.

---

## 8. Outputs

```
<out>/
  Dataset<ID>_<NAME>/
    templates/<case>_hr<voxel>mm.nii.gz      HR templates (--seg-dir cache, or
                                             --save-hr-template-dir)
    templates/<case>_hr<voxel>mm.json        how each one was built
    imagesTr/<prefix>_XXXX_0000.nii.gz       synthetic image, float32 [0, 1]
    labelsTr/<prefix>_XXXX.nii.gz            labels, indices 0..K-1 (uint16)
    specs/<prefix>_XXXX.json                 every parameter of that image
    dataset.json                             nnU-Net v2 dataset description
    generator_config.json, tree.json         copies of the inputs used
```

Image and labels share the affine of the training grid (the template's
world coordinates).

**`dataset.json`**: `channel_names`, `labels` (name -> index),
`numTraining` (cases present in `labelsTr`), `file_ending`, and
`original_label_values` (raw label value of each index).

**`specs/<case>.json`**:

| Key | Content |
|---|---|
| `case`, `image`, `label` | Case name and file paths (relative to the dataset). |
| `run_seed`, `image_number`, `seed` | `--seed`, image index, and the image's own seed. |
| `template`, `template_index` | Path of the label map / HR template used, and its index. |
| `shape`, `voxel_mm`, `affine` | Output grid. |
| `config` | Config file name (`generator_config.json`). |
| `contrast`, `tree_resolution_mm` | Contrast and resolution used for node visibility. |
| `colors`, `colors_mu`, `colors_sigma` | Principal color names and drawn `C`, `V`. |
| `acquisition` | `mode` (`none` / `iso` / `aniso`), `axis`, `spacing_mm`, `thickness_mm`, `alpha`, `sigma_vox`. |
| `morphology` | List of `{grow, shrink, n_iter}` applied. |
| `bias_field`, `motion` | Whether they were applied. |
| `noise_sigma`, `noise_space` | Noise std drawn and where it was added. |
| `labels` | Per index: raw `value`, `name`, `mu`, `sigma`, `generation_class`. |
| `generation_classes` | Tree node of each generation class (labels with the same class share a Gaussian in this image). |

---

## 9. Reproducibility

- Image `i` uses its own seed from `(--seed, i)`, stored in its spec. The
  same command with `--start i -n 1 --overwrite` regenerates it.
- The segmentation of a regenerated image is identical. The intensities can
  differ slightly (up to ~0.1 of the [0, 1] range at a few voxels) because
  some GPU operations are not deterministic.
- Keep the exact templates with the dataset: `--seg-dir` always caches
  them; with `--hr-template-dir`, add `--save-hr-template-dir`.
- The dataset keeps copies of the config and tree used; each spec records
  the template path.

---

## 10. GPU memory and speed

Peak GPU memory of one image on the largest template (`case-3`,
667 x 1024 x 869 at 0.2 mm), `target_voxel` 0.2, `factor` 1,
`field_voxel_mm` 0.6:

| Step | Peak |
|---|---|
| morphology | 11 GB |
| deformation | 7 GB (~125 GB estimated without `field_voxel_mm`) |
| bias field | 9 GB (108 GB without `field_voxel_mm`) |
| motion | 47 GB |
| resolution + noise | 16 GB |

Memory and time grow with the number of output voxels: 0.4 mm has 8x fewer
voxels than 0.2 mm, 0.6 mm 27x fewer. Typical times: ~5 s per image at
0.4 mm on `case-1`, 30-90 s at 0.2 mm on the full templates.

The GPUs are shared: check free memory with `nvidia-smi` and choose
`--device`. If you run out of memory: set `field_voxel_mm`, use a coarser
`target_voxel`, a smaller `template_margin`, or `motion_prob: 0`.
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` does not help (the
memory is really used, not fragmented).

---

## 11. Troubleshooting

| Error / symptom | Cause | Fix |
|---|---|---|
| `No module named 'interpol'` | Environment without `torch-interpol` (e.g. `base`). | `conda activate nxb`. |
| `CUDA out of memory` | Too many voxels for the free GPU memory (section 10). | `field_voxel_mm`, coarser grid, another GPU. |
| `voxel [...] mm, but the config expects HR templates at ... mm` | `--hr-template-dir` files not at `target_voxel / factor`. | Fix `target_voxel` / `factor`, or use `--seg-dir`. |
| `shape '[...]' is invalid for input of size ...` | HR template shape not a multiple of `factor` (templates not built by this code). | Use `--seg-dir` or `--hr-template-dir` (both pad). |
| `Template labels [...] are not in the tree` | Labels missing from the tree with `unassigned: "error"`. | Add them to the tree or set `unassigned: "background"`. |
| `Unknown config fields: [...]` | Typo or field from another version. | Check the names in section 6. |
| `WARNING ...: background margin ...` | `--hr-template-dir` map with little background around the brain. | Use `--seg-dir` with a `template_margin`. |
| `template settings changed [...], rebuilding` | Config changed since the template was cached. | Expected; earlier images of that dataset used the old template (use a new dataset ID to keep them separate). |
| `FileNotFoundError` on `--config` | Config renamed or deleted. | Check `data/inputs/`. |
