# Modular generator: structure proposal

Status: **implemented** in `nxgen/` (see README.md and docs/). The proposal
below is kept as the design record. Section "Implementation decisions"
lists where the code differs from it, after the requirements "as simple as
possible", "fast enough for training", "tests apart" and "well documented".

## Implementation decisions (iteration 3)

| Proposal | Implementation | Why |
|---|---|---|
| Module classes with `Params` dataclasses | One file per module with plain functions `sample(...) -> dict` and `apply(...)`; params are plain JSON dicts | Simpler; plans serialise as-is |
| `required_input(roi)` in every module | Only `spatial.source_box` + `morphology.halo`; the pipeline crops the template once | The only steps that need more input than their output region |
| Configurable artefact order | Fixed order bias, motion, resolution, noise (as v2) | Simpler; no use case yet |
| `template.margin` with `auto` / quantile | Fixed `margin_mm` only | v2's automatic margins were complex and not exact (problems_to_solve #6-8); outside = background anyway |
| Cornucopia for deterministic primitives (Q1) | Not used at all. Own B-spline fields, rigid motion, Gaussian blur | Full parameter control, no dependency on its internal layout. Resolution step matches v2 to < 0.02 |
| Elastic and bias on a coarse grid (`field_voxel_mm`) | Separable B-spline evaluated directly at the needed voxels | Exact on patches, small memory, no extra parameter |
| PV + synthesis fused for speed | Separate modules, sparse top-k PV, histogram without K-sized arrays | Already fast enough (0.18 s full volume at 0.4 mm) |
| Geometry reuse (§9.3), `torch.compile` (§9.4) | Not implemented | Not needed at current speed (0.07 s per 160³ patch) |
| Tree noise patch-consistent | Drawn per patch (seeded); patch == crop only with sigma 0 | A counter-based noise would add complexity |
| Producer process (§9.5) | `SyntheticStream` (in-process) only | Simplest; a producer can wrap it later |
| `checks/` package | `nxgen/checks.py`, used by the tests only | Tests are separate from the generator |
| Adjacency rule "CSF must not touch WM" | `checks.adjacency_violations(rules)`, not enforced | In NextBrain, ventricular CSF does touch WM (45 694 faces on `case-1`): the rule must target external CSF only |

Answers used for the open questions: Q1 (no cornucopia), Q2 (10 GB met:
7 GB full volume at 0.4 mm, 2.4 GB per 160³ patch), Q3 (plan up front),
Q4 (online, patch mode, in-process; nnU-Net trainer still to write), Q5
(one template per HR voxel, padded to the factor at load time), Q6
(Gaussian window, seg unchanged), Q7 (explicit ranges, converter table in
docs/config.md), Q8 (`nxgen`), Q9 (motion approximated on the patch), Q10
(no reuse).

---

Original proposal (iteration 2): Based on `generator_structure.txt` (the initial
proposal), the code of `../generator_v2/` (`tree_generator.py`,
`tree_sampling.py`), and the known flaws listed in
`../../testing/generating_merged_seg_images/problems_to_solve.md`.

Names (package, modules, config keys) are placeholders. The package is called
`nxgen` below.

---

## 0. Summary of changes relative to the initial proposal

The six modules of the initial proposal are kept. The main changes are:

1. **Every random module is split into `sample()` and `apply()`.**
   `sample(rng, ctx) -> Params` draws every random number.
   `apply(x, params) -> y` is deterministic. This one rule covers several of
   the requirements: "return the parameters used", "set specific values for
   predictable results", per-module unit tests, and reproducibility.
2. **A new step 0, `acquisition`, samples the scan before anything is
   rendered.** In v2 the simulated resolution decides which tree nodes are
   visible (step 4) *and* how the image is degraded (step 5). The initial
   proposal leaves this coupling implicit. Here it is a separate module, and
   both consumers read its output.
3. **Two-phase pipeline: sample a `Plan`, then render it.** All the
   parameters of one image (template, acquisition, morphology, deformation,
   tree Gaussians, bias, motion, noise, ...) are drawn first, into one
   JSON-serialisable `Plan`. Rendering is then deterministic. A saved plan
   replays an image exactly, and a config can pin any part of it.
4. **Independent random streams per module.** v2 draws everything from the
   global `random` / `numpy` / `torch` state. Turning one step off (e.g.
   morphology) therefore changes every draw after it, which makes A/B
   debugging hard. Here each module gets its own generator, derived from
   `(image_seed, module_name)`.
5. **Sparse partial volumes.** With NextBrain (~333 labels), dense PV maps do
   not fit the 10 GB budget (estimate in section 4.3). Module 3 outputs the
   top-k labels per voxel instead. Both PV methods of the initial proposal
   (downsampling and boundary smoothing) are then the same operation, a
   weighted neighbourhood label histogram, with different windows.
6. **Shared core objects:** `LabelSpace` (raw value <-> index, label groups,
   the `unassigned` policy), `Grid` (shape + affine + voxel size), config
   loading/validation, RNG, I/O, and a `Recorder` for intermediate outputs.
   In v2 these concerns are spread over `prepare_template`,
   `TreeGenerator.__init__` and `TreeSampling`.
7. **Module 4 also outputs the generation classes** (labels that share a
   Gaussian at this resolution). WP4 needs them for the loss, and the initial
   proposal does not list them as an output.
8. **Fast enough to generate images during training** (§9). Templates stay
   on the GPU. Generation can be restricted to the training patch: the
   deformation is a pull-back, so only the part of the HR template that the
   patch needs is processed. One deformed label map can also be reused for
   several intensity draws.

---

## 1. What v2 does well (to keep)

- Memory-aware implementation: per-label bounding boxes in the
  super-resolution, slab-wise PV histogram, deformation and bias fields drawn
  on a coarse grid (`field_voxel_mm`), integer index maps (int16).
- The acquisition is sampled separately from the transform that applies it
  (`SynthSegLowResTransform.sample` / `make_final`). This is the pattern to
  generalise to every module.
- Template build records, uint16 label I/O, explicit voxel-size checks.
- The segmentation target is never degraded.

## 2. Limitations of v2 that drive this design

| # | Limitation in v2 | Consequence | Addressed by |
|---|---|---|---|
| L1 | Steps are private methods of `TreeGenerator` that read `self.cfg` and the global RNG. | A step cannot be run on its own with chosen inputs/params. | `sample`/`apply` modules (§3) |
| L2 | Most drawn parameters are not returned: affine matrix, elastic field, bias field and motion parameters are cornucopia internals (only "applied: yes/no" is logged). | An image cannot be fully explained or replayed. | `Params` dataclasses, `Plan` (§3, §5) |
| L3 | One global random stream; `TreeSampling` has its own RNG, which `generate_images.py` reseeds by hand. | Switching off a step changes all later draws. Reproducibility depends on call order. | Per-module RNG (§6) |
| L4 | PV and intensity synthesis are fused in `_synthesize`. | PV cannot be inspected or tested on its own. | Separate module 3 with sparse output (§4.3) |
| L5 | Flat config with ~50 fields. Some values are not validated: e.g. `nextbrain_configs.json` uses `"unassigned": "background_l1"`, which is neither `"error"` nor `"background"`. It silently skips the template restriction and maps labels to background at generation time. | Typos and invalid values change behaviour without an error. | Sectioned config with per-section schemas (§7) |
| L6 | Template building depends on `target_voxel` / `factor` / the deformation config. | Offline and online settings are entangled. Changing the output resolution can rebuild templates. | The offline module is parametrised by the HR voxel only (§4.1) |
| L7 | The label index depends on the templates found in the current run (`output_labels: "templates"`). | Index meaning can change between runs of the same dataset (problems_to_solve #2). | `LabelSpace` frozen and saved per dataset (§4.0) |
| L8 | `return_steps` copies every step to CPU memory. | Inspection is all-or-nothing and memory-heavy. | `Recorder` (§8) |
| L9 | Motion peaks at 47 GB at 0.2 mm. Morphology 11 GB. | Not compatible with the 10 GB target at 0.2 mm. | Open question Q2 |

---

## 3. Module contract

Every online module follows the same pattern:

```python
@dataclass
class MorphologyParams:           # JSON-serialisable (to_dict / from_dict)
    ops: list[dict]               # [{grow: [...], shrink: [...], n_iter: 2}, ...]

class Morphology:
    name = 'morphology'           # also the config section and RNG stream name

    def __init__(self, cfg: MorphologyConfig, labels: LabelSpace): ...

    def sample(self, rng: np.random.Generator, ctx: Context) -> MorphologyParams:
        """Draw every random choice. Uses no tensor data, only the context
        (shapes, voxel sizes, label space, acquisition)."""

    def apply(self, x: Volume, params: MorphologyParams) -> Volume:
        """Deterministic. Same inputs + params -> same output."""

    def __call__(self, x, rng=None, params=None, return_params=False):
        """Convenience: sample (unless params are given) then apply."""
```

Rules:

- `sample` never touches voxel data. It only needs a `Context` (template
  grid, target grid, label space, acquisition...). The whole `Plan` can
  therefore be drawn before rendering, and it can be checked cheaply.
- For voxel-wise random fields (Gaussian noise, tree noise ε), `Params`
  stores a **seed**, not the field. For smooth fields (elastic, bias), it
  stores the **control points** (e.g. 7×7×7×3 floats: small).
- `apply` uses torch, runs on any device, and does not use any global RNG.
  Where it needs randomness (seeded noise), it builds a local
  `torch.Generator` from the seed in `Params`.
- Modules exchange `Volume` objects (tensor + `Grid`), so an intermediate
  can always be saved with the right affine (HR grid, target grid, low-res
  grid).
- Cornucopia is used only for **deterministic primitives** (e.g. `smoothnd`,
  `apply_flow`, k-space motion given the motion parameters), never for
  sampling. This removes the cornucopia RNG and its internal parameter
  layout from the critical path (problems_to_solve #13).
- `apply` accepts an optional **region of interest** (`roi`: a box on the
  output grid). The module computes only that region and asks for the input
  region it needs (`required_input(roi, params) -> roi_in`, including any
  halo). This is what makes patch generation possible (§9.2).
- `apply` must not force GPU -> CPU syncs (`.item()`, `nonzero`,
  `torch.unique`, `.cpu()`) in the hot path. Anything data-dependent that can
  be precomputed per template (label bounding boxes, labels present, LUTs) is
  computed once when the template is loaded.

---

## 4. Modules

### 4.0 Core (not a pipeline step)

| File | Content |
|---|---|
| `core/grid.py` | `Grid(shape, affine, voxel_mm)`, `Volume(data, grid)`; helpers for pooling (HR -> target) and coarse field grids. |
| `core/label_space.py` | `LabelSpace`: raw value <-> index LUT (index 0 = background), names, `resolve(group)` (tree node / raw value / list -> indices), `unassigned` policy (validated: `"error"` or `"background"`). Built once from tree + templates, **saved with the dataset** and reloaded on later runs (L7). |
| `core/tree.py` | `LabelTree` (moved from `tree_sampling.py` mostly unchanged: it is already clean and tested). |
| `core/config.py` | Load JSON -> one dataclass per section, unknown keys raise, value checks, distribution specs (§7). |
| `core/rng.py` | `image_seed(run_seed, i)`, `module_rng(image_seed, name)` via `np.random.SeedSequence`. |
| `core/io.py` | NIfTI read/write (uint16 labels, float32 images), JSON params, sparse PV save. |
| `core/recorder.py` | Intermediate outputs (§8). |
| `checks/` | Reusable validators used both in tests and optionally at run time: label set preserved, per-label volume change, label adjacency rules (e.g. WM must not touch CSF), Gaussian fit per label. |

### 4.1 Template super-resolution (offline)

Objective: input label map -> HR template on a cropped grid.

Steps (same as v2 `prepare_template`, split into functions):
1. `restrict_labels` (per `unassigned`).
2. `crop_to_foreground(margin)`. The margin is in mm, voxels, or
   computed by `margin.py` (`auto` / quantile, which reads the `spatial`
   section of the same config).
3. `super_resolve(seg, hr_voxel_mm, sigma_mm)`: one-hot -> sinc upsampling
   -> Gaussian -> argmax, per-label bounding boxes.
4. Pad the shape to a multiple of the allowed pooling factors (see Q5).

Changes:
- **Parameter: `hr_voxel_mm`, not `target_voxel / factor`** (L6). The online
  generator derives `factor = target_voxel / hr_voxel` and errors if it is
  not an integer.
- The build record also stores a hash of the source file (problems_to_solve
  #3, #4), and the HR template file name contains that hash, so each image
  spec can point to the exact template version.
- Returns `(Volume, TemplateParams)`. `TemplateParams` records the crop box,
  margin in mm/voxels, padding, sigma, source hash, and label volumes
  before/after.

Tests:
- No new labels (as proposed).
- **Vanished / shrunk labels.** Sinc + argmax can delete thin or tiny
  structures, which matters for small NextBrain nuclei. Report the volume
  ratio per label and fail if a label disappears.
- Identity: input already at `hr_voxel`, `sigma=0` -> unchanged labels.
- Equivalence with v2 `super_resolve_labels` (bit-exact on a small map).
- Time and peak memory (CPU and GPU).

Future (as proposed): morphological consistency fix-up, using the adjacency
rules from `checks/`.

### 4.2 Morphology (online, HR grid)

Two modules, as proposed:

**4.2a `morphology`**: erosion/dilation between label groups (v2
`_morphology`).
- Params: list of `{grow, shrink, n_iter}`.
- Config: `pairs`, `n_iter` (distribution), `prob`.

**4.2b `spatial`**: affine + elastic.
- Params: the affine matrix (4×4, in mm, world convention documented) and the
  elastic control points (+ their grid shape). The affine is built by our
  own code (not cornucopia's `affine_linear` layout), so `margin.py` and the
  generator share the same function.
- `apply`: control points -> cubic B-spline field on the coarse field grid
  -> trilinear upsampling slab by slab -> nearest resampling (v2 `_deform`).

Tests:
- `n_iter=0` / identity affine and zero elastic -> output == input.
- Label set preserved (no new labels), volume change bounded by zoom.
- Adjacency rules (proposed "CSF must not touch WM"): implemented once in
  `checks/adjacency.py`, configured with
  `"forbidden_adjacency": [["cerebral_wm", "csf"]]`. Reported, not enforced,
  in this version.
- Field of view: report the fraction of foreground pushed out of the grid
  (ties to the margin computation).

### 4.3 Partial volume (online, HR -> target grid)

Both methods of the initial proposal are the same operation: for each output
voxel, a weighted histogram of the HR labels in a window.

| Method | Window | Stride | Hard segmentation |
|---|---|---|---|
| `downsample` (HR finer than target, the paper's approach) | box `factor³` | `factor` | majority label |
| `smooth` (HR ≈ target, `factor = 1`) | Gaussian, `sigma_mm` (truncated) | 1 | original labels (unchanged) |

**Output representation: sparse top-k.** Dense PV maps are too big. Example:
`case-1` at 0.4 mm = 239×439×289 ≈ 30 M voxels; with ~333 labels, a dense
float32 PV map is ~40 GB. Instead:

```
PartialVolume:
    index : (k, X, Y, Z) int16    label index, sorted by weight
    weight: (k, X, Y, Z) float16  fractions, sum to 1 (renormalised)
    seg   : (X, Y, Z) int16       hard segmentation
    lost  : (X, Y, Z) float16     mass dropped by the top-k truncation (diagnostic)
```

With `k = 8` this is ~1 GB at 0.4 mm. `k = min(factor³, 8)` is exact for
`factor ≤ 2`. Hard labels are the special case `k = 1`, so module 4 has
only one input type.

The computation is the v2 slab-wise `scatter_add` histogram, followed by
`topk` per slab, so memory stays bounded by the slab size.

Tests:
- `factor=1, method=downsample` -> `k=1`, weight 1, seg == input.
- Mass conservation: sum of PV over the image == HR label volume /
  `factor³`, per label (exact for `downsample`).
- Equivalence of `seg` with v2 `_synthesize` seg.
- Histogram of `lost` (how much top-k truncation costs).

Open: the `smooth` method needs one parameter (`sigma_mm`), and there is no
paper prior for it. See Q6.

### 4.4 Image synthesis from the tree (online, target grid)

Split into two parts:

1. **`tree_intensities.sample`** (no voxels): principal colours `C`, `V`,
   then per-node `(mu, sigma)` via `LabelTree` visibility at the
   acquisition's resolution and contrast (v2 `TreeSampling.sample_node_stats`,
   rewritten to take an explicit `rng`).
   Params: `C`, `V`, `resolution_mm`, `contrast`, per-label `mu`, `sigma`,
   `generation_class` per label, class names, and the `noise_seed` used for
   the voxel noise.
2. **`synthesis.apply(pv, params)`**:
   `img = Σ_classes pv_class · (mu_class + sigma_class · ε_class)`, with one ε
   field per *generation class* as in v2. Output on the target grid, in tree
   units, then divided by `intensity_scale`.

Outputs: image `Volume`, plus the params above. The `generation_class`
mapping is what WP4 needs for the loss.

Tests (refining the initial proposal):
- **Gaussian fit on pure voxels only** (PV weight == 1). There, the intensity
  must follow `N(mu, sigma)` exactly (KS test). Mixed voxels are expected to
  deviate, so they are reported separately.
- Collapsed nodes share exactly the parent's `(mu, sigma)` (as proposed), and
  their voxels share the same ε field.
- With a fixed seed, visible nodes get the same values at any resolution
  (already a property of v2's sampling, keep it as a test).
- `null` nodes -> 0.

### 4.5 Intensity augmentations (online, target grid)

Four independent modules, as proposed, plus their order:

| Module | Params | Note |
|---|---|---|
| `bias` | control points (`bias_ctrl³`), coarse grid factor | multiplicative, exp or linear (decide) |
| `motion` | axis, shots, per-shot rotation/translation, k-space pattern, centre shot kept at reference | v2 workaround for the centre shot kept |
| `resolution` | the acquisition (from step 0): spacing, thickness, alpha (mode and sigma derived; None = not degraded) | v2 `LowResAcquisitionFinalTransform`; also outputs the low-res grid image |
| `noise` | `sigma`, `seed`, `space` (`lowres` / `highres`) | with `lowres`, called *inside* `resolution` on the low-res grid |

The order is a config list (`"order": ["bias", "motion", "resolution",
"noise"]`, v2 default), so effects can be swapped or removed without code
changes. The pipeline enforces the one constraint that exists:
`noise.space = "lowres"` requires `resolution` to be present.

Tests: each with params set to identity (bias ≡ 1, no motion, spacing =
target, sigma 0) returns its input. `resolution` single-axis ==
`cc.LowResSliceFinalTransform` (existing v2 test). `noise` sigma estimated
from the output matches the param.

### 4.6 Normalisation

`minmax` (v2) or `percentile` (clip to [p_lo, p_hi] then rescale: more robust
to noise outliers). Params: the values used for the rescaling.

In patch mode (§9.2), the statistics of a patch differ from those of the full
image. So the rescaling values are either computed from the plan (expected
intensity range from the tree Gaussians, bias and noise), or the
normalisation is left to the training pipeline (e.g. z-score per patch, as
nnU-Net does).

---

## 5. Data flow

```
                      ┌────────────── Plan (sampled first, JSON) ───────────────┐
 config.json ──►      │ template_id, acquisition, morphology, spatial,          │
 image_seed  ──►      │ tree_intensities, bias, motion, resolution, noise, norm │
                      └─────────────────────────────────────────────────────────┘
                                         │ (deterministic rendering)
 HR template ──► morphology ──► spatial ──► partial_volume ──► synthesis ──► [bias ─ motion ─ resolution(+noise) ─ noise]* ──► normalise ──► image
   (HR grid)       (HR)          (HR)        (HR → target)      (target)            (* order from config)                              (target)
                                                  │
                                                  └──► seg (target grid, never degraded)
 acquisition ──► tree_intensities (visibility) and resolution (degradation)
```

`Plan` = `{module_name: Params}` + `image_seed`, `template` (path + hash),
config hash, and code version. It is saved as `specs/<case>.json`. It replaces
the v2 spec, and it can be fed back in.

---

## 6. Randomness and reproducibility

- `image_seed = SeedSequence([run_seed, i])` (as v2).
- `rng_module = SeedSequence([image_seed, crc32(module_name)])`: each module
  has its own stream. Disabling or changing one module does not change the
  others' draws (L3).
- Voxel-wise noise uses `torch.Generator(device).manual_seed(params.seed)`.
- Pinning, by priority (later wins):
  1. defaults < `config.json` < CLI overrides (as v2);
  2. a full or partial plan: `--params plan.json` or
     `generate(plan={'resolution': {...}})`. The given modules skip
     `sample`.
- Known limit (as in v2): some CUDA ops that use float atomics are not
  bit-deterministic. The segmentation is exact, the intensities can differ
  slightly. Offer a `--deterministic` flag
  (`torch.use_deterministic_algorithms`) for debugging.

---

## 7. Config file

One JSON file shared by the offline and online launchers, with one section per
module. Each section maps to a dataclass. Unknown keys raise. Enum values are
validated.

**Distribution specs** (for any random parameter):

| JSON | Meaning |
|---|---|
| `5.0` | fixed value |
| `[lo, hi]` | uniform |
| `{"choice": [a, b, c]}` | uniform choice |
| `{"uniform": [lo, hi]}` | explicit form (needed when the value itself is a list) |

Note: v2 uses "max rotation 10 -> U(-10, 10)". Here a range is explicit
(`[-10, 10]`), so that `5` can mean "exactly 5°". This changes the meaning of
existing configs. A converter from v2 configs is cheap.

```json
{
  "grid":      {"hr_voxel_mm": 0.2, "target_voxel_mm": 0.4, "field_voxel_mm": 0.8},
  "labels":    {"tree": "tree.json", "background": 0, "unassigned": "background",
                "output_labels": "templates"},
  "template":  {"sigma_mm": 0.1, "margin": {"auto": 0.99}},

  "acquisition": {"lowres_prob": 0.5, "iso_prob": 0.2, "max_iso_mm": 1.0,
                  "max_aniso_mm": 1.0, "alpha": [0.9709, 1.03],
                  "random_thickness": true, "contrasts": null,
                  "tree_resolution": "acquisition"},
  "morphology":  {"enabled": true, "pairs": [["cerebral_gm", "cerebral_wm"]],
                  "n_iter": {"choice": [1, 2, 3]}, "prob": 0.5},
  "spatial":     {"rotation_deg": [-5, 5], "zoom": [0.95, 1.05], "shear": 0,
                  "translation_mm": 0, "elastic_dmax_mm": 7.5, "elastic_ctrl": [2, 7]},
  "partial_volume": {"method": "downsample", "top_k": 8},
  "synthesis":   {"colors_mu": [0, 255], "colors_sigma": [0.255, 2.55],
                  "min_sigma": 0, "intensity_scale": 255},
  "artifacts": {
    "order": ["bias", "motion", "resolution", "noise"],
    "bias":   {"prob": 1.0, "vmax": 0.5, "ctrl": 5},
    "motion": {"prob": 0.3, "shots": 3, "rotation_deg": [-10, 10], "translation_mm": [-2, 2]},
    "noise":  {"sigma": [0.01, 0.07], "space": "lowres"}
  },
  "normalization": {"method": "minmax"},
  "checks": {"forbidden_adjacency": [["cerebral_wm", "csf_ventricles"]]}
}
```

---

## 8. Intermediate outputs (`Recorder`)

The pipeline calls `recorder.record(step, volume, params)` after each module.

| Recorder | Behaviour |
|---|---|
| `NullRecorder` (default) | nothing, zero cost |
| `MemoryRecorder(steps=[...], device='cpu')` | keeps selected steps (notebooks, figures) |
| `DiskRecorder(dir, steps=[...])` | writes `<case>/<nn>_<step>.nii.gz` with the correct grid + `params.json` |

Recorded PV is saved as viewable summaries: `pv_max` (largest fraction) and
`pv_n` (number of labels in the voxel), plus the optional raw sparse arrays.

Every recorder also records the **time and peak GPU memory of each module**
(CUDA events, `torch.cuda.max_memory_allocated`). This gives the profile
needed for §9 for free.

---

## 9. Speed: generation during training

Goal: produce a training sample faster than the network consumes one, so the
GPU never waits for data. v2 takes ~5 s per full image at 0.4 mm (`case-1`),
which is far too slow for that. Target: TBD with the training setup (Q4); the
order of magnitude to aim for is ≤ 0.5 s per training patch.

### 9.1 Everything stays on the GPU

- HR templates are loaded on the GPU once, as int16 label indices. That is
  ~0.5 GB per template at 0.2 mm (`case-1`, 478×878×578), so several
  templates fit. At 0.1 mm a template is 8× larger (see Q5).
- Per-template precomputation at load time: bounding box of every label
  group used by `morphology`, labels present, foreground box.
- `sample()` runs on the CPU with numpy and is cheap (no voxels). Only the
  small params (control points, matrices, Gaussian tables) are sent to the
  GPU.
- No host-device syncs inside `render` (§3). The v2 morphology calls `_bbox`
  (`nonzero`, a sync) for every pair of every image: this moves to the
  per-template precomputation.

### 9.2 Patch mode (`render(plan, roi=patch)`)

Training uses patches (e.g. 128³–192³), so generating a full 0.4 mm volume
(~30 M voxels) to keep a ~2–7 M voxel patch wastes most of the work. The
pipeline runs backwards to find what each step needs, then forwards on crops
only:

1. The patch location is part of the plan (output grid).
2. `spatial.required_input(patch)`: evaluate the deformation only at the
   patch voxels (B-spline of the control points + affine, at those
   coordinates). This gives the source box in the HR template.
3. Crop the HR template to that box + `n_iter` halo, then apply `morphology`
   to the crop. This is **exact**: morphology is local (it reaches `n_iter`
   voxels), so the result inside the box equals the full-volume result.
4. Resample, partial volume, synthesis: patch only.
5. Bias field: evaluate the smooth field at the patch only. Exact.
6. Resolution: patch + a halo of ~3σ of the slice profile along the
   degraded axis, with the low-res grid aligned to the full-volume grid.
   Exact up to the boundary handling.
7. Noise: patch only. Exact (seeded per patch).
8. Motion: k-space ghosting spans the whole field of view along the
   phase-encode axis, so on a patch it is only an approximation. Options:
   (a) accept the approximation (ghosts wrap inside the patch);
   (b) compute motion on the full volume at a coarse grid and add the
   difference to the patch; (c) render the full volume when motion is
   drawn (prob 0.3 in the current config). Decide after benchmarking (Q9).
9. Normalisation: min-max on a patch differs from min-max on the whole
   image. Use either fixed or percentile normalisation, or compute it from
   the plan (expected intensity range), so the patch is normalised as the
   full image would be.

Patch mode also solves most of the memory problem (L9, Q2): the cost scales
with the patch, not the template, so 0.2 mm becomes possible under 10 GB.
Full-volume mode (`roi=None`) stays the default for offline datasets and for
debugging. A test checks that patch mode equals the same crop of a
full-volume render (except motion).

### 9.3 Reuse the expensive part

The geometry (morphology + deformation + PV) costs much more than the
intensities. Option `intensity_draws_per_geometry: n` (default 1) renders `n`
images with different tree Gaussians, contrasts and artefacts from one
deformed label map. This divides the geometric cost by `n`, at the price of
correlated shapes inside a batch.

### 9.4 Kernels

- PV + synthesis fused in one pass per slab, as in v2 (histogram ->
  top-k -> gather mu/sigma). The separate PV output of §4.3 is only
  materialised when it is recorded or requested.
- Batch the small per-label work (morphology pairs, Gaussian table lookup)
  instead of looping in Python.
- `torch.compile` / half precision (fp16 intensities, int16 labels) where the
  benchmarks show a gain. Not before the modules are correct and tested.

### 9.5 Integration with training

Two options (Q4):
- **In-process**: the training loop calls `gen.render(gen.sample_plan(seed),
  roi=patch)` on the training GPU. Simple, no copies, but generation and
  training share GPU time and memory.
- **Producer process**: one or more generator processes (another GPU, or the
  same one) fill a queue of samples. Training reads from the queue and can
  fall back to repeating recent samples if the queue is empty.

Either way, the generator exposes a `torch.utils.data.IterableDataset`
wrapper with a worker-safe seed (`run_seed`, worker id, sample counter).

### 9.6 Benchmarks (`tests/ -m bench`)

Time and peak memory per module and per sample, for each target resolution
(0.2, 0.4, 0.6 mm), full volume and patch mode, on the real NextBrain
templates. They are run after every change to a module, with results saved
so regressions are visible.

---

## 10. Launchers

**Offline** (`nxgen-template` / `build_templates(...)`):
```
python -m nxgen.cli.build_templates --seg-dir maps/ --config config.json \
       --out templates/ --device cpu [--overwrite]
```
Reads the `grid`, `labels`, `template` sections (and `spatial`/`morphology`
only for `margin: auto`). Writes templates + build records with hashes.

**Online** (`nxgen-generate` / `Generator`):
```
python -m nxgen.cli.generate -n 100 --templates templates/ --config config.json \
       --out data/outputs --dataset-id 1 [--seed 0] [--start 0] \
       [--params plan.json] [--save-steps morphology,pv,synthesis] [--device cuda:0]
```
```python
gen = Generator.from_config('config.json', templates='templates/', device='cuda')
out = gen(seed=42)                    # out.image, out.seg, out.plan, out.grid
out = gen(plan=old_plan)              # exact replay
plan = gen.sample_plan(seed=42)       # inspect / edit before rendering
plan['motion'] = None                 # disable one step for this image
out = gen.render(plan, recorder=DiskRecorder('debug/'))
```
- Writes the nnU-Net v2 layout (as `generate_images.py`), plus the frozen
  `label_space.json` and a config copy **per config hash** (fixes
  problems_to_solve #1, #2).
- `Generator` can also be wrapped in a `torch.utils.data.IterableDataset` to
  generate online during training (see Q4).

---

## 11. Package layout

```
generator_v2_modular/
├── generator_structure.txt / .md
├── nxgen/
│   ├── core/        grid.py label_space.py tree.py config.py rng.py io.py recorder.py
│   ├── offline/     template.py margin.py
│   ├── online/      acquisition.py morphology.py spatial.py partial_volume.py
│   │                tree_intensities.py synthesis.py normalization.py
│   │                artifacts/ bias.py motion.py resolution.py noise.py
│   ├── checks/      labels.py adjacency.py intensity.py
│   ├── pipeline.py  Plan, Generator (sample_plan / render)
│   └── cli/         build_templates.py generate.py
├── configs/         default.json, nextbrain_l1.json
├── tests/
│   ├── phantoms.py  concentric-sphere label maps + tiny tree (CPU, seconds)
│   ├── test_<module>.py
│   └── test_v2_equivalence.py   deterministic parts vs generator_v2
└── notebooks/       one per module (inspect inputs/params/outputs)
```

**Testing strategy.** Unit tests run on small synthetic phantoms (CPU,
seconds). Equivalence tests compare the deterministic parts to v2:
super-resolution, PV seg, low-res transform, and morphology/deformation with
the same params. This shows the refactor did not change behaviour before
anything new is added. Memory/time benchmarks are separate (`-m bench`,
GPU, real templates).

---

## 12. Suggested implementation order

1. `core` (grid, label space, config, rng, io) + phantoms.
2. Offline template module + equivalence test vs v2.
3. `acquisition` + `tree_intensities` (no voxels: fast to test).
4. `partial_volume` + `synthesis` -> first clean images.
5. `morphology`, `spatial`.
6. `bias`, `noise`, `resolution`, `motion`, `normalization`.
7. `pipeline` (full-volume mode) + recorder (with timing) + launchers.
8. First benchmarks on real NextBrain templates (time + memory per module).
9. Patch mode (`roi` in every module) + test: patch == crop of the full
   render.
10. Speed work driven by the benchmarks (§9.3, §9.4), then the training
    integration (§9.5).

Correctness first, speed second: the per-module tests of steps 1–7 are the
reference that the optimised versions must still pass.

---

## 13. Open questions for the next iteration

- **Q1 Cornucopia.** Keep it only for deterministic primitives (proposed), or
  keep its random transforms and fork the global RNG per module? The first
  gives full params. The second is less code.
- **Q2 Memory target.** "< 10 GB for any supported resolution": does the
  limit apply to patch mode (easy, §9.2) or also to full volumes (hard at
  0.2 mm: on the largest template, ~593 M voxels, one float32 volume is
  2.4 GB and one complex FFT 4.7 GB)?
- **Q3 Plan up front vs sampling inside the pipeline.** Up front (proposed)
  requires that `sample` needs no voxel data. That is true for every v2 step.
  Is anything planned that would break it (e.g. morphology driven by
  local thickness)?
- **Q4 Training setup** (answered in part: generation must be fast enough
  for training). Still needed: which training framework? nnU-Net v2 reads
  preprocessed files, so online generation needs a custom trainer and
  dataloader. What patch size, batch size and step time? This sets the time
  budget per sample. In-process generation or a producer process (§9.5)?
  Same GPU or a dedicated one?
- **Q5 Grids.** Should one HR template serve several target resolutions
  (pad to a multiple of e.g. 6 = lcm(2, 3)), or one template per target?
- **Q6 PV `smooth` method.** Gaussian window (proposed: the same code as
  `downsample`), or grey-level morphology? What sigma, and should the hard
  seg stay identical to the HR labels?
- **Q7 Range semantics change** (`rotation 10` = max in v2 vs fixed here):
  OK to break compatibility with v2 configs, with a converter?
- **Q8 Package name** and whether `LabelTree` stays a separate file shared
  with WP1 tree building.
- **Q9 Motion in patch mode** (§9.2 step 8): approximate on the patch,
  coarse full-volume difference, or full render when motion is drawn?
- **Q10 Geometry reuse** (§9.3): is it acceptable for several samples to
  share the same deformed label map, and if so, how many?
