# Tree generator (WP3)

The SIAM generator from `../augmentation/pipeline` (super-resolution → erosion/dilation →
affine/elastic → partial volume → intensities → bias / motion / noise), with two changes:

- **Intensities come from the label tree** of `../simple_generator/tree_sampling.py`, not from an
  independent Gaussian per tissue. At the sampled resolution and contrast, labels whose node collapses
  share one Gaussian (one *generation class*).
- **It works with any label set.** Raw label values are kept (no grouping into the paper's 12
  tissues). The only requirement is that the template labels are in the tree.

![steps](data/outputs/steps.png)

## Files

```
generator/
├── tree_generator.py     library + CLI (TreeGenerator, TreeGeneratorConfig, prepare_template, ...)
├── run_example.py        step-by-step figure + 4 samples on the SIAM example template
├── configs/freesurfer.json   morphology rules (as tree nodes) for the FreeSurfer/SynthSeg trees
└── data/outputs/         example outputs (HR template cached in templates/)
```

Imports `tree_sampling.py` from `../simple_generator` (path added automatically). Environment: `nxb`
(torch + cornucopia 0.5.1).

## Usage

```bash
conda activate nxb
python run_example.py -n 4

python tree_generator.py --seg seg1.nii.gz seg2.nii.gz \
    --tree ../simple_generator/data/inputs/synthseg_ctx_tree.json \
    --config configs/freesurfer.json --out /path/DatasetXXX -n 100
```

```python
import tree_generator as tg
cfg = tg.TreeGeneratorConfig.from_json('configs/freesurfer.json')
templates = tg.build_templates(['seg.nii.gz'], tree_json, cfg, 'cache/', device='cuda')
gen = tg.TreeGenerator(templates, tree_json, cfg, device='cuda')
img, seg, affine, info = gen()
# seg: indices into gen.labels; info['group_of'][i] = generation class of label i
```

## What changed with respect to the SIAM pipeline

| Step | SIAM pipeline | Tree generator |
|---|---|---|
| Labels | FreeSurfer → 12 tissues (`FREESURFER_TO_TISSUE`) | raw values; outputs are indices into `gen.labels` (0 = background) |
| Super-resolution | one full-volume FFT per tissue | one FFT per label **in its bounding box**; background = 1 − Σ others (exact, by linearity) |
| Erosion/dilation | fixed tissue pairs | `morpho_pairs`: pairs of tree nodes (whole subtree), label values or lists; a growing group with several labels propagates the neighbouring label |
| Partial volume | K one-hot volumes + avg-pool | label histogram per training voxel, slab by slab (memory independent of K) |
| Intensities | `mu ~ U(0,1)`, `sigma ~ U(0.001, 0.01)` per tissue | principal colors `C ~ U(0,255)`, `V ~ U(0.255, 2.55)` (the paper's ranges × 255, in tree units) → `TreeSampling.compute_stats`; one Gaussian per generation class |
| Background | `zero_background` flag | set by the tree (`null` nodes) |
| Artefacts | bias, motion, noise, min-max | same (image / `intensity_scale` first, so the paper's noise range still applies) |

New config fields: `background`, `output_labels` (`'templates'` or `'tree'`), `unassigned`
(`'error'` or `'background'`), `colors_mu`, `colors_sigma`, `min_sigma`, `intensity_scale`,
`resolution` (resolution used to decide node visibility: `None` = `target_voxel`, a number, or a
`(lo, hi)` range), `contrasts` (list to sample from, for per-contrast `r_max`).

## Notes

- Checked on the example template: the 0.25 mm template agrees with the 1 mm labels on 99.94% of
  foreground voxels. All 99 labels are kept, and at 0.75 mm there are 52 generation classes
  (12 at 3 mm). One sample takes about 0.9 s on an H200 (the super-resolution runs once per template and takes a few seconds; the result is cached).
- `--raw-labels` writes raw label values. The `dataset.json` label map is then wrong for
  nnU-Net, which needs the default indices (the raw values are in `original_label_values`).
- The tree's `wr_sigma` values (0.5–3) are larger than the paper's sigma range, so the per-class
  noise is mostly set by the tree.
- The image is still synthesised on the 0.75 mm grid. `resolution` only changes which nodes
  collapse; there is no blur or downsampling to that resolution (same as the SIAM pipeline).
