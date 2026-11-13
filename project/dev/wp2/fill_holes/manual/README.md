# Classical (no deep learning) restoration of NextBrain reconstructions

Fixes the artefacts of the 5 ex-vivo NextBrain hemispheres in
`../data/inputs/recon_holes_0.2mm/` (0.2 mm, full label set, ~600 M voxels)
with geometry and simple anatomical rules. No training is involved.

## What the artefacts are
NextBrain hemispheres are reconstructed from ~1 cm tissue blocks cut in near-coronal slabs.
In the label maps this produces:

| artefact | what it looks like |
|---|---|
| **block gaps** | thin (0.4–2 mm) background slabs every ~11–13 mm, tilted a few degrees from coronal, often crossing only part of the brain |
| **block steps** | in-plane misalignment of a block relative to its neighbours |
| **cracks** | thin background cracks inside blocks |
| **holes** | enclosed background cavities |

The main difficulty is that **sulci are also thin background sheets**, so any rule that fills
"thin background" also closes sulci (see the naive baseline below).

## Method (`restore.py`)
1. **Gap planes.** Take bounded short background runs (≤ 2.4 mm) along the A-P axis.
   - Grid-search the section normal so that their midpoints form sharp peaks.
   - Fit one plane per peak (RANSAC, ≤ 10° from the global normal).
   - Keep a plane only if ≥ 8% of its runs are bounded by **non-cortical** tissue at both
     ends: real gaps cross white matter, while sulci are lined by cortex.
2. **Block steps.** For each gap plane, estimate the in-plane shift between the label slices
   on both sides (FFT cross-correlation over the L1 classes).
   - Subtract the natural anatomical drift, measured inside both neighbouring blocks over the
     same distance.
   - Solve the block translations by regularised least squares.
   - Apply only shifts ≥ 0.4 mm, in integer voxels, so labels are never interpolated.
3. **Gap filling.** The voxels of each gap slab get the nearest known label (3D distance
   transform). Sulcus background next to the gap stays "known", so **sulci crossing a gap stay open**.
4. **Cracks.** Fill background runs ≤ 1 mm along any axis that are bounded at both ends by the
   **same non-cortical class** (white matter, deep GM, brainstem, …).
5. **Enclosed holes.** Fill background components that are not connected to the outside, whose
   border is mostly non-cortical and that don't touch CSF. Most of the lateral ventricle is
   unlabelled background in NextBrain, so CSF-adjacent holes are left alone.

All steps work on the **full label set**. The L1 classes, needed to know which labels are cortex,
come from `wp1/create_merged_segmentations/merge_labels.py` (NextBrain ontology + `mapping_l1.json`),
plus the 3 raw labels missing from the ontology (1→Brainstem, 2→CSF, 3→cerebral cortex).
This reproduces `changed_recon_holes_0.2mm_L1` exactly.

## Usage
```bash
# one volume -> restored labels + per-voxel step codes + json report
python restore.py input.nii.gz output.nii.gz            # add --help for all parameters
# use --input-is-l1 if necessary
#   output_steps.nii.gz : 1 gap, 2 crack, 3 hole, 4 shift
#   output_report.json  : planes found, misalignments, voxel counts

# naive baseline: fill every background voxel inside the closed brain with the nearest label
python baseline_naive.py input.nii.gz output.nii.gz [--closing-iter 3] [--no-fill-holes] [--save-mask]

# evaluation (benchmark/)
python benchmark/run_all.py             # the 5 real cases -> benchmark/results/restored/ (+ QC figures)
python benchmark/benchmark.py --source 4 <FOMO seg.left.nii.gz> --downsample 1 --seed 1
python benchmark/benchmark.py ... --no-shifts      # ablation of the shift step
python benchmark/compare_naive.py <FOMO seg.left.nii.gz> --closing-iter 3 6
```
`baseline_naive.py` never looks at label classes, so it needs no `--input-is-l1`.
The benchmark scripts assume full-label inputs (the real cases or FOMO segmentations).
Useful switches: `--no-shifts`, `--no-gaps`, `--no-cracks`, `--no-holes`, `--all-axes`
(also look for gaps along the other two axes; off by default, see limitations),
`--max-gap-mm`, `--max-crack-mm`, `--min-noncortex`, `--min-shift-mm`.

Use: `--input-is-l1` if the input is L1.

Runtime is 3–4 min per real case at 0.2 mm (single process).

## Does it work? Benchmark with known corruptions (`benchmark/benchmark.py`)
The real cases have no ground truth, so a reference label map gets synthetic NextBrain-like
corruption with known positions:
- block gaps every 9–13 mm, 0.6–2 mm wide, tilted, partial;
- random block shifts (SD 0.6 mm);
- cracks and blob holes from `corrupt_nifti.py`.

The restored map is scored against the reference **only where the synthetic corruption
changed something**.

Two kinds of reference:
- **Clean FOMO NextBrain segmentations** (4 hemispheres, 4 datasets, 0.4 mm). These have no
  pre-existing artefacts, so this is also the only valid test of damage: the method is run on
  the clean volume and we count sulcal background that gets filled.
- **The real cases themselves** at 0.4 mm. Realistic anatomy, but they contain real gaps,
  so damage can't be measured on them.

L1 voxel accuracy restored (% of corrupted voxels fixed), seed 1:

| reference | gaps | shifted blocks | holes/cracks | block-step RMS (vox) | clean volume: voxels changed / sulci filled |
|---|---|---|---|---|---|
| FOMO PT017 HBN-SSI | 86.7 | 82.9 | 50.0 | 3.50 → 1.03 | 889 / 0.00% |
| FOMO PT027 NKI | 84.5 | 70.9 | 37.8 | 3.61 → 1.41 | 1,349 / 0.02% |
| FOMO PT018 HBN | 86.9 | 74.6 | 43.4 | 3.14 → 1.13 | 1,742 / 0.09% |
| FOMO PT020 HCP | 86.8 | 65.6 | 57.8 | 2.84 → 1.33 | 3,008 / 0.05% |
| real case-1 | 77.6 | 58.2 | 22.5 | 2.79 → 1.73 | – |
| real case-2 | 81.5 | 69.1 | 32.7 | 3.50 → 2.77 | – |
| real case-3 | 77.0 | 61.8 | 45.2 | 3.54 → 2.16 | – |
| real case-4 | 76.0 | 66.5 | 27.1 | 2.83 → 1.82 | – |
| real case-5 | 79.9 | 70.8 | 29.8 | 3.46 → 1.84 | – |

**Ablation, without the shift step:** only ~2% of shifted voxels are fixed and the block steps
don't change. The shift step is what fixes them.

**Naive baseline** (`baseline_naive.py`, scored with `benchmark/compare_naive.py`), FOMO PT017 / PT020:

| | gaps | holes/cracks | sulci filled on the clean volume |
|---|---|---|---|
| closing 3 iterations | 81–84 | 80–83 | **83–87%** |
| closing 6 iterations | 85–88 | 88–92 | **91–97%** |

The naive approach fills holes better but destroys the sulcal anatomy. `restore.py` matches it
on gaps and keeps the sulci (≤ 0.1% filled).

Holes broken down by location (FOMO PT017):
- inside white matter > 1 mm from cortex: **83%** fixed;
- inside cortex: **~1%**. A hole in cortex opens into a sulcus or the outside and can't be told
  apart from a sulcus by local rules.

## Results on the 5 real cases (`benchmark/results/restored/`)

| case | gap planes | gap voxels filled | crack voxels | hole voxels | block shifts applied |
|---|---|---|---|---|---|
| 1 | 11 | 586,819 (4.7 cm³) | 285,221 | 2,187 | 1 block, 1 voxel |
| 2 | 12 | 428,407 (3.4 cm³) | 129,137 | 965 | 1 block, 1 voxel |
| 3 | 11 | 645,527 (5.2 cm³) | 190,485 | 2,695 | 4 blocks, 1–2 voxels |
| 4 | 16 | 2,019,465 (16.2 cm³) | 389,621 | 2,686 | 3 blocks, 1 voxel |
| 5 | 21 | 2,154,353 (17.2 cm³) | 338,926 | 9,075 | 10 blocks, 1 voxel |

QC figures (`*_qc.png`, `*_qc_zoom.png`) show input, restored and the changed voxels coloured
by step. Visually, most block gaps disappear and the sulci stay open.

On the real cases the detected block misalignments are small (≤ 0.4 mm), so few shifts are
applied. The steps visible at some block boundaries are probably local, non-rigid deformations
that one translation per block can't capture.

## Conclusion
Yes, most of the problem can be solved without deep learning:
- **Block gaps:** 76–87% of corrupted voxels fixed.
- **Block steps:** reduced 21–71% (50–71% on the clean references).
- **Holes in white matter:** fixed.
- **Damage:** essentially no sulci closed.

Everything is deterministic and explainable, keeps all 301 labels and runs in minutes.

What it can't do, and where a learned or atlas prior would help:
1. **Holes and cracks inside cortex** (~1% fixed). They are indistinguishable from sulci locally;
   restoring the missing cortical ribbon needs an anatomical prior: multi-atlas from the other
   hemispheres, or a model trained to inpaint.
2. **Gaps in other orientations** (e.g. axial-oriented cracks near the brainstem in case-1) and
   cracks tilted > 10° from their block's plane. `--all-axes` looks for them, but on the
   secondary axes real and false planes can't be separated with the current features: it added
   one false plane on a clean FOMO volume, filling 1.9% of its sulci. So it's off by default.
3. **Non-rigid misalignment** inside a block boundary. Only one in-plane translation per block
   is estimated (no rotation, no deformation). The cerebellum, cut separately in NextBrain, gets
   the translation of the cerebral block it lies in. Only small shifts (≤ 2 voxels) were applied
   on the real cases.
4. **Missed planes:** about 10–15% of synthetic gap planes are missed (low coverage), and their
   gaps stay unfilled.

A sensible combination is to use this pipeline first, and keep a learned or atlas inpainting
model only for what's left: cortical holes, with `*_steps.nii.gz` showing what was already handled.

## Files
```
manual/
├── restore.py            the method (library + CLI)
├── baseline_naive.py     naive baseline (library + CLI: input -> output)
├── README.md
└── benchmark/
    ├── benchmark.py      synthetic-corruption benchmark of restore.py -> results/benchmark*/
    ├── compare_naive.py  same benchmark for the naive baseline     -> results/baseline_naive.log
    ├── run_all.py        restore the 5 real cases + QC figures     -> results/restored/
    ├── qc.py             QC figure helper
    └── results/          logs (bench_*.log, run_all.log), benchmark figures/json, restored cases
```
The corruption generator is imported from `../image_corruption/corrupt_nifti.py`.
