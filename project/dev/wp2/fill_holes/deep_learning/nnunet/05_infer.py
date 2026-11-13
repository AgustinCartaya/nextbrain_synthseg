#!/usr/bin/env python
"""
Step 5 — restore label volumes (the 5 real test cases by default).

For every input NIfTI (already in the training label space, see
02_prepare_test_cases.py, and at the training resolution):
  <name>.nii.gz            restored labels (same grid, affine and header)
  <name>_changed.nii.gz    mask of the voxels the model changed
  <name>_qc.png            sagittal / coronal / axial slices: input | output | changed
and, for all cases together:
  report.md, stats.json    how much was changed, suspicious holes before/after,
                           volume per label before/after, and a place for notes

Sliding window with overlap and Gaussian weighting (section `inference`).
The network definition and the patch size are read from the checkpoint.

  python 05_infer.py --config configs/default.yaml                 # best.pt, test cases
  python 05_infer.py --checkpoint results/l1_resenc/last.pt --input a.nii.gz b.nii.gz --output out/
  python 05_infer.py --set inference.passes=2 inference.tta_flip_lr=true
"""
import glob
import json
import os

import nibabel as nib
import numpy as np
import torch

from lib.config import checkpoint_config, config_from_args, parser
from lib.external import corrupt_module
from lib.labels import assert_writable
from lib.model import load_checkpoint
from lib.qc import qc_figure, restoration_stats, stats_markdown
from lib.sliding import restore_passes


def main():
    p = parser(__doc__)
    p.add_argument("--checkpoint", default=None, help="default: <run folder>/best.pt")
    p.add_argument("--input", nargs="+", default=None,
                   help="NIfTI files or one folder (default: paths.test_l1_dir)")
    p.add_argument("--output", default=None, help="default: <run folder>/predictions")
    args = p.parse_args()
    cfg = config_from_args(args)
    run_dir = os.path.join(cfg["paths"]["runs_dir"], cfg["train"]["run_name"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, ckpt_cfg, ckpt = load_checkpoint(args.checkpoint or os.path.join(run_dir, "best.pt"), device)
    cfg = checkpoint_config(ckpt_cfg, cfg)
    cn = corrupt_module(cfg)
    num_classes, names = cfg["labels"]["num_classes"], cfg["labels"]["names"]

    inputs = args.input or [cfg["paths"]["test_l1_dir"]]
    if len(inputs) == 1 and os.path.isdir(inputs[0]):
        inputs = sorted(glob.glob(os.path.join(inputs[0], "*.nii*")))
    out_dir = assert_writable(args.output or os.path.join(run_dir, "predictions"))
    os.makedirs(out_dir, exist_ok=True)
    print(f"checkpoint epoch {ckpt['epoch']} | {len(inputs)} volumes -> {out_dir}")

    all_stats, report = {}, ["# Restored volumes", ""]
    for path in inputs:
        name = os.path.basename(path).replace(".nii.gz", "").replace(".nii", "")
        img = nib.load(path)
        vol = np.asarray(img.dataobj)
        if not np.issubdtype(vol.dtype, np.integer):
            vol = np.rint(vol).astype(np.uint8 if num_classes <= 256 else np.uint16)
        if vol.max() >= num_classes:
            raise ValueError(f"{path}: label {vol.max()} >= num_classes {num_classes}. "
                             "Convert it first with 02_prepare_test_cases.py")
        spacing = np.sqrt((img.affine[:3, :3] ** 2).sum(0))
        if not np.allclose(spacing, cfg["data"]["target_spacing"], rtol=0.1):
            print(f"[warning] {name}: voxel size {spacing.round(3)} mm, the model was trained "
                  f"at {cfg['data']['target_spacing']} mm")

        print(f"[{name}] shape {vol.shape}", flush=True)
        out = restore_passes(model, vol, cfg, device, progress=True)

        header = img.header.copy()
        header.set_data_dtype(out.dtype)
        nib.save(nib.Nifti1Image(out, img.affine, header), os.path.join(out_dir, f"{name}.nii.gz"))
        nib.save(nib.Nifti1Image((out != vol).astype(np.uint8), img.affine),
                 os.path.join(out_dir, f"{name}_changed.nii.gz"))
        qc_figure(vol, out, os.path.join(out_dir, f"{name}_qc.png"), num_classes,
                  n_slices=cfg["inference"]["qc_slices"], title=name)
        all_stats[name] = restoration_stats(cn, vol, out, num_classes)
        report.append(stats_markdown(name, all_stats[name], names))
        print(f"[{name}] changed {all_stats[name]['voxels_changed']:,} voxels, suspicious holes "
              f"{all_stats[name]['suspicious_holes_input']:,} -> "
              f"{all_stats[name]['suspicious_holes_output']:,}", flush=True)

        # written after every case, so partial results survive an interruption
        with open(os.path.join(out_dir, "stats.json"), "w") as f:
            json.dump(all_stats, f, indent=2)
        with open(os.path.join(out_dir, "report.md"), "w") as f:
            f.write("\n".join(report))


if __name__ == "__main__":
    main()
