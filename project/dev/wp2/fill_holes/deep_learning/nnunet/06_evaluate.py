#!/usr/bin/env python
"""
Step 6 — evaluate on held-out FOMO hemispheres with synthetic corruption.

Two modes (both use subjects never seen in training, and fixed seeds):

  --mode patch   (fast) evaluation.patch_samples corrupted patches, generated
                 exactly like the training patches.
  --mode volume  (realistic) evaluation.volumes whole hemispheres: upsample to
                 0.2 mm, corrupt the WHOLE volume, restore it with the sliding
                 window of 05_infer.py, compare with the clean volume.

Reported (metrics.md, metrics.json, per_label.csv in the output folder):
  * Dice per label over all voxels, and restricted to corrupted voxels
  * Dice of the corrupted input itself (baseline: what doing nothing gives)
  * fraction of hole voxels filled (and filled with the right label)
  * accuracy on corrupted voxels and "damage" (correct input voxels made wrong)
  volume mode also writes per_volume.csv, and --save-volumes N stores
  clean / corrupted / restored NIfTIs of the first N volumes for viewing.

  python 06_evaluate.py --config configs/default.yaml --mode patch
  python 06_evaluate.py --config configs/default.yaml --mode volume --save-volumes 2
"""
import os

import nibabel as nib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from lib.config import checkpoint_config, config_from_args, parser
from lib.corruption import sample_corruption_args
from lib.data import RestorationPatches, load_items, worker_init
from lib.external import corrupt_module
from lib.labels import (VolumeStore, assert_writable, upsample_factor, upsample_labels,
                              upsampled_affine)
from lib.losses import RestorationMetrics, evaluate_loader, write_metrics
from lib.model import load_checkpoint
from lib.sliding import restore_passes


def evaluate_patches(model, cfg, split, device):
    """Metrics over seeded corrupted patches of `split`."""
    e = cfg["evaluation"]
    data = RestorationPatches(cfg, split, e["patch_samples"], deterministic=True, seed=e["seed"])
    loader = DataLoader(data, batch_size=cfg["inference"]["batch_size"],
                        num_workers=cfg["inference"]["num_workers"], worker_init_fn=worker_init)
    metrics, loss = evaluate_loader(model, loader, device, cfg)
    return metrics, dict(mode="patch", split=split, samples=len(data), loss=loss)


def evaluate_volumes(model, cfg, split, device, out_dir, save_volumes):
    """Metrics over whole corrupted hemispheres of `split` (one row per volume in per_volume.csv)."""
    e = cfg["evaluation"]
    cn = corrupt_module(cfg)
    store = VolumeStore(cfg)
    items, _ = load_items(cfg, split)
    order = np.random.default_rng(e["seed"]).permutation(len(items))[:e["volumes"]]
    total = RestorationMetrics(cfg["labels"]["num_classes"])
    rows = []
    for n, idx in enumerate(order):
        item = items[idx]
        rng = np.random.default_rng([e["seed"], n])
        seg, affine = store.load(item)
        factor = upsample_factor(affine, cfg["data"]["target_spacing"])
        clean = upsample_labels(seg, factor)
        affine = upsampled_affine(affine, factor)
        angle_range, axis, kw, info = sample_corruption_args(rng, cfg["corruption"],
                                                             blob_scale=e["volume_blob_scale"])
        corrupted, params, _ = cn.corrupt_array(clean, affine, angle_range, axis, rng,
                                                brain=clean > 0, fg_box=cn.bbox(clean > 0), **kw)
        restored = restore_passes(model, corrupted, cfg, device)

        one = RestorationMetrics(total.c)
        one.update(*(torch.from_numpy(a) for a in (restored, corrupted, clean)))
        for k in total.conf:
            total.conf[k] += one.conf[k]
        s = one.summary()
        rows.append(dict(dataset=item["dataset"], subject=item["subject"], session=item["session"],
                         hemi=item["hemi"], severity=info["severity"], axis=axis,
                         angle_deg=params["angle_deg"], thickness_mm=params["thickness_mm"],
                         events=len(params["events"]),
                         **{k: s[k] for k in ("corrupted_fraction", "dice_corrupted_mean",
                                              "dice_all_mean", "dice_input_mean", "hole_filled",
                                              "hole_correct", "acc_corrupted", "damage")}))
        print(f"[{n + 1}/{len(order)}] {item['subject']} {item['hemi']} severity {info['severity']:.2f} "
              f"| Dice corrupted {s['dice_corrupted_mean']:.4f} | Dice all {s['dice_all_mean']:.4f} "
              f"(input {s['dice_input_mean']:.4f}) | holes filled {s['hole_filled']:.3f}", flush=True)
        pd.DataFrame(rows).to_csv(os.path.join(out_dir, "per_volume.csv"), index=False)
        if n < save_volumes:
            for tag, arr in (("clean", clean), ("corrupted", corrupted), ("restored", restored)):
                nib.save(nib.Nifti1Image(arr, affine), os.path.join(out_dir, f"vol{n:02d}_{tag}.nii.gz"))
    return total, dict(mode="volume", split=split, volumes=len(order))


def main():
    p = parser(__doc__)
    p.add_argument("--checkpoint", default=None, help="default: <run folder>/best.pt")
    p.add_argument("--mode", choices=["patch", "volume"], default="patch")
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--output", default=None, help="default: <run folder>/eval_<mode>")
    p.add_argument("--save-volumes", type=int, default=0,
                   help="volume mode: save NIfTIs of the first N volumes")
    args = p.parse_args()
    cfg = config_from_args(args)
    run_dir = os.path.join(cfg["paths"]["runs_dir"], cfg["train"]["run_name"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, ckpt_cfg, ckpt = load_checkpoint(args.checkpoint or os.path.join(run_dir, "best.pt"), device)
    cfg = checkpoint_config(ckpt_cfg, cfg)
    out_dir = assert_writable(args.output or os.path.join(run_dir, f"eval_{args.mode}"))
    os.makedirs(out_dir, exist_ok=True)

    if args.mode == "patch":
        metrics, extra = evaluate_patches(model, cfg, args.split, device)
    else:
        metrics, extra = evaluate_volumes(model, cfg, args.split, device, out_dir, args.save_volumes)
    extra["checkpoint_epoch"] = ckpt["epoch"]
    write_metrics(out_dir, metrics, cfg["labels"]["names"], extra)
    print(open(os.path.join(out_dir, "metrics.md")).read())


if __name__ == "__main__":
    main()
