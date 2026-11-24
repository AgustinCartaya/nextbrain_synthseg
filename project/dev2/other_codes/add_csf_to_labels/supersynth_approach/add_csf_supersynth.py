#!/usr/bin/env python
"""Approach 2 (TODO.txt): paste SuperSynth ventricles / CSF into nxbsimplified labels.

Synthetic images were generated from the nxbsimplified segmentations and segmented
with SuperSynth (data/inputs/supersynth, FreeSurfer labels, 0.4 mm, same field of
view as the 0.2 mm label maps). Those segmentations already contain ventricles
(and, if SuperSynth was run in a mode that predicts it, CSF). They are put into the
nxbsimplified labels like a puzzle:

Ventricles (final label 50)
    1. SuperSynth ventricle labels (lateral 4/43, inferior lateral 5/44,
       3rd 14, 4th 15) -> binary mask, resampled to the label grid
       (linear interpolation of the mask + 0.5 threshold, i.e. a smooth nearest).
    2. dilated a bit (--dilate-mm).
    3. pasted on the segmentation, keeping only voxels that are background there
       (everything already occupied by tissue is removed).

CSF (final label 24)
    SuperSynth CSF (label 24) is resampled and pasted directly, keeping only
    voxels that are background and not ventricle. NOTE: the current SuperSynth
    outputs contain no CSF label, so this step is a no-op until they do (a
    warning is printed).

Usage
    python add_csf_supersynth.py                         # all cases
    python add_csf_supersynth.py --cases case-1 --save-intermediates
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parent.parent  # add_csf_to_labels/ (shared data/)
DEFAULT_LABELS = ROOT / "data" / "inputs" / "nxbsimplified" / "labels"
DEFAULT_SUPERSYNTH = ROOT / "data" / "inputs" / "supersynth"
DEFAULT_OUTPUT = ROOT / "data" / "outputs" / "supersynth_approach"

VENTRICLE_LABEL = 50  # NextBrainLabels.yaml: anterior_horn_of_lateral_ventricle
CSF_LABEL = 24        # FreeSurfer CSF
# FreeSurfer: Left/Right-Lateral-Ventricle, Left/Right-Inf-Lat-Vent, 3rd, 4th
SS_VENTRICLE_LABELS = (4, 5, 43, 44, 14, 15)
SS_CSF_LABELS = (24,)


def resample_mask(mask: np.ndarray, src_affine, dst_affine, dst_shape) -> np.ndarray:
    """Resample a binary mask onto another grid (linear + 0.5 threshold)."""
    if not mask.any():
        return np.zeros(dst_shape, bool)
    m = np.linalg.inv(src_affine) @ dst_affine  # dst voxel -> src voxel
    out = ndi.affine_transform(mask.astype(np.float32), m[:3, :3], m[:3, 3],
                               output_shape=dst_shape, order=1, mode="constant", cval=0.0)
    return out >= 0.5


def dilate_mm(mask: np.ndarray, r: float, spacing) -> np.ndarray:
    if r <= 0 or not mask.any():
        return mask.copy()
    # crop around the mask to keep the EDT cheap
    pad = int(np.ceil(r / min(spacing))) + 2
    idx = np.nonzero(mask)
    sl = tuple(slice(max(int(i.min()) - pad, 0), min(int(i.max()) + pad + 1, s))
               for i, s in zip(idx, mask.shape))
    out = np.zeros_like(mask)
    out[sl] = ndi.distance_transform_edt(~mask[sl], sampling=spacing) <= r
    return out


def add_csf_supersynth(labels, lab_affine, spacing, ss, ss_affine,
                       ss_ventricle_labels=SS_VENTRICLE_LABELS, ss_csf_labels=SS_CSF_LABELS,
                       dilate_mm_r=0.6, ventricle_label=VENTRICLE_LABEL, csf_label=CSF_LABEL):
    """Return (new_labels, steps) where steps is a dict of boolean masks."""
    spacing = tuple(float(s) for s in spacing)
    bg = labels == 0

    ss_vent = resample_mask(np.isin(ss, ss_ventricle_labels), ss_affine, lab_affine, labels.shape)
    ss_vent_dil = dilate_mm(ss_vent, dilate_mm_r, spacing)
    vent = ss_vent_dil & bg

    ss_csf = resample_mask(np.isin(ss, ss_csf_labels), ss_affine, lab_affine, labels.shape)
    csf = ss_csf & bg & ~vent

    new = labels.copy()
    new[vent] = ventricle_label
    new[csf] = csf_label
    steps = {
        "ss_ventricles": ss_vent,
        "ss_ventricles_dilated": ss_vent_dil,
        "ventricles": vent,
        "ss_csf": ss_csf,
        "csf": csf,
    }
    return new, steps


def process_case(lab_path: Path, ss_path: Path, out_dir: Path, params: dict, save_intermediates: bool):
    name = lab_path.name.replace(".nii.gz", "")
    t = time.time()
    img = nib.load(str(lab_path))
    labels = np.asarray(img.dataobj)
    spacing = img.header.get_zooms()[:3]
    ss_img = nib.load(str(ss_path))
    ss = np.rint(np.asarray(ss_img.dataobj)).astype(np.int32)  # stored as float32

    present = set(np.unique(ss).tolist())
    vent_found = sorted(present & set(params["ss_ventricle_labels"]))
    csf_found = sorted(present & set(params["ss_csf_labels"]))
    if not csf_found:
        print(f"[{name}] WARNING: no SuperSynth CSF label {list(params['ss_csf_labels'])} in {ss_path.name}; "
              f"only ventricles are added", flush=True)

    new, steps = add_csf_supersynth(labels, img.affine, spacing, ss, ss_img.affine, **params)

    (out_dir / "labels").mkdir(parents=True, exist_ok=True)
    hdr = img.header.copy()
    hdr.set_data_dtype(labels.dtype)
    nib.save(nib.Nifti1Image(new, img.affine, hdr), str(out_dir / "labels" / lab_path.name))

    if save_intermediates:
        idir = out_dir / "intermediates" / name
        idir.mkdir(parents=True, exist_ok=True)
        for k, m in steps.items():
            h = img.header.copy()
            h.set_data_dtype(np.uint8)
            nib.save(nib.Nifti1Image(m.astype(np.uint8), img.affine, h), str(idir / f"{k}.nii.gz"))

    vox = float(np.prod(spacing)) / 1000  # cm3
    s = {
        "case": name,
        "ss_ventricle_labels_found": vent_found,
        "ss_csf_labels_found": csf_found,
        "ss_ventricle_cm3": float(steps["ss_ventricles"].sum() * vox),
        "ss_ventricle_on_tissue_cm3": float((steps["ss_ventricles"] & (labels > 0)).sum() * vox),
        "ventricle_cm3": float(steps["ventricles"].sum() * vox),
        "csf_cm3": float(steps["csf"].sum() * vox),
        "seconds": round(time.time() - t, 1),
    }
    print(f"[{name}] done in {s['seconds']}s: ventricles {s['ventricle_cm3']:.2f} cm3 "
          f"(SuperSynth {s['ss_ventricle_cm3']:.2f}, {s['ss_ventricle_on_tissue_cm3']:.2f} on tissue removed), "
          f"CSF {s['csf_cm3']:.2f} cm3", flush=True)
    return s


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--labels", type=Path, default=DEFAULT_LABELS, help="nxbsimplified label maps")
    p.add_argument("--supersynth", type=Path, default=DEFAULT_SUPERSYNTH, help="SuperSynth segmentations (same names)")
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--cases", nargs="*", default=None, help="case names (e.g. case-1); default all")
    p.add_argument("--ss-ventricle-labels", type=int, nargs="+", default=list(SS_VENTRICLE_LABELS))
    p.add_argument("--ss-csf-labels", type=int, nargs="+", default=list(SS_CSF_LABELS))
    p.add_argument("--dilate-mm", type=float, default=0.6, help="dilation of the SuperSynth ventricles")
    p.add_argument("--ventricle-label", type=int, default=VENTRICLE_LABEL)
    p.add_argument("--csf-label", type=int, default=CSF_LABEL)
    p.add_argument("--save-intermediates", action="store_true", help="save every step as a binary mask")
    p.add_argument("--workers", type=int, default=5)
    a = p.parse_args()

    files = sorted(a.labels.glob("*.nii.gz"))
    if a.cases:
        files = [f for f in files if f.name.replace(".nii.gz", "") in a.cases]
    pairs = [(f, a.supersynth / f.name) for f in files]
    missing = [str(s) for _, s in pairs if not s.exists()]
    if missing:
        raise SystemExit(f"missing SuperSynth files: {missing}")
    if not pairs:
        raise SystemExit(f"no cases found in {a.labels}")

    params = dict(ss_ventricle_labels=tuple(a.ss_ventricle_labels), ss_csf_labels=tuple(a.ss_csf_labels),
                  dilate_mm_r=a.dilate_mm, ventricle_label=a.ventricle_label, csf_label=a.csf_label)
    a.output.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=min(a.workers, len(pairs))) as ex:
        futs = [ex.submit(process_case, f, s, a.output, params, a.save_intermediates) for f, s in pairs]
        summaries = [f.result() for f in futs]

    with open(a.output / "summary.json", "w") as fh:
        json.dump({"params": params, "cases": summaries}, fh, indent=2)
    print(f"saved {len(summaries)} cases to {a.output}")


if __name__ == "__main__":
    main()
