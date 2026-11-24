#!/usr/bin/env python
"""Add ventricle and CSF labels to NextBrain (nxbsimplified) hemisphere segmentations.

The input label maps have no ventricle / CSF labels: those regions are background (0).
This script adds them following TODO.txt:

Ventricles
    "All background regions inside the WM are ventricle."
    1. inner   = brain voxels that are NOT cortex-like (cerebral cortex 3 and the
                 cerebellar cortex layers 595/597 are excluded, so sulci, which are
                 lined by cortex, are never enclosed).
    2. inner is closed with a small radius (bridges 1-2 voxel gaps in the WM).
    3. holes of that mask are filled (a) in 3D and (b) slice by slice along the
       left-right axis. (b) is needed because the specimens are cut hemispheres:
       the lateral ventricles are open on the medial cut face, so in 3D they are
       not enclosed, but in every sagittal slice they are.
    4. ventricle candidates = background voxels inside the filled mask. Connected
       components smaller than --min-ventricle-mm3 (cracks / tiny pockets in the
       WM) are not ventricle: they are filled with the nearest tissue label.
    5. the ventricle mask is closed (--vent-close-mm, background voxels only) to
       bridge the few sagittal slices where the ventricle is open even in 2D.

CSF
    1. brain mask   = all labelled voxels + ventricles
    2. close the cortex: morphological closing of the brain mask (--close-mm),
       followed by 3D hole filling -> sulci become part of the mask.
    3. dilate a bit (--dilate-mm).
    4. CSF = dilated mask - brain mask   (ventricles are part of the brain mask,
       so they are excluded).

Morphology is done with Euclidean distance transforms in mm (exact balls, works
with anisotropic voxels and large radii fast).

Usage
    python add_csf_to_labels.py                       # all cases, default paths
    python add_csf_to_labels.py --cases case-1 --save-intermediates
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

HERE = Path(__file__).resolve().parent.parent  # add_csf_to_labels/ (shared data/)
DEFAULT_INPUT = HERE / "data" / "inputs" / "nxbsimplified" / "labels"
DEFAULT_OUTPUT = HERE / "data" / "outputs" / "morphological_approach"

# NextBrainLabels.yaml: anterior_horn_of_lateral_ventricle = 50
VENTRICLE_LABEL = 50
# FreeSurfer LUT: CSF = 24 (unused in NextBrain)
CSF_LABEL = 24
# cerebral cortex + cerebellar cortex (molecular / granular layers)
CORTEX_LABELS = (3, 595, 597)


# --------------------------------------------------------------------------- #
# morphology in mm (distance-transform based)
# --------------------------------------------------------------------------- #
def dilate_mm(mask: np.ndarray, r: float, spacing) -> np.ndarray:
    if r <= 0:
        return mask.copy()
    return ndi.distance_transform_edt(~mask, sampling=spacing) <= r


def erode_mm(mask: np.ndarray, r: float, spacing) -> np.ndarray:
    if r <= 0:
        return mask.copy()
    return ndi.distance_transform_edt(mask, sampling=spacing) > r


def close_mm(mask: np.ndarray, r: float, spacing) -> np.ndarray:
    """Closing with a ball of radius r (mm). Caller must pad by >= r."""
    return erode_mm(dilate_mm(mask, r, spacing), r, spacing)


def fill_holes_2d(mask: np.ndarray, axis: int) -> np.ndarray:
    out = np.zeros_like(mask)
    for i in range(mask.shape[axis]):
        sl = [slice(None)] * mask.ndim
        sl[axis] = i
        sl = tuple(sl)
        out[sl] = ndi.binary_fill_holes(mask[sl])
    return out


def lr_axis(affine: np.ndarray) -> int:
    """Voxel axis that is most aligned with world left-right (x)."""
    return int(np.argmax(np.abs(affine[:3, :3])[0]))


def bbox_with_pad(mask: np.ndarray, pad: int):
    idx = np.nonzero(mask)
    lo = [max(int(i.min()) - pad, 0) for i in idx]
    hi = [min(int(i.max()) + pad + 1, s) for i, s in zip(idx, mask.shape)]
    return tuple(slice(a, b) for a, b in zip(lo, hi))


# --------------------------------------------------------------------------- #
# core
# --------------------------------------------------------------------------- #
def add_csf(
    labels: np.ndarray,
    spacing,
    lr: int,
    cortex_labels=CORTEX_LABELS,
    bridge_mm: float = 0.4,
    vent_close_mm: float = 2.0,
    min_ventricle_mm3: float = 1.0,
    close_mm_r: float = 2.0,
    dilate_mm_r: float = 1.0,
    ventricle_label: int = VENTRICLE_LABEL,
    csf_label: int = CSF_LABEL,
    verbose: bool = True,
):
    """Return (new_labels, steps) where steps is a dict of boolean masks."""
    spacing = tuple(float(s) for s in spacing)
    vox_mm3 = float(np.prod(spacing))
    t0 = time.time()

    def log(msg):
        if verbose:
            print(f"    [{time.time() - t0:6.1f}s] {msg}", flush=True)

    # Work on a padded crop around the brain: padding avoids border effects in
    # the closing / dilation, cropping saves memory and time.
    rmax = max(close_mm_r, bridge_mm, vent_close_mm) + dilate_mm_r
    pad = int(np.ceil(rmax / min(spacing))) + 3
    brain0 = labels > 0
    crop = bbox_with_pad(brain0, 0)
    lab = np.pad(labels[crop], pad)
    brain = lab > 0
    bg = ~brain

    # ---------------- ventricles ----------------
    inner = brain & ~np.isin(lab, cortex_labels)
    inner_c = close_mm(inner, bridge_mm, spacing)
    log("inner (non-cortex) mask closed")
    filled = ndi.binary_fill_holes(inner_c)
    filled |= fill_holes_2d(inner_c, lr)
    log("3D + 2D (sagittal) hole filling done")
    cand = filled & bg

    cc, n = ndi.label(cand)
    sizes = np.bincount(cc.ravel())
    sizes[0] = 0
    min_vox = int(np.ceil(min_ventricle_mm3 / vox_mm3))
    keep = sizes >= min_vox
    keep[0] = False
    vent = keep[cc]
    small = cand & ~vent
    del cc
    # bridge sagittal slices where the ventricle is open even in 2D (otherwise
    # they would become CSF stripes inside the ventricle): close the ventricle
    # mask and add the background voxels it covers.
    if vent_close_mm > 0:
        vent |= close_mm(vent, vent_close_mm, spacing) & bg
        small &= ~vent
    log(f"{n} candidate components, {int(keep.sum())} kept as ventricle "
        f"(>= {min_vox} vox), {int(small.sum())} vox of small pockets to fill")

    out = lab.copy()
    if small.any():
        # nearest tissue label, computed only in the bbox of the small pockets
        sb = bbox_with_pad(small, 3)
        sub = lab[sb]
        _, ind = ndi.distance_transform_edt(sub == 0, sampling=spacing, return_indices=True)
        nearest = sub[tuple(ind)]
        o = out[sb]
        m = small[sb]
        o[m] = nearest[m]
        out[sb] = o
        del ind, nearest
    out[vent] = ventricle_label

    # ---------------- CSF ----------------
    brain_v = out > 0  # tissue + ventricles (+ filled pockets)
    closed = close_mm(brain_v, close_mm_r, spacing)
    closed = ndi.binary_fill_holes(closed)
    log("brain mask closed (sulci filled)")
    dilated = dilate_mm(closed, dilate_mm_r, spacing)
    log("closed mask dilated")
    csf = dilated & ~brain_v
    out[csf] = csf_label

    # back to the original grid
    def uncrop(a, fill=0):
        full = np.full(labels.shape, fill, dtype=a.dtype)
        full[crop] = a[tuple(slice(pad, -pad) for _ in range(a.ndim))]
        return full

    new = uncrop(out.astype(labels.dtype))
    steps = {
        "brain_mask": uncrop(brain),
        "inner_mask": uncrop(inner_c),
        "inner_filled": uncrop(filled),
        "ventricles": uncrop(vent),
        "small_pockets": uncrop(small),
        "closed_mask": uncrop(closed),
        "dilated_mask": uncrop(dilated),
        "csf": uncrop(csf),
    }
    # anything that falls in the pad (outside the original FOV) is dropped
    return new, steps


def process_case(path: Path, out_dir: Path, params: dict, save_intermediates: bool):
    name = path.name.replace(".nii.gz", "")
    print(f"[{name}] start", flush=True)
    t = time.time()
    img = nib.load(str(path))
    labels = np.asarray(img.dataobj)
    spacing = img.header.get_zooms()[:3]
    lr = lr_axis(img.affine)

    new, steps = add_csf(labels, spacing, lr, **params)

    lab_dir = out_dir / "labels"
    lab_dir.mkdir(parents=True, exist_ok=True)
    hdr = img.header.copy()
    hdr.set_data_dtype(labels.dtype)
    nib.save(nib.Nifti1Image(new, img.affine, hdr), str(lab_dir / path.name))

    if save_intermediates:
        idir = out_dir / "intermediates" / name
        idir.mkdir(parents=True, exist_ok=True)
        for k, m in steps.items():
            h = img.header.copy()
            h.set_data_dtype(np.uint8)
            nib.save(nib.Nifti1Image(m.astype(np.uint8), img.affine, h), str(idir / f"{k}.nii.gz"))

    vox = float(np.prod(spacing))
    summary = {
        "case": name,
        "shape": list(labels.shape),
        "spacing_mm": [float(s) for s in spacing],
        "lr_axis": lr,
        "brain_cm3": float(steps["brain_mask"].sum() * vox / 1000),
        "ventricle_cm3": float(steps["ventricles"].sum() * vox / 1000),
        "small_pockets_filled_cm3": float(steps["small_pockets"].sum() * vox / 1000),
        "csf_cm3": float(steps["csf"].sum() * vox / 1000),
        "seconds": round(time.time() - t, 1),
    }
    print(f"[{name}] done in {summary['seconds']}s: ventricles {summary['ventricle_cm3']:.2f} cm3, "
          f"CSF {summary['csf_cm3']:.2f} cm3", flush=True)
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="folder with label maps (*.nii.gz)")
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="output folder")
    p.add_argument("--cases", nargs="*", default=None, help="case names (e.g. case-1); default all")
    p.add_argument("--ventricle-label", type=int, default=VENTRICLE_LABEL)
    p.add_argument("--csf-label", type=int, default=CSF_LABEL)
    p.add_argument("--cortex-labels", type=int, nargs="+", default=list(CORTEX_LABELS),
                   help="labels that line the sulci (never enclose ventricles)")
    p.add_argument("--bridge-mm", type=float, default=0.4,
                   help="closing radius of the non-cortex mask before hole filling")
    p.add_argument("--vent-close-mm", type=float, default=2.0,
                   help="closing radius of the ventricle mask (bridges slices where it is open in 2D)")
    p.add_argument("--min-ventricle-mm3", type=float, default=1.0,
                   help="smaller enclosed background pockets are filled with the nearest label")
    p.add_argument("--close-mm", type=float, default=2.0, help="closing radius to close the cortex (sulci)")
    p.add_argument("--dilate-mm", type=float, default=1.0, help="dilation of the closed brain mask")
    p.add_argument("--save-intermediates", action="store_true", help="save every step as a binary mask")
    p.add_argument("--workers", type=int, default=5)
    a = p.parse_args()

    files = sorted(a.input.glob("*.nii.gz"))
    if a.cases:
        files = [f for f in files if f.name.replace(".nii.gz", "") in a.cases]
    if not files:
        raise SystemExit(f"no cases found in {a.input}")

    params = dict(
        cortex_labels=tuple(a.cortex_labels),
        bridge_mm=a.bridge_mm,
        vent_close_mm=a.vent_close_mm,
        min_ventricle_mm3=a.min_ventricle_mm3,
        close_mm_r=a.close_mm,
        dilate_mm_r=a.dilate_mm,
        ventricle_label=a.ventricle_label,
        csf_label=a.csf_label,
        verbose=len(files) == 1 or a.workers == 1,
    )
    a.output.mkdir(parents=True, exist_ok=True)

    with ProcessPoolExecutor(max_workers=min(a.workers, len(files))) as ex:
        futs = [ex.submit(process_case, f, a.output, params, a.save_intermediates) for f in files]
        summaries = [f.result() for f in futs]

    report = {"params": {k: v for k, v in params.items() if k != "verbose"}, "cases": summaries}
    with open(a.output / "summary.json", "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"saved {len(summaries)} cases to {a.output}")


if __name__ == "__main__":
    main()
