#!/usr/bin/env python
"""Approach 3 (combined, TODO.txt): SuperSynth ventricles + CSF grown from the cortex.

Ventricles (label 50)
    Same as the supersynth approach (supersynth_approach/add_csf_supersynth.py):
    SuperSynth ventricles (4/43, 5/44, 14, 15) resampled to the label grid,
    dilated a bit (--vent-dilate-mm) and pasted only on background voxels.

CSF (label 24)
    Like approach 1, but the CSF must never touch WM / deep GM directly (TODO
    observation 2: on the cut faces of a hemisphere a brain-mask dilation puts
    CSF against exposed WM, which a complete brain would not have). So the CSF
    is grown only from cortical tissue:
    1. cortical mask = cerebral cortex (3) + every label under `cerebral_cortex`
       in the nxbsimplified tree (hippocampal formation, piriform/olfactory
       cortex, ...) + every label under `cerebellar_cortex` + cortical
       amygdala areas (--extra-cortical-labels).
    2. dilate it a bit (--dilate-mm): thin CSF layer on the surface.
    3. (optional, --close-cortex, default on) closing of the cortical mask
       (--close-cortex-mm): fills the gaps between nearby cortex (wide sulci)
       without thickening the outer layer.
    4. remove the brain mask (all tissue + ventricles) -> CSF.
    5. (optional, --fill-holes, default on) background pockets fully enclosed by
       brain + CSF become CSF, unless more than --max-noncortical-boundary of
       their boundary is non-cortical tissue (pockets inside WM / deep GM stay
       background, to respect observation 2).
    6. TODO observation 1: connected components of (brain + CSF) smaller than
       --min-component-frac of the largest one are detached objects; their
       CSF is removed (the tissue itself is left untouched).

Usage
    python add_csf_combined.py --save-intermediates
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parent.parent  # add_csf_to_labels/ (shared data/)
sys.path.insert(0, str(ROOT / "supersynth_approach"))
from add_csf_supersynth import SS_VENTRICLE_LABELS, resample_mask  # noqa: E402
from add_csf_supersynth import dilate_mm as dilate_mm_cropped  # noqa: E402

DEFAULT_LABELS = ROOT / "data" / "inputs" / "nxbsimplified" / "labels"
DEFAULT_SUPERSYNTH = ROOT / "data" / "inputs" / "supersynth"
DEFAULT_OUTPUT = ROOT / "data" / "outputs" / "combined_approach"
DEFAULT_TREE = (ROOT.parents[1] / "simplified_trees_and_labels" / "data" / "outputs" / "nxbsimplified"
                / "nextbrain_tree_allen_gyral_simplified.json")

VENTRICLE_LABEL = 50
CSF_LABEL = 24
CORTICAL_NODES = ("cerebral_gyri_and_lobules_l2", "cerebral_cortex", "cerebellar_cortex")
# amygdala areas with cortical structure that reach the surface (not under `cerebral_cortex`)
EXTRA_CORTICAL_LABELS = (238, 242)  # ventral subdivision of CoA, amygdalocortical transition area
# fallback if the tree is not available (resolved from the tree on 2026-11-23)
FALLBACK_CORTICAL_LABELS = (
    3, 99, 100, 125, 157, 326, 339, 340, 341, 342, 343, 344, 345, 346, 347, 364, 365, 367, 368, 369,
    370, 371, 372, 373, 374, 375, 404, 405, 407, 408, 409, 410, 411, 418, 419, 420, 421, 422, 432, 558,
    559, 561, 562, 563, 564, 565, 566, 567, 568, 569, 571, 572, 573, 574, 575, 576, 595, 597)


def cortical_labels_from_tree(tree_path: Path, nodes=CORTICAL_NODES) -> list[int]:
    """All labels in the subtrees whose node name starts with one of `nodes`."""
    if not tree_path.exists():
        print(f"WARNING: tree not found ({tree_path}); using the fallback cortical label list")
        return list(FALLBACK_CORTICAL_LABELS)
    t = json.load(open(tree_path))["nodes"]

    def collect(k):
        out = list(t[k]["labels"])
        for c in t[k]["children"]:
            out += collect(c)
        return out

    roots = [k for k in t if any(k == n or k.startswith(n + "_l") for n in nodes)]
    return sorted(set(lab for r in roots for lab in collect(r)))


def edt_dilate(mask, r, spacing):
    return ndi.distance_transform_edt(~mask, sampling=spacing) <= r


def add_csf_combined(labels, lab_affine, spacing, ss, ss_affine, cortical_labels,
                     ss_ventricle_labels=SS_VENTRICLE_LABELS, vent_dilate_mm=0.6, dilate_mm_r=1.0,
                     close_cortex=True, close_cortex_mm=3.0, fill_holes=True, max_noncortical_boundary=0.5,
                     min_component_frac=0.3, ventricle_label=VENTRICLE_LABEL, csf_label=CSF_LABEL):
    spacing = tuple(float(s) for s in spacing)

    # ---------------- ventricles (supersynth approach) ----------------
    ss_vent = resample_mask(np.isin(ss, ss_ventricle_labels), ss_affine, lab_affine, labels.shape)
    vent = dilate_mm_cropped(ss_vent, vent_dilate_mm, spacing) & (labels == 0)

    # ---------------- CSF from the cortex ----------------
    # padded crop around the brain (no border effects, cheaper EDT)
    rmax = max(dilate_mm_r, close_cortex_mm if close_cortex else 0.0)
    pad = int(np.ceil(rmax / min(spacing))) + 3
    idx = np.nonzero(labels)
    crop = tuple(slice(int(i.min()), int(i.max()) + 1) for i in idx)
    lab = np.pad(labels[crop], pad)
    brain = np.pad((labels > 0)[crop] | vent[crop], pad)
    cortex = np.isin(lab, cortical_labels)
    cortex_dil = edt_dilate(cortex, dilate_mm_r, spacing)
    if close_cortex and close_cortex_mm > 0:
        # closing via distance transforms: dilate by r, then erode by r
        d = ndi.distance_transform_edt(~cortex, sampling=spacing)
        cortex_closed = ndi.distance_transform_edt(d <= close_cortex_mm, sampling=spacing) > close_cortex_mm
        del d
    else:
        cortex_closed = cortex.copy()
    csf_dil = cortex_dil & ~brain
    csf_raw = (cortex_dil | cortex_closed) & ~brain

    holes_kept = np.zeros_like(brain)
    holes_rejected = np.zeros_like(brain)
    if fill_holes:
        holes = ndi.binary_fill_holes(brain | csf_raw) & ~brain & ~csf_raw
        if holes.any():
            hc, nh = ndi.label(holes)
            # boundary of each pocket: voxels just outside it, assigned to the pocket
            ring_lab = ndi.grey_dilation(hc, size=(3, 3, 3))
            ring = (hc == 0) & (ring_lab > 0)
            noncort = brain & ~cortex & (lab != 0)  # WM / deep GM (ventricles excluded)
            tot = np.bincount(ring_lab[ring], minlength=nh + 1)
            bad = np.bincount(ring_lab[ring], weights=noncort[ring], minlength=nh + 1)
            frac = np.divide(bad, tot, out=np.zeros_like(bad), where=tot > 0)
            ok = frac <= max_noncortical_boundary
            ok[0] = False
            holes_kept = ok[hc]
            holes_rejected = holes & ~holes_kept
            del hc, ring_lab, ring
        csf_raw |= holes_kept

    # observation 1: drop the CSF of small detached objects
    cc, n = ndi.label(brain | csf_raw)
    sizes = np.bincount(cc.ravel())
    sizes[0] = 0
    keep = sizes >= min_component_frac * sizes.max()
    keep[0] = False
    csf = csf_raw & keep[cc]
    del cc

    def uncrop(a):
        full = np.zeros(labels.shape, dtype=a.dtype)
        full[crop] = a[tuple(slice(pad, -pad) for _ in range(3))]
        return full

    new = labels.copy()
    new[vent] = ventricle_label
    csf_full = uncrop(csf)
    new[csf_full] = csf_label
    steps = {
        "ss_ventricles": ss_vent,
        "ventricles": vent,
        "brain_mask": uncrop(brain),
        "cortical_mask": uncrop(cortex),
        "cortical_dilated": uncrop(cortex_dil),
        "cortical_closed": uncrop(cortex_closed),
        "csf_dilation_only": uncrop(csf_dil),
        "holes_filled": uncrop(holes_kept),
        "holes_rejected": uncrop(holes_rejected),
        "csf_raw": uncrop(csf_raw),
        "csf": csf_full,
    }
    info = {"n_components": int(n), "n_components_kept": int(keep.sum())}
    return new, steps, info


def process_case(lab_path, ss_path, out_dir, params, save_intermediates):
    name = lab_path.name.replace(".nii.gz", "")
    t = time.time()
    img = nib.load(str(lab_path))
    labels = np.asarray(img.dataobj)
    spacing = img.header.get_zooms()[:3]
    ss_img = nib.load(str(ss_path))
    ss = np.rint(np.asarray(ss_img.dataobj)).astype(np.int32)

    new, steps, info = add_csf_combined(labels, img.affine, spacing, ss, ss_img.affine, **params)

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

    vox = float(np.prod(spacing)) / 1000
    wm_like = (labels > 0) & ~np.isin(labels, params["cortical_labels"])
    touch = ndi.binary_dilation(steps["csf"]) & wm_like  # non-cortical tissue voxels adjacent to CSF
    s = {"case": name,
         "ventricle_cm3": float(steps["ventricles"].sum() * vox),
         "csf_dilation_only_cm3": float(steps["csf_dilation_only"].sum() * vox),
         "csf_added_by_closing_cm3": float((steps["csf_raw"] & ~steps["csf_dilation_only"]
                                            & ~steps["holes_filled"]).sum() * vox),
         "csf_added_by_hole_filling_cm3": float(steps["holes_filled"].sum() * vox),
         "holes_rejected_noncortical_cm3": float(steps["holes_rejected"].sum() * vox),
         "csf_raw_cm3": float(steps["csf_raw"].sum() * vox),
         "csf_cm3": float(steps["csf"].sum() * vox),
         "csf_removed_small_objects_cm3": float((steps["csf_raw"] & ~steps["csf"]).sum() * vox),
         "noncortical_tissue_touching_csf_cm3": float(touch.sum() * vox),
         **info,
         "seconds": round(time.time() - t, 1)}
    print(f"[{name}] done in {s['seconds']}s: ventricles {s['ventricle_cm3']:.2f} cm3, CSF {s['csf_cm3']:.2f} cm3 "
          f"(+{s['csf_added_by_closing_cm3']:.2f} closing, +{s['csf_added_by_hole_filling_cm3']:.2f} holes, "
          f"-{s['csf_removed_small_objects_cm3']:.2f} small objects)", flush=True)
    return s


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    p.add_argument("--supersynth", type=Path, default=DEFAULT_SUPERSYNTH)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--tree", type=Path, default=DEFAULT_TREE, help="nxbsimplified tree (to find cortical labels)")
    p.add_argument("--extra-cortical-labels", type=int, nargs="*", default=list(EXTRA_CORTICAL_LABELS))
    p.add_argument("--cases", nargs="*", default=None)
    p.add_argument("--ss-ventricle-labels", type=int, nargs="+", default=list(SS_VENTRICLE_LABELS))
    p.add_argument("--vent-dilate-mm", type=float, default=0.6, help="dilation of the SuperSynth ventricles")
    p.add_argument("--dilate-mm", type=float, default=1.0, help="dilation of the cortical mask (CSF thickness)")
    p.add_argument("--close-cortex", action=argparse.BooleanOptionalAction, default=True,
                   help="closing of the cortical mask to fill wide sulci (--no-close-cortex to disable)")
    p.add_argument("--close-cortex-mm", type=float, default=3.0, help="closing radius of the cortical mask")
    p.add_argument("--fill-holes", action=argparse.BooleanOptionalAction, default=True,
                   help="enclosed background pockets become CSF (--no-fill-holes to disable)")
    p.add_argument("--max-noncortical-boundary", type=float, default=0.5,
                   help="pockets with a larger fraction of WM / deep GM boundary are not filled")
    p.add_argument("--min-component-frac", type=float, default=0.3,
                   help="objects smaller than this fraction of the largest lose their CSF")
    p.add_argument("--ventricle-label", type=int, default=VENTRICLE_LABEL)
    p.add_argument("--csf-label", type=int, default=CSF_LABEL)
    p.add_argument("--save-intermediates", action="store_true")
    p.add_argument("--workers", type=int, default=5)
    a = p.parse_args()

    cortical = sorted(set(cortical_labels_from_tree(a.tree)) | set(a.extra_cortical_labels))
    files = sorted(a.labels.glob("*.nii.gz"))
    if a.cases:
        files = [f for f in files if f.name.replace(".nii.gz", "") in a.cases]
    pairs = [(f, a.supersynth / f.name) for f in files]
    missing = [str(s) for _, s in pairs if not s.exists()]
    if missing or not pairs:
        raise SystemExit(f"nothing to do / missing SuperSynth files: {missing}")

    params = dict(cortical_labels=tuple(cortical), ss_ventricle_labels=tuple(a.ss_ventricle_labels),
                  vent_dilate_mm=a.vent_dilate_mm, dilate_mm_r=a.dilate_mm,
                  close_cortex=a.close_cortex, close_cortex_mm=a.close_cortex_mm,
                  fill_holes=a.fill_holes, max_noncortical_boundary=a.max_noncortical_boundary,
                  min_component_frac=a.min_component_frac, ventricle_label=a.ventricle_label,
                  csf_label=a.csf_label)
    print(f"{len(cortical)} cortical labels: {cortical}")
    a.output.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=min(a.workers, len(pairs))) as ex:
        futs = [ex.submit(process_case, f, s, a.output, params, a.save_intermediates) for f, s in pairs]
        summaries = [f.result() for f in futs]
    with open(a.output / "summary.json", "w") as fh:
        json.dump({"params": params, "cases": summaries}, fh, indent=2)
    print(f"saved {len(summaries)} cases to {a.output}")


if __name__ == "__main__":
    main()
