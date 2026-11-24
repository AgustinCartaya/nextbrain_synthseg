"""Fill some labels in a NextBrain segmentation to nearest neighbors or specific values.

Usage:
    python nxb_label_filling.py -i seg.nii.gz -o out.nii.gz --mapping groups.yaml
                                [--margin-mm 2] [--background-competes] [--report report.json]

--mapping : YAML with groups of NextBrain labels:
                null:   [7, 68]     # filled with the label of the nearest kept region
                                    # (also accepted as  nn: [7, 68]  or  nearest: [7, 68])
                0:      [50]        # filled with 0
                4:      [33, 36]    # filled with 4
            The "Combined Structures:" list layout of nxb_label_mapping.py is also accepted.

Constant fills are applied first, so their new values count as kept regions for the
nearest-neighbor fill. The nearest-neighbor fill always looks at the segmentation as it was
before any nearest-neighbor filling: voxels of removed labels are never used as sources, even
after they have been filled. Each removed voxel therefore gets the label of the truly nearest
kept region, the result does not depend on the order of the labels in the YAML, and it is
equivalent to filling all removed labels at once (the removed area is split between its
neighboring kept regions like a Voronoi partition).
"""

import argparse
import json

import nibabel as nib
import numpy as np
import scipy.ndimage as ndi
import yaml

NN_KEYS = (None, "null", "none", "nn", "nearest")


def load_groups(yaml_path):
    """Return {target: [nextbrain_labels]} where target is None (nearest neighbor) or an int."""
    with open(yaml_path) as f:
        data = yaml.safe_load(f)
    # "Combined Structures:" layout: a single key holding a list of one-entry dicts
    if (len(data) == 1 and isinstance(next(iter(data.values())), list)
            and all(isinstance(g, dict) for g in next(iter(data.values())))):
        data = {k: v for g in next(iter(data.values())) for k, v in g.items()}

    groups = {}
    for key, labels in data.items():
        k = key.strip().lower() if isinstance(key, str) else key
        target = None if k in NN_KEYS else int(k)
        groups.setdefault(target, []).extend(int(l) for l in (labels or []))

    seen = {}
    for target, labels in groups.items():
        for l in labels:
            if l in seen and seen[l] != target:
                raise ValueError(f"Label {l} appears in two groups ({seen[l]} and {target})")
            seen[l] = target
    return groups


def fill_nearest(seg, removed_map, lab, bbox, zooms, margin_mm, background_competes):
    """
    Fill the voxels of one removed label with the label of the nearest kept region.

    seg          label map, the removed labels are already 0 (modified in place)
    removed_map  original label of the removed voxels, 0 elsewhere
    lab          the removed label to fill
    bbox         tuple of slices, bounding box of `lab` (from ndi.find_objects)
    Returns (number of voxels, {fill label: number of voxels}).
    """
    zooms = np.asarray(zooms, float)
    margin = margin_mm
    while True:
        pad = np.ceil(margin / zooms).astype(int)
        box = tuple(slice(max(0, s.start - p), min(n, s.stop + p))
                    for s, p, n in zip(bbox, pad, seg.shape))
        whole = all(b.start == 0 and b.stop == n for b, n in zip(box, seg.shape))
        sub = seg[box]
        rm = removed_map[box]
        target = rm == lab
        # kept regions: voxels that were not removed in the original segmentation
        # (+ background if it competes). Voxels of removed labels that were already filled
        # are never sources, so the result does not depend on the filling order.
        known = (rm == 0) if background_competes else ((sub > 0) & (rm == 0))
        if known.any():
            # EDT of "not known": distance (mm) to the nearest known voxel and its index
            dist, ind = ndi.distance_transform_edt(~known, sampling=zooms, return_indices=True)
            d_max = float(dist[target].max())
            # every voxel of the label is >= margin from the box border, so a nearest voxel
            # found at distance <= margin is the true (whole-volume) nearest voxel
            if d_max <= margin or whole:
                values = sub[tuple(i[target] for i in ind)]
                sub[target] = values
                fill, cnt = np.unique(values, return_counts=True)
                return int(target.sum()), {int(f): int(c) for f, c in zip(fill, cnt)}
        elif whole:
            return int(target.sum()), {}         # nothing kept in the whole volume
        margin *= 2


def fill_labels(seg, groups, zooms=(1, 1, 1), margin_mm=2.0, background_competes=False):
    """Return a copy of `seg` with the labels in `groups` filled, and a report dict.

    groups: {target: [labels]}, target None = nearest kept region, int = that value.
    """
    seg = np.asarray(seg)
    if np.issubdtype(seg.dtype, np.floating):
        seg = np.rint(seg).astype(np.int32)
    seg = seg.copy()
    report = {"constant": {}, "nearest": {}}

    # 1. constant fills
    for target, labels in groups.items():
        if target is None:
            continue
        for lab in labels:
            m = seg == lab
            if m.any():
                seg[m] = target
                report["constant"][str(lab)] = dict(voxels=int(m.sum()), filled_with=target)

    # 2. nearest-neighbor fills
    nn_labels = [l for l in groups.get(None, []) if l > 0]
    if nn_labels:
        removed_any = np.isin(seg, nn_labels)
        removed_map = np.where(removed_any, seg, 0).astype(seg.dtype)
        seg[removed_any] = 0
        bboxes = ndi.find_objects(removed_map)   # bboxes[l - 1] = bbox of label l (or None)
        for lab in nn_labels:
            if lab > len(bboxes) or bboxes[lab - 1] is None:
                continue
            n_vox, filled = fill_nearest(seg, removed_map, lab, bboxes[lab - 1], zooms,
                                         margin_mm, background_competes)
            report["nearest"][str(lab)] = dict(voxels=n_vox, filled_with=filled)
        report["still_unfilled"] = int((removed_any & (seg == 0)).sum())
    return seg, report


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-i", required=True, help="NextBrain segmentation")
    p.add_argument("-o", required=True, help="Output segmentation")
    p.add_argument("--mapping", required=True, help="YAML with groups of NextBrain labels")
    p.add_argument("--margin-mm", type=float, default=2.0,
                   help="initial margin around each label bbox (default 2 mm, doubled if needed)")
    p.add_argument("--background-competes", action="store_true",
                   help="background (0) also competes in the nearest-region fill")
    p.add_argument("--report", help="optional JSON report of what was filled with what")
    args = p.parse_args()

    groups = load_groups(args.mapping)

    img = nib.load(args.i)
    out, report = fill_labels(np.asanyarray(img.dataobj), groups, img.header.get_zooms()[:3],
                              args.margin_mm, args.background_competes)
    nib.save(nib.Nifti1Image(out.astype(img.get_data_dtype()), img.affine, img.header), args.o)
    if args.report:
        with open(args.report, "w") as f:
            json.dump(report, f, indent=1)


if __name__ == "__main__":
    main()
