"""Map a NextBrain segmentation to grouped labels.

Usage:
    python next_brain_mapping.py -i seg.nii.gz -o out.nii.gz --mapping groups.yaml [--labels labels.json] [--unknown 0|same|error]

--mapping : YAML with groups of NextBrain labels, either
                Group-A: [7, 68]            or     Combined Structures:
                Group-B: [2001]                    - Group-A: [7, 68]
--labels  : optional JSON (or dict) {group_name: output_label}.
            If omitted, groups are numbered 0, 1, 2, ... in the order they appear.
--unknown : what to do with labels in the segmentation that are not in the mapping:
            a number (set them to it, default 0), "same" (keep their value) or "error" (stop).
"""

import argparse
import json

import nibabel as nib
import numpy as np
import yaml


def load_groups(yaml_path):
    """Return {group_name: [nextbrain_labels]} in file order."""
    with open(yaml_path) as f:
        data = yaml.safe_load(f)
    if len(data) == 1 and isinstance(next(iter(data.values())), list):  # "Combined Structures:" layout
        data = {k: v for g in next(iter(data.values())) for k, v in g.items()}
    return {name: labels or [] for name, labels in data.items()}


def map_segmentation(seg, groups, labels=None, unknown=0):
    """Replace each NextBrain label in `seg` by its group's output label.

    unknown: int, "same" or "error" -- handling of labels not in `groups`.
    """
    if labels is None:
        labels = {name: i for i, name in enumerate(groups)}
    seg = np.asarray(seg).astype(np.int64)
    mapped = {l for v in groups.values() for l in v}

    missing = sorted(set(np.unique(seg).tolist()) - mapped)
    if missing and unknown == "error":
        raise ValueError(f"Labels in segmentation but not in mapping: {missing}")

    lut = np.zeros(max(seg.max(), max(mapped)) + 1, dtype=np.int32)
    lut[missing] = missing if unknown == "same" else int(unknown)
    for name, nb_labels in groups.items():
        lut[nb_labels] = labels[name]
    return lut[seg]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-i", required=True, help="NextBrain segmentation")
    p.add_argument("-o", required=True, help="Output segmentation")
    p.add_argument("--mapping", required=True, help="YAML with groups of NextBrain labels")
    p.add_argument("--labels", help="JSON {group_name: output_label}")
    p.add_argument("--unknown", default="same", help='Number, "same" or "error" (default 0)')
    args = p.parse_args()

    groups = load_groups(args.mapping)
    labels = json.load(open(args.labels)) if args.labels else None
    unknown = args.unknown if args.unknown in ("same", "error") else int(args.unknown)

    img = nib.load(args.i)
    out = map_segmentation(np.asanyarray(img.dataobj), groups, labels, unknown)
    nib.save(nib.Nifti1Image(out, img.affine, img.header), args.o)


if __name__ == "__main__":
    main()
