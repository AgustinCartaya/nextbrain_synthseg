"""Map a NextBrain segmentation to grouped labels.

Usage:
    python nxb_label_mapping.py -i seg.nii.gz -o out.nii.gz --mapping groups.yaml [--labels labels.json] [--unknown 0|same|error]

--mapping : "nxb6" / "nxb16" (built-in mappings in mapping/, labels numbered 0..N in file order),
            "nxbsupersynth" (SuperSynth ids of the right side, see nxb_supersynth_label_mapping.py), or
            a YAML with groups of NextBrain labels, either
                Group-A: [7, 68]            or     Combined Structures:
                Group-B: [2001]                    - Group-A: [7, 68]
            The presets expect a simplified NextBrain segmentation: if it contains any label of
            mapping/labels_to_remove.yaml, those labels are first filled with nxb_label_filling.py
            (nearest kept region), then the mapping is applied.
--labels  : optional JSON (or dict) {group_name: output_label}.
            If omitted, groups are numbered 0, 1, 2, ... in the order they appear.
--unknown : what to do with labels in the segmentation that are not in the mapping:
            a number (set them to it, default 0), "same" (keep their value) or "error" (stop).
"""

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import yaml

try:
    from .nxb_label_filling import fill_labels, load_groups as load_fill_groups
    from .nxb_supersynth_label_mapping import map_segmentation as supersynth_map_segmentation
except ImportError:
    from nxb_label_filling import fill_labels, load_groups as load_fill_groups
    from nxb_supersynth_label_mapping import map_segmentation as supersynth_map_segmentation


MAPPING_DIR = Path(__file__).resolve().parent / "mapping"
PRESETS = {
    "nxb6": MAPPING_DIR / "nextbrain_to_nxb6.yaml",
    "nxb16": MAPPING_DIR / "nextbrain_to_nxb16.yaml",
    "nxbsupersynth": MAPPING_DIR / "nextbrain_to_nxbsupersynth.yaml",
}
LABELS_TO_REMOVE = MAPPING_DIR / "labels_to_remove.yaml"


def is_simplified(seg):
    """Return True if `seg` does not contain any label of labels_to_remove.yaml."""
    to_remove = [l for v in load_fill_groups(LABELS_TO_REMOVE).values() for l in v]
    return not np.isin(seg, to_remove).any()


def simplify(seg, zooms):
    """Fill the labels of labels_to_remove.yaml with nxb_label_filling.

    zooms: voxel size (mm) of `seg`, e.g. img.header.get_zooms()[:3] (required: the nearest
           region depends on it if the image is not isotropic).
    """
    if zooms is None:
        raise ValueError("zooms are needed because the nextbrain segmentation is not simplified (e.g. zooms=img.header.get_zooms()[:3] or zooms=(1, 1, 1) for isotropic)")
    return fill_labels(seg, load_fill_groups(LABELS_TO_REMOVE), zooms)[0]


def load_groups(yaml_path):
    """Return {group_name: [nextbrain_labels]} in file order.

    yaml_path: path to a YAML file or a preset name ("nxb6", "nxb16", "nxbsupersynth").
    """
    yaml_path = PRESETS.get(str(yaml_path), yaml_path)
    with open(yaml_path) as f:
        data = yaml.safe_load(f)
    # "Combined Structures:" layout: a single key holding a list of one-entry dicts
    if (len(data) == 1 and isinstance(next(iter(data.values())), list)
            and all(isinstance(g, dict) for g in next(iter(data.values())))):
        data = {k: v for g in next(iter(data.values())) for k, v in g.items()}
    return {name: labels or [] for name, labels in data.items()}


def map_segmentation(seg, groups, labels=None, unknown=0, zooms=None):
    """Replace each NextBrain label in `seg` by its group's output label.

    groups: {group_name: [nextbrain_labels]}, a YAML path or a preset name
            ("nxb6", "nxb16", "nxbsupersynth").
    unknown: int, "same" or "error" -- handling of labels not in `groups`.
    zooms: voxel size (mm) of `seg`, e.g. img.header.get_zooms()[:3]. Only needed when `groups`
           is a preset and `seg` is not simplified (contains labels of labels_to_remove.yaml,
           which are then filled first with the nearest kept region); a ValueError is raised
           if it is needed and missing.
    """
    seg = np.rint(np.asarray(seg)).astype(np.int64)
    if not isinstance(groups, dict):
        if str(groups) in PRESETS and not is_simplified(seg):
            seg = simplify(seg, zooms)
        if str(groups) == "nxbsupersynth":    # fixed SuperSynth ids (right side), no `labels`
            if labels is not None:
                raise ValueError("`labels` cannot be used with the nxbsupersynth preset")
            return supersynth_map_segmentation(seg, unknown=unknown)
        groups = load_groups(groups)
    if labels is None:
        labels = {name: i for i, name in enumerate(groups)}
    mapped = {l for v in groups.values() for l in v}

    missing = sorted(set(np.unique(seg).tolist()) - mapped)
    if missing and unknown == "error":
        raise ValueError(f"Labels in segmentation but not in mapping: {missing}")

    lut = np.zeros(max(seg.max(), max(mapped)) + 1, dtype=np.int32)
    if missing:
        lut[missing] = missing if unknown == "same" else int(unknown)
    for name, nb_labels in groups.items():
        lut[nb_labels] = labels[name]
    return lut[seg]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-i", required=True, help="NextBrain segmentation")
    p.add_argument("-o", required=True, help="Output segmentation")
    p.add_argument("--mapping", required=True, help='"nxb6", "nxb16", "nxbsupersynth" or a YAML with groups of NextBrain labels')
    p.add_argument("--labels", help="JSON {group_name: output_label}")
    p.add_argument("--unknown", default="0", help='Number, "same" or "error" (default 0)')
    args = p.parse_args()

    groups = args.mapping if args.mapping in PRESETS else load_groups(args.mapping)
    labels = json.load(open(args.labels)) if args.labels else None
    unknown = args.unknown if args.unknown in ("same", "error") else int(args.unknown)

    img = nib.load(args.i)
    out = map_segmentation(np.asanyarray(img.dataobj), groups, labels, unknown,
                           zooms=img.header.get_zooms()[:3])
    nib.save(nib.Nifti1Image(out, img.affine, img.header), args.o)


if __name__ == "__main__":
    main()
