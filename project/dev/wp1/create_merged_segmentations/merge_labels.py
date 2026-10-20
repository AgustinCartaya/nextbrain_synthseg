"""Collapse NextBrain segmentation labels into their ontology parents.

Every leaf label (allen_id) is replaced by the index assigned to its ancestor
at the selected ontology level, e.g. at level L1 all labels under "Deep GM"
become a single value.

Levels map to these columns of the ontology CSV:
    L1 -> L1_tissue, L2 -> L2_division, L3 -> L3_structure,
    L4 -> L4_subregion, L5 -> leaf_name

Usage (python):
    from merge_labels import merge_folder
    merge_folder(
        input_dir="data/inputs/recon_0.2mm",
        output_dir="data/outputs/recon_0.2mm_L1",
        ontology_csv="data/inputs/nextbrain_ontology_v1.csv",
        level="L1",
        mapping={"Background": 0, "Deep GM": 1, "Cerebral GM": 2, ...},
    )

Usage (CLI):
    python merge_labels.py --level L1 --mapping '{"Deep GM": 1, "Cerebral GM": 2}' \
        --input data/inputs/recon_0.2mm --output data/outputs/recon_0.2mm_L1
"""
import argparse
import json
import os
import warnings

import nibabel as nib
import numpy as np
import pandas as pd

LEVEL_COLUMNS = {
    "L1": "L1_tissue",
    "L2": "L2_division",
    "L3": "L3_structure",
    "L4": "L4_subregion",
    "L5": "leaf_name",
}


def load_ontology(ontology_csv, extra_labels=None):
    """Load the ontology, optionally adding rows for labels the CSV lacks.

    extra_labels: list of dicts with at least allen_id, leaf_name and the L1-L4 columns.
    """
    df = pd.read_csv(ontology_csv)
    if extra_labels:
        clash = set(row["allen_id"] for row in extra_labels) & set(df["allen_id"])
        if clash:
            raise ValueError(f"extra_labels ids already in the ontology: {sorted(clash)}")
        df = pd.concat([df, pd.DataFrame(extra_labels)], ignore_index=True)
    df["allen_id"] = df["allen_id"].astype(int)
    return df


def valid_names(ontology, level):
    """Return the sorted list of node names available at `level`."""
    return sorted(ontology[_level_column(level)].dropna().unique())


def _level_column(level):
    if level not in LEVEL_COLUMNS:
        raise ValueError(f"level must be one of {list(LEVEL_COLUMNS)}, got {level!r}")
    return LEVEL_COLUMNS[level]


def build_label_map(ontology, level, mapping):
    """Return {allen_id: new_index} for every leaf whose ancestor at `level` is in `mapping`.

    Raises if any name in `mapping` is not a node of `level`.
    """
    column = _level_column(level)
    names = set(ontology[column].dropna())
    invalid = [name for name in mapping if name not in names]
    if invalid:
        raise ValueError(
            f"Names not found at level {level} ({column}): {invalid}\n"
            f"Valid names: {sorted(names)}"
        )
    for name, index in mapping.items():
        if not isinstance(index, (int, np.integer)) or index < 0:
            raise ValueError(f"Index for {name!r} must be a non-negative int, got {index!r}")

    rows = ontology[ontology[column].isin(mapping.keys())]
    return {int(a): int(mapping[p]) for a, p in zip(rows["allen_id"], rows[column])}


def build_lut(ontology, label_map, max_label, unmapped_value=0, unknown_value=None):
    """Build a lookup table so that merged = lut[segmentation].

    unmapped_value: value for ontology labels whose parent is not in the mapping.
    unknown_value:  value for labels missing from the ontology (None keeps them unchanged).
    """
    size = max(max_label, int(ontology["allen_id"].max())) + 1
    lut = np.arange(size, dtype=np.int64) if unknown_value is None else np.full(size, unknown_value, dtype=np.int64)
    lut[ontology["allen_id"].to_numpy()] = unmapped_value
    for allen_id, index in label_map.items():
        lut[allen_id] = index
    return lut


def merge_segmentation(seg, ontology, label_map, unmapped_value=0, unknown_value=None):
    """Return (merged array, list of labels present in `seg` but missing from the ontology)."""
    seg = np.asarray(seg)
    if not np.issubdtype(seg.dtype, np.integer):
        if not np.array_equal(seg, np.round(seg)):
            raise ValueError("Segmentation contains non-integer labels")
        seg = seg.astype(np.int64)
    present = np.unique(seg)
    if present.min() < 0:
        raise ValueError("Segmentation contains negative labels")
    unknown = sorted(set(present.tolist()) - set(ontology["allen_id"].tolist()))
    lut = build_lut(ontology, label_map, int(present.max()), unmapped_value, unknown_value)
    return lut[seg], unknown


def _output_dtype(max_value):
    for dtype in (np.uint8, np.uint16, np.int32):
        if max_value <= np.iinfo(dtype).max:
            return dtype
    return np.int64


def write_complete_mapping(path, ontology, level, mapping, unknown_ids=(),
                           unmapped_value=0, unknown_value=None):
    """Write a CSV with one row per original label: original_id, original_name, new_id, new_name."""
    column = _level_column(level)
    names_by_index = {}
    for name, index in mapping.items():
        names_by_index.setdefault(index, []).append(name)

    def new_name(index, default):
        return " + ".join(names_by_index[index]) if index in names_by_index else default

    rows = []
    for allen_id, leaf, parent in zip(ontology["allen_id"], ontology["leaf_name"], ontology[column]):
        if parent in mapping:
            rows.append((allen_id, mapping[parent], leaf, parent))
        else:
            rows.append((allen_id, unmapped_value, leaf, new_name(unmapped_value, "unmapped")))
    for allen_id in unknown_ids:
        index = allen_id if unknown_value is None else unknown_value
        rows.append((allen_id, index, "unknown (not in ontology)", new_name(index, "unknown")))

    df = pd.DataFrame(rows, columns=["original_id", "new_id", "original_name", "new_name"])
    df.sort_values(["new_id", "original_id"]).to_csv(path, index=False)


def merge_folder(input_dir, output_dir, ontology_csv, level, mapping,
                 unmapped_value=0, unknown_value=None, extra_labels=None):
    """Merge every .nii/.nii.gz segmentation in `input_dir` and write it to `output_dir`."""
    ontology = load_ontology(ontology_csv, extra_labels)
    label_map = build_label_map(ontology, level, mapping)

    unmapped = sorted(set(valid_names(ontology, level)) - set(mapping))
    if unmapped:
        print(f"[info] Level {level} nodes not in mapping (-> {unmapped_value}): {unmapped}")

    os.makedirs(output_dir, exist_ok=True)
    files = sorted(f for f in os.listdir(input_dir) if f.endswith((".nii", ".nii.gz")))
    if not files:
        raise FileNotFoundError(f"No NIfTI files found in {input_dir}")

    all_unknown = set()
    for fname in files:
        img = nib.load(os.path.join(input_dir, fname))
        merged, unknown = merge_segmentation(
            np.asarray(img.dataobj), ontology, label_map, unmapped_value, unknown_value
        )
        all_unknown.update(unknown)
        if unknown:
            action = "kept unchanged" if unknown_value is None else f"set to {unknown_value}"
            warnings.warn(f"{fname}: labels not in ontology {unknown} were {action}")

        merged = merged.astype(_output_dtype(int(merged.max())))
        header = img.header.copy()
        header.set_data_dtype(merged.dtype)
        header.set_slope_inter(1, 0)
        nib.save(nib.Nifti1Image(merged, img.affine, header), os.path.join(output_dir, fname))
        print(f"[ok] {fname}: labels {sorted(np.unique(merged).tolist())}")

    with open(os.path.join(output_dir, "mapping.json"), "w") as f:
        json.dump({"level": level, "column": LEVEL_COLUMNS[level], "mapping": mapping,
                   "unmapped_value": unmapped_value, "unknown_value": unknown_value,
                   "extra_labels": extra_labels or []}, f, indent=2)
    write_complete_mapping(os.path.join(output_dir, "complete_mapping.csv"), ontology, level, mapping,
                           sorted(all_unknown), unmapped_value, unknown_value)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", default=os.path.join(here, "data/inputs/recon_0.2mm"))
    parser.add_argument("--output", default=None, help="Default: data/outputs/<input name>_<level>")
    parser.add_argument("--ontology", default=os.path.join(here, "data/inputs/nextbrain_ontology_v1.csv"))
    parser.add_argument("--level", required=True, choices=list(LEVEL_COLUMNS))
    parser.add_argument("--mapping", help="JSON dict string or path to a JSON file: {\"parent name\": index}")
    parser.add_argument("--unmapped-value", type=int, default=0,
                        help="Value for ontology labels whose parent is not in the mapping")
    parser.add_argument("--unknown-value", type=int, default=None,
                        help="Value for labels not in the ontology (default: keep unchanged)")
    parser.add_argument("--list-names", action="store_true", help="Print valid names for --level and exit")
    args = parser.parse_args()

    if args.list_names:
        print("\n".join(valid_names(load_ontology(args.ontology), args.level)))
        return
    if args.mapping is None:
        parser.error("--mapping is required")

    if os.path.isfile(args.mapping):
        with open(args.mapping) as f:
            mapping = json.load(f)
    else:
        mapping = json.loads(args.mapping)

    output = args.output or os.path.join(
        here, "data/outputs", f"{os.path.basename(os.path.normpath(args.input))}_{args.level}")
    merge_folder(args.input, output, args.ontology, args.level, mapping,
                 args.unmapped_value, args.unknown_value)


if __name__ == "__main__":
    main()
