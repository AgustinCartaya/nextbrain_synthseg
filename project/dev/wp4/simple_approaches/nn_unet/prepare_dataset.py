"""Link the generated dataset into nnUNet_raw and write a template-grouped 5-fold split.

The 1000 synthetic cases come from 5 anatomical templates (case-1..case-5, ~200 each).
A random split would put images of the same anatomy in train and val, so instead each
fold holds out one template (leave-one-template-out). Use --random_split to keep nnU-Net's
default random 5-fold split.
"""
import argparse
import json
import os
from collections import defaultdict
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", default=os.environ.get("SOURCE_DATASET"))
    p.add_argument("--name", default=os.environ.get("DATASET_NAME", "Dataset001_NextBrain_L1"))
    p.add_argument("--random_split", action="store_true")
    args = p.parse_args()

    src = Path(args.source)
    raw = Path(os.environ["nnUNet_raw"]) / args.name
    pre = Path(os.environ["nnUNet_preprocessed"]) / args.name
    raw.mkdir(parents=True, exist_ok=True)

    # Symlink instead of copying (imagesTr is ~55 GB)
    for item in ("imagesTr", "labelsTr", "dataset.json"):
        dst = raw / item
        if dst.is_symlink() or dst.exists():
            dst.unlink()
        dst.symlink_to(src / item)
    print(f"Linked {src} -> {raw}")

    n_images = len(list((src / "imagesTr").glob("*_0000.nii.gz")))
    n_labels = len(list((src / "labelsTr").glob("*.nii.gz")))
    n_declared = json.load(open(src / "dataset.json"))["numTraining"]
    assert n_images == n_labels == n_declared, (n_images, n_labels, n_declared)

    if args.random_split:
        print("Using nnU-Net default random 5-fold split")
        return

    by_template = defaultdict(list)
    for spec in sorted((src / "specs").glob("*.json")):
        s = json.load(open(spec))
        by_template[Path(s["template"]).name.split(".")[0]].append(s["case"])

    templates = sorted(by_template)
    all_cases = sorted(c for cs in by_template.values() for c in cs)
    splits = []
    for t in templates:
        val = sorted(by_template[t])
        splits.append({"train": [c for c in all_cases if c not in set(val)], "val": val})
        print(f"fold {len(splits) - 1}: val = {t} ({len(val)} cases), train = {len(all_cases) - len(val)}")

    pre.mkdir(parents=True, exist_ok=True)
    with open(pre / "splits_final.json", "w") as f:
        json.dump(splits, f, indent=2)
    with open(pre / "splits_templates.json", "w") as f:
        json.dump({i: t for i, t in enumerate(templates)}, f, indent=2)
    print(f"Wrote {pre / 'splits_final.json'}")


if __name__ == "__main__":
    main()
