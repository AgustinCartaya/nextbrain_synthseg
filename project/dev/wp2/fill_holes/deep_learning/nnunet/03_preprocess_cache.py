#!/usr/bin/env python
"""
Step 3 (optional) — fill the cache of merged training labels in parallel.

For every hemisphere of the index: read seg.<hemi>.nii.gz, merge the labels
(e.g. to L1), crop to the bounding box and save a small uint8 NIfTI under
paths.cache_dir. The dataset does the same on first use (data.write_cache), so
this step only moves that cost before the training (~1.5 MB per hemisphere).

The FOMO files are only read.

  python 03_preprocess_cache.py --config configs/default.yaml --workers 64
"""
from multiprocessing import Pool

from lib.config import config_from_args, parser
from lib.data import load_items
from lib.labels import VolumeStore

_store = None


def _init(cfg):
    global _store
    cfg["data"]["write_cache"] = True
    _store = VolumeStore(cfg)


def _one(item):
    """Cache one hemisphere; returns an error message or None."""
    try:
        _store.load(item)
        return None
    except Exception as e:                                   # noqa: BLE001
        return f"{item['nextbrain_dir']} ({item['hemi']}): {e!r}"


def main():
    p = parser(__doc__)
    p.add_argument("--workers", type=int, default=32)
    p.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    args = p.parse_args()
    cfg = config_from_args(args)

    items = []
    for split in args.splits:
        try:
            items += load_items(cfg, split)[0]
        except RuntimeError as e:                            # an empty split is not an error here
            print(e)
    print(f"{len(items)} hemispheres -> {cfg['paths']['cache_dir']}")
    errors = []
    with Pool(args.workers, initializer=_init, initargs=(cfg,)) as pool:
        for i, err in enumerate(pool.imap_unordered(_one, items, chunksize=4)):
            if err:
                errors.append(err)
            if (i + 1) % 500 == 0 or i + 1 == len(items):
                print(f"  {i + 1}/{len(items)}  errors: {len(errors)}", flush=True)
    for err in errors:
        print("[error]", err)


if __name__ == "__main__":
    main()
