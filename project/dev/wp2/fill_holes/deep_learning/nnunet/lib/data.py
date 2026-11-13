"""
Scan index (filter, subject split, sampling weights) and the patch dataset that
builds (corrupted, clean) pairs on the fly.
"""
import hashlib
import math
import warnings

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .corruption import hole_channel, sample_corruption_args
from .external import corrupt_module
from .labels import VolumeStore, shift_affine, upsample_factor, upsample_labels, upsampled_affine


# =============================================================================
# Index
# =============================================================================
def subject_split(dataset, subject, seed, val_fraction, test_fraction):
    """'train' | 'val' | 'test' for one subject.

    The split is a hash of (seed, dataset, subject): all sessions and both
    hemispheres of a subject fall in the same split, and a subject keeps its
    split when the index is rebuilt with another resolution threshold.
    """
    digest = hashlib.md5(f"{seed}:{dataset}/{subject}".encode()).hexdigest()
    u = int(digest[:12], 16) / 16 ** 12
    if u < test_fraction:
        return "test"
    if u < test_fraction + val_fraction:
        return "val"
    return "train"


def load_items(cfg, split):
    """Hemispheres of one split, with their sampling weights.

    Reads paths.index_csv (written by 01_build_index.py) and applies
    data.max_voxel_mm again, so a stricter threshold only needs a config change.

    Returns (items, weights):
      items   list of dicts (dataset, subject, session, hemi, nextbrain_dir, max_vox)
      weights array summing to 1: (1 / max_vox) ** data.weight_gamma, so that
              sub-millimetre scans are drawn more often
    """
    d = cfg["data"]
    df = pd.read_csv(cfg["paths"]["index_csv"])
    df = df[(df["split"] == split) & (df["max_vox"] <= d["max_voxel_mm"] + d["voxel_tolerance"])]
    items = []
    for row in df.itertuples(index=False):
        for hemi in d["hemispheres"]:
            if getattr(row, f"has_{hemi}"):
                items.append(dict(dataset=row.dataset, subject=row.subject, session=row.session,
                                  hemi=hemi, nextbrain_dir=row.nextbrain_dir,
                                  max_vox=float(row.max_vox)))
    if not items:
        raise RuntimeError(f"No '{split}' hemispheres in {cfg['paths']['index_csv']}")
    weights = np.array([(1.0 / it["max_vox"]) ** d["weight_gamma"] for it in items])
    return items, weights / weights.sum()


# =============================================================================
# Patches
# =============================================================================
def crop_pad(volume, start, size):
    """Cube of side `size` starting at voxel `start` (may lie partly outside: zero padded)."""
    out = np.zeros((size,) * 3, volume.dtype)
    src, dst = [], []
    for s, n in zip(start, volume.shape):
        lo, hi = max(int(s), 0), min(int(s) + size, n)
        if hi <= lo:
            return out
        src.append(slice(lo, hi))
        dst.append(slice(lo - int(s), hi - int(s)))
    out[tuple(dst)] = volume[tuple(src)]
    return out


class RestorationPatches(Dataset):
    """(corrupted patch, hole channel, clean patch) samples generated on the fly.

    One sample (all on one CPU core, ~3.5 s for a 256^3 context):
      1. pick a hemisphere (weighted by resolution) and a patch centre
      2. crop a context region at the native ~0.4 mm (context / factor voxels)
      3. upsample it to 0.2 mm: linear interpolation of one-hot channels + argmax
      4. corrupt the context crop with `corrupt_array`, using the affine of the
         crop, so that the section geometry is defined in world coordinates
      5. compute the hole channel on the corrupted context crop
      6. keep the central patch (the margin absorbs border effects of 4 and 5)

    split:         'train' | 'val' | 'test'
    length:        number of samples of one "epoch" (the data is random, so this
                   only sets how many items a DataLoader yields)
    deterministic: True -> sample i is always the same (seeded by `seed` and i),
                   used for validation and evaluation. False -> fresh random
                   samples from OS entropy (different in every worker and epoch).

    Each item is a dict of tensors:
      x     [P, P, P] corrupted labels (uint8)        -> network input (one-hot on the GPU)
      hole  [P, P, P] suspicious-hole channel (uint8)
      y     [P, P, P] clean labels (uint8)            -> target
      clean 1 if the patch was left uncorrupted
      severity  severity level in [0, 1] (-1 for clean patches)
    """

    def __init__(self, cfg, split, length, deterministic=False, seed=0):
        self.cfg = cfg
        self.items, self.weights = load_items(cfg, split)
        self.length = int(length)
        self.deterministic = deterministic
        self.seed = int(seed)
        self.store = VolumeStore(cfg)
        self.patch = int(cfg["patch"]["size"])
        self.context = int(cfg["patch"]["context"])
        if self.context < self.patch:
            raise ValueError("patch.context must be >= patch.size")
        # fixed order of volumes for the deterministic mode (uniform over hemispheres)
        self.order = np.random.default_rng(self.seed).permutation(len(self.items))

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        if self.deterministic:
            rng = np.random.default_rng([self.seed, i])
            return self.make_sample(self.items[self.order[i % len(self.items)]], rng)
        rng = np.random.default_rng()
        for attempt in range(5):               # an unreadable file must not stop the training
            item = self.items[int(rng.choice(len(self.items), p=self.weights))]
            try:
                return self.make_sample(item, rng)
            except Exception as e:             # noqa: BLE001
                warnings.warn(f"Skipping {item['nextbrain_dir']} ({item['hemi']}): {e!r}")
        raise RuntimeError("5 consecutive samples failed, see the warnings above")

    # -------------------------------------------------------------------------
    def context_crop(self, item, rng):
        """Steps 1-3: clean context crop at the target resolution and its affine."""
        d = self.cfg["data"]
        seg, affine = self.store.load(item)
        factor = upsample_factor(affine, d["target_spacing"])
        n_low = math.ceil(self.context / factor)          # context size at native resolution

        shape = np.array(seg.shape)
        centre = rng.integers(0, shape)
        if rng.random() < d["foreground_center_prob"]:
            for _ in range(100):                          # rejection sampling of a labelled voxel
                if seg[tuple(centre)] > 0:
                    break
                centre = rng.integers(0, shape)
        start = centre - n_low // 2
        clean = upsample_labels(crop_pad(seg, start, n_low), factor)
        return clean, upsampled_affine(shift_affine(affine, start), factor)

    def make_sample(self, item, rng):
        cn = corrupt_module(self.cfg)
        ccfg = self.cfg["corruption"]
        clean, affine = self.context_crop(item, rng)

        # step 4: corruption (a fraction of the patches is kept intact)
        keep_clean = rng.random() < ccfg["p_clean"] or not clean.any()
        severity = -1.0
        if keep_clean:
            corrupted = clean
        else:
            angle_range, axis, kw, info = sample_corruption_args(rng, ccfg)
            severity = info["severity"]
            corrupted, _, _ = cn.corrupt_array(clean, affine, angle_range, axis, rng,
                                               brain=clean > 0, fg_box=cn.bbox(clean > 0), **kw)
        # step 5: hole channel, from the corrupted labels only
        hole = hole_channel(cn, corrupted)

        # step 6: central patch (+ left-right mirror, applied to all three alike)
        m = (clean.shape[0] - self.patch) // 2
        sl = (slice(m, m + self.patch),) * 3
        flip = rng.random() < self.cfg["data"]["flip_lr_prob"]

        def out(a):
            a = a[sl]
            if flip:
                a = a[::-1]
            # uint8 up to 256 classes, int16 above (finer label levels)
            dtype = np.uint8 if self.cfg["labels"]["num_classes"] <= 256 else np.int16
            return torch.from_numpy(np.ascontiguousarray(a).astype(dtype))

        return dict(x=out(corrupted), hole=out(hole), y=out(clean),
                    clean=torch.tensor(int(keep_clean)), severity=torch.tensor(severity))


def worker_init(_):
    """DataLoader worker setup: one thread per worker (the workers are the parallelism)."""
    torch.set_num_threads(1)
