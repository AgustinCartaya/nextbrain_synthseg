"""
Label handling: ontology merge (e.g. 257 NextBrain labels -> 9 L1 tissues),
one-hot linear upsampling, and cached loading of the FOMO training volumes.
"""
import json
import os
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F

from .external import merge_module

#: nothing in this project may write below this folder
READ_ONLY_ROOT = "/home/Data"


def assert_writable(path):
    """Raise if `path` is inside the read-only data folder (safety net for every writer)."""
    resolved = os.path.realpath(path)
    if resolved == READ_ONLY_ROOT or resolved.startswith(READ_ONLY_ROOT + os.sep):
        raise PermissionError(f"Refusing to write inside {READ_ONLY_ROOT}: {path}")
    return path


def label_dtype(num_classes):
    """Smallest unsigned integer type that holds the class indices."""
    return np.uint8 if num_classes <= 256 else np.uint16


# =============================================================================
# Merge
# =============================================================================
class LabelMerger:
    """Collapse leaf labels into the classes of one ontology level, with a lookup table.

    Uses merge_labels.py as a library (`load_ontology`, `build_label_map`,
    `build_lut`), so the result is the same as its CLI. Only the treatment of
    labels missing from the ontology differs: they go to 0 (background) unless
    listed in `extra`, and are reported, instead of being kept unchanged (which
    would create values outside the class range).
    """

    def __init__(self, cfg):
        ml = merge_module(cfg)
        ontology = ml.load_ontology(cfg["paths"]["ontology_csv"])
        with open(cfg["paths"]["mapping_json"]) as f:
            mapping = json.load(f)
        label_map = ml.build_label_map(ontology, cfg["labels"]["level"], mapping)
        self.num_classes = int(cfg["labels"]["num_classes"])
        self.dtype = label_dtype(self.num_classes)
        self.lut = ml.build_lut(ontology, label_map, 0, unmapped_value=0, unknown_value=0)
        self.known = np.zeros(len(self.lut), bool)
        self.known[ontology["allen_id"].to_numpy()] = True
        if self.lut.max() >= self.num_classes:
            raise ValueError(f"The mapping produces index {self.lut.max()} but "
                             f"labels.num_classes is {self.num_classes}")

    def __call__(self, seg, extra=None):
        """Merge one label array.

        seg:   integer array, or float array holding integer values
        extra: optional {raw label: merged index} for labels not in the ontology
        Returns (merged array, sorted list of labels that were unknown and set to 0).
        """
        seg = np.asarray(seg)
        if not np.issubdtype(seg.dtype, np.integer):
            seg = np.rint(seg).astype(np.int32)
        if seg.min() < 0:
            raise ValueError("Segmentation contains negative labels")
        lut, known = self.lut, self.known
        top = int(seg.max())
        if top >= len(lut):                       # labels above the largest ontology id
            lut = np.concatenate([lut, np.zeros(top + 1 - len(lut), lut.dtype)])
            known = np.concatenate([known, np.zeros(top + 1 - len(known), bool)])
        if extra:
            lut, known = lut.copy(), known.copy()
            for raw, new in extra.items():
                if int(raw) < len(lut) and not known[int(raw)]:
                    lut[int(raw)] = int(new)
                    known[int(raw)] = True
        present = np.flatnonzero(np.bincount(seg.ravel(), minlength=len(lut)))
        unknown = [int(v) for v in present if not known[v]]
        return lut.astype(self.dtype)[seg], unknown


# =============================================================================
# Resolution
# =============================================================================
def upsample_factor(affine, target_spacing):
    """Integer factor that brings the voxel size of `affine` closest to `target_spacing`."""
    spacing = np.sqrt((np.asarray(affine)[:3, :3] ** 2).sum(0)).mean()
    return max(1, int(round(spacing / target_spacing)))


def upsampled_affine(affine, factor):
    """Affine of the grid produced by `upsample_labels`.

    New voxel i is centred at old index (i + 0.5) / factor - 0.5 (the convention
    of `F.interpolate(..., align_corners=False)`): both grids cover the same box.
    """
    scale = np.eye(4)
    scale[:3, :3] /= factor
    scale[:3, 3] = (1.0 / factor - 1.0) / 2.0
    return np.asarray(affine, float) @ scale


def upsample_labels(seg, factor):
    """Upsample a label array by an integer factor: linear interpolation of every
    one-hot channel followed by argmax (smooth borders, unlike nearest neighbour).

    Channels are interpolated one at a time keeping a running maximum, so the
    memory is 2 float volumes instead of num_classes. Only labels present in
    `seg` are processed. Ties go to the lowest label.
    """
    if factor == 1:
        return seg.copy()
    src = torch.from_numpy(np.ascontiguousarray(seg))
    out, best = None, None
    for c in np.flatnonzero(np.bincount(seg.ravel())):
        up = F.interpolate((src == int(c))[None, None].float(), scale_factor=factor,
                           mode="trilinear", align_corners=False)[0, 0]
        if best is None:
            best = up
            out = torch.full(up.shape, int(c), dtype=src.dtype)
        else:
            out[up > best] = int(c)
            best = torch.maximum(best, up)
    return out.numpy()


def shift_affine(affine, start):
    """Affine of a crop that starts at voxel `start` of the original grid."""
    out = np.array(affine, float)
    out[:3, 3] = out[:3, :3] @ np.asarray(start, float) + out[:3, 3]
    return out


# =============================================================================
# Training volumes
# =============================================================================
class VolumeStore:
    """Loads one FOMO hemisphere as merged labels, cropped to its bounding box.

    The first read merges `seg.<hemi>.nii.gz` (float32, ~257 labels) and, if
    data.write_cache is on, saves the result as a small uint8 NIfTI in
    paths.cache_dir; later reads take ~0.1 s instead of ~0.5 s.
    The source files are only read.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.cache_root = Path(cfg["paths"]["cache_dir"]) / cfg["labels"]["level"]
        self.write_cache = bool(cfg["data"]["write_cache"])
        self._merger = None                     # built lazily (once per worker process)

    @property
    def merger(self):
        if self._merger is None:
            self._merger = LabelMerger(self.cfg)
        return self._merger

    def cache_path(self, item):
        return (self.cache_root / str(item["dataset"]) / str(item["subject"])
                / str(item["session"]) / f"seg.{item['hemi']}.nii.gz")

    def load(self, item):
        """item: dict with dataset, subject, session, hemi, nextbrain_dir.
        Returns (labels [X, Y, Z] uint8, affine 4x4)."""
        path = self.cache_path(item)
        if path.exists():
            img = nib.load(path)
            return np.asarray(img.dataobj), img.affine

        src = nib.load(os.path.join(item["nextbrain_dir"], f"seg.{item['hemi']}.nii.gz"))
        seg, _ = self.merger(np.asarray(src.dataobj))
        fg = np.nonzero(seg)
        if len(fg[0]) == 0:
            raise ValueError(f"Empty segmentation: {src.get_filename()}")
        lo = [int(c.min()) for c in fg]
        hi = [int(c.max()) + 1 for c in fg]
        seg = np.ascontiguousarray(seg[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]])
        affine = shift_affine(src.affine, lo)

        if self.write_cache:
            assert_writable(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_name(f".tmp{os.getpid()}_{path.name}")
            nib.save(nib.Nifti1Image(seg, affine), tmp)
            os.replace(tmp, path)               # atomic: workers never see half a file
        return seg, affine
