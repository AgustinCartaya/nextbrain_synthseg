"""
File input/output: NIfTI volumes, JSON, file hashes.

Label maps are written as uint16 when the values fit (0..65535), else int32,
never uint8 (uint8 silently wraps values > 255; NextBrain has values up to
~2000). Images are written as float32.
"""

import hashlib
import json

import nibabel as nib
import numpy as np
import torch


def load_nifti(path):
    """
    Load a NIfTI volume.

    Returns
    -------
    data : ndarray (as stored, not scaled for integer label maps)
    affine : (4, 4) ndarray, voxel -> world (mm)
    voxel_mm : (3,) ndarray, voxel size
    """
    nii = nib.load(path)
    return np.asarray(nii.dataobj), nii.affine, \
        np.asarray(nii.header.get_zooms()[:3], float)


def load_labels(path):
    """Load an integer label map. Returns (int64 array, affine, voxel_mm)."""
    data, affine, voxel = load_nifti(path)
    return np.rint(data).astype(np.int64), affine, voxel


def _numpy(x):
    """Tensor or array -> numpy array (on the CPU)."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def save_labels(labels, affine, path):
    """
    Save an integer label map (uint16 if the values fit, else int32).

    Parameters
    ----------
    labels : (X, Y, Z) integer tensor or array
    affine : (4, 4) array
    path : str (.nii or .nii.gz)
    """
    arr = _numpy(labels).astype(np.int64)
    lo, hi = (int(arr.min()), int(arr.max())) if arr.size else (0, 0)
    dtype = np.uint16 if lo >= 0 and hi <= 65535 else np.int32
    nii = nib.Nifti1Image(arr.astype(dtype), np.asarray(affine, float))
    nii.set_data_dtype(dtype)
    nii.header.set_slope_inter(1, 0)   # stored as is, no scaling
    nib.save(nii, path)


def save_image(image, affine, path):
    """Save a float volume (X, Y, Z) as float32 NIfTI."""
    arr = _numpy(image).astype(np.float32)
    nib.save(nib.Nifti1Image(arr, np.asarray(affine, float)), path)


def to_jsonable(x):
    """Convert numpy / torch values (recursively) to plain JSON types."""
    if isinstance(x, dict):
        return {str(k): to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_jsonable(v) for v in x]
    if isinstance(x, torch.Tensor):
        return to_jsonable(x.detach().cpu().numpy())
    if isinstance(x, np.ndarray):
        return to_jsonable(x.tolist())
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        return float(x)
    return x


def save_json(obj, path):
    """Write `obj` as indented JSON (numpy/torch values converted)."""
    with open(path, 'w') as f:
        json.dump(to_jsonable(obj), f, indent=2)


def load_json(path):
    """Read a JSON file."""
    with open(path) as f:
        return json.load(f)


def file_sha256(path, chunk=1 << 20):
    """SHA-256 of a file (hex), to identify the exact source of a template."""
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(chunk), b''):
            h.update(block)
    return h.hexdigest()
