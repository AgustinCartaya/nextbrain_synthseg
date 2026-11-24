"""
Offline module: HR template from a label map.

    label map (e.g. 0.2 mm)
      -> restrict_labels      labels not in the tree -> background (optional)
      -> crop_to_foreground   bounding box of the brain + margin (mm)
      -> super_resolve        one-hot -> sinc upsampling -> Gaussian -> argmax
      -> HR template (hr_voxel_mm), saved as uint16 + a build record (JSON)

The algorithm is the one of v2 (`prepare_template`, `super_resolve_labels`),
split into functions. Differences with v2:

* Parametrised by the HR voxel only (`template.hr_voxel_mm`), not by the
  output resolution. The same template serves any output voxel that is an
  integer multiple of it. The generator pads the template to a multiple of
  the partial-volume factor when it loads it.
* Margins are fixed (mm). The v2 automatic margins ('auto', {'auto': q})
  were removed: they were the most complex part of v2 and not exact
  (problems_to_solve #6-8). In the modular generator, anything moved
  outside the template is background, and `nxgen.checks` can measure how
  much foreground a deformation pushes out.
* The build record stores a SHA-256 of the source file; a cached template is
  reused only if the source and the settings are unchanged
  (problems_to_solve #3).
* The record also stores the volume of every label before and after, and
  the labels that vanished (thin or tiny structures can disappear at the
  argmax).
"""

import math
import os

import numpy as np
import torch

from ..core.io import file_sha256, load_json, load_labels, save_json, \
    save_labels


def restrict_labels(seg, allowed, background=0):
    """
    Map every label not in `allowed` to `background`.

    Parameters
    ----------
    seg : int ndarray, raw label values
    allowed : iterable of int
    background : int

    Returns
    -------
    seg : int64 ndarray (copy)
    removed : sorted list of the raw values that were mapped to background
    """
    seg = np.asarray(seg).astype(np.int64)
    present = set(np.unique(seg).tolist())
    removed = sorted(present - set(int(v) for v in allowed) - {background})
    if removed:
        seg[np.isin(seg, removed)] = background
    return seg, removed


def crop_to_foreground(seg, affine, voxel_mm, margin_mm, background=0):
    """
    Crop around the foreground, keeping `margin_mm` of background.

    Where the margin goes past the edge of the map, the map is padded with
    background.

    Parameters
    ----------
    seg : (X, Y, Z) int ndarray
    affine : (4, 4) ndarray
    voxel_mm : float, isotropic voxel size of `seg`
    margin_mm : float or 3 floats
    background : int

    Returns
    -------
    seg : cropped (and padded) map
    affine : affine of the crop
    box : dict(lo=[3], hi=[3]) voxel box in the input map (may be < 0 or
        > shape where padded)
    """
    fg = np.argwhere(seg != background)
    if len(fg) == 0:
        raise ValueError('The label map has no foreground (only background)')
    margin = np.broadcast_to(np.asarray(margin_mm, float), (3,))
    margin = np.ceil(margin / voxel_mm - 1e-6).astype(int)
    lo = fg.min(0) - margin
    hi = fg.max(0) + 1 + margin
    pad_lo = np.maximum(-lo, 0)
    pad_hi = np.maximum(hi - np.array(seg.shape), 0)
    seg = np.pad(seg, list(zip(pad_lo, pad_hi)), constant_values=background)
    a, b = lo + pad_lo, hi + pad_lo
    seg = seg[a[0]:b[0], a[1]:b[1], a[2]:b[2]]
    shift = np.eye(4)
    shift[:3, 3] = lo
    return seg, np.asarray(affine, float) @ shift, \
        dict(lo=lo.tolist(), hi=hi.tolist())


def _bbox(mask, margin):
    """Bounding box (tuple of slices) of a 3D bool tensor, + margin voxels."""
    box = []
    for d in range(3):
        other = [i for i in range(3) if i != d]
        nz = mask.any(dim=other[1]).any(dim=other[0]).nonzero()
        lo = max(int(nz[0]) - margin, 0)
        hi = min(int(nz[-1]) + 1 + margin, mask.shape[d])
        box.append(slice(lo, hi))
    return tuple(box)


def sinc_upsample_smooth(x, factor, sigma_mm, out_voxel_mm):
    """
    Band-limited (sinc) upsampling with Gaussian smoothing, in Fourier.

    The centred spectrum is zero-padded to `factor` times the size (sinc
    interpolation) and multiplied by the transfer function of a Gaussian of
    std `sigma_mm`. Output voxel j sits at input coordinate j / factor.

    Parameters
    ----------
    x : (X, Y, Z) float tensor
    factor : int (>= 1)
    sigma_mm : float (0 = no smoothing)
    out_voxel_mm : float, voxel size of the output

    Returns
    -------
    (factor*X, factor*Y, factor*Z) float tensor
    """
    shape = x.shape
    new_shape = [s * factor for s in shape]
    spec = torch.fft.fftshift(torch.fft.fftn(x))
    padded = torch.zeros(new_shape, dtype=spec.dtype, device=spec.device)
    # keep the DC component (index s // 2 after fftshift) aligned
    where = tuple(slice(n // 2 - s // 2, n // 2 - s // 2 + s)
                  for n, s in zip(new_shape, shape))
    padded[where] = spec
    del spec
    if sigma_mm:
        for d, n in enumerate(new_shape):
            f = torch.fft.fftshift(torch.fft.fftfreq(
                n, d=out_voxel_mm, device=x.device))          # cycles / mm
            g = torch.exp(-2 * (math.pi * sigma_mm * f) ** 2)
            view = [1, 1, 1]
            view[d] = n
            padded *= g.view(view)
    y = torch.fft.ifftn(torch.fft.ifftshift(padded)).real
    return y * factor ** 3


def super_resolve(seg, factor, voxel_mm, sigma_mm=0.0, background=0):
    """
    One-hot -> sinc upsampling -> Gaussian smoothing -> argmax.

    Every foreground label is processed inside its own bounding box (plus a
    margin past the kernel support), so the cost scales with the size of the
    structures, not with the number of labels. The background channel is
    never computed: upsampling and smoothing are linear and the one-hot
    channels sum to 1, so background = 1 - sum(foreground).

    With factor = 1 and sigma_mm = 0 the labels are returned unchanged.

    Parameters
    ----------
    seg : (X, Y, Z) integer tensor of raw label values
    factor : int, upsampling factor
    voxel_mm : float, voxel size of `seg`
    sigma_mm : float, smoothing std (mm)
    background : int

    Returns
    -------
    (factor*X, factor*Y, factor*Z) int32 tensor of raw label values
    """
    if factor == 1 and not sigma_mm:
        return seg.to(torch.int32).clone()
    out_voxel = voxel_mm / factor
    margin = max(4, int(math.ceil(4 * sigma_mm / voxel_mm)) + 2)
    hr_shape = [s * factor for s in seg.shape]
    best = torch.zeros(hr_shape, dtype=torch.float32, device=seg.device)
    total = torch.zeros_like(best)
    out = torch.full(hr_shape, background, dtype=torch.int32,
                     device=seg.device)
    for v in torch.unique(seg).tolist():
        if v == background:
            continue
        mask = seg == v
        box = _bbox(mask, margin)
        y = sinc_upsample_smooth(mask[box].float(), factor, sigma_mm,
                                 out_voxel)
        del mask
        hr_box = tuple(slice(b.start * factor, b.stop * factor) for b in box)
        better = y > best[hr_box]
        out[hr_box].masked_fill_(better, v)
        best[hr_box] = torch.maximum(best[hr_box], y)
        total[hr_box] += y
        del y, better
    out.masked_fill_(1 - total > best, background)
    return out


def label_volumes(seg, voxel_mm, background=0):
    """{raw value: volume in mm^3} of every foreground label."""
    values, counts = np.unique(np.asarray(seg), return_counts=True)
    return {int(v): float(c) * voxel_mm ** 3
            for v, c in zip(values, counts) if v != background}


def build_template(seg_path, out_dir, hr_voxel_mm, sigma_mm=0.0,
                   margin_mm=16.0, allowed=None, background=0,
                   device='cpu', overwrite=False):
    """
    Build (or reuse) the HR template of one label map.

    Output files in `out_dir`:
        <name>_hr<voxel>mm.nii.gz   HR template, raw label values, uint16
        <name>_hr<voxel>mm.json     build record (see below)

    Parameters
    ----------
    seg_path : str, input label map (isotropic voxels, an integer multiple
        of hr_voxel_mm)
    out_dir : str
    hr_voxel_mm : float
    sigma_mm : float, smoothing before the argmax (mm)
    margin_mm : float or 3 floats, background margin (mm)
    allowed : iterable of int, optional
        Allowed raw labels (those of the tree). Others go to background.
        None = keep every label.
    background : int
    device : str, where the super-resolution runs ('cpu' has more memory)
    overwrite : bool, rebuild even if a matching cached template exists

    Returns
    -------
    path : str, the HR template
    record : dict with
        source, source_sha256 : input file and its hash
        settings : hr_voxel_mm, sigma_mm, margin_mm, background, allowed
            (sorted list or None)
        input_voxel_mm, upsampling_factor, crop_box, shape
        removed_labels : labels mapped to background (not allowed)
        volumes_in_mm3, volumes_out_mm3 : per label, before / after
        vanished_labels : labels present before but not after
    """
    os.makedirs(out_dir, exist_ok=True)
    name = os.path.basename(seg_path).split('.')[0]
    path = os.path.join(out_dir, f'{name}_hr{hr_voxel_mm:g}mm.nii.gz')
    record_path = path[:-len('.nii.gz')] + '.json'
    settings = dict(hr_voxel_mm=hr_voxel_mm, sigma_mm=sigma_mm,
                    margin_mm=margin_mm, background=background,
                    allowed=None if allowed is None
                    else sorted(int(v) for v in allowed))
    sha = file_sha256(seg_path)

    if not overwrite and os.path.exists(path) and os.path.exists(record_path):
        old = load_json(record_path)
        if old.get('source_sha256') == sha and old.get('settings') == settings:
            print(f'{path}: reused (same source and settings)')
            return path, old
        print(f'{path}: source or settings changed, rebuilding')

    seg, affine, voxel = load_labels(seg_path)
    if np.ptp(voxel) > 1e-4:
        raise ValueError(f'{seg_path}: voxels must be isotropic, got {voxel}')
    voxel_mm = round(float(voxel[0]), 6)     # NIfTI float32: 0.3 -> 0.30000001
    up = voxel_mm / hr_voxel_mm
    if abs(up - round(up)) > 1e-3 or round(up) < 1:
        raise ValueError(f'{seg_path}: voxel {voxel_mm} mm is not an integer '
                         f'multiple of the HR voxel {hr_voxel_mm} mm')
    up = int(round(up))

    removed = []
    if allowed is not None:
        seg, removed = restrict_labels(seg, allowed, background)
    vol_in = label_volumes(seg, voxel_mm, background)
    seg, affine, box = crop_to_foreground(seg, affine, voxel_mm, margin_mm,
                                          background)
    print(f'{name}: crop {box}, super-resolution x{up}, sigma {sigma_mm} mm')

    with torch.no_grad():
        hr = super_resolve(torch.as_tensor(seg, device=device), up, voxel_mm,
                           sigma_mm, background).cpu().numpy()
    # HR voxel j sits at input voxel coordinate j / up
    hr_affine = affine @ np.diag([1 / up, 1 / up, 1 / up, 1])
    save_labels(hr, hr_affine, path)

    vol_out = label_volumes(hr, hr_voxel_mm, background)
    record = dict(source=os.path.abspath(seg_path), source_sha256=sha,
                  settings=settings, input_voxel_mm=voxel_mm,
                  upsampling_factor=up, crop_box=box, shape=list(hr.shape),
                  removed_labels=removed,
                  # JSON keys are strings: str(raw value)
                  volumes_in_mm3={str(k): v for k, v in vol_in.items()},
                  volumes_out_mm3={str(k): v for k, v in vol_out.items()},
                  vanished_labels=sorted(set(vol_in) - set(vol_out)))
    save_json(record, record_path)
    if record['vanished_labels']:
        print(f'{name}: WARNING labels vanished in the super-resolution: '
              f'{record["vanished_labels"]}')
    return path, record
