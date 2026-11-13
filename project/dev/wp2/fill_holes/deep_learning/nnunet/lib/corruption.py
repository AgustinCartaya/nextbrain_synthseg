"""
Random arguments for `corrupt_nifti.corrupt_array`, and the extra input channel.

The generator itself (corrupt_nifti.py) is used unchanged. This module only
decides, for every training sample, WHICH geometry and severity to ask for
(section `corruption` of the config), so that the network sees everything from
mild to severe corruption, each corruption type alone and combined.
"""
import numpy as np

#: parameters that are (min, max) integer ranges
_INT_RANGES = ("block_sections", "n_blobs", "thin_sections", "slab_sections")


def sample_corruption_args(rng, ccfg, blob_scale=1.0):
    """Draw one set of arguments for `corrupt_array`.

    rng:        numpy Generator
    ccfg:       the `corruption` section of the config
    blob_scale: multiplies n_blobs (the count is per image, so whole volumes
                need more blobs than a crop for the same density)

    Returns (angle_range, axis, kw, info):
      corrupt_array(seg, affine, angle_range, axis, rng, **kw)
      info = dict(severity=..., dropped=[...]) for logging.
    """
    # ---- section geometry ---------------------------------------------------
    modes = ccfg["geometry"]["modes"]
    p = np.array([m["p"] for m in modes], float)
    mode = modes[int(rng.choice(len(modes), p=p / p.sum()))]
    kw = dict(tilt_about=mode["tilt_about"], random_sign=bool(ccfg["geometry"]["random_sign"]))

    th = ccfg["thickness"]
    if rng.random() < th["p_voxel"]:
        kw["thickness"] = None                               # one voxel thick sections
    else:
        lo, hi = np.log(th["range_mm"])
        kw["thickness"] = float(np.exp(rng.uniform(lo, hi)))  # log-uniform

    # ---- severity -----------------------------------------------------------
    severity = float(rng.random())
    mild, severe = ccfg["mild"], ccfg["severe"]

    def lerp(name):
        """Value of `name` between its mild and severe setting, at a jittered severity."""
        t = float(np.clip(severity + rng.normal(0.0, ccfg["jitter"]), 0.0, 1.0))
        a, b = mild[name], severe[name]
        if isinstance(a, (list, tuple)):
            v = [x + (y - x) * t for x, y in zip(a, b)]
            if name in _INT_RANGES:
                v = [int(round(x)) for x in v]
            return (v[0], max(v[0], v[1]))
        return a + (b - a) * t

    for name in mild:
        kw[name] = lerp(name)
    kw["thin_prob"] = float(rng.uniform(*ccfg["thin_prob"]))
    kw["n_blobs"] = tuple(int(round(v * blob_scale)) for v in kw["n_blobs"])

    # ---- switch whole families off, so each type is also seen alone -----------
    off = dict(section_shift=("p_section", 0.0), block_shift=("p_block", 0.0),
               holes=("n_blobs", (0, 0)), missing_sections=("p_slab", 0.0))
    dropped = [name for name in off if rng.random() < ccfg["p_drop_type"]]
    for name in dropped:
        key, value = off[name]
        kw[key] = value

    info = dict(severity=severity, dropped=dropped)
    return tuple(mode["angle"]), mode["axis"], kw, info


def hole_channel(cn, labels):
    """The "suspicious holes" input channel of a (corrupted) label array.

    True for background voxels inside the closed brain mask, i.e.
    `brain_mask(labels, fill_holes=True) & (labels == 0)`: closing (3 iterations)
    + hole filling of the foreground. It needs no knowledge of the corruption,
    so it is computed in exactly the same way (on a context crop of the same
    size) for training patches and for the windows of a real test case.

    cn: the corrupt_nifti module.
    """
    return cn.brain_mask(labels, fill_holes=True) & (labels == 0)
