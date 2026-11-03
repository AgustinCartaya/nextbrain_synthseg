"""Helpers shared by the nextbrain-utils tutorial notebooks.

- ``make_phantom_hemi``: build a toy NextBrain hemisphere label map that uses
  *real* NextBrain label IDs, so every conversion in the library can be
  exercised without needing a real segmentation.
- ``lut_dict`` / ``show_seg``: draw label maps with their LUT colours.
- ``patched_nextbrain_lut``: work around the ``ctx-`` / ``ctx-rh-`` naming
  mismatch that currently breaks ``to_allen`` with the gyral/developmental
  cortex ontologies (see notebook 04).
"""
from contextlib import contextmanager

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from matplotlib.patches import Patch

# ----------------------------------------------------------------------
#   Synthetic phantom
# ----------------------------------------------------------------------

SHAPE = (48, 56, 48)  # (x, y, z) voxels, 1 mm isotropic, RAS

# (label, centre (m, y, z), radii) -- m = distance from the midline.
# Structures listed later overwrite earlier ones.
_STRUCTURES = [
    (50, (6, 30, 28), (3, 10, 4)),    # anterior_horn_of_lateral_ventricle
    (48, (10, 34, 28), (4, 5, 4)),    # head_of_caudate
    (218, (6, 20, 22), (5, 7, 5)),    # thalamus
    (79, (20, 30, 22), (4, 8, 6)),    # putamen
    (119, (15, 28, 21), (2, 5, 4)),   # external_segment_of_globus_pallidus
    (102, (27, 30, 22), (1, 8, 6)),   # claustrum
    (214, (18, 36, 12), (4, 4, 3)),   # amygdaloid_complex
    (345, (20, 24, 12), (3, 7, 3)),   # stratum_pyramidale_of_rostral_ca1
    (384, (1, 16, 9), (4, 5, 7)),     # midbrain_(mesencephalon)
]


def _ellipsoid(m, y, z, centre, radii):
    return sum(((a - c) / r) ** 2 for a, c, r in zip((m, y, z), centre, radii)) <= 1


def make_phantom_hemi(side="R"):
    """Return a toy NextBrain segmentation of one hemisphere.

    The right hemisphere occupies world x in [0, 48) mm and the left one
    x in [-46, 2) mm, so the two volumes overlap by 2 voxels at the midline.
    Like real NextBrain output, both hemispheres use the *right* (2xxx)
    cortical label IDs.
    """
    side = side.upper()[0]
    i, y, z = np.meshgrid(*map(np.arange, SHAPE), indexing="ij")
    m = i if side == "R" else (SHAPE[0] - 1 - i)  # mirror for the left side

    seg = np.zeros(SHAPE, dtype=np.int16)

    # cerebrum: half-ellipsoid, 3-voxel cortical ribbon around white matter
    brain = _ellipsoid(m, y, z, (0, 30, 26), (42, 25, 20))
    inner = _ellipsoid(m, y, z, (0, 30, 26), (39, 22, 17))
    seg[brain] = 7                                       # white_matter_of_forebrain
    ribbon = brain & ~inner
    seg[ribbon & (z >= 26) & (y >= 30)] = 2028           # ctx-superiorfrontal
    seg[ribbon & (z >= 26) & (y < 30)] = 2024            # ctx-precentral
    seg[ribbon & (z < 26)] = 2030                        # ctx-superiortemporal
    seg[ribbon & (z < 26) & (m < 14)] = 2006             # ctx-entorhinal
    seg[ribbon & (z >= 18) & (z < 26) & (m > 30)] = 2035  # ctx-insula

    # cerebellum: granular layer around hindbrain white matter
    cereb = _ellipsoid(m, y, z, (12, 7, 7), (14, 7, 6))
    seg[cereb] = 597                                     # granular_cell_layer_of_pva
    seg[_ellipsoid(m, y, z, (12, 7, 7), (10, 4, 3))] = 846  # white_matter_of_hindbrain

    for label, centre, radii in _STRUCTURES:
        seg[_ellipsoid(m, y, z, centre, radii)] = label

    affine = np.eye(4)
    affine[0, 3] = 0.0 if side == "R" else -46.0
    return nib.Nifti1Image(seg, affine)


# ----------------------------------------------------------------------
#   Plotting
# ----------------------------------------------------------------------

def lut_dict(lut):
    """Normalise a LUT to ``{label: (name, (r, g, b))}``.

    Accepts a ``nextbrain_utils.io.LUT`` or the list of tuples returned by
    ``nextbrain_utils.lut.allen_lut``.
    """
    if hasattr(lut, "dtype") and lut.dtype.names:
        return {
            int(r["ID"]): (r["NAME"].decode(), (int(r["R"]), int(r["G"]), int(r["B"])))
            for r in lut
        }
    return {int(lab): (name, (r, g, b)) for lab, name, r, g, b, *_ in lut}


def _fallback_color(label):
    rng = np.random.default_rng(int(label) % (2**32))
    return tuple(int(c) for c in rng.integers(40, 230, 3))


def show_seg(volumes, lut, axis=2, index=None, titles=None, legend=True,
             max_legend=25, figsize=None):
    """Show one slice of one or several label maps, coloured with ``lut``.

    ``volumes`` may be arrays or nibabel images; ``lut`` may be a single LUT
    or a list with one LUT per volume.
    """
    if not isinstance(volumes, (list, tuple)):
        volumes = [volumes]
    luts = lut if isinstance(lut, (list, tuple)) and not _is_tuple_lut(lut) else [lut] * len(volumes)
    luts = [lut_dict(x) for x in luts]
    titles = titles or [""] * len(volumes)
    n = len(volumes)
    fig, axes = plt.subplots(1, n, figsize=figsize or (5.2 * n, 5.4), squeeze=False)

    for ax, vol, lt, title in zip(axes[0], volumes, luts, titles):
        dat = np.asarray(vol.dataobj if hasattr(vol, "dataobj") else vol)
        idx = dat.shape[axis] // 2 if index is None else index
        sl = np.take(dat, idx, axis=axis).T  # transpose so rows = 2nd axis
        rgb = np.zeros(sl.shape + (3,), dtype=np.uint8)
        present = [int(v) for v in np.unique(sl) if v != 0]
        for lab in present:
            rgb[sl == lab] = lt.get(lab, (None, _fallback_color(lab)))[1]
        ax.imshow(rgb, origin="lower", interpolation="nearest")
        ax.set_title(f"{title}\n(axis {axis}, slice {idx})", fontsize=10)
        ax.set_xticks([]), ax.set_yticks([])
        if legend:
            handles = [
                Patch(color=np.array(lt.get(lab, (None, _fallback_color(lab)))[1]) / 255,
                      label=f"{lab}  {lt.get(lab, ('<not in LUT>',))[0]}"[:48])
                for lab in present[:max_legend]
            ]
            ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0, -0.02),
                      fontsize=7, frameon=False, ncol=1)
    plt.tight_layout()
    plt.show()


def _is_tuple_lut(x):
    return len(x) > 0 and isinstance(x[0], tuple)


# ----------------------------------------------------------------------
#   Workaround for the ctx- / ctx-rh- mismatch in to_allen
# ----------------------------------------------------------------------

@contextmanager
def patched_nextbrain_lut():
    """Make ``to_allen`` see cortical names as ``ctx-rh-<name>``.

    ``get_nextbrain2allen_map`` strips the first 7 characters of cortical
    names (``"ctx-rh-"``), but since commit 310dbf2 ("Align LUT") the bundled
    NextBrainLUT.txt names them ``ctx-<name>``. This temporarily re-prefixes
    them so the gyral/developmental mappings work, without editing the library.
    """
    import nextbrain_utils.to_allen as ta
    from nextbrain_utils.io import load_lut

    original = ta.load_lut

    def _load_lut(*args, **kwargs):
        lut = load_lut(*args, **kwargs).copy()
        for k, name in enumerate(lut["NAME"]):
            if name.startswith(b"ctx-") and not name.startswith((b"ctx-rh-", b"ctx-lh-")):
                lut["NAME"][k] = b"ctx-rh-" + name[4:]
        return lut

    ta.load_lut = _load_lut
    try:
        yield
    finally:
        ta.load_lut = original
