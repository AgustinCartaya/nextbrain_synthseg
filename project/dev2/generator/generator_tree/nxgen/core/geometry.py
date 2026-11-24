"""
Small geometry helpers: rotation/affine matrices and grid affines.

Conventions
-----------
* Volumes are indexed [x, y, z] (array axes 0, 1, 2).
* Spatial transforms are expressed in *voxel-axis millimetres*: the array
  axes scaled by the voxel size, centred on the centre of the volume. They
  do not depend on the orientation stored in the NIfTI affine.
* Angles are in degrees. Rotations are applied as R = Rx @ Ry @ Rz (a
  rotation about z first, then y, then x).
"""

import numpy as np


def rotation_matrix(angles_deg):
    """
    3x3 rotation matrix R = Rx @ Ry @ Rz.

    Parameters
    ----------
    angles_deg : 3 floats, rotation about the x, y and z axes (degrees)
    """
    ax, ay, az = np.deg2rad(np.asarray(angles_deg, float))
    cx, sx = np.cos(ax), np.sin(ax)
    cy, sy = np.cos(ay), np.sin(ay)
    cz, sz = np.cos(az), np.sin(az)
    rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return rx @ ry @ rz


def affine_matrix(rotation_deg, zoom, shear):
    """
    Linear part M = R @ S @ Z of the random affine.

    Parameters
    ----------
    rotation_deg : 3 floats (see `rotation_matrix`)
    zoom : 3 floats, scale per axis (1 = none)
    shear : 3 floats, (xy, xz, yz) entries of the upper-triangular shear
        matrix S = [[1, xy, xz], [0, 1, yz], [0, 0, 1]]

    Returns
    -------
    (3, 3) ndarray
    """
    s = np.eye(3)
    s[0, 1], s[0, 2], s[1, 2] = shear
    return rotation_matrix(rotation_deg) @ s @ np.diag(np.asarray(zoom, float))


def pooled_affine(affine, factor):
    """
    Affine of the grid obtained by pooling `factor`^3 voxels into one.

    Output voxel i covers input voxels factor*i .. factor*i + factor - 1,
    so its centre is input voxel factor*i + (factor - 1) / 2.
    """
    pool = np.diag([factor, factor, factor, 1.0])
    pool[:3, 3] = (factor - 1) / 2
    return np.asarray(affine, float) @ pool


def shifted_affine(affine, start):
    """Affine of a crop whose first voxel is voxel `start` of `affine`."""
    shift = np.eye(4)
    shift[:3, 3] = start
    return np.asarray(affine, float) @ shift
