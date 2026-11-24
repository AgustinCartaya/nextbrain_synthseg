"""
Diagnostics for generated data.

These functions are never called by the generator. The tests (tests/) use
them, and they can be run by hand on any output, e.g. on the steps kept by
a `Recorder`.

    labels_present(lab)                    set of label indices
    label_volumes(lab)                     voxels per label index
    new_labels(before, after)              labels in `after` but not `before`
    vanished_labels(before, after)         labels in `before` but not `after`
    contacts(lab, group_a, group_b)        voxel faces between two groups
    adjacency_violations(lab, rules, ...)  forbidden contacts (e.g. WM-CSF)
    gaussian_fit_pure(image, pv, tree_params)
                                           intensity stats on pure voxels
"""

import numpy as np
import torch


def labels_present(lab):
    """Set of the values in a label map (tensor)."""
    return set(torch.unique(lab).tolist())


def label_volumes(lab):
    """{label: number of voxels}."""
    values, counts = torch.unique(lab, return_counts=True)
    return dict(zip(values.tolist(), counts.tolist()))


def new_labels(before, after):
    """Labels present in `after` but not in `before`."""
    return sorted(labels_present(after) - labels_present(before))


def vanished_labels(before, after):
    """Labels present in `before` but not in `after`."""
    return sorted(labels_present(before) - labels_present(after))


def contacts(lab, group_a, group_b):
    """
    Number of voxel faces (6-neighbourhood) between two label groups.

    Parameters
    ----------
    lab : (X, Y, Z) integer tensor
    group_a, group_b : lists of label values
    """
    a = torch.isin(lab, torch.as_tensor(group_a, device=lab.device,
                                        dtype=lab.dtype))
    b = torch.isin(lab, torch.as_tensor(group_b, device=lab.device,
                                        dtype=lab.dtype))
    n = 0
    for d in range(3):
        lo = [slice(None)] * 3
        hi = [slice(None)] * 3
        lo[d], hi[d] = slice(0, -1), slice(1, None)
        lo, hi = tuple(lo), tuple(hi)
        n += int((a[lo] & b[hi]).sum() + (b[lo] & a[hi]).sum())
    return n


def adjacency_violations(lab, rules, label_space, tree):
    """
    Forbidden contacts, e.g. rules = [['cerebral_wm_l1', 'csf_ventricles_l1']]
    (white matter must not touch CSF: grey matter should lie between them).

    Parameters
    ----------
    lab : (X, Y, Z) tensor of label indices
    rules : list of [group_a, group_b] (tree node names, raw values, lists)
    label_space : LabelSpace
    tree : LabelTree

    Returns
    -------
    {'a|b': number of contact faces} for every rule
    """
    out = {}
    for a, b in rules:
        ia = label_space.resolve(a, tree)
        ib = label_space.resolve(b, tree)
        out[f'{a}|{b}'] = contacts(lab, ia, ib) if ia and ib else 0
    return out


def gaussian_fit_pure(image, pv, tree_params, min_voxels=100):
    """
    Intensity statistics of every label on its *pure* voxels.

    A voxel is pure when one label has the whole fraction (weight 1). There,
    the synthesised intensity (before any artefact) must follow
    N(mu, sigma) / intensity_scale exactly. Mixed voxels are expected to
    deviate and are excluded.

    Parameters
    ----------
    image : (X, Y, Z) tensor, output of `synthesis.apply`
    pv : dict from `partial_volume.apply`
    tree_params : dict from `tree_intensities.sample`
    min_voxels : int, labels with fewer pure voxels are skipped

    Returns
    -------
    {label index: dict(n, mean, std, mu, sigma, ks)} (expected mu, sigma
    already divided by intensity_scale; ks = Kolmogorov-Smirnov statistic
    against the expected Gaussian, None without scipy or if sigma = 0)
    """
    scale = tree_params['intensity_scale']
    pure = pv['weight'][0] > 1 - 1e-6
    lab = pv['index'][0][pure].cpu().numpy()
    val = image[pure].cpu().numpy().astype(float)
    try:
        from scipy import stats
    except ImportError:
        stats = None
    out = {}
    for i in np.unique(lab):
        v = val[lab == i]
        if len(v) < min_voxels:
            continue
        mu = tree_params['mu'][i] / scale
        sd = tree_params['sigma'][i] / scale
        ks = None
        if stats is not None and sd > 0:
            ks = float(stats.kstest(v, 'norm', args=(mu, sd)).statistic)
        out[int(i)] = dict(n=len(v), mean=float(v.mean()),
                           std=float(v.std()), mu=mu, sigma=sd, ks=ks)
    return out
