"""
Step 1: morphology (HR grid).

Random growth of one label group into another. For each configured pair
[a, b], with probability `prob`, one of the two groups (chosen at random)
grows into the other by `n_iter` voxel layers, and the other shrinks by the
same amount. Growth uses the 26-neighbourhood. When the growing group has
several labels, each new voxel copies one neighbouring growing label (the
one with the largest index).

Same algorithm as v2 `_morphology`. Differences:
* the label groups are resolved to indices when the parameters are drawn,
  and stored in the plan;
* it runs on whatever crop it is given (the pipeline gives the part of the
  template the patch needs, plus a halo of `halo(params)` voxels). The
  operation is local (n_iter voxels), so inside the halo the result equals
  the full-volume result.
"""

import torch
import torch.nn.functional as F

from ..core.rng import draw


def sample(cfg, rng, label_space, tree):
    """
    Draw the morphology operations of one image.

    Parameters
    ----------
    cfg : dict, config section 'morphology'
    rng : numpy Generator
    label_space : LabelSpace
    tree : LabelTree

    Returns
    -------
    None if no operation is applied, else
    dict(ops=[dict(grow=<group spec>, shrink=<group spec>, n_iter=int,
                   grow_idx=[indices], shrink_idx=[indices]), ...])
    """
    ops = []
    for a, b in cfg['pairs']:
        # always draw the three numbers, so each pair's draw does not depend
        # on what happened to the previous pairs
        apply = rng.random() < cfg['prob']
        swap = rng.random() < 0.5
        n_iter = int(draw(cfg['n_iter'], rng))
        if not apply or n_iter < 1:
            continue
        grow, shrink = (b, a) if swap else (a, b)
        grow_idx = label_space.resolve(grow, tree)
        shrink_idx = label_space.resolve(shrink, tree)
        if grow_idx and shrink_idx:
            ops.append(dict(grow=grow, shrink=shrink, n_iter=n_iter,
                            grow_idx=grow_idx, shrink_idx=shrink_idx))
    return dict(ops=ops) if ops else None


def halo(params):
    """
    Voxels around a crop needed for an exact result inside it.

    Each layer looks one voxel further, and the operations are applied one
    after the other, so the reach is the total number of layers.
    """
    if params is None:
        return 0
    return sum(op['n_iter'] for op in params['ops'])


def _member(indices, n_labels, device):
    """Bool lookup table: table[i] is True if index i is in `indices`."""
    table = torch.zeros(n_labels, dtype=torch.bool, device=device)
    table[torch.as_tensor(indices, device=device, dtype=torch.long)] = True
    return table


def apply(lab, params, n_labels):
    """
    Apply the operations of `params` to an index map.

    Parameters
    ----------
    lab : (X, Y, Z) int16 tensor of label indices
    params : dict from `sample`, or None (nothing done)
    n_labels : int, size of the label space

    Returns
    -------
    (X, Y, Z) int16 tensor (a new tensor; `lab` is not modified)
    """
    if params is None:
        return lab
    for op in params['ops']:
        grow = _member(op['grow_idx'], n_labels, lab.device)
        shrink = _member(op['shrink_idx'], n_labels, lab.device)
        for _ in range(op['n_iter']):
            idx = lab.long()
            # index + 1 of the growing labels, 0 elsewhere; the max over the
            # 3x3x3 neighbourhood picks one neighbouring growing label
            val = torch.where(grow[idx], lab.float() + 1, 0.)
            nbr = F.max_pool3d(val[None, None], 3, stride=1, padding=1)[0, 0]
            take = shrink[idx] & (nbr > 0)
            lab = torch.where(take, (nbr - 1).to(lab.dtype), lab)
            del idx, val, nbr, take
    return lab
