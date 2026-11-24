"""
Step 3: partial volume (HR grid -> output grid).

Each output voxel gets the fraction of every label in a window of HR
voxels. Both methods are the same operation, a weighted label histogram per
window; only the window changes:

    method       window                       stride   output grid   hard label
    downsample   factor^3 box, equal weights  factor   HR / factor   largest fraction
    smooth       Gaussian (std smooth_sigma_  1        HR            the HR label
                 vox, truncated at 2 std)                            (unchanged)

'downsample' is the partial volume of the paper (HR labels pooled to the
output resolution). 'smooth' approximates partial volumes when the HR grid
is the output grid (factor = 1): voxels near a boundary get a mix of the
labels around them.

Output: sparse top-k
--------------------
Dense maps (one fraction per label per voxel) are too big for NextBrain:
~333 labels x 30 M voxels (0.4 mm) x 4 bytes = 40 GB. Only the `top_k`
largest fractions of each voxel are kept:

    index   (k, X, Y, Z) int16    label indices (0 where the weight is 0)
    weight  (k, X, Y, Z) float32  fractions, renormalised to sum to 1
    seg     (X, Y, Z) int16       hard segmentation
    lost    (X, Y, Z) float32     total fraction dropped by the truncation

With 'downsample', k = factor^3 is exact (one voxel cannot hold more labels
than HR voxels). Hard labels are the case k = 1.

How the histogram is computed (no K-sized arrays, so the cost does not
depend on the number of labels): for the W labels L_1..L_W of a window with
weights g_1..g_W, the weight of the label of entry i is
sum_j g_j [L_j == L_i]. Entries that repeat an earlier label are zeroed,
then the k largest are kept. Memory is W^2 per voxel, so the work is done
in slabs.

Ties for the hard label (e.g. 4 + 4 of 8 voxels) go to the smallest index,
as in v2 (`argmax` of the histogram).
"""

import math

import torch
import torch.nn.functional as F


def window(method, factor, sigma_vox):
    """
    Window of a method.

    Returns
    -------
    size : int, window side (voxels)
    stride : int
    weights : (size^3,) float tensor summing to 1, in [x, y, z] order
    """
    if method == 'downsample':
        return factor, factor, torch.full((factor ** 3,), 1 / factor ** 3)
    if factor != 1:
        raise ValueError("partial_volume.method 'smooth' needs the HR grid to "
                         'be the output grid (factor 1)')
    r = max(1, math.ceil(2 * sigma_vox))
    d = torch.arange(-r, r + 1, dtype=torch.float32)
    g1 = torch.exp(-d ** 2 / (2 * sigma_vox ** 2))
    g = g1[:, None, None] * g1[None, :, None] * g1[None, None, :]
    return 2 * r + 1, 1, (g / g.sum()).reshape(-1)


def _histogram_topk(labels, g, k):
    """
    Top-k weighted label histogram of N windows.

    Parameters
    ----------
    labels : (N, W) long tensor, labels of each window
    g : (W,) float tensor, window weights
    k : int (<= W)

    Returns
    -------
    index : (N, k) long, weight : (N, k) float (renormalised),
    best : (N,) long, label with the largest weight (ties: smallest index),
    lost : (N,) float, weight dropped by the truncation
    """
    W = labels.shape[1]
    eq = labels[:, :, None] == labels[:, None, :]                  # (N, W, W)
    weight = eq.float() @ g                                        # (N, W)
    earlier = torch.ones(W, W, dtype=torch.bool,
                         device=labels.device).tril(-1)            # j < i
    dup = (eq & earlier).any(-1)
    del eq
    weight = weight.masked_fill(dup, 0.)
    # rank by weight; equal weights -> smaller label first. The offset
    # (<= 0.0033 for 32767 labels) only orders ties: the stored weights are
    # the exact ones.
    key = weight - labels.float() * 1e-7
    pos = key.topk(k, dim=1).indices
    val = weight.gather(1, pos)
    index = torch.where(val > 0, labels.gather(1, pos), 0)
    best = labels.gather(1, pos[:, :1])[:, 0]
    total = val.sum(1)
    return index, val / total[:, None], best, 1 - total


def apply(lab, method='downsample', factor=1, top_k=8, sigma_vox=0.5,
          budget_bytes=1 << 29):
    """
    Partial volume of a deformed HR label map.

    Parameters
    ----------
    lab : (fX, fY, fZ) int16 tensor of label indices ('downsample': shape
        a multiple of factor)
    method : 'downsample' or 'smooth'
    factor : int, HR voxels per output voxel along each axis
    top_k : int, labels kept per voxel (capped at the window size)
    sigma_vox : float, 'smooth' only
    budget_bytes : int, memory of the per-slab work arrays

    Returns
    -------
    dict(index, weight, seg, lost) (see the module docstring)
    """
    size, stride, g = window(method, factor, sigma_vox)
    g = g.to(lab.device)
    W = size ** 3
    k = min(top_k, W)
    if method == 'downsample':
        X, Y, Z = (s // factor for s in lab.shape)
    else:
        X, Y, Z = lab.shape
        r = size // 2
        # background (index 0) outside the volume
        padded = F.pad(lab[None, None].float(), [r] * 6)[0, 0].to(lab.dtype)
    dev = lab.device
    index = torch.empty((k, X, Y, Z), dtype=torch.int16, device=dev)
    weight = torch.empty((k, X, Y, Z), dtype=torch.float32, device=dev)
    seg = torch.empty((X, Y, Z), dtype=torch.int16, device=dev)
    lost = torch.empty((X, Y, Z), dtype=torch.float32, device=dev)

    per_voxel = W * W * 6 + W * 24          # eq (bool) + float copy + misc
    step = max(1, budget_bytes // (per_voxel * Y * Z))
    for i0 in range(0, X, step):
        i1 = min(i0 + step, X)
        s = i1 - i0
        if method == 'downsample':
            f = factor
            blk = lab[i0 * f:i1 * f].long().view(s, f, Y, f, Z, f)
            blk = blk.permute(0, 2, 4, 1, 3, 5).reshape(-1, W)
        else:
            blk = padded[i0:i1 + 2 * r].long()
            blk = blk.unfold(0, size, 1).unfold(1, size, 1).unfold(2, size, 1)
            blk = blk.reshape(-1, W)
        idx, wgt, best, lst = _histogram_topk(blk, g, k)
        index[:, i0:i1] = idx.T.reshape(k, s, Y, Z).to(torch.int16)
        weight[:, i0:i1] = wgt.T.reshape(k, s, Y, Z)
        lost[i0:i1] = lst.view(s, Y, Z)
        if method == 'downsample':
            seg[i0:i1] = best.view(s, Y, Z).to(torch.int16)
        del blk, idx, wgt, best, lst
    if method == 'smooth':
        seg = lab.clone()
    return dict(index=index, weight=weight, seg=seg, lost=lost)
