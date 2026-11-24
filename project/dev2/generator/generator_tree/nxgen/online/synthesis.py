"""
Step 4b: synthesis (output grid).

The image is the partial-volume mix of the label Gaussians:

    image(x) = sum_i w_i(x) * (mu[l_i] + sigma[l_i] * eps_i(x)) / intensity_scale

over the top-k entries (l_i, w_i) of the voxel (step 3). eps is standard
normal noise, one value per voxel and per *generation class*: two labels of
the same class in one voxel share the same eps (as in v2, where the
fractions were first summed per class). With hard labels (k = 1) a voxel
is simply mu + sigma * eps of its label.

The noise comes from a torch generator seeded with `noise_seed` (stored in
the plan), so it is reproduced exactly for the same plan and device.
"""

import torch

from ..core.rng import torch_generator


def apply(pv, params):
    """
    Synthetic image from partial volumes and label Gaussians.

    Parameters
    ----------
    pv : dict from `partial_volume.apply` (uses 'index' and 'weight')
    params : dict from `tree_intensities.sample`

    Returns
    -------
    (X, Y, Z) float32 tensor
    """
    index, weight = pv['index'], pv['weight']
    dev = index.device
    mu = torch.tensor(params['mu'], dtype=torch.float32, device=dev)
    sd = torch.tensor(params['sigma'], dtype=torch.float32, device=dev)
    cls = torch.tensor(params['class_of'], dtype=torch.long, device=dev)
    g = torch_generator(params['noise_seed'], dev)
    eps = torch.randn(index.shape, generator=g, device=dev)   # (k, X, Y, Z)
    img = torch.zeros(index.shape[1:], device=dev)
    # one entry at a time (memory: a few volumes, not k of each)
    for i in range(index.shape[0]):
        li = index[i].long()
        ci = cls[li]
        # entries of the same class in a voxel take the eps of the first one
        for j in range(i):
            same = ci == cls[index[j].long()]
            eps[i] = torch.where(same, eps[j], eps[i])
        img += weight[i] * (mu[li] + sd[li] * eps[i])
        del li, ci
    return img / params['intensity_scale']
