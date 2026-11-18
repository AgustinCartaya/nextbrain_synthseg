"""
Step 4a: tree intensities (no voxels).

Draws one Gaussian (mu, sigma) per output label from the label tree, for
one image:

    C[i] ~ U(colors_mu), V[i] ~ U(colors_sigma)        one per tree color
    level 1   : mu = sum_i w_mu[i] C[i] + wr_mu N(0,1)
                sd = sum_i w_sigma[i] V[i] + wr_sigma N(0,1)
    level >= 2: visible     -> mu = mu_parent + wr_mu N(0,1)
                                sd = sd_parent + wr_sigma N(0,1)
                not visible -> exactly (mu_parent, sd_parent)
    sd = max(sd, min_sigma); 'null' nodes (and their subtrees): mu = sd = 0

Visibility uses the resolution of the image: by default the coarsest
spacing of the acquisition (step 0), so labels the scan cannot resolve
share one Gaussian. Labels that share a Gaussian form a *generation class*
(named after the visible node they use); WP4 merges them in the loss.

Two normal numbers are drawn for every node, visible or not, in the tree's
breadth-first order. With a fixed seed the visible nodes therefore get the
same values at any resolution.

Same model as v2 `TreeSampling.sample_node_stats` / `label_set_lut`.
"""

import numpy as np

from ..core.rng import draw, new_seed


def node_stats(tree, C, V, rng, resolution, contrast, min_sigma):
    """
    (mu, sigma) of every non-root node, top-down.

    Parameters
    ----------
    tree : LabelTree
    C, V : (n_colors,) arrays, principal color means and stds
    rng : numpy Generator
    resolution : float (mm)
    contrast : str or None
    min_sigma : float

    Returns
    -------
    stats : {node: (mu, sigma)}
    """
    vis = tree.visibility(resolution, contrast)
    stats, is_null = {}, {tree.root: False}
    for name in tree.order()[1:]:
        n = tree.nodes[name]
        e_mu, e_sd = rng.standard_normal(2)
        is_null[name] = n.null or is_null[n.parent]
        if is_null[name]:
            stats[name] = (0.0, 0.0)
            continue
        if n.level == 1:
            w_mu = np.asarray(n.w_mu)
            w_sd = np.asarray(n.w_sigma if n.w_sigma is not None else n.w_mu)
            mu = float(w_mu @ C) + n.wr_mu * e_mu
            sd = float(w_sd @ V) + n.wr_sigma * e_sd
        elif vis[name]:
            mu_p, sd_p = stats[n.parent]
            mu = mu_p + n.wr_mu * e_mu
            sd = sd_p + n.wr_sigma * e_sd
        else:
            stats[name] = stats[n.parent]   # collapsed: parent's Gaussian
            continue
        stats[name] = (float(mu), float(max(sd, min_sigma)))
    return stats


def sample(cfg, rng, tree, label_space, acquisition):
    """
    Draw the Gaussians of one image.

    Parameters
    ----------
    cfg : dict, config section 'tree'
    rng : numpy Generator
    tree : LabelTree
    label_space : LabelSpace
    acquisition : dict from `acquisition.sample`

    Returns
    -------
    dict with
        colors_mu, colors_sigma : C and V (one per tree color)
        resolution_mm, contrast : used for node visibility
        mu, sigma : per label index (tree units)
        class_of : generation class of each label index
        class_names : visible node of each class ('unassigned' for labels
            without a node, which get mu = sigma = 0)
        intensity_scale : the image is divided by it after the synthesis
        noise_seed : seed of the per-voxel noise of step 4b
    """
    n = tree.n_colors
    C = np.array([draw(cfg['colors_mu'], rng) for _ in range(n)], float)
    V = np.array([draw(cfg['colors_sigma'], rng) for _ in range(n)], float)
    if cfg['resolution'] == 'acquisition':
        # coarsest axis: a structure must be resolved along every axis
        res = float(max(acquisition['spacing_mm']))
    else:
        res = float(draw(cfg['resolution'], rng))
    contrast = acquisition['contrast']
    stats = node_stats(tree, C, V, rng, res, contrast, cfg['min_sigma'])
    rep = tree.representative(res, contrast)
    l2n = tree.label_to_node

    mu, sigma, class_of, class_names = [], [], [], []
    for value in label_space.values:
        node = l2n.get(value)
        name = rep[node] if node is not None else 'unassigned'
        m, s = stats[node] if node is not None else (0.0, 0.0)
        if name not in class_names:
            class_names.append(name)
        mu.append(m)
        sigma.append(s)
        class_of.append(class_names.index(name))
    return dict(colors_mu=C.tolist(), colors_sigma=V.tolist(),
                resolution_mm=res, contrast=contrast, mu=mu, sigma=sigma,
                class_of=class_of, class_names=class_names,
                intensity_scale=float(cfg['intensity_scale']),
                noise_seed=new_seed(rng))
