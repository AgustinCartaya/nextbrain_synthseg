"""
Draw a LabelTree and show which nodes are visible at a given resolution / contrast.

    from tree_plot import plot_tree
    plot_tree(tree, resolution=0.7, contrast='T1', save='tree_0.7mm.png')

Visible nodes are drawn with a solid colored box (one color per generation class).
Collapsed nodes are faded, dashed, use the color of the node whose Gaussian they copy and
show "-> <node>". Null nodes are drawn in dark grey.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb
from matplotlib.patches import Patch

from tree_sampling import INF, LabelTree, _as_resolution


def _layout(tree):
    """Tidy layout: leaves get consecutive slots, parents are centered on their children."""
    pos, next_slot = {}, [0]

    def _rec(name):
        children = tree.nodes[name].children
        if not children:
            pos[name] = next_slot[0]
            next_slot[0] += 1
        else:
            for c in children:
                _rec(c)
            pos[name] = 0.5 * (pos[children[0]] + pos[children[-1]])

    _rec(tree.root)
    return pos, next_slot[0]


def _node_colors(tree, cmap=None):
    """One color per node, by tree order; consecutive nodes get clearly different colors."""
    nodes = [n for n in tree.order() if n != tree.root]
    if cmap is None:
        t10 = [c for i, c in enumerate(plt.get_cmap('tab10').colors) if i != 7]   # no grey (= null)
        if len(nodes) <= len(t10):
            cols = t10
        else:  # tab20 without greys, strong and light shades far apart in the cycle
            t20 = [c for i, c in enumerate(plt.get_cmap('tab20').colors) if i not in (14, 15)]
            cols = list(t20[0::2]) + list(t20[1::2])
    else:
        pal = plt.get_cmap(cmap)
        cols = [pal(i / max(1, len(nodes) - 1)) for i in range(len(nodes))] if pal.N > 32 else \
               [pal(i % pal.N) for i in range(len(nodes))]
    return {n: cols[i % len(cols)] for i, n in enumerate(nodes)}


_LUT_CACHE = {}


def load_freesurfer_lut(path=None):
    """Reads FreeSurferColorLUT.txt -> {label: (name, (r, g, b))}, colors in [0, 1].

    path=None looks in $FREESURFER_HOME, next to this file and in the working directory.
    """
    if path is None:
        candidates = [os.path.join(os.environ.get('FREESURFER_HOME', ''), 'FreeSurferColorLUT.txt'),
                      os.path.join(os.path.dirname(os.path.abspath(__file__)), 'FreeSurferColorLUT.txt'),
                      'FreeSurferColorLUT.txt']
        path = next((p for p in candidates if os.path.isfile(p)), None)
        if path is None:
            raise FileNotFoundError('FreeSurferColorLUT.txt not found; pass lut=<path>')
    if path in _LUT_CACHE:
        return _LUT_CACHE[path]

    lut = {}
    with open(path) as f:
        for line in f:
            tok = line.split('#', 1)[0].split()
            if len(tok) < 6 or not tok[0].isdigit():
                continue
            try:
                r, g, b = (int(v) for v in tok[-4:-1])
            except ValueError:
                continue
            lut[int(tok[0])] = (' '.join(tok[1:-4]), (r / 255, g / 255, b / 255))
    _LUT_CACHE[path] = lut
    return lut


def _node_color_freesurfer(tree, lut=None, resolution=None, contrast=None, fallback=None):
    """One color per node taken from the FreeSurfer LUT.

    - A node with labels uses the LUT color of its first label.
    - A node without labels (pure group, e.g. 'deep_gm') gets the mean color of its children.
    - A label missing from the LUT falls back to `fallback` (dict, default: _node_colors(tree)).
    - resolution (mm, scalar or per axis) and contrast: if given, every node that is not visible
      takes the color of the node whose Gaussian it copies, i.e. collapsed children get the
      color of their (deepest visible) parent. None = every node keeps its own color.
    Returns {node: (r, g, b)} for every non-root node.
    """
    table = lut if isinstance(lut, dict) else load_freesurfer_lut(lut)
    fallback = _node_colors(tree) if fallback is None else fallback

    color = {}
    for name in reversed(tree.order()):              # children before parents
        if name == tree.root:
            continue
        n = tree.nodes[name]
        if n.labels:
            entry = table.get(n.labels[0])
            color[name] = entry[1] if entry else tuple(fallback[name][:3])
        elif n.children:
            color[name] = tuple(float(v) for v in np.mean([color[c] for c in n.children], axis=0))
        else:
            color[name] = tuple(fallback[name][:3])

    if resolution is not None:
        rep = tree.representative(resolution, contrast)
        color = {name: color[rep[name]] for name in color}
    return color


def _text_color(face):
    """Black or white text, whichever reads better on `face`."""
    r, g, b = to_rgb(face)
    return 'white' if 0.299 * r + 0.587 * g + 0.114 * b < 0.5 else 'black'


def _fmt_r(r):
    return 'always' if r == INF else f'{r:g} mm'


def plot_tree(tree, resolution, contrast=None, orientation='horizontal', show_labels=True,
              show_root=False, ax=None, save=None, cmap=None, colors=None, lut=None):
    """Draws `tree` highlighting the nodes visible at `resolution` (mm) and `contrast`.

    tree        : LabelTree, dict or path to the JSON file.
    resolution  : voxel size in mm (scalar or per-axis spacing; the coarsest axis is used).
    contrast    : key of r_max to use (falls back to 'default').
    orientation : 'horizontal' (levels left -> right, good for big trees) or 'vertical'.
    cmap        : None (tab10 / tab20) or any matplotlib colormap name.
    colors      : None (palette from cmap), 'freesurfer' (FreeSurfer LUT colors, see `lut`)
                  or a dict {node: color}.
    lut         : path to FreeSurferColorLUT.txt (or an already loaded LUT dict).
    Returns the matplotlib Axes.
    """
    tree = LabelTree.load(tree)
    res = _as_resolution(resolution)
    vis = tree.visibility(res, contrast)
    rep = tree.representative(res, contrast)
    pos, n_slots = _layout(tree)
    names = [n for n in tree.order() if show_root or n != tree.root]
    depth = max(tree.nodes[n].level for n in names)

    # Fixed color per node (independent of the resolution, so plots are comparable)
    # Generation classes = distinct nodes whose Gaussian is used by at least one label
    classes = sorted({rep[n] for n in names if tree.nodes[n].labels})
    if colors is None:
        color = _node_colors(tree, cmap)
    elif isinstance(colors, str) and colors.lower() == 'freesurfer':
        color = _node_color_freesurfer(tree, lut, fallback=_node_colors(tree, cmap))
    else:
        color = dict(colors)

    horizontal = orientation == 'horizontal'
    if ax is None:
        h_slot = 0.95 if show_labels else 0.7                  # room for up to 4 text lines per box
        size = (3.0 * (depth + 1), max(3, h_slot * n_slots)) if horizontal else \
               (max(5, 1.7 * n_slots), 2.0 * (depth + 1))
        _, ax = plt.subplots(figsize=size)

    def xy(name):
        lvl, slot = tree.nodes[name].level, pos[name]
        return (lvl, -slot) if horizontal else (slot, -lvl)

    # Edges
    for name in names:
        p = tree.nodes[name].parent
        if p is None or (p == tree.root and not show_root):
            continue
        (x0, y0), (x1, y1) = xy(p), xy(name)
        if horizontal:
            xm = 0.5 * (x0 + x1)
            xs, ys = [x0, xm, xm, x1], [y0, y0, y1, y1]
        else:
            ym = 0.5 * (y0 + y1)
            xs, ys = [x0, x0, x1, x1], [y0, ym, ym, y1]
        ax.plot(xs, ys, color='0.3' if vis[name] else '0.75', lw=1.2 if vis[name] else 0.8,
                ls='-' if vis[name] else '--', zorder=1)

    # Nodes
    for name in names:
        n = tree.nodes[name]
        x, y = xy(name)
        r_txt = _fmt_r(n.r_max_for(contrast))
        lines = [name, f'r_max {r_txt}']
        if show_labels and n.labels:
            labs = n.labels if len(n.labels) <= 3 else n.labels[:3] + ['...']
            lines.append('labels ' + ', '.join(map(str, labs)))

        if name == tree.root:
            face, edge, alpha, style = 'white', 'black', 1.0, '-'
        elif n.null or any(tree.nodes[a].null for a in tree.ancestors(name)):
            face, edge, alpha, style = '0.25', 'black', 1.0, '-'
        elif vis[name]:
            face, edge, alpha, style = color[name], 'black', 1.0, '-'
        else:
            face, edge, alpha, style = color[rep[name]], '0.5', 0.35, '--'
            lines.append(f'-> {rep[name]}')

        text_color = _text_color(face) if alpha == 1.0 else 'black'
        ax.text(x, y, '\n'.join(lines), ha='center', va='center', fontsize=8, color=text_color,
                zorder=3, bbox=dict(boxstyle='round,pad=0.35', fc=face, ec=edge, ls=style,
                                    lw=1.2, alpha=alpha))

    n_leaves_vis = sum(vis[l] for l in tree.leaves())
    title = (f'Resolution {res:g} mm' + (f', {contrast}' if contrast else '') +
             f': {len(classes)} generation classes, {n_leaves_vis}/{len(tree.leaves())} leaves visible')
    ax.set_title(title, fontsize=11)
    ax.legend(handles=[Patch(fc='0.6', ec='black', label='visible (own Gaussian)'),
                       Patch(fc='0.6', ec='0.5', ls='--', alpha=0.35, label='collapsed (copies node after ->)'),
                       Patch(fc='0.25', ec='black', label='null color')],
              loc='upper center', bbox_to_anchor=(0.5, -0.01), ncol=3, fontsize=8, frameon=False)

    lo = 1 if not show_root else 0
    if horizontal:
        ax.set_xlim(lo - 0.6, depth + 0.6)
        ax.set_ylim(-n_slots + 0.4, 0.6)
    else:
        ax.set_xlim(-0.8, n_slots - 0.2)
        ax.set_ylim(-depth - 0.6, -lo + 0.6)
    ax.axis('off')

    if save:
        ax.figure.tight_layout()
        ax.figure.savefig(save, dpi=110, bbox_inches='tight')
    return ax


if __name__ == '__main__':
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else 'toy_tree.json'
    r = float(sys.argv[2]) if len(sys.argv) > 2 else 0.7
    c = sys.argv[3] if len(sys.argv) > 3 else None
    plot_tree(path, r, c, save=f'tree_{r:g}mm{"_" + c if c else ""}.png')


import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot a tree.')

    parser.add_argument('--i', required=True, help='Input JSON file.')
    parser.add_argument('--o', default=None, help='Output image file.')
    parser.add_argument('--resolution', type=float, default=0.7, help='Resolution in mm.')
    parser.add_argument('--contrast', default=None, help='Contrast key.')
    parser.add_argument('--orientation', choices=['horizontal', 'vertical'],
                        default='horizontal', help='Tree orientation.')
    parser.add_argument('--show-labels', action=argparse.BooleanOptionalAction,
                        default=True, help='Show node labels.')
    parser.add_argument('--show-root', action='store_true', help='Show root node.')
    parser.add_argument('--cmap', default=None, help='Matplotlib colormap name.')
    parser.add_argument('--lut', default=None, help='Path to FreeSurferColorLUT.txt (or None).')

    args = parser.parse_args()

    output = args.o or (
        f'tree_{args.resolution:g}mm'
        f'{"_" + args.contrast if args.contrast else ""}.png'
    )

    plot_tree(
        tree=args.i,
        resolution=args.resolution,
        contrast=args.contrast,
        orientation=args.orientation,
        show_labels=args.show_labels,
        show_root=args.show_root,
        save=output,
        cmap=args.cmap,
        lut=args.lut
    )

# CLI:
# python tree_plot.py --i /home/agustin/phd/fast_nextbrain2/project/wp3/simple_generator/data/inputs/synthseg_tree.json --resolution 1 --orientation horizontal

# CLI:
# python tree_plot.py /home/agustin/phd/fast_nextbrain2/project/wp3/simple_generator/data/inputs/nextbrain_tree_example.json