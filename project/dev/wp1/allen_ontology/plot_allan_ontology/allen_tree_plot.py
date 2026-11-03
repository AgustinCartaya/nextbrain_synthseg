"""
Draw an AllenTree (Allen structure graph or NextBrain-Allen tree).

Equivalent of `create_nextbraintree/tree_plot.py` for the Allen format.

    from allen_tree_plot import plot_tree
    plot_tree('AllenBrainOntologyDev.json', root='THM', max_depth=3, save='thalamus.png')

The Allen ontology has no resolution model, so instead of visible / collapsed nodes the plot shows
where the drawing was cut:
  - boxes use the Allen color of each structure (or a palette with `cmap`);
  - a node whose children are not drawn (beyond `max_depth`) is dashed and shows "+N below";
  - NextBrain leaves (NextBrain-Allen tree) have a thick border and show their label.
The full ontology (3317 nodes) is too big for one readable figure: select a part with `root`,
`max_depth` or `keep` (only these structures and their ancestors), or save as .svg / .pdf.
"""

import argparse
import textwrap
import warnings

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.patches import Patch

from allen_tree import AllenTree

MAX_PIXELS = 65000            # Agg limit is 2**16 pixels per side


def _layout(tree, children):
    """Tidy layout: leaves get consecutive slots, parents are centered on their children."""
    pos, next_slot = {}, [0]

    def _rec(name):
        ch = children[name]
        if not ch:
            pos[name] = next_slot[0]
            next_slot[0] += 1
        else:
            for c in ch:
                _rec(c)
            pos[name] = 0.5 * (pos[ch[0]] + pos[ch[-1]])

    _rec(tree.root)
    return pos, next_slot[0]


def _node_colors(names, cmap=None):
    """One color per node, by tree order; consecutive nodes get clearly different colors."""
    if cmap is None:
        t20 = [c for i, c in enumerate(plt.get_cmap('tab20').colors) if i not in (14, 15)]
        cols = list(t20[0::2]) + list(t20[1::2])
    else:
        pal = plt.get_cmap(cmap)
        cols = [pal(i / max(1, len(names) - 1)) for i in range(len(names))] if pal.N > 32 else \
               [pal(i % pal.N) for i in range(len(names))]
    return {n: cols[i % len(cols)] for i, n in enumerate(names)}


def _text_color(face):
    """Black or white text, whichever reads better on `face`."""
    r, g, b = to_rgb(face)
    return 'white' if 0.299 * r + 0.587 * g + 0.114 * b < 0.5 else 'black'


def _wrap(text, width=26, max_lines=2):
    lines = textwrap.wrap(text, width) or ['']
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] = lines[-1][:width - 3] + '...'
    return lines


def plot_tree(tree, root=None, max_depth=None, keep=None, orientation='horizontal', show_ids=True,
              show_names=True, ax=None, save=None, cmap=None, colors=None, dpi=300):
    """Draws (part of) an Allen tree.

    tree        : AllenTree, dict or path to the JSON file.
    root        : node to start from (key, Allen id, acronym or exact name). None = tree root.
    max_depth   : levels drawn below `root` (None = all). Cut nodes are dashed with "+N below".
    keep        : list of structures; only they and their ancestors are drawn (applied first).
    orientation : 'horizontal' (levels left -> right, good for big trees) or 'vertical'.
    show_ids    : add the Allen id (or NextBrain label) to each box.
    show_names  : add the full name under the acronym.
    colors      : None (Allen colors), 'palette' (colors from `cmap`) or a dict {node key: color}.
    cmap        : matplotlib colormap name, used when colors='palette' (default tab20).
    Returns the matplotlib Axes.
    """
    full = AllenTree.load(tree)
    t = full.prune_to(keep) if keep else full
    if root is not None:
        t = t.subtree(root)
    base = t.nodes[t.root].depth

    def drawn(k):
        return max_depth is None or t.nodes[k].depth - base <= max_depth

    names = [k for k in t.order() if drawn(k)]
    children = {k: [c for c in t.nodes[k].children if drawn(c)] for k in names}
    hidden = {k: len(t.descendants(k)) for k in names if t.nodes[k].children and not children[k]}
    pos, n_slots = _layout(t, children)
    depth = max(t.nodes[k].depth for k in names) - base

    if colors is None:
        fallback = _node_colors(names, cmap)
        color = {k: t.nodes[k].rgb or fallback[k] for k in names}
    elif isinstance(colors, str) and colors.lower() == 'palette':
        color = _node_colors(names, cmap)
    else:
        color = dict(colors)

    horizontal = orientation == 'horizontal'
    if ax is None:
        n_lines = 1 + 2 * show_names + 1                       # acronym, name (2), extra line
        h_slot = 0.25 * n_lines
        size = (3.2 * (depth + 1), max(3, h_slot * n_slots)) if horizontal else \
               (max(5, 1.9 * n_slots), 0.5 * n_lines * (depth + 1))
        _, ax = plt.subplots(figsize=size)

    def xy(name):
        lvl, slot = t.nodes[name].depth - base, pos[name]
        return (lvl, -slot) if horizontal else (slot, -lvl)

    # Edges
    for name in names:
        p = t.nodes[name].parent
        if p is None or name == t.root:
            continue
        (x0, y0), (x1, y1) = xy(p), xy(name)
        if horizontal:
            xm = 0.5 * (x0 + x1)
            xs, ys = [x0, xm, xm, x1], [y0, y0, y1, y1]
        else:
            ym = 0.5 * (y0 + y1)
            xs, ys = [x0, x0, x1, x1], [y0, ym, ym, y1]
        ax.plot(xs, ys, color='0.3', lw=1.0, zorder=1)

    # Nodes
    for name in names:
        n = t.nodes[name]
        x, y = xy(name)
        if n.kind == 'leaf':
            head = f'{n.id} {n.name}' if show_ids else n.name
            lines = _wrap(head, 30, 2) + ([f'Allen {n.acronym}'] if n.acronym else [])
        else:
            lines = [f'{n.acronym} ({n.id})' if show_ids else n.acronym]
            if show_names:
                lines += _wrap(n.name)
        if name in hidden:
            lines.append(f'+{hidden[name]} below')

        face = color[name]
        edge, lw, style = 'black', (2.2 if n.kind == 'leaf' else 1.0), ('--' if name in hidden else '-')
        ax.text(x, y, '\n'.join(lines), ha='center', va='center', fontsize=7, color=_text_color(face),
                zorder=3, bbox=dict(boxstyle='round,pad=0.3', fc=face, ec=edge, ls=style, lw=lw))

    r = t.nodes[t.root]
    title = (f'{r.acronym or r.name} ({r.id}): {len(names)} of {len(t.nodes)} nodes drawn, '
             f'depth {depth}' + (f' (cut at {max_depth})' if max_depth is not None else ''))
    ax.set_title(title, fontsize=11)
    handles = [Patch(fc='0.8', ec='black', label='Allen structure'),
               Patch(fc='0.8', ec='black', ls='--', label='children not drawn (+N below)')]
    if any(t.nodes[k].kind == 'leaf' for k in names):
        handles.append(Patch(fc='0.8', ec='black', lw=2.2, label='NextBrain leaf'))
    ax.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, -0.01), ncol=len(handles),
              fontsize=8, frameon=False)

    if horizontal:
        ax.set_xlim(-0.6, depth + 0.6)
        ax.set_ylim(-n_slots + 0.4, 0.6)
    else:
        ax.set_xlim(-0.8, n_slots - 0.2)
        ax.set_ylim(-depth - 0.6, 0.6)
    ax.axis('off')

    if save:
        fig = ax.figure
        fig.tight_layout()
        if not save.lower().endswith(('.svg', '.pdf')):
            dpi = min(dpi, int(MAX_PIXELS / max(fig.get_size_inches())))
            if dpi < 72:
                warnings.warn(f'Figure is {fig.get_size_inches().round(0)} in: saving at {dpi} dpi. '
                              f'Use root / max_depth / keep, or save as .svg / .pdf.')
        fig.savefig(save, dpi=dpi, bbox_inches='tight')
    return ax


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot (part of) the Allen ontology.')

    parser.add_argument('--i', required=True, help='Input JSON (Allen structure graph or nextbrain_allen_tree).')
    parser.add_argument('--o', default=None, help='Output image file (.png, .svg, .pdf).')
    parser.add_argument('--root', default=None, help='Start node: Allen id, acronym or exact name.')
    parser.add_argument('--max-depth', type=int, default=None, help='Levels drawn below the root.')
    parser.add_argument('--keep', nargs='+', default=None,
                        help='Only these structures (id / acronym / name) and their ancestors.')
    parser.add_argument('--orientation', choices=['horizontal', 'vertical'],
                        default='horizontal', help='Tree orientation.')
    parser.add_argument('--show-ids', action=argparse.BooleanOptionalAction, default=True,
                        help='Show Allen ids / NextBrain labels.')
    parser.add_argument('--show-names', action=argparse.BooleanOptionalAction, default=True,
                        help='Show full structure names.')
    parser.add_argument('--palette', action='store_true', help='Use a palette instead of Allen colors.')
    parser.add_argument('--cmap', default=None, help='Matplotlib colormap name (with --palette).')
    parser.add_argument('--print', action='store_true', help='Also print the drawn tree as text.')

    args = parser.parse_args()

    output = args.o or (
        f'allen_tree{"_" + args.root if args.root else ""}'
        f'{f"_d{args.max_depth}" if args.max_depth is not None else ""}.png'
    )

    if args.print:
        t = AllenTree.load(args.i)
        t = t.prune_to(args.keep) if args.keep else t
        t.print_tree(args.root, args.max_depth)

    plot_tree(
        tree=args.i,
        root=args.root,
        max_depth=args.max_depth,
        keep=args.keep,
        orientation=args.orientation,
        show_ids=args.show_ids,
        show_names=args.show_names,
        save=output,
        cmap=args.cmap,
        colors='palette' if args.palette else None,
    )
    print(output)

# CLI:
# python allen_tree_plot.py --i ../data/inputs/AllenBrainOntologyDev.json --max-depth 5 --o outputs/allen_ontology_d5.png
# python allen_tree_plot.py --i ../data/inputs/AllenBrainOntologyDev.json --root THM --max-depth 3 --print
# python allen_tree_plot.py --i ../data/inputs/AllenBrainOntologyDev.json --keep Pu Ca GP THM
# python allen_tree_plot.py --i outputs/nextbrain_allen_tree_gyral.json --root BN --o outputs/nextbrain_allen_BN.png
# python allen_tree_plot.py --i ../data/inputs/AllenBrainOntologyDev.json --o outputs/allen_ontology_full.svg --root brain
# python allen_tree_plot.py --i ../data/inputs/AllenBrainOntologyDev.json --o outputs/allen_ontology_hss.png --root HSS
# python allen_tree_plot.py --i ../data/inputs/AllenBrainOntologyDev.json --o outputs/allen_ontology_cb.png --root CB
