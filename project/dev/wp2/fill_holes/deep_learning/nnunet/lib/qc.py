"""Quality control of a restored volume: figure (input / output / changed) and numbers."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402

from .corruption import hole_channel  # noqa: E402

#: view name -> array axis that is fixed (volumes are RAS: x = L-R, y = P-A, z = I-S)
VIEWS = {"sagittal": 0, "coronal": 1, "axial": 2}


def label_cmap(num_classes):
    """Background black, then well separated colours."""
    base = plt.get_cmap("tab10" if num_classes <= 11 else "gist_ncar")
    colours = [(0, 0, 0, 1)] + [base(i / max(num_classes - 2, 1)) if num_classes > 11 else base(i)
                                for i in range(num_classes - 1)]
    return ListedColormap(colours)


def qc_figure(inp, out, path, num_classes, n_slices=3, title=""):
    """Save a PNG with sagittal, coronal and axial slices.

    Columns: input | output | changed voxels (red, over the dimmed input).
    Rows:    n_slices per view, evenly spaced inside the bounding box of the labels.
    """
    fg = np.nonzero((inp > 0) | (out > 0))
    if len(fg[0]) == 0:
        return
    lo = [int(c.min()) for c in fg]
    hi = [int(c.max()) + 1 for c in fg]
    box = tuple(slice(a, b) for a, b in zip(lo, hi))
    inp, out = inp[box], out[box]
    cmap = label_cmap(num_classes)

    rows = len(VIEWS) * n_slices
    fig, axes = plt.subplots(rows, 3, figsize=(13, 4.2 * rows), squeeze=False)
    r = 0
    for view, axis in VIEWS.items():
        for q in np.linspace(0, 1, n_slices + 2)[1:-1]:
            pos = int(q * (inp.shape[axis] - 1))
            a = np.take(inp, pos, axis=axis).T          # .T + origin="lower": superior/anterior up
            b = np.take(out, pos, axis=axis).T
            changed = a != b
            kw = dict(cmap=cmap, vmin=0, vmax=num_classes - 1, origin="lower",
                      interpolation="nearest")
            axes[r, 0].imshow(a, **kw)
            axes[r, 1].imshow(b, **kw)
            axes[r, 2].imshow(a, alpha=0.35, **kw)
            axes[r, 2].imshow(np.ma.masked_equal(changed, 0), cmap=ListedColormap(["red"]),
                              origin="lower", interpolation="nearest")
            for c, name in enumerate(("input", "output", f"changed ({int(changed.sum())} px)")):
                axes[r, c].set_title(f"{view} {lo[axis] + pos} | {name}", fontsize=9)
                axes[r, c].axis("off")
            r += 1
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.985))      # leave room for the title
    fig.savefig(path, dpi=110)
    plt.close(fig)


def restoration_stats(cn, inp, out, num_classes):
    """Numbers describing what the model changed in a volume WITHOUT ground truth.

    "suspicious holes" = background voxels inside the closed brain mask, here
    computed on the whole volume. Those that remain in the output are the first
    place to look for artefacts that were not repaired.
    """
    changed = inp != out
    holes_in = hole_channel(cn, inp)
    holes_out = hole_channel(cn, out)
    n_fg = int((inp > 0).sum())
    stats = dict(
        voxels_labelled_input=n_fg,
        voxels_changed=int(changed.sum()),
        changed_fraction_of_labelled=float(changed.sum() / max(n_fg, 1)),
        filled=int((changed & (inp == 0)).sum()),                    # background -> label
        removed=int((changed & (out == 0)).sum()),                   # label -> background
        relabelled=int((changed & (inp > 0) & (out > 0)).sum()),     # label -> other label
        suspicious_holes_input=int(holes_in.sum()),
        suspicious_holes_output=int(holes_out.sum()),
        suspicious_holes_filled_fraction=float((holes_in & (out > 0)).sum() / max(holes_in.sum(), 1)),
        voxels_per_class_input=np.bincount(inp.ravel(), minlength=num_classes).tolist(),
        voxels_per_class_output=np.bincount(out.ravel(), minlength=num_classes).tolist(),
    )
    return stats


def stats_markdown(name, stats, names):
    """Markdown section for one case (used in the inference report)."""
    s = stats
    lines = [
        f"## {name}", "",
        f"- changed voxels: {s['voxels_changed']:,} "
        f"({100 * s['changed_fraction_of_labelled']:.2f} % of the labelled input voxels)",
        f"- filled (background -> label): {s['filled']:,}; removed (label -> background): "
        f"{s['removed']:,}; relabelled: {s['relabelled']:,}",
        f"- suspicious holes (background inside the closed brain mask): "
        f"{s['suspicious_holes_input']:,} -> {s['suspicious_holes_output']:,} "
        f"({100 * s['suspicious_holes_filled_fraction']:.1f} % of the input ones were filled)",
        "", "| label | name | voxels in | voxels out | change % |", "|---|---|---|---|---|",
    ]
    for c, (a, b) in enumerate(zip(s["voxels_per_class_input"], s["voxels_per_class_output"])):
        lines.append(f"| {c} | {names[c] if c < len(names) else ''} | {a:,} | {b:,} "
                     f"| {100 * (b - a) / max(a, 1):+.2f} |")
    lines += ["", "Notes on remaining artefacts: _fill in after looking at the QC figure_", ""]
    return "\n".join(lines)
