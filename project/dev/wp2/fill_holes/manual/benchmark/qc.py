"""QC figures: before / after / changed voxels coloured by restoration step."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

L1_CMAP = ListedColormap(["black"] + [plt.cm.tab10(i) for i in range(10)])
# step codes: 1 gap, 2 crack, 3 hole, 4 shift
STEP_CMAP = ListedColormap([(0, 0, 0, 0), (1, .15, .15, 1), (.2, .6, 1, 1), (.2, 1, .3, 1), (1, .9, .1, 1)])
VIEWS = {"sagittal": 0, "coronal": 1, "axial": 2}


def _take(v, view, i, crop=None):
    s = np.take(v, i, axis=VIEWS[view]).T
    return s if crop is None else s[crop]


def qc_figure(before_l1, after_l1, steps, path, at=None, views=("sagittal", "coronal", "axial"),
              crop=None, title="", labels=("input", "restored", "changed: red gap, blue crack, green hole, yellow shift")):
    """before/after are L1 label arrays (for colours); steps = step codes."""
    if at is None:
        fg = before_l1 > 0
        at = [int(np.mean(np.nonzero(fg.any(axis=tuple(a for a in range(3) if a != ax)))))
              for ax in range(3)]
    fig, ax = plt.subplots(len(views), 3, figsize=(15, 4.6 * len(views)), squeeze=False)
    for r, v in enumerate(views):
        i = at[VIEWS[v]]
        cr = None if crop is None else crop.get(v)
        ax[r, 0].imshow(_take(before_l1, v, i, cr), cmap=L1_CMAP, vmin=0, vmax=10, origin="lower", interpolation="nearest")
        ax[r, 1].imshow(_take(after_l1, v, i, cr), cmap=L1_CMAP, vmin=0, vmax=10, origin="lower", interpolation="nearest")
        ax[r, 2].imshow(_take(before_l1, v, i, cr) > 0, cmap="gray", vmin=0, vmax=3, origin="lower", interpolation="nearest")
        ax[r, 2].imshow(_take(steps, v, i, cr), cmap=STEP_CMAP, vmin=0, vmax=4, origin="lower", interpolation="nearest")
        for c, t in enumerate(labels):
            ax[r, c].set_title(f"{v} {i} - {t}", fontsize=9)
            ax[r, c].axis("off")
    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(path, dpi=75)
    plt.close(fig)
