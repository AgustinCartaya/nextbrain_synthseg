"""Typical thickness, maximum thickness and minimum resolution of every node of an AllenLabelTree.

For every node of the tree (leaves and parents; root and background are skipped) the node mask is
the union of its own NextBrain labels and those of all its descendants. On each template
(a NextBrain segmentation, e.g. changed_recon_0.2mm/case-1.nii.gz) the mask is measured with the
Euclidean distance transform (EDT): EDT(x) = distance in mm from voxel x to the closest voxel
outside the mask. With v = voxel size (mm):

    max_thickness     = 2 * max(EDT) - v
        Diameter of the largest ball that fits in the structure: its thickest part.
    typical_thickness = 4 * mean(EDT) - 2v, never above max_thickness and never below v
        Thickness of the flat slab with the same mean distance to the border (in a slab of
        thickness t the mean EDT is t/4). For compact structures (balls, cylinders) it is smaller
        than the diameter, so the criterion is conservative (it asks for a finer resolution).
        The -2v (and -v above) corrects the discretization: the EDT of a voxel at the border is
        already v instead of v/2.
    min_resolution    = typical_thickness / 2
        Largest isotropic voxel that still has at least 2 voxels across the typical thickness of
        the structure. It is necessary but not sufficient: a structure whose border has no image
        contrast (delineated by geometry / atlas) does not appear at any resolution.

Each value is the median over the templates where the node is present (n_templates).
A value below the template voxel size cannot really be measured on that template (the structure is
less than 2 voxels thick); `below_voxel` flags those rows.
Nodes whose labels are absent from every template get empty values (e.g. on the
changed_recon_0.2mm cases the whole cortex is the single label 2002, so the other gyral labels
are missing).

Large masks are subsampled (by an integer factor f, so that the bounding box has at most
--max-voxels voxels) before the EDT; the voxel size used is then f * v.

Output (by the extension of --o): .csv, .tsv or .md (markdown table), with columns
    node, parent, level, is_leaf, labels, allen_id, allen_acronym, allen_name, n_templates,
    volume_mm3, typical_thickness_mm, max_thickness_mm, min_resolution_mm, below_voxel
in tree order (every parent before its children).

    python node_thickness.py --i /home/agustin/phd/synthseg_nextbrain/data/basics/changed_recon_0.2mm --o data/outputs/node_thickness.csv
    python node_thickness.py --i case-1.nii.gz case-2.nii.gz --o thickness.md --tree my_tree.json
"""
import argparse
import csv
from multiprocessing import Pool
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage

from allen_tree import AllenLabelTree

SKIP = ('root', 'background')
COLUMNS = ['node', 'parent', 'level', 'is_leaf', 'labels', 'allen_id', 'allen_acronym', 'allen_name',
           'n_templates', 'volume_mm3', 'typical_thickness_mm', 'max_thickness_mm', 'min_resolution_mm',
           'below_voxel']


def node_labels(tree):
    """{node: NextBrain labels of the node and all its descendants}, root and background excluded."""
    return {n: [lab for k in [n] + tree.descendants(n) for lab in tree.nodes[k].labels]
            for n in tree.order() if n not in SKIP}


def measure_mask(mask, spacing):
    """Volume (mm3), typical and maximum thickness (mm) of a boolean mask with voxel size `spacing`."""
    v = float(np.mean(spacing))
    edt = ndimage.distance_transform_edt(np.pad(mask, 1), sampling=spacing)
    inside = edt[edt > 0]
    max_thick = 2 * inside.max() - v
    typical = max(min(4 * inside.mean() - 2 * v, max_thick), v)
    return float(mask.sum() * np.prod(spacing)), float(typical), float(max_thick)


def measure_template(path, labels, max_voxels=6e7):
    """{node: dict(volume, typical, max, voxel)} for one template; nodes absent from it are left out."""
    im = nib.load(path)
    seg = np.asanyarray(im.dataobj).astype(np.int32)
    spacing = np.array(im.header.get_zooms()[:3], float)
    objects = ndimage.find_objects(seg)   # objects[l - 1] = bounding box of label l
    out = {}
    for node, labs in labels.items():
        boxes = [objects[l - 1] for l in labs if 0 < l <= len(objects) and objects[l - 1] is not None]
        if not boxes:
            continue
        # bounding box of all the node labels, one voxel of margin
        box = tuple(slice(max(min(b[i].start for b in boxes) - 1, 0), max(b[i].stop for b in boxes) + 1)
                    for i in range(3))
        mask = np.isin(seg[box], labs)
        volume = float(mask.sum() * np.prod(spacing))
        f = 1
        while mask.size / f ** 3 > max_voxels:
            f += 1
        _, typical, max_thick = measure_mask(mask[::f, ::f, ::f], spacing * f)
        out[node] = dict(volume=volume, typical=typical, max=max_thick, voxel=float(np.mean(spacing)))
    print(f'measured {path}', flush=True)
    return out


def node_table(tree, templates, workers=1, max_voxels=6e7):
    """One row (dict with COLUMNS) per node; values are medians over the templates."""
    labels = node_labels(tree)
    with Pool(workers) as pool:
        per_template = pool.starmap(measure_template, [(t, labels, max_voxels) for t in templates])
    rows = []
    for node in labels:
        n = tree.nodes[node]
        vals = [m[node] for m in per_template if node in m]
        row = dict(node=node, parent=n.parent, level=n.level, is_leaf=not n.children,
                   labels=' '.join(map(str, n.labels)), allen_id=n.allen_id,
                   allen_acronym=n.allen_acronym, allen_name=n.allen_name, n_templates=len(vals))
        if vals:
            med = {k: float(np.median([v[k] for v in vals])) for k in ('volume', 'typical', 'max', 'voxel')}
            row.update(volume_mm3=round(med['volume'], 3), typical_thickness_mm=round(med['typical'], 3),
                       max_thickness_mm=round(med['max'], 3), min_resolution_mm=round(med['typical'] / 2, 3),
                       below_voxel=med['typical'] / 2 < med['voxel'])
        rows.append(row)
    return rows


def save_table(rows, path):
    """Writes the rows as .csv, .tsv or .md (markdown table), chosen by the extension of `path`."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cell = lambda r, c: '' if r.get(c) is None else str(r[c])
    if path.suffix == '.md':
        lines = ['| ' + ' | '.join(COLUMNS) + ' |', '|' + '---|' * len(COLUMNS)]
        lines += ['| ' + ' | '.join(cell(r, c).replace('|', '\\|') for c in COLUMNS) + ' |' for r in rows]
        path.write_text('\n'.join(lines) + '\n')
    else:
        with open(path, 'w', newline='') as fh:
            writer = csv.writer(fh, delimiter='\t' if path.suffix == '.tsv' else ',')
            writer.writerow(COLUMNS)
            writer.writerows([[cell(r, c) for c in COLUMNS] for r in rows])


def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--i', type=Path, nargs='+', required=True,
                        help='template segmentations .nii(.gz), and/or folders of them')
    parser.add_argument('--o', type=Path, required=True, help='output table: .csv, .tsv or .md')
    parser.add_argument('--tree', type=Path, default=here / 'data/outputs/nextbrain_tree_allen_gyral.json',
                        help='AllenLabelTree json (default: %(default)s)')
    parser.add_argument('--workers', type=int, default=1, help='templates measured in parallel (default: 1)')
    parser.add_argument('--max-voxels', type=float, default=6e7,
                        help='bounding boxes larger than this are subsampled before the EDT (default: 6e7)')
    args = parser.parse_args()

    templates = [f for p in args.i for f in (sorted(p.glob('*.nii*')) if p.is_dir() else [p])]
    if not templates:
        parser.error('no .nii / .nii.gz template found')
    tree = AllenLabelTree.load(args.tree)
    rows = node_table(tree, templates, args.workers, args.max_voxels)
    save_table(rows, args.o)
    print(f'{len(rows)} nodes, {len(templates)} templates -> {args.o}')


if __name__ == '__main__':
    main()


# python node_thickness.py \
#     --i /home/agustin/phd/synthseg_nextbrain/data/basics/changed_recon_0.2mm \
#     --o /home/agustin/phd/synthseg_nextbrain/project/dev/wp1/allen_ontology/data/outputs/node_thickness.csv \
#     --workers 5
