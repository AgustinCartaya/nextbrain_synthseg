#!/usr/bin/env python3
"""
Convert an nxb16, nxbsupersynth or real SuperSynth segmentation to the common label space
defined in common_space.yaml (see README.md). The tests that prove it works are in
test_conversion/.

CLI
---
    python to_common_space.py --ontology nxb16 -i seg.nii.gz -o seg_common.nii.gz
    python to_common_space.py --ontology nxbsupersynth -i dir/ -o out_dir/
    python to_common_space.py --ontology supersynth -i ss.nii.gz -o ss_common.nii.gz

  Labels not in common_space.yaml raise an error (--unknown ignore maps them to ignore_label).
  The output keeps the grid of the input (uint8).

Python
------
    from to_common_space import load_common_space, to_common, comparable_mask
    cs = load_common_space()
    a = to_common(seg16, 'nxb16', cs)
    b = to_common(segss, 'nxbsupersynth', cs)
    valid = comparable_mask(a, b, 'nxb16', 'nxbsupersynth', cs)   # voxels to use in metrics
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import nibabel as nib
import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
DEFAULT_CS = HERE / 'common_space.yaml'
ONTOLOGIES = ('nxb16', 'nxbsupersynth', 'supersynth')


def load_common_space(path=DEFAULT_CS):
    """
    Read and check common_space.yaml. Adds:
      cs['lut'][ontology]        {source id: common id} (ignored ids -> ignore_label)
      cs['available'][ontology]  common ids that the ontology can produce
    """
    with open(path) as f:
        cs = yaml.safe_load(f)
    ign = cs['ignore_label']
    ids = [c['id'] for c in cs['classes']]
    if len(set(ids)) != len(ids) or ign in ids:
        raise ValueError('Common ids must be unique and different from ignore_label')
    cs['lut'], cs['available'] = {}, {}
    for ont in ONTOLOGIES:
        lut = {}
        for c in cs['classes']:
            for i in c.get(ont) or []:
                if i in lut:
                    raise ValueError(f'{ont} label {i} is in more than one common class')
                lut[i] = c['id']
        for i in cs['ignore'].get(ont) or []:
            if i in lut:
                raise ValueError(f'{ont} label {i} is both in a common class and ignored')
            lut[i] = ign
        cs['lut'][ont] = lut
        cs['available'][ont] = {c['id'] for c in cs['classes'] if c.get(ont)}
    return cs


def to_common(seg, ontology, cs, unknown='error'):
    """
    Relabel an nxb16 / nxbsupersynth / supersynth label array to the common space (uint8).
    unknown: 'error' (stop if a label is not in common_space.yaml) or 'ignore' (-> ignore_label).
    """
    seg = np.rint(np.asarray(seg)).astype(np.int64)
    lut = cs['lut'][ontology]
    present = np.unique(seg)
    missing = sorted(set(present.tolist()) - set(lut))
    if missing and unknown == 'error':
        raise ValueError(f'{ontology} labels not in the common space: {missing}')
    table = np.full(max(int(present.max()), max(lut)) + 1, cs['ignore_label'], dtype=np.uint8)
    for src, dst in lut.items():
        table[src] = dst
    return table[seg]


def comparable_mask(a, b, ont_a, ont_b, cs):
    """
    Voxels on which two common-space maps can be compared: not ignore_label in either map and
    not in a class that the other ontology does not have (e.g. SN/STN/RN against supersynth).
    """
    valid = (a != cs['ignore_label']) & (b != cs['ignore_label'])
    for x, ont_other in ((a, ont_b), (b, ont_a)):
        missing = [c['id'] for c in cs['classes'] if c['id'] not in cs['available'][ont_other]]
        if missing:
            valid &= ~np.isin(x, missing)
    return valid


# ================================================================================================ #

def convert_file(src, dst, ontology, cs_path=DEFAULT_CS, unknown='error'):
    cs = load_common_space(cs_path)
    img = nib.load(src)
    out = to_common(np.asanyarray(img.dataobj), ontology, cs, unknown)
    hdr = img.header.copy()
    hdr.set_data_dtype(np.uint8)
    hdr['scl_slope'], hdr['scl_inter'] = 1, 0
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(out, img.affine, hdr), dst)
    return Path(dst).name, np.unique(out).tolist()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--ontology', choices=ONTOLOGIES, required=True)
    p.add_argument('-i', '--input', type=Path, required=True, help='file or directory')
    p.add_argument('-o', '--output', type=Path, required=True, help='file or directory')
    p.add_argument('--common-space', type=Path, default=DEFAULT_CS)
    p.add_argument('--unknown', choices=['error', 'ignore'], default='error')
    p.add_argument('--jobs', type=int, default=5)
    args = p.parse_args()

    if args.input.is_dir():
        srcs = sorted(args.input.glob('*.nii*'))
        dsts = [args.output / f.name for f in srcs]
    else:
        srcs, dsts = [args.input], [args.output]
    n = len(srcs)
    with ProcessPoolExecutor(args.jobs) as ex:
        for name, present in ex.map(convert_file, srcs, dsts, [args.ontology] * n,
                                    [args.common_space] * n, [args.unknown] * n):
            print(f'{name}: common labels {present}', flush=True)


if __name__ == '__main__':
    main()
