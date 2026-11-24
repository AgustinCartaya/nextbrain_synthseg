"""Map nnU-Net predictions (label indices 0..N-1) back to the raw label values of the tree.

The networks are trained on label INDICES; label_space.json of the raw dataset gives the
raw value of every index. For nxb18 both are equal, for
nxbsimplified_csf they are not. 06_predict.sh calls it.

    python map_labels.py --input pred_indices/ --output pred_raw/
"""
import argparse
import json
import os
from pathlib import Path

import nibabel as nib
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--input', required=True, help='folder of *.nii.gz with label indices')
    p.add_argument('--output', required=True)
    p.add_argument('--label_space', default=None,
                   help='default: $nnUNet_raw/$DATASET_NAME/label_space.json')
    args = p.parse_args()

    ls_path = args.label_space or Path(os.environ['nnUNet_raw']) / os.environ['DATASET_NAME'] / 'label_space.json'
    values = np.array(json.loads(Path(ls_path).read_text())['values'])
    dtype = np.uint8 if values.max() < 256 else np.uint16 if values.max() < 65536 else np.int32

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    files = sorted(Path(args.input).glob('*.nii.gz'))
    for f in files:
        img = nib.load(f)
        idx = np.asarray(img.dataobj).astype(np.int64)
        assert idx.min() >= 0 and idx.max() < len(values), f'{f.name}: index outside label_space'
        raw = nib.Nifti1Image(values[idx].astype(dtype), img.affine, img.header)
        raw.set_data_dtype(dtype)
        nib.save(raw, out / f.name)
    print(f'{len(files)} predictions mapped to raw values ({ls_path}) -> {out}')


if __name__ == '__main__':
    main()
