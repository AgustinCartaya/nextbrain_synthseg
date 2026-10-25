"""
Offline launcher: build HR templates from label maps.

    python -m nxgen.cli.build_templates --config configs/nextbrain.json \
        --seg data/case-1.nii.gz [more maps] --out templates/ [--device cpu]
    python -m nxgen.cli.build_templates --config ... --seg-dir maps/ --out ...

Reads the 'template' and 'labels' sections of the config:
    template.hr_voxel_mm, template.sigma_mm, template.margin_mm,
    labels.tree, labels.background, labels.unassigned
With labels.unassigned = 'background', labels that are not in the tree are
mapped to the background in the template; with 'error' they are kept (and
the generator raises when it loads the template).

A template is rebuilt only if its source file or its settings changed (or
with --overwrite). See nxgen.offline.template.build_template.
"""

import argparse
import glob
import os
import sys

from ..core.config import load_config
from ..core.tree import LabelTree
from ..offline.template import build_template


def build_parser():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[1],
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('--config', required=True, help='config JSON')
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument('--seg', nargs='+', help='label map(s) (.nii/.nii.gz)')
    src.add_argument('--seg-dir', help='folder of label maps (*.nii.gz)')
    p.add_argument('--out', required=True, help='output folder')
    p.add_argument('--device', default='cpu',
                   help="device of the super-resolution (default 'cpu': "
                        'large maps need a lot of memory)')
    p.add_argument('--overwrite', action='store_true',
                   help='rebuild even if a matching template exists')
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config)
    paths = args.seg or sorted(glob.glob(os.path.join(args.seg_dir, '*.nii*')))
    if not paths:
        sys.exit('No label maps found')
    allowed = None
    if cfg['labels']['unassigned'] == 'background':
        if cfg['labels']['tree'] is None:
            sys.exit("labels.unassigned = 'background' needs labels.tree")
        allowed = list(LabelTree.from_json(cfg['labels']['tree'])
                       .label_to_node)
    t = cfg['template']
    for path in paths:
        out, record = build_template(
            path, args.out, t['hr_voxel_mm'], t['sigma_mm'], t['margin_mm'],
            allowed, cfg['labels']['background'], args.device, args.overwrite)
        print(f'{out}: shape {record["shape"]}, removed labels '
              f'{record["removed_labels"]}, vanished labels '
              f'{record["vanished_labels"]}')


if __name__ == '__main__':
    main()
