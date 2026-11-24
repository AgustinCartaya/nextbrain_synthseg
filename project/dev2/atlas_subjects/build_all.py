#!/usr/bin/env python3
"""Run build_nn.py or build_signed_distance.py for every NextBrain case.

Cases are the folders <root>/NextBrain-Case-*/<subject>/ that contain labels_blocks/.
Outputs: <out-dir>/rec_<method>_<res>mm/case-1.nii.gz, case-2.nii.gz, ...
Any argument not recognised here is passed through unchanged to the build script
(e.g. --res, --save-steps, --cerebellum-remap ...).

Usage:
    python build_all.py --method nn --res 0.2
    python build_all.py --method sd --res 0.2 --save-steps -j 2
    python build_all.py --method sd --cases 1 3      # only NextBrain-Case-1 and -3
"""
import argparse
import glob
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = {'nn': 'build_nn.py', 'sd': 'build_signed_distance.py'}
ROOT = '/home/agustin/phd/synthseg_nextbrain/project/nextbrain_atlas_files/registered_images'
OUT = os.path.join(HERE, 'data', 'outputs', 'originals')


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--method', required=True, choices=SCRIPTS,
                   help='nn = build_nn.py, sd = build_signed_distance.py')
    p.add_argument('--root', default=ROOT, help=f'Folder with the NextBrain-Case-* repos (default {ROOT})')
    p.add_argument('--out-dir', default=OUT, help=f'Output folder (default {OUT})')
    p.add_argument('--cases', nargs='*', type=int, default=None,
                   help='Only these case numbers (default: all)')
    p.add_argument('-j', '--jobs', type=int, default=1,
                   help='Cases run in parallel (default 1; each case needs several GB of RAM at 0.2 mm)')
    p.add_argument('--logs', action='store_true',
                   help='Write each case output to <out-dir>/logs/<case>_<method>.log instead of the terminal')
    return p.parse_known_args()


def find_cases(root, numbers):
    cases = []
    for repo in sorted(glob.glob(os.path.join(root, 'NextBrain-Case-*')),
                       key=lambda x: int(x.rsplit('-', 1)[-1])):
        if numbers is not None and int(repo.rsplit('-', 1)[-1]) not in numbers:
            continue
        cases += sorted(d for d in glob.glob(os.path.join(repo, '*'))
                        if os.path.isdir(os.path.join(d, 'labels_blocks')))
    return cases


def case_name(case_dir):
    """.../NextBrain-Case-3/P58-16 -> case-3"""
    return 'case-' + os.path.basename(os.path.dirname(case_dir)).rsplit('-', 1)[-1]


def run(case_dir, args, extra):
    subject = case_name(case_dir)
    cmd = [sys.executable, os.path.join(HERE, SCRIPTS[args.method]),
           '--case-dir', case_dir, '--out-dir', args.out_dir, '--name', subject, *extra]
    t0 = time.time()
    if args.logs:
        log_dir = os.path.join(args.out_dir, 'logs')
        os.makedirs(log_dir, exist_ok=True)
        with open(os.path.join(log_dir, f'{subject}_{args.method}.log'), 'w') as f:
            rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT).returncode
    else:
        rc = subprocess.run(cmd).returncode
    print(f'[{subject}] {"OK" if rc == 0 else f"FAILED (exit {rc})"} in {time.time() - t0:.0f} s', flush=True)
    return subject, rc


def main():
    args, extra = parse_args()
    cases = find_cases(args.root, args.cases)
    if not cases:
        sys.exit(f'No cases found in {args.root}')
    os.makedirs(args.out_dir, exist_ok=True)
    print(f'{SCRIPTS[args.method]} on {len(cases)} cases -> {args.out_dir}')
    for c in cases:
        print(f'  {case_name(c)}: {os.path.relpath(c, args.root)}')
    if extra:
        print(f'  extra arguments: {" ".join(extra)}')

    with ThreadPoolExecutor(max_workers=max(args.jobs, 1)) as ex:
        results = list(ex.map(lambda c: run(c, args, extra), cases))

    failed = [s for s, rc in results if rc != 0]
    print(f'\n{len(results) - len(failed)}/{len(results)} cases done'
          + (f'; failed: {", ".join(failed)}' if failed else ''))
    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    main()
