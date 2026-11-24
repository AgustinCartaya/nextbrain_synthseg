#!/usr/bin/env python3
"""Run restore.py on every NIfTI under an input folder, keeping the folder structure.

    originals/rec_nn_0.2mm/case-1.nii.gz  ->  restored/rec_nn_0.2mm/case-1.nii.gz
    originals/rec_sd_0.2mm/case-1.nii.gz  ->  restored/rec_sd_0.2mm/case-1.nii.gz

*_steps.nii.gz files (step maps from build_signed_distance.py --save-steps) are ignored.

Every option that restore_all.py does not know is passed to restore.py as is, e.g.
--save-steps, --save-reports, --remove-isolated-elements, --bmask, --no-shifts...
Files whose output already exists are skipped (use --overwrite to redo them).

Usage:

    python restore_all.py
    python restore_all.py --jobs 5 --remove-isolated-elements --save-steps --save-reports
"""
import argparse
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESTORE = HERE / 'restore.py'


def run(src, dst, extra):
    dst.parent.mkdir(parents=True, exist_ok=True)
    t = time.time()
    p = subprocess.run([sys.executable, str(RESTORE), str(src), str(dst), *extra],
                       capture_output=True, text=True)
    return src, p.returncode, p.stdout + p.stderr, time.time() - t


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--input-dir', type=Path, default=HERE / 'data/outputs/adria')
    ap.add_argument('--output-dir', type=Path, default=HERE / 'data/outputs/restored_adria')
    ap.add_argument('--jobs', type=int, default=5, help='files processed in parallel (default 5)')
    ap.add_argument('--overwrite', action='store_true', help='redo files whose output already exists')
    args, extra = ap.parse_known_args()

    # *_steps.nii.gz are step maps written by build_signed_distance.py --save-steps, not segmentations
    files = sorted(f for pat in ('*.nii.gz', '*.nii') for f in args.input_dir.rglob(pat)
                   if not f.name.split('.')[0].endswith('_steps'))
    todo = []
    for src in files:
        dst = args.output_dir / src.relative_to(args.input_dir)
        if dst.exists() and not args.overwrite:
            print(f'skip (exists): {dst}')
            continue
        todo.append((src, dst))
    print(f'{len(files)} files found, {len(todo)} to restore, {args.jobs} in parallel')
    print(f'restore.py options: {" ".join(extra) or "(defaults)"}\n', flush=True)

    failed = []
    with ProcessPoolExecutor(args.jobs) as ex:
        futs = [ex.submit(run, src, dst, extra) for src, dst in todo]
        for f in as_completed(futs):
            src, code, out, sec = f.result()
            rel = src.relative_to(args.input_dir)
            status = 'OK' if code == 0 else f'FAILED (exit {code})'
            print(f'==== {rel}: {status} in {sec:.0f} s')
            print(out, flush=True)
            if code:
                failed.append(rel)

    print(f'done: {len(todo) - len(failed)} restored, {len(failed)} failed')
    for rel in failed:
        print(f'  failed: {rel}')
    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    main()
