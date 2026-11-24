"""
Run the tests (only when asked; the generator never runs them).

    python tests/run_tests.py                     every test, on the CPU
    python tests/run_tests.py -k spatial          tests whose name contains 'spatial'
    python tests/run_tests.py --device cuda:0     tests that take a device run on it
    python tests/run_tests.py -x                  stop at the first failure

The test files are plain python (`test_*.py`, functions `test_*`), so
`pytest tests/` works too if pytest is installed. Tests skip themselves
(`Skip`) when an optional dependency is missing (e.g. generator_v2 for the
equivalence tests).

Every test uses the small phantoms of `phantoms.py` and runs in seconds.
Benchmarks on real templates are separate: `tests/benchmark.py`.
"""

import argparse
import glob
import importlib.util
import inspect
import os
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))      # nxgen
sys.path.insert(0, HERE)                        # phantoms

from phantoms import Skip  # noqa: E402


def main():
    p = argparse.ArgumentParser(description='Run the nxgen tests.')
    p.add_argument('-k', default='', help='only tests whose name contains this')
    p.add_argument('--device', default='cpu')
    p.add_argument('-x', action='store_true', help='stop at the first failure')
    args = p.parse_args()

    results = {'passed': 0, 'failed': 0, 'skipped': 0}
    failures = []
    for path in sorted(glob.glob(os.path.join(HERE, 'test_*.py'))):
        name = os.path.basename(path)[:-3]
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        for fname, fn in inspect.getmembers(module, inspect.isfunction):
            full = f'{name}::{fname}'
            if not fname.startswith('test_') or args.k not in full:
                continue
            kwargs = {}
            if 'device' in inspect.signature(fn).parameters:
                kwargs['device'] = args.device
            t0 = time.time()
            try:
                fn(**kwargs)
                status = 'passed'
            except Skip as e:
                status = 'skipped'
                full += f'  ({e})'
            except Exception:
                status = 'failed'
                failures.append((full, traceback.format_exc()))
            results[status] += 1
            print(f'{status.upper():8s} {time.time() - t0:6.2f}s  {full}')
            if status == 'failed' and args.x:
                break
        if failures and args.x:
            break

    for full, tb in failures:
        print(f'\n===== {full}\n{tb}')
    print(f"\n{results['passed']} passed, {results['failed']} failed, "
          f"{results['skipped']} skipped")
    sys.exit(1 if failures else 0)


if __name__ == '__main__':
    main()
