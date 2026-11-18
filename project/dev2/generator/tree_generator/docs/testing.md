# Tests, benchmark, figures

None of these run with the generator; run them when you want them.

## Unit tests

```bash
python tests/run_tests.py                    # all, CPU (~10 s)
python tests/run_tests.py --device cuda:0    # same on a GPU
python tests/run_tests.py -k spatial         # names containing 'spatial'
python tests/run_tests.py -x                 # stop at the first failure
pytest tests/                                # also works if pytest is installed
```

They use small phantoms (`tests/phantoms.py`: concentric spheres, a
5-node tree), so no data is needed.

| file | what |
|---|---|
| `test_core.py` | config (defaults, unknown keys, invalid values, paths), distribution specs, seeds, label space, B-spline fields |
| `test_template.py` | offline module; equivalence with v2 (skipped if v2 cannot be imported) |
| `test_geometry_modules.py` | acquisition priors, morphology, spatial, partial volume |
| `test_intensity_modules.py` | tree intensities, synthesis, bias, motion, resolution (vs v2), noise, normalisation |
| `test_pipeline.py` | shapes/ranges, determinism, plan JSON replay, independent module streams, patch == crop of full volume, recorder, plan editing, CLIs, training stream |

Status at writing: 46 passed on CPU and on GPU.

Adding a test: a function `test_<name>()` in a `tests/test_*.py` file, with
plain `assert`s. Add a `device='cpu'` argument if it should run on the
device given to `run_tests.py`. Raise `phantoms.Skip('reason')` when an
optional dependency is missing.

## Benchmark (real templates, GPU)

```bash
python tests/benchmark.py --templates data/templates \
    --config configs/nextbrain.json --device cuda:0 -n 5 \
    [--patch 160 160 160] [--worst-case]
```

Median time and peak GPU memory of every step. `--worst-case` forces motion
and a degraded acquisition. Results at writing: see
[pipeline.md](pipeline.md#7-speed-and-memory).

## Figures

```bash
python examples/plot_steps.py --templates data/templates \
    --config configs/nextbrain.json --device cuda:0 \
    --seeds 0 1 2 --out data/outputs/steps.png [--worst-case]
```

One row per seed, one column per step (middle slice).

## Checking your own outputs

`nxgen.checks` works on any volume, e.g. the steps kept by a recorder:

```python
from nxgen import Generator, Recorder
from nxgen import checks

gen = Generator(templates, 'configs/nextbrain.json', device='cuda:0')
rec = Recorder(steps=['template_crop', 'spatial', 'pv_seg'])
s = gen(3, recorder=rec)
checks.vanished_labels(rec.steps['template_crop'], s.seg)
checks.adjacency_violations(s.seg, [['cerebral_wm_l1', 'csf_ventricles_l1']],
                            gen.labels, gen.tree)
```
