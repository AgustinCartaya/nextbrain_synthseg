"""
Randomness: seeds, per-module generators, distribution specs.

Every image has one seed. Each module draws its parameters from its own
numpy generator, derived from (image seed, module name). Consequence: turning
one module off, or changing its settings, does not change what the other
modules draw. This makes A/B comparisons meaningful when debugging.

Voxel-wise noise is not stored in the plan; its module stores a seed
instead, and `torch_generator` rebuilds the same noise from it.

No function here touches the global random state of python, numpy or torch.
"""

import zlib

import numpy as np
import torch


def image_seed(run_seed, index):
    """
    Seed of image `index` of a run with seed `run_seed`.

    Independent 32-bit seeds for every (run_seed, index) pair (numpy
    SeedSequence), so image 42 can be regenerated alone.
    """
    return int(np.random.SeedSequence([run_seed, index]).generate_state(1)[0])


def module_rng(seed, name):
    """
    Numpy generator of module `name` for the image with seed `seed`.

    Parameters
    ----------
    seed : int
        Image seed.
    name : str
        Module name (e.g. 'spatial'). Different names give independent
        streams.
    """
    return np.random.default_rng([int(seed), zlib.crc32(name.encode())])


def new_seed(rng):
    """A 31-bit integer seed drawn from `rng` (for voxel-wise noise)."""
    return int(rng.integers(2 ** 31))


def torch_generator(seed, device):
    """torch.Generator on `device`, seeded with `seed`."""
    g = torch.Generator(device=torch.device(device))
    g.manual_seed(int(seed))
    return g


def check_spec(spec, name='value'):
    """
    Raise ValueError if `spec` is not a valid distribution spec.

    Valid specs: a number; [lo, hi] with lo <= hi; {'uniform': [lo, hi]};
    {'randint': [lo, hi]} (integers); {'choice': [non-empty list]}.
    """
    if isinstance(spec, bool):
        raise ValueError(f'{name}: booleans are not distribution specs')
    if isinstance(spec, (int, float)):
        return
    if isinstance(spec, (list, tuple)):
        if len(spec) != 2 or spec[0] > spec[1]:
            raise ValueError(f'{name}: a range must be [lo, hi] with '
                             f'lo <= hi, got {spec!r}')
        return
    if isinstance(spec, dict) and len(spec) == 1:
        kind, value = next(iter(spec.items()))
        if kind in ('uniform', 'randint'):
            if len(value) != 2 or value[0] > value[1]:
                raise ValueError(f'{name}: {kind} needs [lo, hi], got '
                                 f'{value!r}')
            return
        if kind == 'choice':
            if not value:
                raise ValueError(f'{name}: empty choice')
            return
    raise ValueError(f'{name}: invalid distribution spec {spec!r} (number, '
                     f"[lo, hi], {{'uniform': ...}}, {{'randint': ...}} or "
                     f"{{'choice': [...]}})")


def draw(spec, rng):
    """
    One value from a distribution spec (see `check_spec`).

    Parameters
    ----------
    spec : number, [lo, hi] or dict
    rng : numpy Generator

    Returns
    -------
    float, int or element of a choice. A fixed number is returned as is
    (an int stays an int). Nothing is drawn from `rng` for a fixed number.
    """
    if isinstance(spec, (int, float)):
        return spec
    if isinstance(spec, (list, tuple)):
        return float(rng.uniform(spec[0], spec[1]))
    kind, value = next(iter(spec.items()))
    if kind == 'uniform':
        return float(rng.uniform(value[0], value[1]))
    if kind == 'randint':
        return int(rng.integers(value[0], value[1] + 1))
    return value[int(rng.integers(len(value)))]
