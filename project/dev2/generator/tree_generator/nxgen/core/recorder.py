"""
Recorder: intermediate outputs and per-step timing.

The pipeline wraps each step in `with recorder.step(name):` and hands the
result to `recorder.record(name, volume, affine)`. What happens then depends
on how the recorder was built:

    Recorder()                                  nothing (the default; no cost)
    Recorder(steps='all')                       keep every step in memory (CPU)
    Recorder(steps=['pv_seg', 'synthesis'])     keep some steps
    Recorder(steps='all', out_dir='debug/')     also write <nn>_<step>.nii.gz
    Recorder(profile=True)                      time and peak GPU memory of
                                                every step (forces a GPU sync
                                                per step, so only for profiling)

Recorded step names (in order):
    template_crop  HR template crop, before morphology (indices)
    morphology     HR crop after morphology (indices)
    spatial        deformed HR labels of the region (indices)
    pv_seg         output segmentation (indices)
    pv_max         largest partial-volume fraction per voxel
    pv_count       number of labels with a non-zero fraction per voxel
    synthesis      image from the tree Gaussians
    bias, motion   image after each artefact
    lowres         image on the low-resolution grid (own affine)
    resolution     image after the resolution step
    noise          image after the noise
    image          final, normalised image
"""

import os
import time
from contextlib import contextmanager

import torch

from .io import save_image, save_labels


class Recorder:
    """
    Parameters
    ----------
    steps : None, 'all' or list of str
        Steps to keep (None = none).
    out_dir : str, optional
        Also save the kept steps as NIfTI files in this folder.
    profile : bool
        Measure time (s) and peak GPU memory (GB) of every step, in
        `self.timings` = {step: {'time_s': ..., 'peak_gb': ...}}.
    """

    INTEGER_STEPS = ('template_crop', 'morphology', 'spatial', 'pv_seg',
                     'pv_count')

    def __init__(self, steps=None, out_dir=None, profile=False):
        self.steps_wanted = steps
        self.out_dir = out_dir
        self.profile = profile
        self.steps = {}
        self.affines = {}
        self.timings = {}
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

    def wants(self, name):
        """True if step `name` must be kept."""
        if self.steps_wanted is None:
            return False
        return self.steps_wanted == 'all' or name in self.steps_wanted

    @contextmanager
    def step(self, name, device=None):
        """Time a step (only with profile=True)."""
        if not self.profile:
            yield
            return
        cuda = device is not None and torch.device(device).type == 'cuda'
        if cuda:
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        t0 = time.perf_counter()
        yield
        if cuda:
            torch.cuda.synchronize(device)
        self.timings[name] = dict(
            time_s=time.perf_counter() - t0,
            peak_gb=torch.cuda.max_memory_allocated(device) / 1e9
            if cuda else None)

    def record(self, name, volume, affine):
        """Keep (and save) `volume` if step `name` is wanted."""
        if volume is None or not self.wants(name):
            return
        volume = volume.detach().cpu()
        self.steps[name] = volume
        self.affines[name] = affine
        if self.out_dir:
            n = len(self.steps)
            path = os.path.join(self.out_dir, f'{n:02d}_{name}.nii.gz')
            if name in self.INTEGER_STEPS:
                save_labels(volume, affine, path)
            else:
                save_image(volume.float(), affine, path)
