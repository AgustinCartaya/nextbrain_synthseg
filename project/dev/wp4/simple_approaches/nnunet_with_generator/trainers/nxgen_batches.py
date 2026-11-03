"""nnU-Net training batches generated on the fly by nxgen.

    batches = GeneratorBatches(generator, batch_size=2, axes=[2, 1, 0], ...)
    batch = next(batches)     # {'data': (B, 1, *patch), 'target': [(B, 1, *patch * scale) per DS scale]}

What a batch holds is what nnU-Net's own training loader would produce after its
augmentations, so the trainer's train_step runs unchanged:

1. one nxgen sample per batch element: `generator(image_seed(run_seed, i))`, a patch of the
   output grid (the generator config has `patch.size` = the nnU-Net patch, in NIfTI axes);
2. NIfTI array axes -> nnU-Net axes (`axes`, see `nnunet_axes`);
3. random mirroring (nnU-Net mirrors every axis with p = 0.5 and uses the same axes for
   test-time mirroring);
4. z-score normalisation (nnU-Net's ZScoreNormalization, computed on the patch);
5. segmentation downsampled for deep supervision ('nearest-exact', like
   DownsampleSegForDSTransform).

Sample i of the stream always has the same seed, so a run is reproducible and resuming at
epoch e continues the stream instead of replaying it (`start`).

nxgen is imported lazily (only `build_from_plans` needs it), so nnU-Net can import the
trainer for inference without the generator.
"""
import json
import os
import queue
import threading
import time

import numpy as np
import torch
import torch.nn.functional as F


def nnunet_axes(transpose_forward):
    """
    Permutation NIfTI array axes -> nnU-Net array axes.

    nnU-Net readers (SimpleITKIO, NibabelIO) return the NIfTI array with reversed axes
    (z, y, x); preprocessing then applies `transpose_forward`. Axis i of a preprocessed
    nnU-Net array is therefore NIfTI axis 2 - transpose_forward[i].
    """
    return [2 - int(t) for t in transpose_forward]


def ds_scales_from_plans(pool_op_kernel_sizes):
    """Deep-supervision scales from ConfigurationManager.pool_op_kernel_sizes (as nnUNetTrainer)."""
    return [list(s) for s in 1 / np.cumprod(np.vstack(pool_op_kernel_sizes), axis=0)][:-1]


class GeneratorBatches:
    """
    Endless iterator of nnU-Net training batches.

    Parameters
    ----------
    generator : nxgen Generator in patch mode (its patch size is the nnU-Net patch in NIfTI axes)
    batch_size : int
    axes : 3 ints, NIfTI axes -> nnU-Net axes (`nnunet_axes`)
    ds_scales : list of 3-lists or None, deep-supervision scales (None: target is a tensor)
    mirror_axes : tuple of nnU-Net axes to mirror randomly (p = 0.5 each), or None
    use_mask_for_norm : bool, z-score on the non-zero voxels only
    out_device : device of the returned tensors (the training GPU)
    run_seed : int, seed of the stream
    start : int, index of the first batch (resume)
    rank, world_size : DDP, each rank takes a different slice of the stream
    prefetch : int, batches generated ahead in a background thread (0: synchronous)
    """

    def __init__(self, generator, batch_size, axes, ds_scales=None, mirror_axes=(0, 1, 2),
                 use_mask_for_norm=False, out_device='cuda', run_seed=0, start=0, rank=0, world_size=1,
                 prefetch=2):
        from nxgen.core.rng import image_seed, module_rng
        self._image_seed, self._module_rng = image_seed, module_rng
        self.generator = generator
        self.batch_size = batch_size
        self.axes = list(axes)
        self.ds_scales = ds_scales
        self.mirror_axes = tuple(mirror_axes) if mirror_axes else ()
        self.use_mask_for_norm = use_mask_for_norm
        self.out_device = torch.device(out_device)
        self.run_seed = run_seed
        self.batch_index = start
        self.rank, self.world_size = rank, world_size
        self.prefetch = prefetch
        self.wait_s = 0.0          # time spent waiting for batches (reset by the trainer)
        self.gen_s = 0.0           # time spent generating batches
        self._queue = self._thread = None
        self._stop = threading.Event()

    # -- one batch ------------------------------------------------------------

    def sample_indices(self, b):
        """Stream indices of the samples of batch b on this rank."""
        first = (b * self.world_size + self.rank) * self.batch_size
        return list(range(first, first + self.batch_size))

    def make_sample(self, i):
        """Sample i of the stream -> (image (1, *patch), seg (1, *patch)) on the generator device."""
        seed = self._image_seed(self.run_seed, i)
        s = self.generator(seed)
        img = s.image.permute(*self.axes).float()
        seg = s.seg.permute(*self.axes)
        r = self._module_rng(seed, 'nnunet_mirror')
        flips = [a for a in self.mirror_axes if r.random() < 0.5]
        if flips:
            img, seg = img.flip(flips), seg.flip(flips)
        mask = img != 0 if self.use_mask_for_norm else None
        vals = img[mask] if mask is not None else img
        mean, std = vals.mean(), vals.std().clamp_min(1e-8)
        if mask is not None:
            img = torch.where(mask, (img - mean) / std, img)
        else:
            img = (img - mean) / std
        return img[None], seg[None]

    def make_batch(self, b):
        t0 = time.perf_counter()
        pairs = [self.make_sample(i) for i in self.sample_indices(b)]
        data = torch.stack([p[0] for p in pairs]).to(self.out_device, non_blocking=True)
        seg = torch.stack([p[1] for p in pairs]).to(self.out_device, non_blocking=True)
        target = seg.float()
        if self.ds_scales is not None:
            shape = target.shape[2:]
            target = [target if all(s == 1 for s in sc) else
                      F.interpolate(target, [round(n * s) for n, s in zip(shape, sc)], mode='nearest-exact')
                      for sc in self.ds_scales]
        if self.out_device.type == 'cuda':
            torch.cuda.synchronize(self.out_device)
        self.gen_s += time.perf_counter() - t0
        return {'data': data, 'target': target, 'keys': [f'nxgen_{i}' for i in self.sample_indices(b)]}

    # -- iterator ---------------------------------------------------------------

    def _worker(self):
        try:
            while not self._stop.is_set():
                batch = self.make_batch(self.batch_index)
                self.batch_index += 1
                while not self._stop.is_set():
                    try:
                        self._queue.put(batch, timeout=0.5)
                        break
                    except queue.Full:
                        pass
        except BaseException as e:      # surfaced in the training process by __next__
            self._queue.put(e)

    def __iter__(self):
        return self

    def __next__(self):
        t0 = time.perf_counter()
        if self.prefetch <= 0:
            batch = self.make_batch(self.batch_index)
            self.batch_index += 1
        else:
            if self._thread is None:
                self._queue = queue.Queue(maxsize=self.prefetch)
                self._thread = threading.Thread(target=self._worker, name='nxgen-batches', daemon=True)
                self._thread.start()
            batch = self._queue.get()
            if isinstance(batch, BaseException):
                raise RuntimeError('nxgen batch generation failed') from batch
        self.wait_s += time.perf_counter() - t0
        return batch

    def close(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=10)
            self._thread = None


def build_from_plans(preprocessed_base, plans, configuration, fold, batch_size, ds_scales, mirror_axes,
                     out_device, run_seed=0, start=0, rank=0, world_size=1, prefetch=2, gen_device=None,
                     log=print):
    """
    GeneratorBatches matching an nnU-Net plan.

    Reads `<preprocessed_base>/nxgen.json` (written by prepare_dataset.py splits): templates,
    generator config, label space and the template held out by each fold.

    Parameters
    ----------
    preprocessed_base : str, $nnUNet_preprocessed/<dataset>
    plans : dict (plans JSON)
    configuration : str, e.g. '3d_fullres'
    fold : int or 'all'. Fold i trains on every template but `fold_holdout[i]`.
    gen_device : str, device of the generator (default: out_device)
    other parameters : see GeneratorBatches
    """
    from nxgen import Generator, load_config

    info = json.load(open(os.path.join(preprocessed_base, 'nxgen.json')))
    conf = plans['configurations'][configuration]
    patch = [int(p) for p in conf['patch_size']]
    if len(patch) != 3:
        raise ValueError(f'{configuration}: nxgen makes 3D patches, got patch size {patch}')
    if conf['normalization_schemes'] != ['ZScoreNormalization']:
        raise ValueError(f"only ZScoreNormalization is implemented, plan has {conf['normalization_schemes']}")

    axes = nnunet_axes(plans['transpose_forward'])
    size_nifti = [0, 0, 0]
    for i, a in enumerate(axes):
        size_nifti[a] = patch[i]
    cfg = load_config(info['config'], patch={'size': size_nifti})

    spacing = conf['spacing']
    if max(abs(s - cfg['grid']['target_voxel_mm']) for s in spacing) > 1e-3:
        raise ValueError(f'plan spacing {spacing} != generator grid.target_voxel_mm '
                         f"{cfg['grid']['target_voxel_mm']}")

    templates, names = info['templates'], info['template_names']
    if fold == 'all':
        keep = list(range(len(templates)))
    else:
        held = info['fold_holdout'][str(fold)]
        keep = [i for i, n in enumerate(names) if n != held]
    gen = Generator([templates[i] for i in keep], cfg, info['label_space'], gen_device or out_device)
    for t in gen.templates:
        if any(o < s for o, s in zip(t['out_shape'], size_nifti)):
            raise ValueError(f"{t['path']}: output grid {t['out_shape']} is smaller than the patch {size_nifti}")

    log(f'nxgen: templates {[names[i] for i in keep]}, patch {patch} (NIfTI axes {size_nifti}), '
        f'axes NIfTI->nnU-Net {axes}, {len(gen.labels)} labels, generator on {gen.device}, '
        f'run seed {run_seed}, first batch {start}, prefetch {prefetch}')
    return GeneratorBatches(gen, batch_size, axes, ds_scales, mirror_axes,
                            bool(conf['use_mask_for_norm'][0]), out_device, run_seed, start, rank, world_size,
                            prefetch)
