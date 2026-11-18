"""
Generation during training.

    gen = Generator(templates, 'config.json', device='cuda')   # patch.size set
    stream = SyntheticStream(gen, run_seed=0)
    loader = torch.utils.data.DataLoader(stream, batch_size=2, num_workers=0)
    for image, seg in loader:          # (B, 1, X, Y, Z) float, (B, 1, X, Y, Z) long
        ...

Every sample has its own seed, `image_seed(run_seed, i)` with i = 0, 1, 2...
(`start` sets the first i), so a training run can be reproduced and any
sample regenerated alone (`gen(image_seed(run_seed, i))`).

GPU generation and DataLoader workers: CUDA cannot be used in forked worker
processes. With a CUDA generator use `num_workers=0`: generation then runs
in the training process, on the training GPU (or on another GPU: build the
Generator with `device='cuda:1'`; the samples are moved to `out_device`).
With a CPU generator, workers are fine: each worker takes every
num_workers-th index, so no two workers produce the same sample.
"""

import torch

from .core.rng import image_seed


class SyntheticStream(torch.utils.data.IterableDataset):
    """
    Endless stream of (image, seg) pairs.

    Parameters
    ----------
    generator : Generator
    run_seed : int
    start : int, index of the first sample
    out_device : str, optional, device of the returned tensors (default:
        the generator's)
    return_plan : bool, also yield the plan (a dict; collate with care)

    Yields
    ------
    image : (1, X, Y, Z) float32 tensor
    seg : (1, X, Y, Z) int64 tensor of label indices
    (plan : dict, if return_plan)
    """

    def __init__(self, generator, run_seed=0, start=0, out_device=None,
                 return_plan=False):
        super().__init__()
        self.generator = generator
        self.run_seed = run_seed
        self.start = start
        self.out_device = out_device
        self.return_plan = return_plan

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        worker, n_workers = (info.id, info.num_workers) if info else (0, 1)
        i = self.start + worker
        while True:
            s = self.generator(image_seed(self.run_seed, i))
            image = s.image[None]
            seg = s.seg[None].long()
            if self.out_device is not None:
                image = image.to(self.out_device)
                seg = seg.to(self.out_device)
            yield (image, seg, s.plan) if self.return_plan else (image, seg)
            i += n_workers
