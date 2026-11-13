"""Sliding-window restoration of a whole label volume (overlap + Gaussian weighting)."""
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .corruption import hole_channel
from .data import worker_init
from .external import corrupt_module


def window_starts(n, patch, step):
    """Start indices of windows of size `patch` covering [0, n) with stride `step`
    (the last window is aligned with the end)."""
    if n <= patch:
        return [0]
    return list(range(0, n - patch, step)) + [n - patch]


def gaussian_weight(patch, sigma_scale=0.125):
    """[P, P, P] weight, 1 at the centre and close to 0 at the border (nnU-Net style),
    so that every voxel is mostly decided by windows in which it is central."""
    x = torch.arange(patch, dtype=torch.float32) - (patch - 1) / 2.0
    g = torch.exp(-0.5 * (x / (patch * sigma_scale)) ** 2)
    w = g[:, None, None] * g[None, :, None] * g[None, None, :]
    return (w / w.max()).clamp(min=1e-4)


class WindowDataset(Dataset):
    """Windows of one padded volume, with the hole channel of each.

    The hole channel is computed on the context crop around the window
    (patch.context voxels) and then cut to the window: exactly the procedure
    used for the training patches. It is the slow part (~0.4 s per window), so
    it runs in DataLoader workers.

    padded: the volume padded with `margin` zeros before every axis (and enough after)
    starts: list of (i, j, k) window origins in the unpadded volume
    """

    def __init__(self, cfg, padded, starts, margin):
        self.cfg, self.padded, self.starts, self.margin = cfg, padded, starts, margin
        self.patch = int(cfg["patch"]["size"])
        self.context = self.patch + 2 * margin

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, n):
        i, j, k = self.starts[n]
        c, m, p = self.context, self.margin, self.patch
        ctx = self.padded[i:i + c, j:j + c, k:k + c]
        hole = hole_channel(corrupt_module(self.cfg), ctx)
        centre = (slice(m, m + p),) * 3
        x = np.ascontiguousarray(ctx[centre])
        return (torch.from_numpy(x.astype(np.int16 if x.dtype.itemsize > 1 else np.uint8)),
                torch.from_numpy(np.ascontiguousarray(hole[centre]).astype(np.uint8)),
                torch.tensor([i, j, k]))


@torch.no_grad()
def restore_volume(model, volume, cfg, device, progress=False):
    """Restore one label volume.

    model:  RestorationNet in eval mode on `device`
    volume: [X, Y, Z] integer labels at the training resolution (0.2 mm)
    cfg:    config (sections patch, labels, inference, train.amp)
    Returns the restored labels, same shape and dtype.

    Steps: crop to the labelled bounding box -> windows with overlap (windows
    without any label are skipped: nothing to restore there) -> softmax of every
    window weighted by a Gaussian and summed -> argmax -> paste back.
    """
    cn = corrupt_module(cfg)
    icfg = cfg["inference"]
    patch = int(cfg["patch"]["size"])
    margin = (int(cfg["patch"]["context"]) - patch) // 2
    num_classes = int(cfg["labels"]["num_classes"])

    box = cn.bbox(volume > 0, pad=8)
    if box is None:
        return volume.copy()
    sub = volume[box]
    shape = tuple(max(n, patch) for n in sub.shape)          # at least one window per axis
    padded = np.pad(sub, [(margin, margin + s - n) for s, n in zip(shape, sub.shape)])

    step = max(1, int(round(patch * (1.0 - icfg["overlap"]))))
    starts = [(i, j, k)
              for i in window_starts(shape[0], patch, step)
              for j in window_starts(shape[1], patch, step)
              for k in window_starts(shape[2], patch, step)
              if padded[margin + i:margin + i + patch, margin + j:margin + j + patch,
                        margin + k:margin + k + patch].any()]

    loader = DataLoader(WindowDataset(cfg, padded, starts, margin),
                        batch_size=int(icfg["batch_size"]), shuffle=False,
                        num_workers=int(icfg["num_workers"]), worker_init_fn=worker_init)
    acc_dev = torch.device(icfg["accum_device"] if torch.cuda.is_available() else "cpu")
    acc = torch.zeros((num_classes,) + shape, dtype=torch.float32, device=acc_dev)
    weight = gaussian_weight(patch).to(acc_dev)
    use_amp = bool(cfg["train"]["amp"]) and torch.device(device).type == "cuda"

    for n, (x, hole, origin) in enumerate(loader):
        x, hole = x.to(device), hole.to(device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            logits = model(x, hole)
            if icfg["tta_flip_lr"]:                           # mirror along the first (L-R) axis
                logits = 0.5 * (logits + model(x.flip(1), hole.flip(1)).flip(2))
        prob = torch.softmax(logits.float(), dim=1).to(acc_dev)
        for b, (i, j, k) in enumerate(origin.tolist()):
            acc[:, i:i + patch, j:j + patch, k:k + patch] += prob[b] * weight
        if progress and (n % 20 == 0 or n == len(loader) - 1):
            print(f"    window batch {n + 1}/{len(loader)}", flush=True)

    # argmax slab by slab (keeps the peak memory low); uncovered voxels stay background
    pred = np.zeros(shape, volume.dtype)
    for i in range(0, shape[0], 64):
        pred[i:i + 64] = acc[:, i:i + 64].argmax(0).cpu().numpy()
    del acc
    out = volume.copy()
    out[box] = pred[:sub.shape[0], :sub.shape[1], :sub.shape[2]]
    return out


def restore_passes(model, volume, cfg, device, progress=False):
    """`restore_volume` applied inference.passes times (the output is fed back)."""
    out = volume
    for _ in range(max(1, int(cfg["inference"]["passes"]))):
        out = restore_volume(model, out, cfg, device, progress)
    return out
