"""Training loss (weighted cross-entropy + soft Dice) and evaluation metrics."""
import numpy as np
import torch
import torch.nn.functional as F


# =============================================================================
# Loss
# =============================================================================
def corrupted_mask(corrupted, clean, dilate_iters=0):
    """[B, X, Y, Z] bool: voxels where the input differs from the target,
    dilated by `dilate_iters` voxels (3x3x3 neighbourhood per iteration)."""
    mask = corrupted != clean
    if dilate_iters > 0:
        m = mask[:, None].float()
        for _ in range(int(dilate_iters)):
            m = F.max_pool3d(m, kernel_size=3, stride=1, padding=1)
        mask = m[:, 0] > 0
    return mask


def restoration_loss(logits, target, corrupted, lcfg):
    """Cross-entropy + soft Dice.

    logits:    [B, C, X, Y, Z] float32
    target:    [B, X, Y, Z] clean labels
    corrupted: [B, X, Y, Z] input labels (only used to find the corrupted voxels)
    lcfg:      section `loss` of the config

    Cross-entropy: voxels where corrupted != clean (dilated by loss.dilate_iters)
    get weight loss.corrupted_weight, the others 1; the loss is the weighted mean.
    Without this the ~95 % of voxels that only need to be copied dominate.

    Dice: soft Dice per class computed over the whole batch (sums over batch
    and space), background excluded, averaged over the classes that are present
    in the target of the batch.

    Returns (loss, dict of the two terms as floats).
    """
    target = target.long()
    weight = 1.0 + (lcfg["corrupted_weight"] - 1.0) * corrupted_mask(
        corrupted, target, lcfg["dilate_iters"]).float()
    ce = (F.cross_entropy(logits, target, reduction="none") * weight).sum() / weight.sum()

    prob = torch.softmax(logits, dim=1)[:, 1:]                       # drop background
    onehot = F.one_hot(target, logits.shape[1]).permute(0, 4, 1, 2, 3)[:, 1:].float()
    dims = (0, 2, 3, 4)
    inter = (prob * onehot).sum(dims)
    denom = prob.sum(dims) + onehot.sum(dims)
    present = onehot.sum(dims) > 0
    dice_c = (2.0 * inter + 1e-5) / (denom + 1e-5)
    dice = 1.0 - (dice_c[present].mean() if present.any() else dice_c.mean())

    loss = lcfg["ce_weight"] * ce + lcfg["dice_weight"] * dice
    return loss, dict(ce=float(ce.detach()), dice=float(dice.detach()))


# =============================================================================
# Metrics
# =============================================================================
class RestorationMetrics:
    """Accumulates confusion matrices over patches or volumes.

    Four [C, C] matrices indexed [target, prediction] are kept:
      all        every voxel, prediction = model output
      corrupted  only voxels where input != target
      holes      only hole voxels: input is background but the target is not
      input      every voxel, "prediction" = the corrupted input (baseline)

    `summary()` derives from them:
      dice_all[c]        Dice of class c over all voxels
      dice_corrupted[c]  Dice of class c restricted to corrupted voxels  (main metric)
      dice_input[c]      Dice of the corrupted input itself (what doing nothing gives)
      hole_filled        fraction of hole voxels that received a label (prediction > 0)
      hole_correct       fraction of hole voxels that received the RIGHT label
      acc_corrupted      fraction of corrupted voxels predicted correctly
      damage             fraction of voxels that were correct in the input and wrong
                         in the output (the model should not touch them)
    """

    NAMES = ("all", "corrupted", "holes", "input")

    def __init__(self, num_classes):
        self.c = int(num_classes)
        self.conf = {k: torch.zeros(self.c, self.c, dtype=torch.float64) for k in self.NAMES}

    def _count(self, target, pred, mask=None):
        if mask is not None:
            target, pred = target[mask], pred[mask]
        idx = target.reshape(-1).long() * self.c + pred.reshape(-1).long()
        return torch.bincount(idx, minlength=self.c * self.c).reshape(self.c, self.c).double().cpu()

    @torch.no_grad()
    def update(self, pred, corrupted, clean):
        """pred, corrupted, clean: integer label tensors of the same shape (any device)."""
        diff = corrupted != clean
        self.conf["all"] += self._count(clean, pred)
        self.conf["corrupted"] += self._count(clean, pred, diff)
        self.conf["holes"] += self._count(clean, pred, diff & (corrupted == 0))
        self.conf["input"] += self._count(clean, corrupted)

    @staticmethod
    def _dice(conf):
        """Per-class Dice from a confusion matrix (NaN where the class is absent)."""
        tp = conf.diag()
        denom = conf.sum(0) + conf.sum(1)
        return torch.where(denom > 0, 2 * tp / denom.clamp(min=1), torch.nan).numpy()

    def summary(self):
        """Dict with the scalar metrics, their foreground means, and per-class arrays."""
        a, cor, hol = self.conf["all"], self.conf["corrupted"], self.conf["holes"]
        n_all, n_cor, n_hol = a.sum().item(), cor.sum().item(), hol.sum().item()
        wrong_all = n_all - a.diag().sum().item()
        wrong_cor = n_cor - cor.diag().sum().item()
        nan = float("nan")
        out = dict(
            dice_all=self._dice(a), dice_corrupted=self._dice(cor),
            dice_input=self._dice(self.conf["input"]),
            hole_filled=hol[:, 1:].sum().item() / n_hol if n_hol else nan,
            hole_correct=hol.diag().sum().item() / n_hol if n_hol else nan,
            acc_corrupted=cor.diag().sum().item() / n_cor if n_cor else nan,
            damage=(wrong_all - wrong_cor) / (n_all - n_cor) if n_all > n_cor else nan,
            corrupted_fraction=n_cor / n_all if n_all else nan,
            n_voxels=int(n_all), n_corrupted=int(n_cor), n_hole=int(n_hol),
        )
        for key in ("dice_all", "dice_corrupted", "dice_input"):    # mean over foreground classes
            fg = out[key][1:]
            out[key + "_mean"] = float(np.nanmean(fg)) if np.isfinite(fg).any() else nan
        return out

    def table(self, names):
        """Markdown table: one row per class with the three Dice values."""
        s = self.summary()
        rows = ["| label | name | Dice all | Dice corrupted voxels | Dice of input |",
                "|---|---|---|---|---|"]
        for c in range(self.c):
            rows.append(f"| {c} | {names[c] if c < len(names) else ''} | {s['dice_all'][c]:.4f} "
                        f"| {s['dice_corrupted'][c]:.4f} | {s['dice_input'][c]:.4f} |")
        rows.append(f"| | **mean (no background)** | {s['dice_all_mean']:.4f} "
                    f"| {s['dice_corrupted_mean']:.4f} | {s['dice_input_mean']:.4f} |")
        return "\n".join(rows)


@torch.no_grad()
def evaluate_loader(model, loader, device, cfg):
    """Run the model on every batch of a patch loader.

    Returns (RestorationMetrics over all patches, mean loss).
    Used for validation during training and by 06_evaluate.py --mode patch.
    """
    metrics = RestorationMetrics(cfg["labels"]["num_classes"])
    use_amp = bool(cfg["train"]["amp"]) and torch.device(device).type == "cuda"
    total, n = 0.0, 0
    for batch in loader:
        x, hole, y = (batch[k].to(device, non_blocking=True) for k in ("x", "hole", "y"))
        with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            logits = model(x, hole)
        loss, _ = restoration_loss(logits, y, x, cfg["loss"])
        metrics.update(logits.argmax(1), x, y)
        total, n = total + float(loss.detach()), n + 1
    return metrics, total / max(n, 1)


def write_metrics(out_dir, metrics, names, extra=None):
    """Save a RestorationMetrics object as metrics.json, per_label.csv and metrics.md.

    extra: optional dict of additional scalars written to the JSON and the markdown.
    Returns the summary dict.
    """
    import json
    import os

    s = metrics.summary()
    os.makedirs(out_dir, exist_ok=True)
    scalars = {k: v for k, v in s.items() if not isinstance(v, np.ndarray)}
    scalars.update(extra or {})
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump({**scalars, **{k: v.tolist() for k, v in s.items() if isinstance(v, np.ndarray)}},
                  f, indent=2)
    with open(os.path.join(out_dir, "per_label.csv"), "w") as f:
        f.write("label,name,dice_all,dice_corrupted,dice_input\n")
        for c in range(metrics.c):
            f.write(f"{c},{names[c] if c < len(names) else ''},{s['dice_all'][c]:.6f},"
                    f"{s['dice_corrupted'][c]:.6f},{s['dice_input'][c]:.6f}\n")
    with open(os.path.join(out_dir, "metrics.md"), "w") as f:
        f.write("# Evaluation with synthetic corruption\n\n")
        for k, v in scalars.items():
            f.write(f"- {k}: {v:.4f}\n" if isinstance(v, float) else f"- {k}: {v}\n")
        f.write("\n" + metrics.table(names) + "\n")
    return s
