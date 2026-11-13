#!/usr/bin/env python
"""
Step 4 — train the restoration network.

Single GPU:
  python 04_train.py --config configs/default.yaml

Multi GPU (DDP, one process per GPU; train.batch_size and train.num_workers are
PER GPU):
  torchrun --nproc_per_node=4 04_train.py --config configs/default.yaml

Continue an interrupted run (same run folder):
  ... 04_train.py --config configs/default.yaml --resume

Outputs in <paths.runs_dir>/<train.run_name>/:
  config.yaml      the full configuration used
  train_log.csv    one row every train.log_every iterations
  val_log.csv      one row per validation (scalars + Dice on corrupted voxels per label)
  tb/              TensorBoard events (only if tensorboard is installed)
  last.pt          newest checkpoint (also used by --resume)
  best.pt          best validation `dice_corrupted_mean` (the main metric)
  epoch_XXXX.pt    every train.save_every epochs

Training data is random and generated on the fly, so an "epoch" is just
train.iters_per_epoch iterations. Validation uses a fixed, seeded set of
corrupted patches of the validation subjects and runs on the first GPU only.
"""
import csv
import os
import time

import torch
import torch.distributed as dist
import yaml
from torch.utils.data import DataLoader

from lib.config import config_from_args, parser
from lib.data import RestorationPatches, worker_init
from lib.labels import assert_writable
from lib.losses import corrupted_mask, evaluate_loader, restoration_loss
from lib.model import build_model


def append_csv(path, row):
    """Append a dict as one CSV row (the header is written when the file is created)."""
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        if new:
            writer.writeheader()
        writer.writerow(row)


def save_checkpoint(path, model, optimizer, scaler, epoch, iteration, best, cfg):
    """Everything needed to resume, plus the config so the weights are self-describing."""
    torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                    scaler=scaler.state_dict(), epoch=epoch, iteration=iteration,
                    best=best, cfg=cfg), path)


def main():
    p = parser(__doc__)
    p.add_argument("--resume", action="store_true", help="continue from last.pt of the run folder")
    args = p.parse_args()
    cfg = config_from_args(args)
    t = cfg["train"]

    # ---- devices / DDP (torchrun sets WORLD_SIZE, RANK and LOCAL_RANK) ---------
    world = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    distributed = world > 1
    if distributed:
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        dist.init_process_group("nccl")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    main_proc = rank == 0
    torch.manual_seed(t["seed"] + rank)
    torch.backends.cudnn.benchmark = True

    run_dir = os.path.join(assert_writable(cfg["paths"]["runs_dir"]), t["run_name"])
    if main_proc:
        os.makedirs(run_dir, exist_ok=True)
        with open(os.path.join(run_dir, "config.yaml"), "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)

    # ---- model, optimiser -----------------------------------------------------------
    model = build_model(cfg).to(device)
    params = [prm for prm in model.parameters() if prm.requires_grad]
    if t["optimizer"] == "sgd":
        optimizer = torch.optim.SGD(params, lr=t["lr"], momentum=0.99, nesterov=True,
                                    weight_decay=t["weight_decay"])
    else:
        optimizer = torch.optim.AdamW(params, lr=t["lr"], weight_decay=t["weight_decay"])
    use_amp = bool(t["amp"]) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    total_iters = t["epochs"] * t["iters_per_epoch"]

    start_epoch, iteration, best = 0, 0, -1.0
    last = os.path.join(run_dir, "last.pt")
    if args.resume and os.path.exists(last):
        ckpt = torch.load(last, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scaler.load_state_dict(ckpt["scaler"])
        start_epoch, iteration, best = ckpt["epoch"] + 1, ckpt["iteration"], ckpt["best"]
        if main_proc:
            print(f"resumed from {last}: epoch {start_epoch}, iteration {iteration}")

    net = model                                   # the bare module (checkpoints, validation)
    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device.index])

    # ---- data ------------------------------------------------------------------------
    train_set = RestorationPatches(cfg, "train", t["iters_per_epoch"] * t["batch_size"])
    train_loader = DataLoader(train_set, batch_size=t["batch_size"], num_workers=t["num_workers"],
                              worker_init_fn=worker_init, pin_memory=True, drop_last=True,
                              persistent_workers=t["num_workers"] > 0,
                              prefetch_factor=2 if t["num_workers"] > 0 else None)
    if main_proc:
        val_set = RestorationPatches(cfg, "val", t["val_samples"], deterministic=True,
                                     seed=t["seed"] + 1)
        val_loader = DataLoader(val_set, batch_size=t["batch_size"],
                                num_workers=min(t["num_workers"], 16), worker_init_fn=worker_init)
        n_par = sum(p.numel() for p in net.parameters())
        print(f"run: {run_dir}\nparameters: {n_par / 1e6:.1f} M | GPUs: {world} | "
              f"train hemispheres: {len(train_set.items)} | val hemispheres: {len(val_set.items)}")
        try:                                      # TensorBoard is optional, the CSVs are always written
            from torch.utils.tensorboard import SummaryWriter
            tb = SummaryWriter(os.path.join(run_dir, "tb"))
        except Exception:                         # noqa: BLE001
            tb = None
            print("tensorboard not available: logging to CSV only")

    # ---- training loop ---------------------------------------------------------------
    for epoch in range(start_epoch, t["epochs"]):
        model.train()
        acc = dict(loss=0.0, ce=0.0, dice=0.0, corrupted_fraction=0.0, data_sec=0.0, n=0)
        t_log = t_data = time.time()
        for batch in train_loader:
            x, hole, y = (batch[k].to(device, non_blocking=True) for k in ("x", "hole", "y"))
            acc["data_sec"] += time.time() - t_data

            lr = t["lr"] * (1.0 - iteration / total_iters) ** t["poly_power"]   # poly decay
            for group in optimizer.param_groups:
                group["lr"] = lr

            with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                logits = model(x, hole)           # float32 logits (see RestorationNet.forward)
            loss, terms = restoration_loss(logits, y, x, cfg["loss"])
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(params, t["grad_clip"])
            scaler.step(optimizer)
            scaler.update()
            iteration += 1

            acc["loss"] += float(loss.detach())
            acc["ce"] += terms["ce"]
            acc["dice"] += terms["dice"]
            acc["corrupted_fraction"] += float(corrupted_mask(x, y).float().mean())
            acc["n"] += 1
            if main_proc and iteration % t["log_every"] == 0:
                n = acc["n"]
                row = dict(iteration=iteration, epoch=epoch, loss=acc["loss"] / n, ce=acc["ce"] / n,
                           dice=acc["dice"] / n, lr=lr,
                           corrupted_fraction=acc["corrupted_fraction"] / n,
                           sec_per_iter=(time.time() - t_log) / n, data_sec=acc["data_sec"] / n)
                append_csv(os.path.join(run_dir, "train_log.csv"), row)
                if tb:
                    for k in ("loss", "ce", "dice", "lr", "sec_per_iter", "data_sec"):
                        tb.add_scalar(f"train/{k}", row[k], iteration)
                print(f"ep {epoch} it {iteration} loss {row['loss']:.4f} (ce {row['ce']:.4f}, "
                      f"dice {row['dice']:.4f}) lr {lr:.2e} {row['sec_per_iter']:.2f} s/it "
                      f"(waiting for data {row['data_sec']:.2f} s)", flush=True)
                acc = dict.fromkeys(acc, 0.0)
                acc["n"] = 0
                t_log = time.time()
            t_data = time.time()

        if not main_proc:
            continue

        # ---- validation + checkpoints (first process only) ---------------------------
        if (epoch + 1) % t["val_every"] == 0 or epoch + 1 == t["epochs"]:
            net.eval()
            metrics, val_loss = evaluate_loader(net, val_loader, device, cfg)
            s = metrics.summary()
            row = dict(epoch=epoch, iteration=iteration, val_loss=val_loss)
            for k in ("dice_corrupted_mean", "dice_all_mean", "dice_input_mean", "hole_filled",
                      "hole_correct", "acc_corrupted", "damage"):
                row[k] = s[k]
            for c in range(metrics.c):
                row[f"dice_corrupted_{c}"] = float(s["dice_corrupted"][c])
            append_csv(os.path.join(run_dir, "val_log.csv"), row)
            if tb:
                for k, v in row.items():
                    if k not in ("epoch", "iteration"):
                        tb.add_scalar(f"val/{k}", v, iteration)
            print(f"[val] ep {epoch} loss {val_loss:.4f} | Dice corrupted voxels "
                  f"{s['dice_corrupted_mean']:.4f} | Dice all {s['dice_all_mean']:.4f} "
                  f"(input {s['dice_input_mean']:.4f}) | holes filled {s['hole_filled']:.3f} "
                  f"(correct {s['hole_correct']:.3f}) | damage {s['damage']:.5f}", flush=True)
            if s["dice_corrupted_mean"] > best:
                best = s["dice_corrupted_mean"]
                save_checkpoint(os.path.join(run_dir, "best.pt"), net, optimizer, scaler,
                                epoch, iteration, best, cfg)
        save_checkpoint(last, net, optimizer, scaler, epoch, iteration, best, cfg)
        if (epoch + 1) % t["save_every"] == 0:
            save_checkpoint(os.path.join(run_dir, f"epoch_{epoch + 1:04d}.pt"), net, optimizer,
                            scaler, epoch, iteration, best, cfg)

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
