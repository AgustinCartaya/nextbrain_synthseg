#!/usr/bin/env bash
# =============================================================================
# The run that produced results/l1_resenc_2h: 40 epochs x 200 iterations on 3 GPUs, followed by the
# evaluation on held-out FOMO patches and the restoration of the 5 real cases.
#
# Measured: 0.56-0.62 s/iteration at batch 4 per GPU (12 patches per step), with
# occasional data-loading stalls. Budgeted at ~0.8 s/iteration:
#   40 epochs x 200 iterations = 8000 steps  ->  about 80-110 minutes.
# GPU 1 is left free because another user's processes hold memory on it.
#
# NOTE: the budget below was wrong. The real run averaged 1.8 s/iteration
# (data loading was the bottleneck) and took 4 h 10 min.
#
#   nohup bash launchers/run_2h.sh > results/l1_resenc_2h_full_run.log 2>&1 &
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."            # the project folder
GPUS="${GPUS:-0,2,3}"
N=$(echo "$GPUS" | tr ',' '\n' | wc -l)
SET="train.run_name=l1_resenc_2h train.epochs=40 train.iters_per_epoch=200 train.batch_size=4 train.num_workers=36 train.val_every=5 train.val_samples=48 train.save_every=10"

echo "== training start: $(date)"
CUDA_VISIBLE_DEVICES=$GPUS torchrun --nproc_per_node=$N 04_train.py --config configs/default.yaml --set $SET
echo "== training end: $(date)"

export CUDA_VISIBLE_DEVICES="${GPUS%%,*}"
echo "== evaluation on held-out patches (synthetic corruption)"
python 06_evaluate.py --config configs/default.yaml --mode patch --set train.run_name=l1_resenc_2h
echo "== restoring the 5 real test cases"
python 05_infer.py --config configs/default.yaml --set train.run_name=l1_resenc_2h
echo "== all done: $(date)"
