#!/usr/bin/env bash
# Generate the 1000 images in the background (survives closing the terminal).
#   ./launch.sh                 every GPU
#   ./launch.sh --gpus 0 2      chosen GPUs (any option of generate.py)
# Follow:  tail -f logs/generate_*.log      Stop:  pkill -f generate_data/generate.py
# Resume after a stop or a crash: run ./launch.sh again (finished cases are skipped).
set -e
cd "$(dirname "$0")"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate nxb
mkdir -p logs
log="logs/generate_$(date +%Y%m%d_%H%M%S).log"
nohup python -u generate.py "$@" > "$log" 2>&1 &
echo "PID $! -> $log"
