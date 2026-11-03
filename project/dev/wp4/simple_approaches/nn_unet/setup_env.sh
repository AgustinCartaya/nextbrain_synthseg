#!/usr/bin/env bash
# Creates a dedicated conda env "nnunet" with PyTorch (CUDA 12.x) and nnU-Net v2.
set -euo pipefail
ENV_NAME=${1:-nnunet}
source "$(conda info --base)/etc/profile.d/conda.sh"
conda create -y -n "$ENV_NAME" python=3.12
conda activate "$ENV_NAME"
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu129
pip install nnunetv2
python -c "import torch, nnunetv2; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
