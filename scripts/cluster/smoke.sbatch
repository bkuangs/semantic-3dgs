#!/bin/bash
# Checks the driver and JIT-compiles gsplat once (shared cache on scratch) before parallel jobs.
#SBATCH --job-name=s3dgs-smoke
#SBATCH --partition=coe-gpu,ice-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/home/hice1/bkuang7/scratch/s3dgs/logs/%x-%j.out
set -euo pipefail
cd "$HOME/dev/semantic-3dgs"
source scripts/cluster/env.sh
nvidia-smi
uv run --no-sync python -c "import torch, gsplat; print(torch.__version__, torch.version.cuda, gsplat.__version__, torch.cuda.get_device_name())"
uv run --no-sync pytest -q tests/scaffold/test_gpu_smoke.py
