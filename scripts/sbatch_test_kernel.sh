#!/bin/bash
#SBATCH --job-name=test-kernel
#SBATCH --partition=edith
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=2
#SBATCH --time=1:00:00
#SBATCH --output=/ubc/cs/research/shield/projects/kren04/MD_DETR_runs/test_kernel/slurm_%j.log

cd /ubc/cs/research/shield/projects/kren04/MD-DETR
mkdir -p /ubc/cs/research/shield/projects/kren04/MD_DETR_runs/test_kernel
eval "$(conda shell.bash hook)"
conda activate MD-DETR

# Redirect JIT compilation cache to project FS (home dir quota is full)
export TORCH_EXTENSIONS_DIR=/ubc/cs/research/shield/projects/kren04/.torch_extensions

# Clear any stale build artifacts from previous failed attempts
rm -rf ${TORCH_EXTENSIONS_DIR}/py313_cu124/MultiScaleDeformableAttention 2>/dev/null || true

# nvcc comes from: conda install -n MD-DETR -c nvidia cuda-toolkit=12.4
# (one-time setup; puts nvcc in the conda env's bin/)

echo "=== Environment ==="
which nvcc || echo "WARN nvcc not in PATH"
python -c "import torch; print('torch.cuda:', torch.version.cuda, 'is_available:', torch.cuda.is_available())"
python -c "from transformers.utils import is_ninja_available; print('ninja:', is_ninja_available())"

echo "=== Kernel load test ==="
python -c "from models.load_custom import load_cuda_kernels; m = load_cuda_kernels(); print('KERNEL LOADED:', m)"
