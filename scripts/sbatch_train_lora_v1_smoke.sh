#!/bin/bash
#SBATCH --job-name=train_lora_v1_smoke
#SBATCH --partition=edith
# Training uses Lightning DDP across 8 GPUs (edith2 has 2 Quadro RTX 6000 + 6 RTX 2080 Ti).
# DDP training works on heterogeneous GPUs per md_detr_ops.md §1 (unlike extraction DDP
# which is broken). Requesting all 8 avoids single-Quadro bottleneck. With batch_size=1
# and n_gpus=8, main.py auto-sets accumulate_grad_batches=4 → effective global batch=32
# (same as 1-GPU × accumulate=32 but ~5-8× faster wall-clock).
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-gpu=2
#SBATCH --time=20-00:00:00
#SBATCH --output=/ubc/cs/research/shield/projects/kren04/MD_DETR_runs/train_lora_v1_smoke/slurm_%j.log

cd /ubc/cs/research/shield/projects/kren04/MD-DETR

eval "$(conda shell.bash hook)"
conda activate MD-DETR

# Redirect JIT compilation cache to project FS (home quota is full).
export TORCH_EXTENSIONS_DIR=/ubc/cs/research/shield/projects/kren04/.torch_extensions
# Reduce GPU memory fragmentation. Required on 11 GB 2080 Tis for batch=1 MD-DETR.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

EXP_NAME=train_lora_v1_smoke
LOG_DIR=/ubc/cs/research/shield/projects/kren04/MD_DETR_runs/${EXP_NAME}
mkdir -p "${LOG_DIR}"

# Phase 1.2 cluster smoke — inherits train_lora_v1.yaml (rank-16 LoRA on decoder fc1 +
# O-LoRA soft-ortho + DP memory E2 coexist + LR asymmetry 2.0) but drops to 1 epoch ×
# 1 task. Purpose: shake out DDP + LoRA end-to-end (forward, backward, optimizer groups,
# checkpoint save, LoRA-norm logging, short_map coverage, unused-parameters under DDP)
# BEFORE committing to ~6 h Gate A run or ~2-3 day full training.
#
# Expected wall-clock: ~30-45 min (vs 6 h on 1 GPU).
#
# Verify after completion (see memory/active_work.md Step 3):
#   - Checkpoint contains fc1.base.weight, fc1.A_list.0, fc1.B_list.0, fc1.A_init_norms
#   - Lightning CSV has lora_norm_ratio_L{0..5} columns (rank-0 only)
#   - No KeyError, NCCL deadlock, or OOM in logs
#   - olora loss is 0 during task 1 (no prior tasks)
python run.py \
    experiment=train_lora_v1 \
    experiment.exp_name=${EXP_NAME} \
    experiment.epochs=1 \
    experiment.n_tasks=1 \
    shared=shield \
    run.local=true \
    sbatch.gpus_per_node=8
