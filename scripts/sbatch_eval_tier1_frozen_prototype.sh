#!/bin/bash
#SBATCH --job-name=eval_t1_frozen
#SBATCH --partition=edith
# Use a fast Quadro RTX 6000 (24 GB). edith2 has mixed GPUs (also has slow RTX 2080 Ti 11 GB)
# — an unlucky allocation on the 2080 Ti would 3×–50× slower. See md_detr_ops.md.
#SBATCH --gres=gpu:quadro_rtx_6000:1
#SBATCH --cpus-per-gpu=16
#SBATCH --time=20-00:00:00
#SBATCH --output=/ubc/cs/research/shield/projects/kren04/MD_DETR_runs/validate_tier1_frozen_prototype/slurm_%j.log

cd /ubc/cs/research/shield/projects/kren04/MD-DETR

eval "$(conda shell.bash hook)"
conda activate MD-DETR

# Redirect JIT compilation cache to project FS (home dir quota is full).
export TORCH_EXTENSIONS_DIR=/ubc/cs/research/shield/projects/kren04/.torch_extensions
# Reduce GPU memory fragmentation (harmless on 24 GB cards; kept for consistency).
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

EXP_NAME=validate_tier1_frozen_prototype
LOG_DIR=/ubc/cs/research/shield/projects/kren04/MD_DETR_runs/${EXP_NAME}
mkdir -p "${LOG_DIR}"

# Tier 1 frozen-baseline full-CL-protocol eval.
# Pipeline: no memory (use_prompts=0) + prototype classifier + single Task-1
# DynamicPrompt checkpoint re-loaded for every task via the symlink mirror
# (see validate_tier1_frozen_prototype.yaml for the one-time symlink setup).
#
# Produces final_stats.txt with T1 C; T2 C/P/A; T3 C/P/A; T4 C/P/A blocks.
# Parse both IoU=0.50 and IoU=0.50:0.95 numbers against the DynamicPrompt
# native baseline (36%/32% T4 C/P @0.5095; 52%/52% @0.50).
python run.py \
    experiment=validate_tier1_frozen_prototype \
    experiment.exp_name=${EXP_NAME} \
    shared=shield \
    run.local=true \
    sbatch.gpus_per_node=1
