#!/bin/bash
#SBATCH --job-name=train_lora_v1_e1
#SBATCH --partition=edith
# E1 ablation — LoRA replaces DP memory (use_prompts=0). 8-GPU Lightning DDP
# training, 4 tasks × 6 epochs. Expected wall-clock ~16h (same as train_lora_v1).
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-gpu=2
#SBATCH --time=20-00:00:00
#SBATCH --output=/ubc/cs/research/shield/projects/kren04/MD_DETR_runs/train_lora_v1_e1/slurm_%j.log

cd /ubc/cs/research/shield/projects/kren04/MD-DETR

eval "$(conda shell.bash hook)"
conda activate MD-DETR

export TORCH_EXTENSIONS_DIR=/ubc/cs/research/shield/projects/kren04/.torch_extensions
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

EXP_NAME=train_lora_v1_e1
LOG_DIR=/ubc/cs/research/shield/projects/kren04/MD_DETR_runs/${EXP_NAME}
mkdir -p "${LOG_DIR}"

# E1 ablation: rank-16 LoRA on decoder fc1, NO DP memory (use_prompts=0).
# Tests whether LoRA alone (EASE-for-detection regime) handles CL better than
# LoRA+DP coexistence did in Phase 1.4 (which regressed vs DP baseline).
python run.py \
    experiment=train_lora_v1_e1 \
    experiment.exp_name=${EXP_NAME} \
    shared=shield \
    run.local=true \
    sbatch.gpus_per_node=8
