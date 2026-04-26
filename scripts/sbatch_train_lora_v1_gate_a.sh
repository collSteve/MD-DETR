#!/bin/bash
#SBATCH --job-name=train_lora_v1_gate_a
#SBATCH --partition=edith
# Phase 1.3 Gate A: Task-1-only training, full 6 epochs, 8-GPU Lightning DDP.
# Purpose: measure (1) T1 C mAP @ IoU=0.50:0.95 (Gate A threshold ≥55%) and
# (2) 6-epoch LoRA-norm ratio trajectory per layer (Gate A threshold >1.5 or
# "monotonically > 1.0" heuristic per phase_1_conceptual_plan.md §2.8 discussion).
# eval_epochs=2 → validation fires at epochs 1, 3, 5 → stats.txt populated.
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-gpu=2
#SBATCH --time=20-00:00:00
#SBATCH --output=/ubc/cs/research/shield/projects/kren04/MD_DETR_runs/train_lora_v1_gate_a/slurm_%j.log

cd /ubc/cs/research/shield/projects/kren04/MD-DETR

eval "$(conda shell.bash hook)"
conda activate MD-DETR

export TORCH_EXTENSIONS_DIR=/ubc/cs/research/shield/projects/kren04/.torch_extensions
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

EXP_NAME=train_lora_v1_gate_a
LOG_DIR=/ubc/cs/research/shield/projects/kren04/MD_DETR_runs/${EXP_NAME}
mkdir -p "${LOG_DIR}"

python run.py \
    experiment=train_lora_v1 \
    experiment.exp_name=${EXP_NAME} \
    experiment.n_tasks=1 \
    shared=shield \
    run.local=true \
    sbatch.gpus_per_node=8
