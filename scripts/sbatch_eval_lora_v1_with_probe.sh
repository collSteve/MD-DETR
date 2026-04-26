#!/bin/bash
#SBATCH --job-name=eval_lora_v1_probe
#SBATCH --partition=edith
# Phase 1.6 — eval the LoRA-trained 4-task checkpoints with probe swapped for class_embed.
#SBATCH --gres=gpu:quadro_rtx_6000:1
#SBATCH --cpus-per-gpu=16
#SBATCH --time=20-00:00:00
#SBATCH --output=/ubc/cs/research/shield/projects/kren04/MD_DETR_runs/eval_lora_v1_with_probe/slurm_%j.log

cd /ubc/cs/research/shield/projects/kren04/MD-DETR

eval "$(conda shell.bash hook)"
conda activate MD-DETR

export TORCH_EXTENSIONS_DIR=/ubc/cs/research/shield/projects/kren04/.torch_extensions
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

EXP_NAME=eval_lora_v1_with_probe
LOG_DIR=/ubc/cs/research/shield/projects/kren04/MD_DETR_runs/${EXP_NAME}
mkdir -p "${LOG_DIR}"

# Assumes:
#   (a) train_lora_v1 has produced Task_{1..4}/checkpoint05.pth
#   (b) tools/extract_features.py produced features.pt and tools/train_linear_probe.py
#       produced linear_probe.pt (Phase 1.6.1 and 1.6.2 of the plan)
python run.py \
    experiment=eval_lora_v1_with_probe \
    experiment.exp_name=${EXP_NAME} \
    shared=shield \
    run.local=true \
    sbatch.gpus_per_node=1
