#!/bin/bash
#SBATCH --job-name=train_lora_v1
#SBATCH --partition=edith
# Training uses Lightning DDP across 8 GPUs (edith2 has 2 Quadro RTX 6000 + 6 RTX 2080 Ti).
# DDP training works on heterogeneous GPUs per md_detr_ops.md §1 (unlike extraction DDP
# which is broken). Requesting all 8 avoids single-Quadro bottleneck. With batch_size=1
# and n_gpus=8, main.py auto-sets accumulate_grad_batches=4 → effective global batch=32
# (same as 1-GPU × accumulate=32 but ~5-8× faster wall-clock). Full 4-task training:
# ~2-3 days at 8 GPUs; 1-GPU would be ~2-3 weeks (unusable).
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-gpu=2
#SBATCH --time=20-00:00:00
#SBATCH --output=/ubc/cs/research/shield/projects/kren04/MD_DETR_runs/train_lora_v1/slurm_%j.log

cd /ubc/cs/research/shield/projects/kren04/MD-DETR

eval "$(conda shell.bash hook)"
conda activate MD-DETR

# Redirect JIT compilation cache to project FS (home quota is full).
export TORCH_EXTENSIONS_DIR=/ubc/cs/research/shield/projects/kren04/.torch_extensions
# Reduce GPU memory fragmentation. Required on 11 GB 2080 Tis for batch=1 MD-DETR.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

EXP_NAME=train_lora_v1
LOG_DIR=/ubc/cs/research/shield/projects/kren04/MD_DETR_runs/${EXP_NAME}
mkdir -p "${LOG_DIR}"

# Path B Variant A v1: rank-16 LoRA on decoder fc1 + O-LoRA soft-ortho + DP memory
# coexistence (E2, use_prompts=1) + LR asymmetry (DP at lr/2, LoRA at lr).
# See docs/MD-DETR/phase_1_conceptual_plan.md and the plan file.
python run.py \
    experiment=train_lora_v1 \
    experiment.exp_name=${EXP_NAME} \
    shared=shield \
    run.local=true \
    sbatch.gpus_per_node=8
