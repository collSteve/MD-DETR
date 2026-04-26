#!/bin/bash
#SBATCH --job-name=diag_null_space
#SBATCH --partition=edith
# Fast Quadro RTX 6000 (24 GB). D2 uses a single final-task checkpoint and
# iterates each of the 4 tasks' training data through it. Expect ~1 hour
# wall-clock. Single-GPU (DDP not implemented for this diagnostic).
# See docs/MD-DETR/path_b_design.md §7.
#SBATCH --gres=gpu:quadro_rtx_6000:1
#SBATCH --cpus-per-gpu=16
#SBATCH --time=20-00:00:00
#SBATCH --output=/ubc/cs/research/shield/projects/kren04/MD_DETR_runs/diagnostic_null_space_viability/slurm_%j.log

cd /ubc/cs/research/shield/projects/kren04/MD-DETR

eval "$(conda shell.bash hook)"
conda activate MD-DETR

# Redirect JIT compilation cache to project FS (home dir quota is full).
export TORCH_EXTENSIONS_DIR=/ubc/cs/research/shield/projects/kren04/.torch_extensions
# Reduce GPU memory fragmentation (harmless on 24 GB, kept for consistency).
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

EXP_NAME=diagnostic_null_space_viability
LOG_DIR=/ubc/cs/research/shield/projects/kren04/MD_DETR_runs/${EXP_NAME}
mkdir -p "${LOG_DIR}"

# Path B Phase 0 diagnostic D2: per-attach-point activation-subspace null-space
# viability. Loads the final Task_4 checkpoint once, then iterates each of the
# 4 tasks' training data through the same fixed model. Captures input activations
# at decoder self-attn and FFN attach points (12 attach points × 4 tasks = 48
# covariance matrices), reports null-space dimensions + feasibility of
# InfLoRA-style init at LoRA ranks {8, 16, 32}.
#
# Why a single fixed model (not per-task checkpoints)? engine.resume() uses
# strict=False, so loading earlier-task checkpoints into a model sized for N
# tasks silently skips the shape-mismatched memory params and leaves late
# tasks' memory at random-init — polluting layer>0 activations. Using one
# fixed model isolates DATA-driven subspace variation, which is the right
# methodology for InfLoRA feasibility analysis.
#
# checkpoint_base/next are the defaults for train_dynamic_correctness_a — D2
# constructs the final-task path by substituting Task_1 → Task_N internally.
python run.py \
    experiment=diagnostic_null_space_viability \
    experiment.exp_name=${EXP_NAME} \
    experiment.checkpoint_dir=/ubc/cs/research/shield/projects/kren04/MD_DETR_runs/train_dynamic_correctness_a/Task_1 \
    experiment.checkpoint_base=checkpoint05.pth \
    experiment.checkpoint_next=checkpoint05.pth \
    shared=shield \
    run.local=true \
    sbatch.gpus_per_node=1
