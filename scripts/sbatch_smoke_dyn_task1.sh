#!/bin/bash
#SBATCH --job-name=smoke_dyn_t1
#SBATCH --partition=edith
# Use a fast Quadro RTX 6000 (24 GB). edith2 has mixed GPUs (also has slow RTX 2080 Ti 11 GB)
# — an unlucky allocation on the 2080 Ti would 3×–50× slower. See md_detr_ops.md.
#SBATCH --gres=gpu:quadro_rtx_6000:1
#SBATCH --cpus-per-gpu=16
#SBATCH --time=20-00:00:00
#SBATCH --output=/ubc/cs/research/shield/projects/kren04/MD_DETR_runs/smoke_dyn_task1/slurm_%j.log

cd /ubc/cs/research/shield/projects/kren04/MD-DETR

eval "$(conda shell.bash hook)"
conda activate MD-DETR

# Redirect JIT compilation cache to project FS (home dir quota is full).
export TORCH_EXTENSIONS_DIR=/ubc/cs/research/shield/projects/kren04/.torch_extensions
# Reduce GPU memory fragmentation (harmless on 24 GB cards; kept for consistency).
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

EXP_NAME=smoke_dyn_task1
LOG_DIR=/ubc/cs/research/shield/projects/kren04/MD_DETR_runs/${EXP_NAME}
mkdir -p "${LOG_DIR}"

# Prerequisite smoke test for the Tier 1 frozen baseline work.
# Task-1-only eval of the DynamicPrompt Task-1 checkpoint with native class_embed.
# Verifies:
#   (1) use_dynamic_prompt branch + 25-unit memory load works end-to-end,
#   (2) benchmark numbers reproduce: T1 Current mAP @ IoU=0.50   ≈ 79%
#                                    T1 Current mAP @ IoU=0.50:0.95 ≈ 58%
# n_tasks=1 restricts the eval loop in main.py:440 to a single iteration.
python run.py \
    experiment=validate_with_prompt_dyn_mem \
    experiment.exp_name=${EXP_NAME} \
    experiment.batch_size=16 \
    experiment.start_task=1 \
    experiment.n_tasks=1 \
    experiment.checkpoint_dir=/ubc/cs/research/shield/projects/kren04/MD_DETR_runs/train_dynamic_correctness_a/Task_1 \
    experiment.checkpoint_base=checkpoint05.pth \
    experiment.checkpoint_next=checkpoint05.pth \
    shared=shield \
    run.local=true \
    sbatch.gpus_per_node=1
