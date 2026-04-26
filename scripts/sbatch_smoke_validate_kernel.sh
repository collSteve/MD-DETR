#!/bin/bash
#SBATCH --job-name=smoke-kernel
#SBATCH --partition=edith
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --time=1:00:00
#SBATCH --output=/ubc/cs/research/shield/projects/kren04/MD_DETR_runs/smoke_kernel_validate/slurm_%j.log

cd /ubc/cs/research/shield/projects/kren04/MD-DETR
EXP_NAME=smoke_kernel_validate
LOG_DIR=/ubc/cs/research/shield/projects/kren04/MD_DETR_runs/${EXP_NAME}
mkdir -p "${LOG_DIR}"

eval "$(conda shell.bash hook)"
conda activate MD-DETR

# Redirect JIT compilation cache to project FS (home dir quota is full)
export TORCH_EXTENSIONS_DIR=/ubc/cs/research/shield/projects/kren04/.torch_extensions

# Task-1-only eval on the SimpleQK baseline at batch_size=16. Tests:
#   (1) absence of "Could not load the custom kernel" warning  -> kernel dispatched
#   (2) no OOM at batch=16                                     -> Python fallback was the memory hog
#   (3) iter/s meaningfully above the 1.7 baseline             -> real speedup
python run.py \
    experiment=validate_with_prompt \
    shared=shield \
    run.local=true \
    sbatch.gpus_per_node=1 \
    experiment.exp_name=${EXP_NAME} \
    experiment.batch_size=16 \
    experiment.start_task=1 \
    experiment.n_tasks=1 \
    experiment.checkpoint_dir=/ubc/cs/research/shield/projects/kren04/MD_DETR_runs/train_proposal_query_memory_simple_qK_mem_u_10_epoch_6/Task_1 \
    experiment.checkpoint_base=checkpoint05.pth
