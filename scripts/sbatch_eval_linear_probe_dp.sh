#!/bin/bash
#SBATCH --job-name=eval_lp_dp
#SBATCH --partition=edith
# Use a fast Quadro RTX 6000 (24 GB). edith2 has mixed GPUs (also has slow RTX 2080 Ti 11 GB)
# — an unlucky allocation on the 2080 Ti would 3×–50× slower. See md_detr_ops.md.
#SBATCH --gres=gpu:quadro_rtx_6000:1
#SBATCH --cpus-per-gpu=16
#SBATCH --time=20-00:00:00
#SBATCH --output=/ubc/cs/research/shield/projects/kren04/MD_DETR_runs/eval_linear_probe_dp/slurm_%j.log

cd /ubc/cs/research/shield/projects/kren04/MD-DETR

eval "$(conda shell.bash hook)"
conda activate MD-DETR

# Redirect JIT compilation cache to project FS (home dir quota is full).
export TORCH_EXTENSIONS_DIR=/ubc/cs/research/shield/projects/kren04/.torch_extensions
# Reduce GPU memory fragmentation (harmless on 24 GB cards; kept for consistency).
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

EXP_NAME=eval_linear_probe_dp
LOG_DIR=/ubc/cs/research/shield/projects/kren04/MD_DETR_runs/${EXP_NAME}
mkdir -p "${LOG_DIR}"

# Stage 3 of the linear-probe upper-bound diagnostic.
# Full 4-task CL eval protocol: per-task DP checkpoints loaded via main.py's
# standard Task_1 → Task_{task_id} replace logic. The trained linear probe
# replaces class_embed logits at eval time via the engine.common_step swap.
# Produces final_stats.txt with T1–T4 Current/Previous/All blocks at both IoU
# thresholds — the full forgetting trajectory under a calibrated linear head.
python run.py \
    experiment=eval_linear_probe_dp \
    experiment.exp_name=${EXP_NAME} \
    shared=shield \
    run.local=true \
    sbatch.gpus_per_node=1
