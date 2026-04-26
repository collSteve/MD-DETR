#!/bin/bash
#SBATCH --job-name=extract_feat_dp_t4
#SBATCH --partition=edith
# Use a fast Quadro RTX 6000 (24 GB). edith2 has mixed GPUs (also has slow RTX 2080 Ti 11 GB)
# — an unlucky allocation on the 2080 Ti would 3×–50× slower. See md_detr_ops.md.
#SBATCH --gres=gpu:quadro_rtx_6000:1
#SBATCH --cpus-per-gpu=16
#SBATCH --time=20-00:00:00
#SBATCH --output=/ubc/cs/research/shield/projects/kren04/MD_DETR_runs/extract_features_dp_t4/slurm_%j.log

cd /ubc/cs/research/shield/projects/kren04/MD-DETR

eval "$(conda shell.bash hook)"
conda activate MD-DETR

# Redirect JIT compilation cache to project FS (home dir quota is full).
export TORCH_EXTENSIONS_DIR=/ubc/cs/research/shield/projects/kren04/.torch_extensions
# Reduce GPU memory fragmentation (harmless on 24 GB cards; kept for consistency).
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

EXP_NAME=extract_features_dp_t4
LOG_DIR=/ubc/cs/research/shield/projects/kren04/MD_DETR_runs/${EXP_NAME}
mkdir -p "${LOG_DIR}"

# Stage 1 of the linear-probe upper-bound diagnostic.
# Loads DP Task_4 checkpoint, iterates all 4 tasks' training data through the
# frozen final-state model, accumulates per-class-capped raw (feature, label)
# pairs via Hungarian matching on Pass 2 outputs. Produces features.pt.
# Next step (Stage 2): run tools/train_linear_probe.py offline on visionsw1.
python run.py \
    experiment=extract_features_dp_t4 \
    experiment.exp_name=${EXP_NAME} \
    shared=shield \
    run.local=true \
    sbatch.gpus_per_node=1
