#!/bin/bash
#SBATCH --job-name=selective_mem
#SBATCH --partition=edith
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-gpu=2
#SBATCH --time=3-00:00:00
#SBATCH --output=/ubc/cs/research/shield/projects/kren04/MD_DETR_runs/selective_softmax_null_bgloss/slurm_%j.log

cd /ubc/cs/research/shield/projects/kren04/MD-DETR

eval "$(conda shell.bash hook)"
conda activate MD-DETR

EXP_NAME=selective_softmax_null_bgloss
LOG_DIR=/ubc/cs/research/shield/projects/kren04/MD_DETR_runs/${EXP_NAME}
mkdir -p "${LOG_DIR}"

python run.py \
    experiment=train_with_prompt_selective \
    experiment.exp_name=${EXP_NAME} \
    shared=shield \
    run.local=true \
    sbatch.gpus_per_node=8
