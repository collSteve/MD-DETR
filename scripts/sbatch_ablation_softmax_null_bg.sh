#!/bin/bash
#SBATCH --job-name=abl_sm_bg
#SBATCH --partition=edith
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-gpu=2
#SBATCH --time=3-00:00:00
#SBATCH --output=/ubc/cs/research/shield/projects/kren04/MD_DETR_runs/ablation_softmax_null_bg_f1_frozen_qf/slurm_%j.log

cd /ubc/cs/research/shield/projects/kren04/MD-DETR

eval "$(conda shell.bash hook)"
conda activate MD-DETR

EXP_NAME=ablation_softmax_null_bg_f1_frozen_qf
LOG_DIR=/ubc/cs/research/shield/projects/kren04/MD_DETR_runs/${EXP_NAME}
mkdir -p "${LOG_DIR}"

python run_test.py \
    experiment=ablation_softmax_null_bg \
    experiment.exp_name=${EXP_NAME} \
    shared=shield \
    run.local=true \
    sbatch.gpus_per_node=8
