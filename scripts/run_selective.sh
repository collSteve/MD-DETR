#!/bin/bash
# Train SelectiveProposalMemory using main.py (standard engine — query components trainable at lr_old)
# Usage: nohup bash scripts/run_selective.sh &

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
    sbatch.gpus_per_node=1 \
    > "${LOG_DIR}/train.log" 2>&1
