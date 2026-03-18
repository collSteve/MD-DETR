#!/bin/bash
# Train SelectiveProposalMemory using main_test.py (fully frozen query function)
# Usage: nohup bash scripts/run_selective_frozen.sh &

cd /ubc/cs/research/shield/projects/kren04/MD-DETR

eval "$(conda shell.bash hook)"
conda activate MD-DETR

EXP_NAME=selective_softmax_null_bgloss_frozen_qf
LOG_DIR=/ubc/cs/research/shield/projects/kren04/MD_DETR_runs/${EXP_NAME}
mkdir -p "${LOG_DIR}"

python run_test.py \
    experiment=train_with_prompt_selective \
    experiment.exp_name=${EXP_NAME} \
    shared=shield \
    run.local=true \
    sbatch.gpus_per_node=1 \
    > "${LOG_DIR}/train.log" 2>&1
