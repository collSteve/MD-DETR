#!/bin/bash
#SBATCH --job-name=eval_proto_sqk
#SBATCH --partition=edith
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=2
#SBATCH --time=6:00:00
#SBATCH --output=/ubc/cs/research/shield/projects/kren04/MD_DETR_runs/eval_prototype_simpleqk/slurm_%j.log

cd /ubc/cs/research/shield/projects/kren04/MD-DETR

eval "$(conda shell.bash hook)"
conda activate MD-DETR

EXP_NAME=eval_prototype_simpleqk
LOG_DIR=/ubc/cs/research/shield/projects/kren04/MD_DETR_runs/${EXP_NAME}
mkdir -p "${LOG_DIR}"

# Eval SimpleProposalMemory (no-ortho, 6ep, standard engine) with prototype classifier
python run.py \
    experiment=validate_with_prompt \
    experiment.exp_name=${EXP_NAME} \
    experiment.use_prototype_classifier=true \
    experiment.prototypes_path=/ubc/cs/research/shield/projects/kren04/MD_DETR_runs/extract_prototypes_simpleqk/prototypes.pt \
    experiment.prototype_temperature=10.0 \
    experiment.checkpoint_dir=/ubc/cs/research/shield/projects/kren04/MD_DETR_runs/train_proposal_query_memory_simple_qK_mem_u_10_epoch_6/Task_1 \
    shared=shield \
    run.local=true \
    sbatch.gpus_per_node=1
