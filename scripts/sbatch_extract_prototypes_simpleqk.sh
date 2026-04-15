#!/bin/bash
#SBATCH --job-name=extract_proto_sqk
#SBATCH --partition=edith
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=2
#SBATCH --time=6:00:00
#SBATCH --output=/ubc/cs/research/shield/projects/kren04/MD_DETR_runs/extract_prototypes_simpleqk/slurm_%j.log

cd /ubc/cs/research/shield/projects/kren04/MD-DETR

eval "$(conda shell.bash hook)"
conda activate MD-DETR

EXP_NAME=extract_prototypes_simpleqk
LOG_DIR=/ubc/cs/research/shield/projects/kren04/MD_DETR_runs/${EXP_NAME}
mkdir -p "${LOG_DIR}"

# Source model: SimpleProposalMemory no-ortho, 6 epochs, standard engine
python run.py \
    experiment=validate_with_prompt \
    experiment.exp_name=${EXP_NAME} \
    experiment.extract_prototypes=true \
    experiment.checkpoint_dir=/ubc/cs/research/shield/projects/kren04/MD_DETR_runs/train_proposal_query_memory_simple_qK_mem_u_10_epoch_6/Task_1 \
    experiment.checkpoint_next=checkpoint05.pth \
    shared=shield \
    run.local=true \
    sbatch.gpus_per_node=1
