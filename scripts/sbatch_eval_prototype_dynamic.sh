#!/bin/bash
#SBATCH --job-name=eval_proto_dyn
#SBATCH --partition=edith
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=2
#SBATCH --time=6:00:00
#SBATCH --output=/ubc/cs/research/shield/projects/kren04/MD_DETR_runs/eval_prototype_dynamic/slurm_%j.log

cd /ubc/cs/research/shield/projects/kren04/MD-DETR

eval "$(conda shell.bash hook)"
conda activate MD-DETR

EXP_NAME=eval_prototype_dynamic
LOG_DIR=/ubc/cs/research/shield/projects/kren04/MD_DETR_runs/${EXP_NAME}
mkdir -p "${LOG_DIR}"

# Eval DynamicPrompt with prototype classifier
python run.py \
    experiment=validate_with_prompt_dyn_mem \
    experiment.exp_name=${EXP_NAME} \
    experiment.use_prototype_classifier=true \
    experiment.prototypes_path=/ubc/cs/research/shield/projects/kren04/MD_DETR_runs/extract_prototypes_dynamic/prototypes.pt \
    experiment.prototype_temperature=10.0 \
    experiment.checkpoint_dir=/ubc/cs/research/shield/projects/kren04/MD_DETR_runs/train_dynamic_correctness_a/Task_1 \
    shared=shield \
    run.local=true \
    sbatch.gpus_per_node=1
