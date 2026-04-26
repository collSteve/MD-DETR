#!/bin/bash
#SBATCH --job-name=eval_lora_v1_e1_with_probe
#SBATCH --partition=edith
# E1 ablation — eval with linear-probe classifier swap. Single Quadro (eval path).
# Expected wall-clock ~2h.
#SBATCH --gres=gpu:quadro_rtx_6000:1
#SBATCH --cpus-per-gpu=16
#SBATCH --time=20-00:00:00
#SBATCH --output=/ubc/cs/research/shield/projects/kren04/MD_DETR_runs/eval_lora_v1_e1_with_probe/slurm_%j.log

cd /ubc/cs/research/shield/projects/kren04/MD-DETR

eval "$(conda shell.bash hook)"
conda activate MD-DETR

export TORCH_EXTENSIONS_DIR=/ubc/cs/research/shield/projects/kren04/.torch_extensions
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

EXP_NAME=eval_lora_v1_e1_with_probe
LOG_DIR=/ubc/cs/research/shield/projects/kren04/MD_DETR_runs/${EXP_NAME}
mkdir -p "${LOG_DIR}"

python run.py \
    experiment=eval_lora_v1_e1_with_probe \
    experiment.exp_name=${EXP_NAME} \
    shared=shield \
    run.local=true \
    sbatch.gpus_per_node=1
