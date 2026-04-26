#!/bin/bash
#SBATCH --job-name=diag_layer_lora_v1
#SBATCH --partition=edith
# Gate B — re-run D1 on the LoRA-trained Task-4 checkpoint.
# Fast (~30 min). See docs/MD-DETR/phase_1_conceptual_plan.md §5 Gate B.
#SBATCH --gres=gpu:quadro_rtx_6000:1
#SBATCH --cpus-per-gpu=16
#SBATCH --time=20-00:00:00
#SBATCH --output=/ubc/cs/research/shield/projects/kren04/MD_DETR_runs/diagnostic_per_layer_lora_v1/slurm_%j.log

cd /ubc/cs/research/shield/projects/kren04/MD-DETR

eval "$(conda shell.bash hook)"
conda activate MD-DETR

export TORCH_EXTENSIONS_DIR=/ubc/cs/research/shield/projects/kren04/.torch_extensions
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

EXP_NAME=diagnostic_per_layer_lora_v1
LOG_DIR=/ubc/cs/research/shield/projects/kren04/MD_DETR_runs/${EXP_NAME}
mkdir -p "${LOG_DIR}"

python run.py \
    experiment=diagnostic_per_layer_lora_v1 \
    experiment.exp_name=${EXP_NAME} \
    shared=shield \
    run.local=true \
    sbatch.gpus_per_node=1
