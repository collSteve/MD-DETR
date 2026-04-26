#!/bin/bash
#SBATCH --job-name=extract_t1_frozen
#SBATCH --partition=edith
# Use a fast Quadro RTX 6000 (24 GB). edith2 has mixed GPUs (also has slow RTX 2080 Ti 11 GB)
# — an unlucky allocation on the 2080 Ti would 3×–50× slower. See md_detr_ops.md.
#SBATCH --gres=gpu:quadro_rtx_6000:1
#SBATCH --cpus-per-gpu=16
#SBATCH --time=20-00:00:00
#SBATCH --output=/ubc/cs/research/shield/projects/kren04/MD_DETR_runs/extract_prototypes_tier1_frozen/slurm_%j.log

cd /ubc/cs/research/shield/projects/kren04/MD-DETR

eval "$(conda shell.bash hook)"
conda activate MD-DETR

# Redirect JIT compilation cache to project FS (home dir quota is full).
export TORCH_EXTENSIONS_DIR=/ubc/cs/research/shield/projects/kren04/.torch_extensions
# Reduce GPU memory fragmentation (harmless on 24 GB cards; kept for consistency).
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

EXP_NAME=extract_prototypes_tier1_frozen
LOG_DIR=/ubc/cs/research/shield/projects/kren04/MD_DETR_runs/${EXP_NAME}
mkdir -p "${LOG_DIR}"

# Tier 1 frozen-baseline prototype extraction.
# Loads DynamicPrompt Task-1 checkpoint with use_prompts=0 (memory disabled).
# Iterates all 4 tasks' training data through the same frozen model,
# accumulates per-class feature means via Hungarian matching on decoder
# hidden states (tools/extract_prototypes.py:89-104).
python run.py \
    experiment=extract_prototypes_tier1_frozen \
    experiment.exp_name=${EXP_NAME} \
    shared=shield \
    run.local=true \
    sbatch.gpus_per_node=1
