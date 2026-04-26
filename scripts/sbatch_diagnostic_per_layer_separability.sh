#!/bin/bash
#SBATCH --job-name=diag_per_layer
#SBATCH --partition=edith
# Fast Quadro RTX 6000 (24 GB). D1 is cheap (~30 min) and doesn't need DDP.
# See docs/MD-DETR/prototype_diagnostic.md + path_b_design.md §7.
#SBATCH --gres=gpu:quadro_rtx_6000:1
#SBATCH --cpus-per-gpu=16
#SBATCH --time=20-00:00:00
#SBATCH --output=/ubc/cs/research/shield/projects/kren04/MD_DETR_runs/diagnostic_per_layer_separability/slurm_%j.log

cd /ubc/cs/research/shield/projects/kren04/MD-DETR

eval "$(conda shell.bash hook)"
conda activate MD-DETR

# Redirect JIT compilation cache to project FS (home dir quota is full).
export TORCH_EXTENSIONS_DIR=/ubc/cs/research/shield/projects/kren04/.torch_extensions
# Reduce GPU memory fragmentation (harmless on 24 GB, kept for consistency).
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

EXP_NAME=diagnostic_per_layer_separability
LOG_DIR=/ubc/cs/research/shield/projects/kren04/MD_DETR_runs/${EXP_NAME}
mkdir -p "${LOG_DIR}"

# Path B Phase 0 diagnostic D1: per-layer class separability on the DynamicPrompt
# baseline checkpoint (train_dynamic_correctness_a). Installs forward hooks on
# each of the 6 decoder layers, accumulates per-class prototypes per layer, then
# prints pairwise cosine statistics at each depth. Decides the LoRA attach point.
#
# checkpoint_base/next overrides are defensive — validate_with_prompt_dyn_mem.yaml
# defaults to checkpoint10.pth (from train_with_promp_dyn_mem_new) which doesn't
# exist for train_dynamic_correctness_a. See md_detr_code_gotchas.md.
python run.py \
    experiment=diagnostic_per_layer_separability \
    experiment.exp_name=${EXP_NAME} \
    experiment.checkpoint_dir=/ubc/cs/research/shield/projects/kren04/MD_DETR_runs/train_dynamic_correctness_a/Task_1 \
    experiment.checkpoint_base=checkpoint05.pth \
    experiment.checkpoint_next=checkpoint05.pth \
    shared=shield \
    run.local=true \
    sbatch.gpus_per_node=1
