#!/bin/bash
#SBATCH --job-name=eval_proto_dyn
#SBATCH --partition=edith
# Use a fast Quadro RTX 6000 (24 GB). edith2 has mixed GPUs (also has slow RTX 2080 Ti 11 GB)
# — an unlucky allocation on the 2080 Ti would 3×–50× slower. See md_detr_ops.md.
#SBATCH --gres=gpu:quadro_rtx_6000:1
#SBATCH --cpus-per-gpu=16
#SBATCH --time=20-00:00:00
#SBATCH --output=/ubc/cs/research/shield/projects/kren04/MD_DETR_runs/eval_prototype_dynamic/slurm_%j.log

cd /ubc/cs/research/shield/projects/kren04/MD-DETR

eval "$(conda shell.bash hook)"
conda activate MD-DETR

# Redirect JIT compilation cache to project FS (home dir quota is full)
export TORCH_EXTENSIONS_DIR=/ubc/cs/research/shield/projects/kren04/.torch_extensions
# Reduce GPU memory fragmentation (harmless on 24 GB cards; kept for consistency)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

EXP_NAME=eval_prototype_dynamic
LOG_DIR=/ubc/cs/research/shield/projects/kren04/MD_DETR_runs/${EXP_NAME}
mkdir -p "${LOG_DIR}"

# Eval DynamicPrompt with prototype classifier.
# use_dynamic_prompt=true is set in validate_with_prompt_dyn_mem.yaml (required to load
# the DynamicPrompt checkpoint with the correct 25-unit memory structure).
# batch_size=16 overrides the config's default of 1 (way too slow at eval time).
#
# checkpoint_base/checkpoint_next must be overridden here. The yaml defaults
# checkpoint_base=checkpoint10.pth, which is correct for train_with_promp_dyn_mem_new
# (T1=11 epochs) but NOT for train_dynamic_correctness_a (T1=6 epochs — only
# checkpoint05.pth exists in each Task_N folder). Without these overrides the eval
# crashes at startup with FileNotFoundError on Task_1/checkpoint10.pth.
python run.py \
    experiment=validate_with_prompt_dyn_mem \
    experiment.exp_name=${EXP_NAME} \
    experiment.batch_size=16 \
    experiment.use_prototype_classifier=true \
    experiment.prototypes_path=/ubc/cs/research/shield/projects/kren04/MD_DETR_runs/extract_prototypes_dynamic/prototypes.pt \
    experiment.prototype_temperature=10.0 \
    experiment.checkpoint_dir=/ubc/cs/research/shield/projects/kren04/MD_DETR_runs/train_dynamic_correctness_a/Task_1 \
    experiment.checkpoint_base=checkpoint05.pth \
    experiment.checkpoint_next=checkpoint05.pth \
    shared=shield \
    run.local=true \
    sbatch.gpus_per_node=1
