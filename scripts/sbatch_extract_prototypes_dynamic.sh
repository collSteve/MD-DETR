#!/bin/bash
#SBATCH --job-name=extract_proto_dyn
#SBATCH --partition=edith
# Request 1 fast GPU (Quadro RTX 6000, 24 GB) — DDP proved unreliable with mixed GPUs on edith2
# If this --gres form is rejected, check `scontrol show node edith2` for the exact gres type name
# (alternatives to try: `gpu:quadro_rtx_6000:1`, or `--gres=gpu:1 --constraint=rtx6000`)
#SBATCH --gres=gpu:quadro_rtx_6000:1
#SBATCH --cpus-per-gpu=16
#SBATCH --time=20-00:00:00
#SBATCH --output=/ubc/cs/research/shield/projects/kren04/MD_DETR_runs/extract_prototypes_dynamic/slurm_%j.log

cd /ubc/cs/research/shield/projects/kren04/MD-DETR

eval "$(conda shell.bash hook)"
conda activate MD-DETR

# Redirect JIT compilation cache to project FS (home dir quota is full)
export TORCH_EXTENSIONS_DIR=/ubc/cs/research/shield/projects/kren04/.torch_extensions
# Reduce GPU memory fragmentation (harmless on 24 GB cards; kept for consistency)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

EXP_NAME=extract_prototypes_dynamic
LOG_DIR=/ubc/cs/research/shield/projects/kren04/MD_DETR_runs/${EXP_NAME}
mkdir -p "${LOG_DIR}"

# Source model: DynamicPrompt (standard engine → use run.py)
# Final checkpoint lives at: train_dynamic_correctness_a/Task_4/checkpoint05.pth
# Single-GPU mode: sbatch.gpus_per_node=1 takes run.py's `else` branch (plain python, no torchrun).
#
# checkpoint_base/checkpoint_next are explicitly overridden because validate_with_prompt_dyn_mem.yaml
# defaults checkpoint_base=checkpoint10.pth — that's correct for train_with_promp_dyn_mem_new
# (T1=11 epochs) but NOT for train_dynamic_correctness_a (T1=6 epochs, only checkpoint05.pth exists).
# In extract mode the override is defensive (extract_prototypes.py uses its own task-N-specific load
# path that bypasses engine.resume() and so doesn't actually read checkpoint_base) — but keeping the
# override here means the script is self-consistent and will not break if someone changes the
# extraction load semantics later.
python run.py \
    experiment=validate_with_prompt_dyn_mem \
    experiment.exp_name=${EXP_NAME} \
    experiment.extract_prototypes=true \
    experiment.extract_num_workers=8 \
    experiment.extract_batch_size=16 \
    experiment.extract_max_samples_per_class=500 \
    experiment.checkpoint_dir=/ubc/cs/research/shield/projects/kren04/MD_DETR_runs/train_dynamic_correctness_a/Task_1 \
    experiment.checkpoint_base=checkpoint05.pth \
    experiment.checkpoint_next=checkpoint05.pth \
    shared=shield \
    run.local=true \
    sbatch.gpus_per_node=1
