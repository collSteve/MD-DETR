#!/bin/bash
#SBATCH --job-name=eval_proto_sqk
#SBATCH --partition=edith
# Use a fast Quadro RTX 6000 (24 GB). edith2 has mixed GPUs (also has slow RTX 2080 Ti 11 GB)
# — an unlucky allocation on the 2080 Ti would 3×–50× slower. See md_detr_ops.md.
#SBATCH --gres=gpu:quadro_rtx_6000:1
#SBATCH --cpus-per-gpu=16
#SBATCH --time=20-00:00:00
#SBATCH --output=/ubc/cs/research/shield/projects/kren04/MD_DETR_runs/eval_prototype_simpleqk/slurm_%j.log

cd /ubc/cs/research/shield/projects/kren04/MD-DETR

eval "$(conda shell.bash hook)"
conda activate MD-DETR

# Redirect JIT compilation cache to project FS (home dir quota is full)
export TORCH_EXTENSIONS_DIR=/ubc/cs/research/shield/projects/kren04/.torch_extensions
# Reduce GPU memory fragmentation (harmless on 24 GB cards; kept for consistency)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

EXP_NAME=eval_prototype_simpleqk
LOG_DIR=/ubc/cs/research/shield/projects/kren04/MD_DETR_runs/${EXP_NAME}
mkdir -p "${LOG_DIR}"

# Eval SimpleProposalMemory (no-ortho, 6ep, standard engine) with prototype classifier.
# batch_size=16 overrides validate_with_prompt.yaml's default of 1 (way too slow at eval time).
python run.py \
    experiment=validate_with_prompt \
    experiment.exp_name=${EXP_NAME} \
    experiment.batch_size=16 \
    experiment.use_prototype_classifier=true \
    experiment.prototypes_path=/ubc/cs/research/shield/projects/kren04/MD_DETR_runs/extract_prototypes_simpleqk/prototypes.pt \
    experiment.prototype_temperature=10.0 \
    experiment.checkpoint_dir=/ubc/cs/research/shield/projects/kren04/MD_DETR_runs/train_proposal_query_memory_simple_qK_mem_u_10_epoch_6/Task_1 \
    shared=shield \
    run.local=true \
    sbatch.gpus_per_node=1
