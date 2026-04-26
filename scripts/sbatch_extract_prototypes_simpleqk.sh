#!/bin/bash
#SBATCH --job-name=extract_proto_sqk
#SBATCH --partition=edith
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-gpu=4
#SBATCH --time=20-00:00:00
#SBATCH --output=/ubc/cs/research/shield/projects/kren04/MD_DETR_runs/extract_prototypes_simpleqk/slurm_%j.log

cd /ubc/cs/research/shield/projects/kren04/MD-DETR

eval "$(conda shell.bash hook)"
conda activate MD-DETR

# Redirect JIT compilation cache to project FS (home dir quota is full)
export TORCH_EXTENSIONS_DIR=/ubc/cs/research/shield/projects/kren04/.torch_extensions
# Make NCCL errors propagate fast instead of hanging on collective ops
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
# Reduce GPU memory fragmentation (needed on smaller 11GB GPUs)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

EXP_NAME=extract_prototypes_simpleqk
LOG_DIR=/ubc/cs/research/shield/projects/kren04/MD_DETR_runs/${EXP_NAME}
mkdir -p "${LOG_DIR}"

# Source model: SimpleProposalMemory no-ortho, 6 epochs, standard engine
python run.py \
    experiment=validate_with_prompt \
    experiment.exp_name=${EXP_NAME} \
    experiment.extract_prototypes=true \
    experiment.extract_num_workers=4 \
    experiment.extract_batch_size=8 \
    experiment.extract_max_samples_per_class=500 \
    experiment.checkpoint_dir=/ubc/cs/research/shield/projects/kren04/MD_DETR_runs/train_proposal_query_memory_simple_qK_mem_u_10_epoch_6/Task_1 \
    experiment.checkpoint_next=checkpoint05.pth \
    shared=shield \
    run.local=true \
    sbatch.gpus_per_node=4
