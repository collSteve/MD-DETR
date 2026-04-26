#!/bin/bash
# Pre-1b — Gate-B variance baseline.
# Runs D1 twice on the DP baseline checkpoint with different seeds (43 and 44) so
# analysis/compute_gate_b_sigma.py can measure the measurement noise σ and set
# Gate B's ±2σ tolerance. See docs/MD-DETR/phase_1_conceptual_plan.md §3.2.
#
# Unlike the other scripts in this directory, this is a *wrapper* that submits
# two sbatch jobs rather than running compute itself. Invoke as:
#
#   bash scripts/sbatch_diagnostic_per_layer_dp_baseline_seeds.sh
#
# (Run from the cluster login node. No GPU needed for the wrapper itself.)

set -euo pipefail

REPO_ROOT=/ubc/cs/research/shield/projects/kren04/MD-DETR
cd "${REPO_ROOT}"

RUNS_ROOT=/ubc/cs/research/shield/projects/kren04/MD_DETR_runs
mkdir -p "${RUNS_ROOT}/diagnostic_per_layer_separability_seed43"
mkdir -p "${RUNS_ROOT}/diagnostic_per_layer_separability_seed44"

# Create two one-off sbatch files (inline heredocs) and submit.
for SEED in 43 44; do
  SBATCH_FILE=$(mktemp /tmp/sbatch_d1_seed${SEED}_XXXX.sh)
  cat > "${SBATCH_FILE}" <<EOF
#!/bin/bash
#SBATCH --job-name=d1_variance_seed${SEED}
#SBATCH --partition=edith
#SBATCH --gres=gpu:quadro_rtx_6000:1
#SBATCH --cpus-per-gpu=16
#SBATCH --time=20-00:00:00
#SBATCH --output=${RUNS_ROOT}/diagnostic_per_layer_separability_seed${SEED}/slurm_%j.log

cd ${REPO_ROOT}
eval "\$(conda shell.bash hook)"
conda activate MD-DETR
export TORCH_EXTENSIONS_DIR=/ubc/cs/research/shield/projects/kren04/.torch_extensions
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python run.py \\
    experiment=diagnostic_per_layer_separability \\
    experiment.exp_name=diagnostic_per_layer_separability_seed${SEED} \\
    experiment.d1_seed=${SEED} \\
    shared=shield \\
    run.local=true \\
    sbatch.gpus_per_node=1
EOF
  echo "Submitting seed=${SEED}: ${SBATCH_FILE}"
  sbatch "${SBATCH_FILE}"
done

echo
echo "Both jobs submitted. When both finish, compute σ with:"
echo "  conda run -n MD-DETR python analysis/compute_gate_b_sigma.py \\"
echo "      --run_a ${RUNS_ROOT}/diagnostic_per_layer_separability_seed43/per_layer_prototypes.pt \\"
echo "      --run_b ${RUNS_ROOT}/diagnostic_per_layer_separability_seed44/per_layer_prototypes.pt"
