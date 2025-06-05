#!/usr/bin/env bash
# ------------------------------------------------------------
# Submit an MD-DETR experiment to Slurm in one command.
#
#  usage:
#     ./launch.sh -e config/demo_agn.env                 # default cluster
#     ./launch.sh -e config/demo_agn.env -p config/cluster_2node.env
#     ./launch.sh -e config/fast_debug.env --dry-run
# ------------------------------------------------------------
set -euo pipefail

# PROJ_ROOT_PATH=/h/stevev/MD-DETR

# ---------- defaults ----------
GLOBAL=config/global.env
SBATCH_CONFIG=config/sbatch/validate.sbatch.env
SCRIPT=submit.sh      # the generic Slurm wrapper
DRYRUN=0

# ---------- arg-parse ----------
usage() {
  echo "Usage: $0 -e <experiment.env> [-p <sbatch_config.env>] [--dry-run]"
  exit 1
}

while [[ $# -gt 0 ]]; do
  case $1 in
    -e|--exp)     EXPERIMENT_CONFIG=$2; shift 2 ;;
    -p|--sbatch_config) SBATCH_CONFIG=$2; shift 2 ;;
    -g|--global)  GLOBAL=$2; shift 2 ;;
    -s|--script)  SCRIPT=$2; shift 2 ;;
    --dry-run)    DRYRUN=1; shift ;;
    -h|--help)    usage ;;
    *) echo "Unknown option $1"; usage ;;
  esac
done

[[ -z "${EXPERIMENT_CONFIG:-}" ]] && { echo "Error: -e <experiment.env> is required"; usage; }

# ---------- load files to derive paths ----------
set -a
source "$GLOBAL"
source "$EXPERIMENT_CONFIG"
source "$SBATCH_CONFIG"
set +a

export EXPERIMENT_CONFIG SBATCH_CONFIG

# print the content of sbatch config file
# echo "Using sbatch config: $SBATCH_CONFIG"
# cat "$SBATCH_CONFIG"




: "${EXP_NAME:?EXP_NAME missing in $EXPERIMENT_CONFIG}"

# ---------- make & export run-time paths ----------
# export EXPERIMENT_CONFIG        # so submit.sh can source it
# export SBATCH_CONFIG
export LOG_DIR="${BASE_RUN_DIR}/${EXP_NAME}/logs"
mkdir -p "$LOG_DIR"

# batch
SBATCH_OPTS="
  --job-name=$EXP_NAME
  --partition=$PARTITION
  --qos=$QOS
  --nodes=$NNODES
  --gres=gpu:$GPUS_PER_NODE
  --ntasks-per-node=$GPUS_PER_NODE
  --cpus-per-task=$CPUS_PER_TASK
  --mem=$MEM_PER_NODE
  --time=$WALL
  --output=$LOG_DIR/%x_%j.out
  --error=$LOG_DIR/%x_%j.err
"


echo "Experiment : $EXPERIMENT_CONFIG"
echo "Cluster    : $SBATCH_CONFIG"
echo "Log dir    : $LOG_DIR"
echo "Submit     : $SCRIPT"

CMD="sbatch --export=ALL $SCRIPT"

if (( DRYRUN )); then
#   echo "[dry-run] would run: $CMD"
    sbatch --test-only --export=ALL $SBATCH_OPTS "$SCRIPT"
else
    sbatch --export=ALL $SBATCH_OPTS "$SCRIPT"
fi