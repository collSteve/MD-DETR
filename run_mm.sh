#!/usr/bin/env bash
set -euo pipefail

# read experiement config
source config/global.env 
EXPERIMENT_CONFIG=${EXPERIMENT_CONFIG:?export EXPERIMENT_CONFIG=config/experiement/*.env}
source "$EXPERIMENT_CONFIG"

# ---------- 2. derived paths ----------
EXP_DIR="${BASE_RUN_DIR}/${EXP_NAME}"


TR_DIR="${TRAIN_DIR}"
VAL_DIR="${VAL_DIR}"
TASK_ANN_DIR="${TASK_ANN_ROOT}/${SPLIT_POINT}"

# ---------- 3. world size ----------
NNODES=${SLURM_JOB_NUM_NODES:-1}
GPUS_PER_NODE=${SLURM_NTASKS_PER_NODE:-1}
WORLD_SIZE=$((NNODES * GPUS_PER_NODE))

freeze='backbone,encoder,decoder'
new_params='class_embed,prompts'

echo ">>> ${EXP_NAME}: ${WORLD_SIZE} GPUs (${NNODES}×${GPUS_PER_NODE})"


VIZ=${VIZ:---viz} 
NEW_PARAMS=${NEW_PARAMS:-class_embed,prompts}
FREEZE=${FREEZE:-backbone,encoder,decoder}
REPO_NAME=${REPO_NAME:-SenseTime/deformable-detr}

# ---------- 4. build arg lists ----------
COMMON="
  --output_dir $EXP_DIR
  --train_img_dir $TR_DIR 
  --test_img_dir $VAL_DIR 
  --task_ann_dir $TASK_ANN_DIR
  --repo_name $REPO_NAME
  --n_gpus $WORLD_SIZE
  --batch_size $BATCH_SIZE 
  --lr 1e-4 --lr_old 1e-5 --n_classes 81 --num_workers 2
  --split_point $SPLIT_POINT
  --use_prompts $USE_PROMPTS --num_prompts $NUM_PROMPTS --prompt_len $PLEN
  --freeze $FREEZE
  $VIZ
  --new_params $NEW_PARAMS
  --start_task 1 
  --n_tasks $N_TASKS 
"

if (( TRAIN )); then
  MODE="
    --epochs $EPOCHS
    --save_epochs $SAVE_EPOCHS   
    --eval_epochs $EVAL_EPOCHS
    --bg_thres $BG_THRES      
    --bg_thres_topk $BG_THRES_TOPK
    --local_query 1               
     --lambda_query $LAMBDA_QUERY
    --checkpoint_dir $CHECKPOINT_DIR        
    --checkpoint_base $CHECKPOINT_BASE
    --checkpoint_next $CHECKPOINT_NEXT
    --resume 0
  "
else
  MODE="
    --eval
    --epochs $EPOCHS
    --save_epochs $SAVE_EPOCHS                  
    --eval_epochs $EVAL_EPOCHS
    --checkpoint_dir $CHECKPOINT_DIR        
    --checkpoint_base $CHECKPOINT_BASE
    --checkpoint_next $CHECKPOINT_NEXT
    --local_query 1
  "
fi


# torchrun \
#   --nnodes $NNODES --nproc_per_node $GPUS_PER_NODE \
#   main.py \
#   $COMMON $MODE
python main.py \
  $COMMON $MODE