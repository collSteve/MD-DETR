#!/bin/bash

use_prompts=0
num_prompts=100
plen=10
train=0
exp_name="demo_agn"
split_point=0

repo_name="SenseTime/deformable-detr"
tr_dir="/scratch/ssd004/datasets/MSCOCO2017/train2017"
val_dir="/scratch/ssd004/datasets/MSCOCO2017/val2017"
# task_ann_dir="/ubc/cs/home/g/gbhatt/borg/cont_learn/data/mscoco/"${split_point}
task_ann_dir="/h/stevev/MD_DETR_runs/upload/mscoco/"${split_point}

freeze='backbone,encoder,decoder'
new_params='class_embed,prompts'

# EXP_DIR=/ubc/cs/home/g/gbhatt/borg/cont_learn/runs/${exp_name}
EXP_DIR=/h/stevev/MD_DETR_runs/${exp_name}

if [[ $train -gt 0 ]]
then
echo "Training ... "${exp_name}
LD_DIR=/ubc/cs/home/g/gbhatt/borg/cont_learn/runs/QL_NN_0.1_10
python main.py \
    --output_dir ${EXP_DIR} --train_img_dir ${tr_dir} --test_img_dir ${val_dir} --task_ann_dir ${task_ann_dir} --repo_name=${repo_name} \
    --n_gpus 8 --batch_size 1 --epochs 6 --lr 1e-4 --lr_old 1e-5 --n_classes=81 --num_workers=2 --split_point=$split_point \
    --use_prompts $use_prompts --num_prompts $num_prompts --prompt_len=$plen --freeze=${freeze} --viz --new_params=${new_params} \
    --start_task=1 --n_tasks=4 --save_epochs=1 --eval_epochs=1  --bg_thres=0.65 --bg_thres_topk=5 --local_query=1 --lambda_query=0.1 \
    --checkpoint_dir ${LD_DIR}'/Task_1' --checkpoint_base 'checkpoint10.pth' --checkpoint_next 'checkpoint05.pth' --resume=0
else
echo "Evaluating ..."
exp_name="outputs"
# EXP_DIR=/ubc/cs/home/g/gbhatt/borg/cont_learn/runs/${exp_name}
# LD_DIR=/ubc/cs/home/g/gbhatt/borg/cont_learn/runs/checkpoints

EXP_DIR=/h/stevev/MD_DETR_runs/${exp_name}
LD_DIR=/h/stevev/MD_DETR_runs/upload/checkpoints

python main.py \
    --output_dir ${EXP_DIR} --train_img_dir ${tr_dir} --test_img_dir ${val_dir} --task_ann_dir ${task_ann_dir} \
    --n_gpus 1 --batch_size 1 --epochs 16 --lr 1e-4 --lr_old 1e-5 --save_epochs=5 --eval_epochs=2 --n_classes=81 --num_workers=2 \
    --use_prompts $use_prompts --num_prompts $num_prompts --prompt_len=$plen --freeze=${freeze} --new_params=${new_params} \
    --checkpoint_dir ${LD_DIR}'/Task_1' --checkpoint_base 'checkpoint10.pth' --checkpoint_next 'checkpoint05.pth' --eval --viz \
    --start_task=1 --n_tasks=4  --local_query=1
fi

# #### Resume from a checkpoint for given task and then train
# python main.py \
#     --output_dir ${EXP_DIR} --train_img_dir ${tr_dir} --test_img_dir ${val_dir} --task_ann_dir ${task_ann_dir} \
#     --n_gpus 8 --batch_size 1 --epochs 26 --lr 1e-4 --lr_old 1e-5 --n_tasks=4 \
#     --use_prompts $use_prompts --num_prompts $num_prompts --prompt_len $plen --save_epochs=5 --eval_epochs=2 \
#     --freeze=${freeze} --viz --new_params=${new_params} --n_classes=81 --num_workers=2 \
#     --start_task=2 --checkpoint_dir ${LD_DIR}'/Task_1' --checkpoint 'checkpoint10.pth' --resume=1