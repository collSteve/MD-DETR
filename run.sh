#!/bin/bash

use_prompts=1
num_prompts=100
plen=10
train=0
exp_name="coco_40_PT"
split_point=40
# repo_name="facebook/deformable-detr-detic"
repo_name="SenseTime/deformable-detr"
tr_dir="/ubc/cs/research/shield/datasets/MSCOCO/2017/train2017"
val_dir="/ubc/cs/research/shield/datasets/MSCOCO/2017/val2017"
task_ann_dir="/ubc/cs/home/g/gbhatt/borg/cont_learn/data/mscoco/"${split_point}
# big_pretrained="SenseTime/deformable-detr"
# freeze='backbone,encoder,decoder,bbox_embed,reference_points,input_proj,level_embed'
freeze='backbone,encoder,decoder'
new_params='class_embed,prompts'

EXP_DIR=/ubc/cs/home/g/gbhatt/borg/cont_learn/runs/${exp_name}

if [[ $train -gt 0 ]]
then
echo "Training ..."
LD_DIR=/ubc/cs/home/g/gbhatt/borg/cont_learn/runs/PT_new
python main.py \
    --output_dir ${EXP_DIR} --train_img_dir ${tr_dir} --test_img_dir ${val_dir} --task_ann_dir ${task_ann_dir} \
    --n_gpus 8 --batch_size 1 --epochs 11 --lr 1e-4 --lr_old 1e-5 --n_classes=81 --num_workers=2 \
    --use_prompts $use_prompts --num_prompts $num_prompts --prompt_len=$plen --freeze=${freeze} --viz --new_params=${new_params} \
    --repo_name=${repo_name} --start_task=2  --n_tasks=2 --save_epochs=1 --eval_epochs=1 --split_point=$split_point \
    --checkpoint_dir ${LD_DIR}'/Task_1' --checkpoint_base 'checkpoint10.pth' --checkpoint_next 'checkpoint10.pth' --resume=0
else
echo "Evaluating ..."
exp_name="eval_PT_40"
EXP_DIR=/ubc/cs/home/g/gbhatt/borg/cont_learn/runs/${exp_name}
LD_DIR=/ubc/cs/home/g/gbhatt/borg/cont_learn/runs/coco_40_PT

python main.py \
    --output_dir ${EXP_DIR} --train_img_dir ${tr_dir} --test_img_dir ${val_dir} --task_ann_dir ${task_ann_dir} \
    --n_gpus 8 --batch_size 1 --epochs 16 --lr 1e-4 --lr_old 1e-5 --save_epochs=5 --eval_epochs=2 --n_classes=81 --num_workers=2 \
    --use_prompts $use_prompts --num_prompts $num_prompts --prompt_len=$plen --freeze=${freeze} --new_params=${new_params} \
    --checkpoint_dir ${LD_DIR}'/Task_1' --checkpoint_base 'checkpoint10.pth' --checkpoint_next 'checkpoint10.pth' --eval --viz \
    --start_task=2 --n_tasks=2 
fi

# #### Resume from a checlpoint for given task and then train
# python main.py \
#     --output_dir ${EXP_DIR} --train_img_dir ${tr_dir} --test_img_dir ${val_dir} --task_ann_dir ${task_ann_dir} \
#     --n_gpus 8 --batch_size 1 --epochs 26 --lr 1e-4 --lr_old 1e-5 --n_tasks=4 \
#     --use_prompts $use_prompts --num_prompts $num_prompts --prompt_len $plen --save_epochs=5 --eval_epochs=2 \
#     --freeze=${freeze} --viz --new_params=${new_params} --n_classes=81 --num_workers=2 \
#     --start_task=2 --checkpoint_dir ${LD_DIR}'/Task_1' --checkpoint 'checkpoint10.pth' --resume=1