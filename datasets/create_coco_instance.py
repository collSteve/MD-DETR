#from coco_hug import create_task_json, task_info_coco
import argparse
from pathlib import Path
import json
import os
import numpy as np

def task_info_coco(split_point=0): # 0, 40, 70

    T1_CLASS_NAMES = [
    "airplane","bicycle","bird","boat","bus","car",
    "cat","cow","dog","horse","motorcycle","sheep","train",
    "elephant","bear","zebra","giraffe","truck","person"
    ] # n_classes = 19

    T2_CLASS_NAMES = [
        "traffic light","fire hydrant","stop sign",
        "parking meter","bench","chair","dining table",
        "potted plant","backpack","umbrella","handbag",
        "tie","suitcase","microwave","oven","toaster","sink",
        "refrigerator","bed","toilet","couch"
    ] # n_classes = 21

    T3_CLASS_NAMES = [
        "frisbee","skis","snowboard","sports ball",
        "kite","baseball bat","baseball glove","skateboard",
        "surfboard","tennis racket","banana","apple","sandwich",
        "orange","broccoli","carrot","hot dog","pizza","donut","cake"
    ] # n_classes = 20

    T4_CLASS_NAMES = [
        "laptop","mouse","remote","keyboard","cell phone","book",
        "clock","vase","scissors","teddy bear","hair drier","toothbrush",
        "wine glass","cup","fork","knife","spoon","bowl","tv","bottle"
    ] # n_classes = 20

    all_classes = T1_CLASS_NAMES+T2_CLASS_NAMES+T3_CLASS_NAMES+T4_CLASS_NAMES

    if split_point == 0:
        task_map = {1:(T1_CLASS_NAMES,0,19), 2:(T2_CLASS_NAMES,19,21), 
                3:(T3_CLASS_NAMES,19+21,20), 4:(T4_CLASS_NAMES,19+21+20,20)}    
    else:
        #np.random.shuffle(all_classes) 
        task_map = {1:(all_classes[0:split_point],0,split_point),
                    2:(all_classes[split_point:],split_point,80-split_point)}

    task_label2name = {}
    for i,j in enumerate(all_classes):
        task_label2name[i] = j

    return task_map, task_label2name

def create_task_json(root_json, cat_names, set_type='train', offset=0, task_id=1, output_dir='', task_label2name=None):

    print ('Creating temp JSON for tasks ',task_id,' ...', set_type)
    temp_json = json.load(open(root_json, 'r'))

    id2id, name2id, flag = {}, {}, False

    if task_label2name:
        flag = True
        for i,j in task_label2name.items():
            name2id[j] = i

    cat_ids, keep_imgs = [], []
    for k,j in enumerate(temp_json['categories']):

        if not flag:
            name2id[j['name']] = j['id']

        if j['name'] in cat_names:
            cat_ids.append(j['id'])
    
    for j,i in enumerate(cat_names):
        id2id[name2id[i]] = offset+j
    
    data = {'images':[], 'annotations':[], 'categories':[], 
            'info':{},'licenses':[]}

    # count = 0
    #print ('total ', len(temp_json['annotations']))
    for i in temp_json['annotations']:
        # if count %100 ==0:
        #     print (count)
        # count+=1
        if i['category_id'] in cat_ids:
            temp = i
            #print (i, temp)
            temp['category_id'] = id2id[temp['category_id']]
            data['annotations'].append(temp)
            keep_imgs.append(i['image_id'])
    
    #print ('here')
            
    keep_imgs = set(keep_imgs)

    for i in temp_json['categories']:
        if i['id'] in cat_ids:
            temp = i
            temp['id'] = id2id[temp['id']]
            data['categories'].append(temp)
            #data['categories'].append(i)
    
    data['info'] = temp_json['info']
    data['licenses'] = temp_json['licenses']

    #count = 0
    print ('total images:', len(temp_json['images']), '  keeping:', len(keep_imgs))
    for i in temp_json['images']:
        if i['id'] in keep_imgs:
            data['images'].append(i)

    #print ('\n dumping \n')
    # with open(output_dir+'/temp_'+set_type+'.json', 'w') as f:
    #     json.dump(data, f)
    
    with open(os.path.join(output_dir,set_type+'_task_'+str(task_id)+'.json'),'w') as f:
        json.dump(data, f)
    

def get_args_parser():
    parser = argparse.ArgumentParser('Deformable DETR Detector', add_help=False)
    parser.add_argument('--n_tasks', default=4, type=int)
    parser.add_argument('--train_ann', default="/ubc/cs/research/shield/datasets/MSCOCO/2017/annotations/instances_train2017.json", type=str)
    parser.add_argument('--test_ann', default="/ubc/cs/research/shield/datasets/MSCOCO/2017/annotations/instances_val2017.json", type=str)
    parser.add_argument('--output_dir', default="/ubc/cs/home/g/gbhatt/borg/cont_learn/data/mscoco", type=str)
    parser.add_argument('--split_point',default=0, type=int)
    
    return parser


def main(args):

    args.output_dir = os.path.join(args.output_dir,str(args.split_point))

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    task_map, task_label2name =  task_info_coco(split_point=args.split_point)

    for task_id in range(1,args.n_tasks+1):
        cur_task = task_map[task_id]

        create_task_json(root_json=args.train_ann,
                        cat_names=cur_task[0], offset=cur_task[1], set_type='train', output_dir=args.output_dir, task_id=task_id)

        create_task_json(root_json=args.test_ann,
                        cat_names=cur_task[0], offset=cur_task[1], set_type='test', output_dir=args.output_dir, task_id=task_id)

    if args.split_point>0:
        create_task_json(root_json=args.test_ann,
                cat_names=task_map[1][0]+task_map[2][0], offset=0, set_type='test', output_dir=args.output_dir, task_id='12')


    for task_id in range(3,args.n_tasks+1):
        cur_task = task_map[task_id]
        known_task_ids = ''.join(str(i) for i in range(1,task_id))
        all_cat_names = []

        for i in range(1,task_id):
            all_cat_names.extend(task_map[i][0])
        
        create_task_json(root_json=args.test_ann,
                        cat_names=all_cat_names, offset=0, set_type='test', output_dir=args.output_dir, task_id=known_task_ids)
    
    known_task_ids = ''.join(str(i) for i in range(1,task_id+1))
    all_cat_names = []

    for i in range(1,task_id+1):
        all_cat_names.extend(task_map[i][0])
    
    create_task_json(root_json=args.test_ann,
                    cat_names=all_cat_names, offset=0, set_type='test', output_dir=args.output_dir, task_id=known_task_ids)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)

    # task_map, task_label2name =  task_info_coco(split_point=args.split_point)

    # cur_task = task_map[4]
    # known_task_ids = ''.join(str(i) for i in range(1,5))
    # all_cat_names = []

    # for i in range(1,5):
    #     all_cat_names.extend(task_map[i][0])
    
    # create_task_json(root_json=args.test_ann,
    #                 cat_names=all_cat_names, offset=0, set_type='test', output_dir=args.output_dir, task_id=known_task_ids)


    print ('\n Done .... \n')