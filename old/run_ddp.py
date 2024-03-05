import os
import tqdm
import torch
import json
import numpy as np
import torchvision
import pdb
import copy
import argparse
import utils
from pathlib import Path

import torch.distributed as dist
#import pytorch_lightning as pl
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
import datasets.samplers as samplers
from datasets.coco_eval import CocoEvaluator
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers.models.deformable_detr.feature_extraction_deformable_detr import DeformableDetrFeatureExtractor 
from transformers import AutoImageProcessor
from transformers.models.deformable_detr.configuration_deformable_detr import DeformableDetrConfig
from transformers.models.deformable_detr.modeling_deformable_detr import DeformableDetrForObjectDetection

T1_CLASS_NAMES = [
    "airplane","bicycle","bird","boat","bus","car",
    "cat","cow","dog","horse","motorcycle","sheep","train",
    "elephant","bear","zebra","giraffe","truck","person"
] # n_classes = 19

T2_CLASS_NAMES = [
    "traffic light","fire hydrant","stop sign",
    "parking meter","bench","chair","diningtable",
    "pottedplant","backpack","umbrella","handbag",
    "tie","suitcase","microwave","oven","toaster","sink",
    "refrigerator","bed","toilet","sofa"
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
    "wine glass","cup","fork","knife","spoon","bowl","tvmonitor","bottle"
] # n_classes = 20

task_map = {1:(T1_CLASS_NAMES,0), 2:(T2_CLASS_NAMES,19), 
            3:(T3_CLASS_NAMES,19+21), 4:(T4_CLASS_NAMES,19+21+20)}

task_label2name = {}
for i,j in enumerate(T1_CLASS_NAMES+T2_CLASS_NAMES+T3_CLASS_NAMES+T4_CLASS_NAMES):
    task_label2name[i] = j

class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, ann_file, processor):
        super(CocoDetection, self).__init__(img_folder, ann_file)
        self.processor = processor

    def __getitem__(self, idx):
        # read in PIL image and target in COCO format
        img, target = super(CocoDetection, self).__getitem__(idx)
        image_id = self.ids[idx]

        target = {'image_id': image_id, 'annotations': target}
        
        # preprocess image and target (converting target to DETR format, resizing + normalization of both image and target)
        encoding = self.processor(images=img, annotations=target, return_tensors="pt")
        pixel_values = encoding["pixel_values"].squeeze() # remove batch dimension
        target = encoding["labels"][0] # remove batch dimension

        return pixel_values, target

repo_name = "SenseTime/deformable-detr" 
#processor = DeformableDetrFeatureExtractor()
processor  = AutoImageProcessor.from_pretrained(repo_name)

# colors for visualization
COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

def collate_fn(batch):
  pixel_values = [item[0] for item in batch]
  #print(pixel_values)
  encoding = processor.pad(pixel_values, return_tensors="pt")
  labels = [item[1] for item in batch]
  batch = {}
  batch['pixel_values'] = encoding['pixel_values']
  batch['pixel_mask'] = encoding['pixel_mask']
  batch['labels'] = labels
  return batch

def plot_results(pil_img, scores, labels, boxes, f_name='out.jpg'):
    plt.figure(figsize=(16,10))
    plt.imshow(pil_img)
    ax = plt.gca()
    colors = COLORS * 100
    for score, label, (xmin, ymin, xmax, ymax),c  in zip(scores.tolist(), labels.tolist(), boxes.tolist(), colors):
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                   fill=False, color=c, linewidth=3))
        text = f'{task_label2name[label]}: {score:0.2f}'
        ax.text(xmin, ymin, text, fontsize=15,
                bbox=dict(facecolor='yellow', alpha=0.5))
    plt.axis('off')
    #plt.show()
    plt.savefig(f_name)

def viz_gt(test_dataset, save_path, id=None, data_root='/ubc/cs/research/shield/datasets/MSCOCO/2017/val2017'):
    # based on https://github.com/woctezuma/finetune-detr/blob/master/finetune_detr.ipynb
    image_ids = test_dataset.coco.getImgIds()
    # let's pick a random image
    #pdb.set_trace()

    #[579321, 170893, 65485, 42276, 109055]
    if not id:
       image_id = image_ids[np.random.randint(0, len(image_ids))]
    else:
        image_id = id
    print('Image n°{}'.format(image_id))
    image = test_dataset.coco.loadImgs(image_id)[0]
    image = Image.open(os.path.join(data_root, image['file_name']))

    annotations = test_dataset.coco.imgToAnns[image_id]
    draw = ImageDraw.Draw(image, "RGBA")

    cats = test_dataset.coco.cats
    id2label = {k: v['name'] for k,v in cats.items()}
    scores, labels, boxes = [],[],[]
    for annotation in annotations:
        box = annotation['bbox']
        class_idx = annotation['category_id']
        x,y,w,h = tuple(box)
        #draw.rectangle((x,y,x+w,y+h), outline='red', width=1)
        #draw.text((x, y), id2label[class_idx], fill='blue')
        scores.append(1.0)
        labels.append(class_idx)
        boxes.append((x,y,x+w,y+h))

    save = os.path.join(save_path, 'GT_'+str(image_id)+'.jpg')

    plot_results(image, np.array(scores), np.array(labels), np.array(boxes), f_name=save)
    #image.save('out_gt.jpg')

def viz_mod(model,test_dataset, save_path, id=None, data_root='/ubc/cs/research/shield/datasets/MSCOCO/2017/val2017', device='cuda'):
    # model.to('cpu')
    model.eval()
    # based on https://github.com/woctezuma/finetune-detr/blob/master/finetune_detr.ipynb
    image_ids = test_dataset.coco.getImgIds()
    # let's pick a random image
    #pdb.set_trace()

    #[579321, 170893, 65485, 42276, 109055]
    if not id:
       image_id = image_ids[np.random.randint(0, len(image_ids))]
    else:
        image_id = id
    print('Image n°{}'.format(image_id))
    image = test_dataset.coco.loadImgs(image_id)[0]
    image = Image.open(os.path.join(data_root, image['file_name']))

    inputs = processor(images=image, return_tensors="pt")
    #pixel_values = inputs["pixel_values"].squeeze()
    inputs['pixel_values'] = inputs['pixel_values'].to(device)
    inputs['pixel_mask'] = inputs['pixel_mask'].to(device)

    with torch.no_grad():
        outputs = model(**inputs)
    # let's only keep predictions with score > 0.3
    results = processor.post_process_object_detection(outputs,target_sizes=[image.size[::-1]],
                                                        threshold=0.3)[0]
    save = os.path.join(save_path, 'mod_'+str(image_id)+'.jpg')
    plot_results(image, results['scores'], results['labels'], results['boxes'], f_name=save)

def convert_to_xywh(boxes):
    xmin, ymin, xmax, ymax = boxes.unbind(1)
    return torch.stack((xmin, ymin, xmax - xmin, ymax - ymin), dim=1)

def prepare_for_coco_detection(predictions):
    coco_results = []
    for original_id, prediction in predictions.items():
        if len(prediction) == 0:
            continue

        boxes = prediction["boxes"]
        boxes = convert_to_xywh(boxes).tolist()
        scores = prediction["scores"].tolist()
        labels = prediction["labels"].tolist()

        coco_results.extend(
            [
                {
                    "image_id": original_id,
                    "category_id": labels[k],
                    "bbox": box,
                    "score": scores[k],
                }
                for k, box in enumerate(boxes)
            ]
        )
    return coco_results

def eval(model, test_dataloader, coco_evaluator):
    print("Running evaluation...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    nbatches = 1000

    for idx, batch in enumerate(tqdm.tqdm(test_dataloader)):
        
        if idx > nbatches:
            break

        # get the inputs
        #print(len(batch['labels'][0]['boxes']))
        pixel_values = batch["pixel_values"].to(device)
        pixel_mask = batch["pixel_mask"].to(device)
        labels = [{k: v.to(device) for k, v in t.items()} for t in batch["labels"]] # these are in DETR format, resized + normalized

        # forward pass
        with torch.no_grad():
            outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask)

        orig_target_sizes = torch.stack([target["orig_size"] for target in labels], dim=0)
        results = processor.post_process(outputs, orig_target_sizes) # convert outputs to COCO api
        res = {target['image_id'].item(): output for target, output in zip(labels, results)}
        res = prepare_for_coco_detection(res)
        #pdb.set_trace()
        #print (res)
        coco_evaluator.update(res)

    coco_evaluator.synchronize_between_processes()
    coco_evaluator.accumulate()
    coco_evaluator.summarize()

def create_task_json(root_json, cat_names, set_type='train', offset=0, task_id=1, output_dir=''):

    print ('Creating temp JSON for tasks ',task_id,' ...', set_type)
    temp_json = json.load(open(root_json, 'r'))

    name2id, id2id = {},{}
    cat_ids, keep_imgs = [], []
    for k,j in enumerate(temp_json['categories']):
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
    with open(output_dir+'/temp_'+set_type+'.json', 'w') as f:
        json.dump(data, f)

def setup(rank, world_size, port='9987'):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = port
    os.environ["NCCL_DEBUG"] = "INFO"

    dist.init_process_group("nccl", rank=rank, world_size=world_size)
        
def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print

class Detr():
     def __init__(self, lr, lr_backbone, weight_decay, train_loader, val_loader, args, device='cuda'):
         super().__init__()
         # replace COCO classification head with custom head
         # we specify the "no_timm" variant here to not rely on the timm library
         # for the convolutional backbone

         detr_config = DeformableDetrConfig()
         detr_config.num_labels = args.n_classes # + 1
         self.model =  DeformableDetrForObjectDetection.from_pretrained('SenseTime/deformable-detr',
                                                                        config=detr_config,
                                                                        ignore_mismatched_sizes=True)

         # see https://github.com/PyTorchLightning/pytorch-lightning/pull/1896
         self.lr = lr
         self.lr_backbone = lr_backbone
         self.weight_decay = weight_decay
         self.train_loader = train_loader
         self.val_loader = val_loader
         self.device = device

     def forward(self, pixel_values, pixel_mask):
       outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask)

       return outputs

     def common_step(self, batch):
       pixel_values = batch["pixel_values"].to(self.device)
       pixel_mask = batch["pixel_mask"].to(self.device)
       labels = [{k: v.to(self.device) for k, v in t.items()} for t in batch["labels"]]

       outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)

       loss = outputs.loss
       loss_dict = outputs.loss_dict

       return loss, loss_dict

     def training_step(self, batch):
        loss, loss_dict = self.common_step(batch)
        # logs metrics for each training_step,
        # and the average across the epoch
        #self.log("training_loss", loss)
        # for k,v in loss_dict.items():
        #   self.log("train_" + k, v.item())

        return loss, loss_dict

     def validation_step(self, batch):
        loss, loss_dict = self.common_step(batch)
        self.log("validation_loss", loss)
        for k,v in loss_dict.items():
          self.log("validation_" + k, v.item())

        return loss
     
     def save(self, epoch, args, model_without_ddp, optimizer):
         torch.save({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    #'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, os.path.join(args.output_dir, f'checkpoint{epoch:02}.pth'))
         
     def load(self, model, args):
        checkpoint = torch.load(args.load_path, map_location='cpu')
        state_dict = checkpoint['model']
        msg = model.load_state_dict(state_dict, strict=False)
        print(msg)
        args.start_epoch = checkpoint['epoch'] + 1
        optimizer = checkpoint['optimizer']

        return model, optimizer

     def configure_optimizers(self):
        param_dicts = [
              {"params": [p for n, p in self.model.named_parameters() if "backbone" not in n and p.requires_grad]},
              {
                  "params": [p for n, p in self.model.named_parameters() if "backbone" in n and p.requires_grad],
                  "lr": self.lr_backbone,
              },
        ]
        optimizer = torch.optim.AdamW(param_dicts, lr=self.lr,
                                  weight_decay=self.weight_decay)

        return optimizer

     def train_dataloader(self):
        return self.train_dataloader

     def val_dataloader(self):
        return self.val_dataloader

def get_args_parser():
    parser = argparse.ArgumentParser('Deformable DETR Detector', add_help=False)
    parser.add_argument('--lr', default=2e-4, type=float)
    parser.add_argument('--new_params', default="", type=str)
    parser.add_argument('--freeze', default="", type=str)
    parser.add_argument('--lr_old', default=2e-5, type=float)
    parser.add_argument('--lr_backbone', default=0, type=float)
    parser.add_argument('--lr_linear_proj_names', default=['reference_points', 'sampling_offsets'], type=str, nargs='+')
    parser.add_argument('--lr_linear_proj_mult', default=0.1, type=float)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--print_freq', default=5, type=int)
    parser.add_argument('--print_class_names', default=0, type=int)
    parser.add_argument('--print_class_ap', default=0, type=int)
    parser.add_argument('--lr_drop', default=40, type=int)
    parser.add_argument('--lr_drop_epochs', default=None, type=int, nargs='+')
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')
    parser.add_argument('--sgd', action='store_true')
    parser.add_argument('--n_gpus', default=1, type=int,
                        help="Number of GPUs available")
    
    parser.add_argument('--n_classes', default=3, type=int) # bal:1, robo:5, COCO:3

    parser.add_argument('--eval', default=1, type=int,
                        help="evaluate")
    parser.add_argument('--num_workers', default=2, type=int,
                        help="evaluate")
    parser.add_argument('--viz', default=1, type=int,
                        help="evaluate")
    parser.add_argument('--output_dir', default="/ubc/cs/home/g/gbhatt/borg/cont_learn/runs/hug2", type=str)
    parser.add_argument('--load_path', default="", type=str)
    
    return parser

def main(rank, world_size, args):

    if args.n_gpus>1:
        device = torch.device('cuda:{}'.format(rank))

        #with socketserver.TCPServer(("localhost", 0), None) as s:
        #    free_port = s.server_address[1]
        #print (free_port, type(free_port))
        setup(rank, world_size, str(8888))
        setup_for_distributed(rank == 0)
        torch.cuda.set_device(rank)
        #dist.barrier()
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.cuda.set_device(rank)


    iou_types = ['bbox']
    keep = ['airplane','bicycle','bird']
    # 2, 17, 72
    #keep = T1_CLASS_NAMES

    cur_task = task_map[1]
    cur_task = [keep,0]

    # create_task_json(root_json='/ubc/cs/research/shield/datasets/MSCOCO/2017/annotations/instances_train2017.json',
    #                 cat_names=cur_task[0], offset=cur_task[1], set_type='train', output_dir=args.output_dir)

    # create_task_json(root_json='/ubc/cs/research/shield/datasets/MSCOCO/2017/annotations/instances_val2017.json',
    #                 cat_names=cur_task[0], offset=cur_task[1], set_type='val', output_dir=args.output_dir)

    # train_dataset = CocoDetection(img_folder='/ubc/cs/research/shield/datasets/MSCOCO/2017/train2017', 
    #                             #ann_file=args.output_dir+'/temp_train.json',
    #                             ann_file='temp_train.json',
    #                             processor=processor)

    # test_dataset = CocoDetection(img_folder='/ubc/cs/research/shield/datasets/MSCOCO/2017/val2017', 
    #                             #ann_file=args.output_dir+'/temp_val.json',
    #                             ann_file='temp_val.json',
    #                             processor=processor)

    # im_tr, ann_tr = 'data/robo/train', 'data/robo/train/_annotations.coco.json'
    # im_val, ann_val = 'data/robo/valid', 'data/robo/valid/_annotations.coco.json'

    im_tr, ann_tr = '/ubc/cs/research/shield/datasets/MSCOCO/2017/train2017', 'temp_train.json'
    im_val, ann_val = '/ubc/cs/research/shield/datasets/MSCOCO/2017/val2017', 'temp_val.json'

    train_dataset = CocoDetection(img_folder=im_tr, 
                                #ann_file=args.output_dir+'/temp_train.json',
                                ann_file=ann_tr,
                                processor=processor)

    test_dataset = CocoDetection(img_folder=im_val, 
                                #ann_file=args.output_dir+'/temp_val.json',
                                ann_file=ann_val,
                                processor=processor)
    
    categories = train_dataset.coco.cats
    task_label2name = {k: v['name'] for k,v in categories.items()}

    if args.n_gpus>1:
        sampler_train = samplers.DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
        sampler_val = samplers.DistributedSampler(test_dataset, shuffle=False, num_replicas=world_size, rank=rank)
    else:
        sampler_train = torch.utils.data.RandomSampler(train_dataset)
        sampler_val = torch.utils.data.SequentialSampler(test_dataset)
        #args.num_workers = 0

    train_dataloader = DataLoader(train_dataset, collate_fn=collate_fn, batch_size=args.batch_size,
                                  num_workers=args.num_workers,
                                   pin_memory=True, drop_last=True, sampler=sampler_train)
    test_dataloader = DataLoader(test_dataset, collate_fn=collate_fn, batch_size=args.batch_size,
                                 num_workers=args.num_workers,
                                   pin_memory=True, drop_last=True, sampler=sampler_val)
    

    #model = DeformableDetrForObjectDetection.from_pretrained('SenseTime/deformable-detr')

    #pdb.set_trace()

    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #model.to(device)

    # id = 42276
    # viz_gt(id=id)
    # viz_mod(model=model, id=id)

    detr = Detr(lr=1e-4, lr_backbone=1e-5, weight_decay=1e-4, train_loader=train_dataloader, 
                args=args, val_loader=test_dataloader, device=device)
    model = detr.model

    if args.n_gpus>1:
        if rank == 0:
            print("\n multi GPUs using DDP\n")
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = model.to(device)
        model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=True)
        model_without_ddp = model.module
    else:
        model = model.cuda()
        model_without_ddp = model

    optimizer = detr.configure_optimizers()
    #coco_evaluator = CocoEvaluator(test_dataset.coco, iou_types)

    for epoch in range(args.epochs):
        
        metric_logger = utils.MetricLogger(delimiter="  ")
        header = 'Epoch: [{}]'.format(epoch)
        
        if args.n_gpus>1:
            sampler_train.set_epoch(epoch)
        count = 0
        #for id, batch in tqdm.tqdm(enumerate(train_dataloader)):
        #for batch in tqdm.tqdm(train_dataloader):
        for batch in metric_logger.log_every(train_dataloader, args.print_freq, header):
            #batch = next(iter(train_dataloader))

            losses, loss_dict = detr.training_step(batch)
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            # if count>10:
            #     break
            # count+=1

            losses_sum = sum(loss_dict.values())
            metric_logger.update(loss=losses_sum, **loss_dict)
        
        metric_logger.synchronize_between_processes()
        print("Averaged stats:", metric_logger)
        #return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
        
        if epoch%5==0:
            coco_evaluator = CocoEvaluator(test_dataset.coco, iou_types)
            #test_dataloader = DataLoader(test_dataset, collate_fn=collate_fn, batch_size=2)
            #batch = next(iter(test_dataloader))
            eval(detr.model, test_dataloader, coco_evaluator=coco_evaluator)
            
            image_ids = test_dataset.coco.getImgIds()

            if args.viz:
                for id in image_ids[0:4]:
                    viz_gt(test_dataset=test_dataset,save_path=args.output_dir, id=id, data_root=im_val)
                    viz_mod(model=model_without_ddp, test_dataset=test_dataset,
                            save_path=args.output_dir, id=id, data_root=im_val)
        
    detr.save(epoch=epoch, model_without_ddp=model_without_ddp, optimizer=optimizer, args=args)
    #model = DeformableDetrForObjectDetection.from_pretrained('SenseTime/deformable-detr')
    # model.to("cuda")

    #batch = next(iter(test_dataloader))
    #pdb.set_trace()

    if args.eval:
        coco_evaluator = CocoEvaluator(test_dataset.coco, iou_types)
        #test_dataloader = DataLoader(test_dataset, collate_fn=collate_fn, batch_size=2)
        #batch = next(iter(test_dataloader))
        eval(detr.model, test_dataloader, coco_evaluator=coco_evaluator)
        
        image_ids = test_dataset.coco.getImgIds()

        if args.viz:
            for id in image_ids[0:4]:
                viz_gt(test_dataset=test_dataset,save_path=args.output_dir, id=id, data_root=im_val)
                viz_mod(model=model_without_ddp, test_dataset=test_dataset,
                        save_path=args.output_dir, id=id, data_root=im_val)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Deformable DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    if args.n_gpus>1:
        mp.spawn(
            main,
            args=(args.n_gpus, args, ),
            nprocs=args.n_gpus,
            daemon=False,
            join=True,
        )
    else:
        main(0, 1, args)
    print ('\n Done .... \n')