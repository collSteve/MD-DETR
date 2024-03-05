import os
import tqdm
import torch
import json
import numpy as np
import torchvision
import pdb
import copy

import torch.distributed as dist
#import pytorch_lightning as pl
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from datasets.coco_eval import CocoEvaluator
from transformers.models.deformable_detr.feature_extraction_deformable_detr import DeformableDetrFeatureExtractor 
from transformers import AutoImageProcessor
from transformers.models.deformable_detr.configuration_deformable_detr import DeformableDetrConfig
from transformers.models.deformable_detr.modeling_deformable_detr import DeformableDetrForObjectDetection

evaluate = 0

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
        text = f'{model.config.id2label[label]}: {score:0.2f}'
        ax.text(xmin, ymin, text, fontsize=15,
                bbox=dict(facecolor='yellow', alpha=0.5))
    plt.axis('off')
    #plt.show()
    plt.savefig(f_name)

def viz_gt(id=None, data_root='/ubc/cs/research/shield/datasets/MSCOCO/2017/val2017'):
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

    plot_results(image, np.array(scores), np.array(labels), np.array(boxes), f_name='out_gt.jpg')
    #image.save('out_gt.jpg')

def viz_mod(model, id=None, data_root='/ubc/cs/research/shield/datasets/MSCOCO/2017/val2017'):
    model.to('cpu')
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
    with torch.no_grad():
        outputs = model(**inputs)
    # let's only keep predictions with score > 0.3
    results = processor.post_process_object_detection(outputs,target_sizes=[image.size[::-1]],
                                                        threshold=0.3)[0]

    plot_results(image, results['scores'], results['labels'], results['boxes'], f_name='out_mod.jpg')

def eval(model, test_dataloader):
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
        outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask)

        orig_target_sizes = torch.stack([target["orig_size"] for target in labels], dim=0)
        results = processor.post_process(outputs, orig_target_sizes) # convert outputs to COCO api
        res = {target['image_id'].item(): output for target, output in zip(labels, results)}
        #pdb.set_trace()
        #print (res)
        coco_evaluator.update(res)

    coco_evaluator.synchronize_between_processes()
    coco_evaluator.accumulate()
    coco_evaluator.summarize()

def create_task_json(root_json, cat_names, set_type='train', offset=0, task_id=1):

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
    with open('temp_'+set_type+'.json', 'w') as f:
        json.dump(data, f)

iou_types = ['bbox']
keep = ['airplane','bicycle','bird']
# 2, 17, 72
#keep = T1_CLASS_NAMES

cur_task = task_map[1]
cur_task = [keep,0]

create_task_json(root_json='/ubc/cs/research/shield/datasets/MSCOCO/2017/annotations/instances_train2017.json',
                 cat_names=cur_task[0], offset=cur_task[1], set_type='train')

create_task_json(root_json='/ubc/cs/research/shield/datasets/MSCOCO/2017/annotations/instances_val2017.json',
                 cat_names=cur_task[0], offset=cur_task[1], set_type='val')

train_dataset = CocoDetection(img_folder='/ubc/cs/research/shield/datasets/MSCOCO/2017/train2017', 
                              ann_file='temp_train.json',
                              processor=processor)

test_dataset = CocoDetection(img_folder='/ubc/cs/research/shield/datasets/MSCOCO/2017/val2017', 
                              ann_file='temp_val.json',
                              processor=processor)

train_dataloader = DataLoader(train_dataset, collate_fn=collate_fn, batch_size=2)
test_dataloader = DataLoader(test_dataset, collate_fn=collate_fn, batch_size=2)

#model = DeformableDetrForObjectDetection.from_pretrained('SenseTime/deformable-detr')

#pdb.set_trace()

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#model.to(device)

# id = 42276
# viz_gt(id=id)
# viz_mod(model=model, id=id)

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
     def __init__(self, lr, lr_backbone, weight_decay, train_loader, val_loader):
         super().__init__()
         # replace COCO classification head with custom head
         # we specify the "no_timm" variant here to not rely on the timm library
         # for the convolutional backbone

         detr_config = DeformableDetrConfig()
         detr_config.num_labels = 3 # + 1
         self.model =  DeformableDetrForObjectDetection.from_pretrained('SenseTime/deformable-detr',
                                                                        config=detr_config,
                                                                        ignore_mismatched_sizes=True)

         # see https://github.com/PyTorchLightning/pytorch-lightning/pull/1896
         self.lr = lr
         self.lr_backbone = lr_backbone
         self.weight_decay = weight_decay
         self.train_loader = train_loader
         self.val_loader = val_loader
         self.device = 'cuda'

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

        return loss

     def validation_step(self, batch):
        loss, loss_dict = self.common_step(batch)
        self.log("validation_loss", loss)
        for k,v in loss_dict.items():
          self.log("validation_" + k, v.item())

        return loss

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

detr = Detr(lr=1e-4, lr_backbone=1e-5, weight_decay=1e-4, train_loader=train_dataloader, val_loader=test_dataloader)
detr.model.to("cuda")

optimizer = detr.configure_optimizers()

for epoch in range(5):
    for batch in tqdm.tqdm(train_dataloader):
        losses = detr.training_step(batch)
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

#model = DeformableDetrForObjectDetection.from_pretrained('SenseTime/deformable-detr')
# model.to("cuda")

#batch = next(iter(test_dataloader))
#pdb.set_trace()
evaluate = 1

if evaluate:
    coco_evaluator = CocoEvaluator(test_dataset.coco, iou_types)
    #coco_evaluator = CocoEvaluator(test_dataset.coco, iou_types)
    #test_dataloader = DataLoader(test_dataset, collate_fn=collate_fn, batch_size=2)
    #batch = next(iter(test_dataloader))
    eval(detr.model,test_dataloader)

print ('\n Done .... \n')