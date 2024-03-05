import os
import tqdm
import torch
import numpy as np
import torchvision
import pdb
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from datasets.coco_eval import CocoEvaluator
from transformers.models.deformable_detr.feature_extraction_deformable_detr import DeformableDetrFeatureExtractor 
from transformers import AutoImageProcessor
from transformers.models.deformable_detr.configuration_deformable_detr import DeformableDetrConfig
from transformers.models.deformable_detr.modeling_deformable_detr import DeformableDetrForObjectDetection 

class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, ann_file, feature_extractor):
        super(CocoDetection, self).__init__(img_folder, ann_file)
        self.feature_extractor = feature_extractor

    def __getitem__(self, idx):
        # read in PIL image and target in COCO format
        img, target = super(CocoDetection, self).__getitem__(idx)
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}
        
        # preprocess image and target (converting target to DETR format, resizing + normalization of both image and target)
        encoding = self.feature_extractor(images=img, annotations=target, return_tensors="pt")
        pixel_values = encoding["pixel_values"].squeeze() # remove batch dimension
        target = encoding["labels"][0] # remove batch dimension
        
        return pixel_values, target

repo_name = "SenseTime/deformable-detr" 
#feature_extractor = DeformableDetrFeatureExtractor()
feature_extractor  = AutoImageProcessor.from_pretrained(repo_name)

test_dataset = CocoDetection(img_folder='/ubc/cs/research/shield/datasets/MSCOCO/2017/val2017', 
                              ann_file='/ubc/cs/research/shield/datasets/MSCOCO/2017/annotations/instances_val2017.json',
                              feature_extractor=feature_extractor)

# colors for visualization
COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

def collate_fn(batch):
  pixel_values = [item[0] for item in batch]
  #print(pixel_values)
  encoding = feature_extractor.pad(pixel_values, return_tensors="pt")
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

def viz_gt(id=None):
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
    image = Image.open(os.path.join('/ubc/cs/research/shield/datasets/MSCOCO/2017/val2017', image['file_name']))

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

def viz_mod(model, id=None):
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
    image = Image.open(os.path.join('/ubc/cs/research/shield/datasets/MSCOCO/2017/val2017', image['file_name']))
    inputs = feature_extractor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    # let's only keep predictions with score > 0.3
    results = feature_extractor.post_process_object_detection(outputs,target_sizes=[image.size[::-1]],
                                                        threshold=0.3)[0]

    plot_results(image, results['scores'], results['labels'], results['boxes'], f_name='out_mod.jpg')

def eval(model, test_dataloader):
    print("Running evaluation...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    nbatches = 5000

    for idx, batch in enumerate(tqdm.tqdm(test_dataloader)):
        
        if idx > nbatches:
            break

        # get the inputs
        pixel_values = batch["pixel_values"].to(device)
        pixel_mask = batch["pixel_mask"].to(device)
        labels = [{k: v.to(device) for k, v in t.items()} for t in batch["labels"]] # these are in DETR format, resized + normalized

        # forward pass
        outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask)

        orig_target_sizes = torch.stack([target["orig_size"] for target in labels], dim=0)
        results = feature_extractor.post_process(outputs, orig_target_sizes) # convert outputs to COCO api
        res = {target['image_id'].item(): output for target, output in zip(labels, results)}
        coco_evaluator.update(res)

    coco_evaluator.synchronize_between_processes()
    coco_evaluator.accumulate()
    coco_evaluator.summarize()

test_dataloader = DataLoader(test_dataset, collate_fn=collate_fn, batch_size=2)
batch = next(iter(test_dataloader))
print(batch.keys())
print (batch['pixel_values'].shape)
print(batch['pixel_mask'].shape)

iou_types = ['bbox']
base_ds = test_dataset.coco
coco_evaluator = CocoEvaluator(base_ds, iou_types)

model = DeformableDetrForObjectDetection.from_pretrained('SenseTime/deformable-detr')

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#model.to(device)
eval(model,test_dataloader)
id = 42276
viz_gt(id=id)
viz_mod(model=model, id=id)

print ('\n Done .... \n')