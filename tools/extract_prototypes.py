"""Prototype extraction: iterate all tasks' training data with a final
trained checkpoint, collect hot-proposal features for matched proposals,
save per-class mean prototypes to disk.
"""

import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.coco_hug import CocoDetection
from models.prototype_classifier import PrototypeStore


def run_prototype_extraction(args, processor, out_dir_root, trainer):
    """Run prototype extraction using an already-constructed trainer.

    trainer: a local_trainer instance with task_id=args.n_tasks so its
             memory modules are initialized for all seen tasks.
    """
    # 1. Resolve checkpoint path
    ckpt_path = args.prototype_checkpoint_path
    if not ckpt_path:
        if args.checkpoint_dir:
            final_dir = args.checkpoint_dir.replace('Task_1', f'Task_{args.n_tasks}')
            ckpt_path = os.path.join(final_dir, args.checkpoint_next or 'checkpoint05.pth')
        else:
            ckpt_path = os.path.join(out_dir_root, f'Task_{args.n_tasks}', 'checkpoint05.pth')

    print(f'[extract_prototypes] loading checkpoint: {ckpt_path}')
    trainer.resume(ckpt_path)

    # 2. Build prototype store (80 foreground classes, no background)
    n_fg_classes = args.n_classes - 1
    store = PrototypeStore(n_fg_classes, d_model=256)

    # 3. Single GPU, eval mode
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    trainer.model.to(device).eval()

    # 4. Iterate all tasks' training data
    for task_id in range(1, args.n_tasks + 1):
        tr_ann = os.path.join(args.task_ann_dir, f'train_task_{task_id}.json')
        if not os.path.exists(tr_ann):
            print(f'[extract_prototypes] WARNING: {tr_ann} not found, skipping task {task_id}')
            continue

        dataset = CocoDetection(img_folder=args.train_img_dir, ann_file=tr_ann, processor=processor)
        loader = DataLoader(dataset, collate_fn=dataset.collate_fn,
                            batch_size=args.batch_size, num_workers=args.num_workers,
                            shuffle=False, pin_memory=True)

        for batch in tqdm(loader, desc=f'extract task {task_id}'):
            pixel_values = batch['pixel_values'].to(device)
            pixel_mask = batch['pixel_mask'].to(device)
            labels = [{k: v.to(device) for k, v in t.items()} for t in batch['labels']]

            with torch.no_grad():
                outputs = trainer.model(
                    pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels,
                    train=False, task_id=args.n_tasks,
                )

            outputs_without_aux = {k: v for k, v in outputs.items()
                                   if k != 'auxiliary_outputs' and k != 'enc_outputs'}
            indices = trainer.model.matcher(outputs_without_aux, labels)
            hs = outputs.last_hidden_state  # (B, 300, 256)

            for i, (pred_idx, tgt_idx) in enumerate(indices):
                if pred_idx.numel() == 0:
                    continue
                feats = hs[i, pred_idx].detach().cpu()
                cls_ids = labels[i]['class_labels'][tgt_idx].detach().cpu()
                store.add(feats, cls_ids)

    # 5. Save and report
    out_path = args.prototypes_out_path or os.path.join(out_dir_root, 'prototypes.pt')
    store.save(out_path)
    store.finalize()
    n_seen = int(store.seen.sum().item())
    n_missing = int((~store.seen).sum().item())
    print(f'[extract_prototypes] wrote {out_path}')
    print(f'[extract_prototypes] classes seen={n_seen}/{store.n_classes}, missing={n_missing}')
    if n_missing > 0:
        missing_ids = torch.nonzero(~store.seen).flatten().tolist()
        missing_names = [args.task_label2name.get(i, f'id_{i}') for i in missing_ids]
        print(f'[extract_prototypes] missing class ids/names: {list(zip(missing_ids, missing_names))}')
