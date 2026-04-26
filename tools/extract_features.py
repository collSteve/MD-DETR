"""Raw per-proposal feature extraction for linear-probe upper-bound diagnostic.

Iterates all tasks' training data through a fixed checkpoint (typically the
final-task DynamicPrompt checkpoint, matching the prototype extraction protocol),
collects hot-proposal features for matched proposals via the Hungarian matcher,
saves per-class-capped (feature, label) pairs to disk for offline probe training.

Deliberately simpler than tools/extract_prototypes.py: single-GPU only (matches
sbatch.gpus_per_node=1 on edith2; DDP is broken there anyway). No all_reduce.
"""

import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.coco_hug import CocoDetection
from models.linear_probe import FeatureStore


def run_feature_extraction(args, processor, out_dir_root, trainer):
    """Run feature extraction using an already-constructed trainer.

    trainer: a local_trainer instance with task_id=args.n_tasks so its
             memory modules are initialized for all seen tasks.
    """
    # 1. Resolve checkpoint path (explicit --feature_extraction_checkpoint_path
    #    preferred; falls back to prototype-extraction resolution logic).
    ckpt_path = getattr(args, 'feature_extraction_checkpoint_path', '') or ''
    if not ckpt_path:
        if args.checkpoint_dir:
            final_dir = args.checkpoint_dir.replace('Task_1', f'Task_{args.n_tasks}')
            ckpt_path = os.path.join(final_dir, args.checkpoint_next or 'checkpoint05.pth')
        else:
            ckpt_path = os.path.join(out_dir_root, f'Task_{args.n_tasks}', 'checkpoint05.pth')

    print(f'[extract_features] loading checkpoint: {ckpt_path}')
    trainer.resume(ckpt_path)

    # 2. Build feature store. Use extract_max_samples_per_class if set; else
    #    default to 2000/class (80 × 2000 = 160k samples ≈ 160 MB on disk).
    n_fg_classes = args.n_classes - 1
    cap = int(getattr(args, 'extract_max_samples_per_class', 0) or 0)
    if cap <= 0:
        cap = 2000
    store = FeatureStore(n_classes=n_fg_classes, d_model=256, max_per_class=cap)
    print(f'[extract_features] per-class cap: {cap}', flush=True)

    # 3. Move model to GPU, eval mode.
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    trainer.model.to(device).eval()

    # 4. Iterate all tasks' training data.
    for task_id in range(1, args.n_tasks + 1):
        tr_ann = os.path.join(args.task_ann_dir, f'train_task_{task_id}.json')
        if not os.path.exists(tr_ann):
            print(f'[extract_features] WARNING: {tr_ann} not found, skipping task {task_id}')
            continue

        dataset = CocoDetection(img_folder=args.train_img_dir, ann_file=tr_ann, processor=processor)
        loader = DataLoader(dataset, collate_fn=dataset.collate_fn,
                            batch_size=args.extract_batch_size,
                            num_workers=args.extract_num_workers,
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

            # Early exit: once every class reached its cap, stop.
            if store.is_full():
                print(f'[extract_features] task {task_id}: all classes full, exiting inner loop early', flush=True)
                break

        if store.is_full():
            print(f'[extract_features] all classes full after task {task_id}, skipping remaining tasks', flush=True)
            break

    # 5. Save.
    out_path = getattr(args, 'features_out_path', '') or os.path.join(out_dir_root, 'features.pt')
    store.finalize()
    store.save(out_path)
    n_seen = int((store.count > 0).sum().item())
    n_missing = int((store.count == 0).sum().item())
    print(f'[extract_features] wrote {out_path}')
    print(f'[extract_features] total samples: {int(store.count.sum().item())}')
    print(f'[extract_features] classes seen={n_seen}/{store.n_classes}, missing={n_missing}')
    print(f'[extract_features] count per class: min={int(store.count.min().item())}, '
          f'max={int(store.count.max().item())}, median={int(store.count.median().item())}')
    if n_missing > 0:
        missing_ids = torch.nonzero(store.count == 0).flatten().tolist()
        missing_names = [args.task_label2name.get(i, f'id_{i}') for i in missing_ids]
        print(f'[extract_features] missing class ids/names: {list(zip(missing_ids, missing_names))}')
