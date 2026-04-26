"""Prototype extraction: iterate all tasks' training data with a final
trained checkpoint, collect hot-proposal features for matched proposals,
save per-class mean prototypes to disk.

Supports DDP: when launched under torchrun (WORLD_SIZE > 1), data is
sharded across ranks via DistributedSampler(drop_last=True), each rank
accumulates into its own PrototypeStore, and partial sum_/count tensors
are merged via all_reduce at the end. Only rank 0 writes prototypes.pt.
"""

import os
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from datasets.coco_hug import CocoDetection
from models.prototype_classifier import PrototypeStore


def run_prototype_extraction(args, processor, out_dir_root, trainer,
                             rank: int = 0, world_size: int = 1, local_rank: int = 0):
    """Run prototype extraction using an already-constructed trainer.

    trainer: a local_trainer instance with task_id=args.n_tasks so its
             memory modules are initialized for all seen tasks.
    rank, world_size, local_rank: DDP metadata. Defaults correspond to
             single-process execution; all DDP-specific paths are skipped
             when world_size == 1.
    """
    is_distributed = world_size > 1

    # 1. Resolve checkpoint path
    ckpt_path = args.prototype_checkpoint_path
    if not ckpt_path:
        if args.checkpoint_dir:
            final_dir = args.checkpoint_dir.replace('Task_1', f'Task_{args.n_tasks}')
            ckpt_path = os.path.join(final_dir, args.checkpoint_next or 'checkpoint05.pth')
        else:
            ckpt_path = os.path.join(out_dir_root, f'Task_{args.n_tasks}', 'checkpoint05.pth')

    if rank == 0:
        print(f'[extract_prototypes] loading checkpoint: {ckpt_path}')
    trainer.resume(ckpt_path)

    # 2. Build prototype store (80 foreground classes, no background)
    n_fg_classes = args.n_classes - 1
    max_per_class = int(getattr(args, 'extract_max_samples_per_class', 0) or 0)
    store = PrototypeStore(n_fg_classes, d_model=256, max_per_class=max_per_class)
    if rank == 0 and max_per_class > 0:
        print(f'[extract_prototypes] subsampling enabled: max {max_per_class} samples per class', flush=True)

    # 3. Move model to this rank's GPU, eval mode
    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
    trainer.model.to(device).eval()

    # 4. Iterate all tasks' training data
    for task_id in range(1, args.n_tasks + 1):
        tr_ann = os.path.join(args.task_ann_dir, f'train_task_{task_id}.json')
        if not os.path.exists(tr_ann):
            if rank == 0:
                print(f'[extract_prototypes] WARNING: {tr_ann} not found, skipping task {task_id}')
            continue

        dataset = CocoDetection(img_folder=args.train_img_dir, ann_file=tr_ann, processor=processor)

        sampler = None
        if is_distributed:
            # drop_last=True avoids the default padding-with-repeated-indices,
            # which would otherwise double-count a few images per class mean.
            sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank,
                                         shuffle=False, drop_last=True)

        loader = DataLoader(dataset, collate_fn=dataset.collate_fn,
                            batch_size=args.extract_batch_size,
                            num_workers=args.extract_num_workers,
                            shuffle=False, sampler=sampler, pin_memory=True)

        # Only rank 0 shows the tqdm progress bar; others iterate silently.
        loader_iter = tqdm(loader, desc=f'extract task {task_id}') if rank == 0 else loader

        for batch in loader_iter:
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

            # Early exit if every class has reached its cap. Saves hours when
            # common classes have long since hit the cap but rare classes are
            # still collecting. This is a per-rank check; in DDP the all_reduce
            # at the end still merges whatever each rank accumulated.
            if store.is_full():
                if rank == 0:
                    print(f'[extract_prototypes] task {task_id}: all classes full, exiting inner loop early', flush=True)
                break

        # Per-rank completion marker (no sync — log line order gives us timing).
        if is_distributed:
            print(f'[rank {rank}] task {task_id} complete', flush=True)

        # If all classes are full, we can skip remaining tasks too.
        if store.is_full():
            if rank == 0:
                print(f'[extract_prototypes] all classes full after task {task_id}, skipping remaining tasks', flush=True)
            break

    # 5. Merge partial stores across ranks (GPU round trip; ~160 KB total)
    if is_distributed:
        sum_gpu = store.sum_.to(device)
        count_gpu = store.count.to(device)
        dist.all_reduce(sum_gpu, op=dist.ReduceOp.SUM)
        dist.all_reduce(count_gpu, op=dist.ReduceOp.SUM)
        store.sum_ = sum_gpu.cpu()
        store.count = count_gpu.cpu()

    # 6. Rank 0 saves; others skip
    if rank == 0:
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

    # 7. Clean shutdown (barrier ensures rank 0 finishes save before anyone exits)
    if is_distributed:
        dist.barrier(device_ids=[local_rank])
        dist.destroy_process_group()
