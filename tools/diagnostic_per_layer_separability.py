"""D1 — per-layer class separability diagnostic.

For each of the 6 decoder layers, accumulate per-class prototypes from
matched proposals, then report pairwise off-diagonal cosine statistics.
Decides the LoRA attach point (path_b_design.md §4 Axis A): if collapse
is gradual across layers, intervene early; if specific to late decoder,
attach there; if already collapsed at layer 1, the encoder is the source
and decoder-only LoRA may be insufficient.

Mirrors tools/extract_prototypes.py's iteration pattern, but installs
forward hooks on each decoder layer and accumulates 6 PrototypeStores
in parallel (one per layer depth).

Output:
  <out_dir>/per_layer_prototypes.pt  — dict {layer_idx -> {sum_, count, n_classes, d_model}}
  <out_dir>/per_layer_separability.txt — printable summary table

The matching uses the FINAL (layer 6) outputs against GT labels — the
same Hungarian matcher extract_prototypes.py uses. Per-layer features
for the same matched proposal are then accumulated against that same
class label.
"""

import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.coco_hug import CocoDetection
from models.prototype_classifier import PrototypeStore


def _pairwise_cosine_stats(prototypes: torch.Tensor, seen: torch.Tensor):
    """Off-diagonal pairwise cosine summary for a per-class prototype matrix.

    prototypes: (n_classes, d_model)
    seen:       (n_classes,) bool
    Returns dict of stats over the off-diagonal cosine matrix restricted to
    seen classes only.
    """
    P = prototypes[seen]
    if P.shape[0] < 2:
        return {'n_seen': int(seen.sum().item()), 'mean': None, 'std': None,
                'min': None, 'max': None, 'top1_other_mean': None}
    Pn = torch.nn.functional.normalize(P, dim=-1)
    M = Pn @ Pn.T
    n = P.shape[0]
    eye = torch.eye(n, dtype=torch.bool)
    off = M[~eye]
    # Each class's nearest other class
    M_ = M.clone()
    M_[eye] = -2.0
    top1 = M_.max(dim=-1).values
    return {
        'n_seen': int(seen.sum().item()),
        'mean': float(off.mean().item()),
        'std': float(off.std().item()),
        'min': float(off.min().item()),
        'max': float(off.max().item()),
        'top1_other_mean': float(top1.mean().item()),
        'top1_other_max': float(top1.max().item()),
    }


def run_per_layer_separability(args, processor, out_dir_root, trainer,
                                rank: int = 0, world_size: int = 1, local_rank: int = 0):
    """Run per-layer class-separability diagnostic.

    Single-process only (DDP not supported — D1 is cheap enough on one GPU).
    Accepts rank/world_size for signature parity with run_prototype_extraction
    but raises if asked to run distributed.
    """
    if world_size > 1:
        raise NotImplementedError(
            'D1 per-layer separability does not support DDP; run with sbatch.gpus_per_node=1.'
        )

    # Pre-1b variance-baseline support: --d1_seed overrides args.seed so two D1
    # runs with different seeds measure the measurement noise floor σ, which is
    # then used to set Gate B's ±2σ tolerance. See phase_1_conceptual_plan.md §3.2.
    d1_seed = getattr(args, 'd1_seed', None)
    if d1_seed is not None:
        import random
        import numpy as np
        import torch as _torch
        print(f'[diag_per_layer] overriding seed with --d1_seed={d1_seed}', flush=True)
        _torch.manual_seed(int(d1_seed))
        if _torch.cuda.is_available():
            _torch.cuda.manual_seed_all(int(d1_seed))
        np.random.seed(int(d1_seed))
        random.seed(int(d1_seed))

    # 1. Resolve checkpoint path. Prefer the explicit Task_4 final checkpoint
    #    (this diagnostic measures the geometry of the FINAL trained model,
    #    so we always want the last-task checkpoint regardless of args.start_task).
    ckpt_path = args.prototype_checkpoint_path
    if not ckpt_path:
        if args.checkpoint_dir:
            final_dir = args.checkpoint_dir.replace('Task_1', f'Task_{args.n_tasks}')
            ckpt_path = os.path.join(final_dir, args.checkpoint_next or 'checkpoint05.pth')
        else:
            ckpt_path = os.path.join(out_dir_root, f'Task_{args.n_tasks}', 'checkpoint05.pth')
    print(f'[diag_per_layer] loading checkpoint: {ckpt_path}', flush=True)
    trainer.resume(ckpt_path)

    # 2. Discover decoder layer count + build a PrototypeStore per layer
    decoder = trainer.model.model.decoder
    n_layers = len(decoder.layers)
    print(f'[diag_per_layer] decoder has {n_layers} layers', flush=True)

    n_fg_classes = args.n_classes - 1
    max_per_class = int(getattr(args, 'extract_max_samples_per_class', 0) or 0)
    if max_per_class <= 0:
        # D1 is a geometry diagnostic — class means converge fast. Cap by default
        # to keep wall-clock reasonable; user can override via the existing flag.
        max_per_class = 500
        print(f'[diag_per_layer] applying default max_per_class={max_per_class}', flush=True)
    stores = [PrototypeStore(n_fg_classes, d_model=256, max_per_class=max_per_class)
              for _ in range(n_layers)]

    # 3. Move to GPU, eval mode
    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
    trainer.model.to(device).eval()

    # 4. Install forward hooks on each decoder layer's output. The decoder layer's
    #    forward returns (hidden_states, *optional_attn). We capture output[0].
    captured = [None] * n_layers

    def _make_hook(idx):
        def _hook(_module, _inputs, output):
            hs = output[0] if isinstance(output, tuple) else output
            captured[idx] = hs.detach()
        return _hook

    handles = [layer.register_forward_hook(_make_hook(i))
               for i, layer in enumerate(decoder.layers)]

    # 5. Iterate all 4 tasks' training data, accumulate per-layer per-class means
    try:
        for task_id in range(1, args.n_tasks + 1):
            tr_ann = os.path.join(args.task_ann_dir, f'train_task_{task_id}.json')
            if not os.path.exists(tr_ann):
                print(f'[diag_per_layer] WARNING: {tr_ann} missing, skipping task {task_id}',
                      flush=True)
                continue

            dataset = CocoDetection(img_folder=args.train_img_dir, ann_file=tr_ann,
                                    processor=processor)
            # When --d1_seed is set (Pre-1b variance baseline), shuffle with a
            # seeded generator so different seeds select a different set of
            # first-500-per-class samples → measurable prototype variance.
            # When d1_seed is None (original D1 run), keep shuffle=False for
            # bit-identical reproducibility of the baseline.
            loader_gen = None
            if d1_seed is not None:
                loader_gen = torch.Generator()
                loader_gen.manual_seed(int(d1_seed) + task_id)
            loader = DataLoader(dataset, collate_fn=dataset.collate_fn,
                                batch_size=args.extract_batch_size,
                                num_workers=args.extract_num_workers,
                                shuffle=(loader_gen is not None),
                                generator=loader_gen, pin_memory=True)
            loader_iter = tqdm(loader, desc=f'diag_per_layer task {task_id}')

            for batch in loader_iter:
                pixel_values = batch['pixel_values'].to(device)
                pixel_mask = batch['pixel_mask'].to(device)
                labels = [{k: v.to(device) for k, v in t.items()} for t in batch['labels']]

                with torch.no_grad():
                    outputs = trainer.model(
                        pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels,
                        train=False, task_id=args.n_tasks,
                    )

                # Hungarian matching uses FINAL outputs, same as extract_prototypes.py.
                # The matched (pred_idx, tgt_idx) pairs are reused for ALL layers —
                # we're asking "what does proposal i (matched to class c at layer 6)
                # look like at intermediate layers?".
                outputs_without_aux = {k: v for k, v in outputs.items()
                                       if k != 'auxiliary_outputs' and k != 'enc_outputs'}
                indices = trainer.model.matcher(outputs_without_aux, labels)

                for img_i, (pred_idx, tgt_idx) in enumerate(indices):
                    if pred_idx.numel() == 0:
                        continue
                    cls_ids = labels[img_i]['class_labels'][tgt_idx].detach().cpu()
                    for layer_i in range(n_layers):
                        feats = captured[layer_i][img_i, pred_idx].detach().cpu()
                        stores[layer_i].add(feats, cls_ids)

                # Early-exit when all stores have all classes filled to the cap.
                if all(s.is_full() for s in stores):
                    print(f'[diag_per_layer] task {task_id}: all stores full, exiting inner loop',
                          flush=True)
                    break

            if all(s.is_full() for s in stores):
                print(f'[diag_per_layer] all stores full after task {task_id}, '
                      f'skipping remaining tasks', flush=True)
                break
    finally:
        for h in handles:
            h.remove()

    # 6. Save raw per-layer prototypes (one combined .pt for easy reloading)
    for s in stores:
        s.finalize()
    out_path = os.path.join(out_dir_root, 'per_layer_prototypes.pt')
    torch.save({
        'n_layers': n_layers,
        'n_classes': n_fg_classes,
        'd_model': 256,
        'max_per_class': max_per_class,
        'per_layer': {
            i: {'prototypes': stores[i].prototypes,
                'seen': stores[i].seen,
                'count': stores[i].count}
            for i in range(n_layers)
        },
    }, out_path)
    print(f'[diag_per_layer] wrote {out_path}', flush=True)

    # 7. Compute pairwise cosine stats per layer + write a printable summary
    summary_lines = []
    summary_lines.append('# D1 — per-layer class separability\n')
    summary_lines.append(f'# checkpoint: {ckpt_path}')
    summary_lines.append(f'# n_layers={n_layers}, n_classes={n_fg_classes}, '
                         f'max_per_class={max_per_class}\n')
    header = (f'{"layer":>5}  {"n_seen":>6}  {"mean":>7}  {"std":>7}  '
              f'{"min":>7}  {"max":>7}  {"top1_oth":>8}  {"top1_max":>8}')
    summary_lines.append(header)
    summary_lines.append('-' * len(header))
    for i, store in enumerate(stores):
        stats = _pairwise_cosine_stats(store.prototypes, store.seen)
        if stats['mean'] is None:
            summary_lines.append(f'{i:>5}  {stats["n_seen"]:>6}  (no pairs)')
            continue
        summary_lines.append(
            f'{i:>5}  {stats["n_seen"]:>6}  {stats["mean"]:>7.4f}  '
            f'{stats["std"]:>7.4f}  {stats["min"]:>7.4f}  {stats["max"]:>7.4f}  '
            f'{stats["top1_other_mean"]:>8.4f}  {stats["top1_other_max"]:>8.4f}'
        )
    summary_lines.append('')
    summary_lines.append('Interpretation guide (from path_b_design.md §4 Axis A):')
    summary_lines.append('  - mean ~ 0.62 + top1 ~ 0.83 at layer 6 reproduces the original diagnostic.')
    summary_lines.append('  - If mean grows monotonically with layer depth: collapse is gradual,')
    summary_lines.append('    LoRA could intervene at any depth, prefer middle-late layers.')
    summary_lines.append('  - If mean jumps in last 1-2 layers: localised collapse, attach there.')
    summary_lines.append('  - If mean is already > 0.5 at layer 0: encoder-sourced, decoder-only')
    summary_lines.append('    LoRA may be insufficient -> consider Variant C (projection head)')
    summary_lines.append('    or extending to encoder.')

    summary = '\n'.join(summary_lines)
    summary_path = os.path.join(out_dir_root, 'per_layer_separability.txt')
    with open(summary_path, 'w') as f:
        f.write(summary + '\n')
    print(summary, flush=True)
    print(f'[diag_per_layer] wrote {summary_path}', flush=True)
