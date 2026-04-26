"""D2 — null-space viability diagnostic for InfLoRA-style LoRA init.

For each candidate LoRA attach-point (decoder self-attention input + decoder
FFN input, at each of the 6 decoder layers), accumulates the input-activation
covariance per task using that task's own final checkpoint. Then computes
pairwise null-space dimensions and reports whether InfLoRA-style structural
orthogonality init is viable at LoRA ranks {8, 16, 32}.

Decides Axis B (rank) and Axis C (task separation: structural null-space init
vs soft orthogonality regularization) — see path_b_design.md §4.

The activation captured at each attach point is the INPUT to the projection
(W_K, W_V, fc1, etc.) — that's what InfLoRA's null-space-init criterion is
defined over: we want new task's LoRA's column space to be in the null space
of old tasks' INPUT activations, so that A_old @ x_new ≈ 0 on data directions
that mattered for old tasks.

Self-attention K, V, Q in MD-DETR all share the same input tensor (the layer's
post-norm hidden state). So one hook on q_proj's input is enough to characterise
all three self-attention projections.

Output:
  <out_dir>/null_space_covariances.pt  — dict {task_id -> {attach_point -> (d,d) cov}}
     where d = attach point's input dim (256 for self_attn/fc1, 1024 for fc2).
  <out_dir>/null_space_viability.txt   — printable summary table
"""

import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.coco_hug import CocoDetection


# Per-task batch cap for activation collection. Covariance estimates converge
# fast — a few hundred batches × 300 proposals = ~100k–300k vectors per task,
# which is far more than needed for a stable 256x256 covariance estimate.
DEFAULT_MAX_BATCHES_PER_TASK = 300


def _energy_threshold_rank(singular_values: torch.Tensor, energy: float = 0.99) -> int:
    """Number of leading singular values needed to capture `energy` fraction of total."""
    # singular values from torch.linalg.svdvals are nonneg, we treat them as standard SVs
    # of the activation matrix. For a covariance C = XᵀX, the eigenvalues of C are σᵢ²
    # — but we feed C directly to SVD so 'singular_values' here ARE the eigenvalues.
    # Use them as energy proxies directly.
    s = singular_values.clone().sort(descending=True).values
    total = s.sum()
    if total <= 0:
        return 0
    cumulative = torch.cumsum(s, dim=0)
    threshold = energy * total
    # smallest k such that cumulative[k] >= threshold
    above = (cumulative >= threshold).nonzero(as_tuple=True)[0]
    return int(above[0].item()) + 1 if above.numel() > 0 else int(s.numel())


def _absolute_threshold_rank(singular_values: torch.Tensor, tol_ratio: float = 1e-4) -> int:
    """Number of singular values above tol_ratio * max_singular_value."""
    if singular_values.numel() == 0:
        return 0
    max_s = singular_values.max()
    if max_s <= 0:
        return 0
    return int((singular_values > tol_ratio * max_s).sum().item())


def run_null_space_viability(args, processor, out_dir_root, trainer,
                              rank: int = 0, world_size: int = 1, local_rank: int = 0):
    """Run null-space viability diagnostic.

    Single-process only. Reloads the checkpoint for each task in turn.
    """
    if world_size > 1:
        raise NotImplementedError(
            'D2 null-space viability does not support DDP; run with sbatch.gpus_per_node=1.'
        )

    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')

    # 1. Resolve the FINAL-task checkpoint. We use a single fixed model state
    #    (Task_4) for all measurements — per-task variation comes from the DATA
    #    distributions passed through it, NOT from per-task model states.
    #
    #    Why not per-task checkpoints? `engine.resume()` uses `load_state_dict(strict=False)`,
    #    so loading Task_1's checkpoint into a 4-task-sized model silently skips the
    #    shape-mismatched memory params and leaves Task_2..4 memory at random-init.
    #    That would pollute layer>0 activations with random-memory output. For InfLoRA
    #    feasibility, the meaningful question is whether different tasks' image
    #    distributions produce distinct activation subspaces — a fixed-model
    #    comparison isolates that, which is the right methodology.
    if not args.checkpoint_dir:
        raise ValueError('D2 requires --checkpoint_dir set to <baseline>/Task_1/.')
    base_dir = args.checkpoint_dir  # e.g. .../train_dynamic_correctness_a/Task_1
    next_name = args.checkpoint_next or 'checkpoint05.pth'
    final_task_dir = base_dir.replace('Task_1', f'Task_{args.n_tasks}')
    final_ckpt = os.path.join(final_task_dir, next_name)
    if not os.path.exists(final_ckpt):
        raise FileNotFoundError(f'D2 requires final-task checkpoint at {final_ckpt}')
    print(f'[diag_null_space] loading final checkpoint (used for all measurements): {final_ckpt}',
          flush=True)
    trainer.resume(final_ckpt)
    trainer.model.to(device).eval()

    # 2. Decide attach points. Pre-hooks fire BEFORE the module's forward, with
    #    input = the activation X about to be projected.
    decoder = trainer.model.model.decoder
    n_layers = len(decoder.layers)

    # Configurable attach-point family via args.d2_attach:
    #   None (default) / "self_attn_fc1" → original D2: self_attn input + fc1 input (12 probes)
    #   "fc1"         → fc1 input only (6 probes; subset of above)
    #   "fc2"         → fc2 input only (6 probes) — used for Pre-1a insurance check
    #   "fc1_fc2"     → fc1 + fc2 inputs (12 probes) — LoRA-ablation-relevant
    #   "self_attn"   → self_attn input only (6 probes)
    # See docs/MD-DETR/phase_1_conceptual_plan.md §3.1 for the Pre-1a fc2 rationale.
    attach_mode = getattr(args, 'd2_attach', None) or 'self_attn_fc1'
    attach_points = {}  # name -> module
    for i, layer in enumerate(decoder.layers):
        if attach_mode in ('self_attn_fc1', 'self_attn'):
            attach_points[f'layer{i}_self_attn_in'] = layer.self_attn.q_proj
        if attach_mode in ('self_attn_fc1', 'fc1', 'fc1_fc2'):
            attach_points[f'layer{i}_fc1_in'] = layer.fc1
        if attach_mode in ('fc2', 'fc1_fc2'):
            attach_points[f'layer{i}_fc2_in'] = layer.fc2
    if not attach_points:
        raise ValueError(f'd2_attach={attach_mode!r} produced no attach points')
    print(f'[diag_null_space] attach_mode={attach_mode}, '
          f'tracking {len(attach_points)} attach points', flush=True)

    # 3. Per-task batch cap (cheap, but 4 checkpoint reloads × full data is too much).
    max_batches = int(getattr(args, 'extract_max_samples_per_class', 0) or 0)
    if max_batches <= 0:
        max_batches = DEFAULT_MAX_BATCHES_PER_TASK
        print(f'[diag_null_space] using default max_batches_per_task={max_batches}', flush=True)
    else:
        # Reuse the existing flag for "how much to collect" — its semantics here are
        # batches per task (not samples per class), since covariance accumulation has
        # no per-class structure.
        print(f'[diag_null_space] max_batches_per_task={max_batches} (from --extract_max_samples_per_class)',
              flush=True)

    # 4. Iterate per-task: install hooks, accumulate covariance. Model state is
    #    fixed (final checkpoint loaded at step 1); only the DATA varies per task.
    per_task_cov = {}  # task_id -> {attach_point_name -> (d, d) covariance tensor on cpu}

    # Per-attach-point input dim. self_attn.q_proj/fc1 are 256-dim; fc2 is 1024-dim
    # (post-FFN-up activation, standard 4× expansion in Deformable DETR).
    attach_dims = {name: int(m.in_features) for name, m in attach_points.items()}
    print(f'[diag_null_space] attach input dims: {attach_dims}', flush=True)

    for task_id in range(1, args.n_tasks + 1):
        print(f'[diag_null_space] === task {task_id}: iterating training data ===', flush=True)

        # Fresh covariance accumulators for this task, sized per-attach-point
        cov_acc = {name: torch.zeros(attach_dims[name], attach_dims[name],
                                     device=device, dtype=torch.float64)
                   for name in attach_points}

        def _make_pre_hook(name):
            def _hook(_module, inputs):
                # inputs is a tuple; first element is the activation X
                x = inputs[0].detach()
                if x.dim() == 3:
                    x_flat = x.reshape(-1, x.shape[-1])  # (B*N, 256)
                elif x.dim() == 2:
                    x_flat = x
                else:
                    x_flat = x.reshape(-1, x.shape[-1])
                # Cast to float64 for numerical stability when summing many outer products
                cov_acc[name] += x_flat.to(torch.float64).T @ x_flat.to(torch.float64)
            return _hook

        handles = [m.register_forward_pre_hook(_make_pre_hook(name))
                   for name, m in attach_points.items()]

        try:
            tr_ann = os.path.join(args.task_ann_dir, f'train_task_{task_id}.json')
            if not os.path.exists(tr_ann):
                print(f'[diag_null_space] WARNING: {tr_ann} missing, skipping task {task_id}',
                      flush=True)
                continue
            dataset = CocoDetection(img_folder=args.train_img_dir, ann_file=tr_ann,
                                    processor=processor)
            loader = DataLoader(dataset, collate_fn=dataset.collate_fn,
                                batch_size=args.extract_batch_size,
                                num_workers=args.extract_num_workers,
                                shuffle=False, pin_memory=True)

            n_done = 0
            for batch in tqdm(loader, desc=f'diag_null_space task {task_id}', total=max_batches):
                if n_done >= max_batches:
                    break
                pixel_values = batch['pixel_values'].to(device)
                pixel_mask = batch['pixel_mask'].to(device)
                labels = [{k: v.to(device) for k, v in t.items()} for t in batch['labels']]
                with torch.no_grad():
                    _ = trainer.model(
                        pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels,
                        train=False, task_id=args.n_tasks,
                    )
                n_done += 1
        finally:
            for h in handles:
                h.remove()

        per_task_cov[task_id] = {name: cov.cpu() for name, cov in cov_acc.items()}
        del cov_acc

    # 5. Save raw covariances for any further offline analysis
    cov_path = os.path.join(out_dir_root, 'null_space_covariances.pt')
    torch.save({
        'n_layers': n_layers,
        'attach_dims': attach_dims,  # {attach_point_name -> input dim}
        'attach_points': list(attach_points.keys()),
        'per_task': per_task_cov,
        'final_ckpt': final_ckpt,
        'max_batches_per_task': max_batches,
    }, cov_path)
    print(f'[diag_null_space] wrote {cov_path}', flush=True)

    # 6. Compute null-space dimensions per attach point per task pair, and report.
    summary_lines = []
    summary_lines.append('# D2 — null-space viability for InfLoRA-style LoRA init\n')
    summary_lines.append(f'# n_layers={n_layers}, batches/task={max_batches}')
    summary_lines.append(f'# attach input dims: {attach_dims}\n')
    summary_lines.append(f'Method: single fixed model state (final checkpoint {final_ckpt})')
    summary_lines.append('used for all measurements; per-task variation comes from the DATA')
    summary_lines.append('distributions passed through it. Per-task covariance C_t = sum_X X^T X')
    summary_lines.append('over input activations to each attach point. Effective rank reported')
    summary_lines.append('with two thresholds:')
    summary_lines.append('  - 99%-energy: smallest k such that top-k singular values capture 99% of trace')
    summary_lines.append('  - 1e-4 ratio: count of singular values > 1e-4 * max_singular_value\n')
    summary_lines.append('For task k, "init in null space of {1..k-1}" requires the cumulative')
    summary_lines.append('covariance C_<k = sum_(t<k) C_t to have null-space dim >= chosen LoRA rank.')
    summary_lines.append('Reported below: null-space dim of C_<k, and feasibility for ranks {8, 16, 32}.\n')

    # Per attach-point analysis
    for ap_name in attach_points.keys():
        summary_lines.append(f'\n## Attach point: {ap_name}')
        # Per-task effective rank
        header = (f'  {"task":>4}  {"rank_99":>7}  {"rank_1e-4":>9}  '
                  f'{"null_99":>7}  {"null_1e-4":>9}')
        summary_lines.append(header)
        summary_lines.append('  ' + '-' * (len(header) - 2))
        for t in sorted(per_task_cov.keys()):
            C = per_task_cov[t][ap_name]
            s = torch.linalg.svdvals(C.to(torch.float64))
            r99 = _energy_threshold_rank(s, energy=0.99)
            r_tol = _absolute_threshold_rank(s, tol_ratio=1e-4)
            d = C.shape[0]
            summary_lines.append(
                f'  {t:>4}  {r99:>7}  {r_tol:>9}  {d - r99:>7}  {d - r_tol:>9}'
            )
        # Cumulative null-space dim (the InfLoRA-relevant quantity for task k+1 init)
        summary_lines.append('')
        summary_lines.append('  Cumulative C_<k = sum of tasks 1..k-1 covariances:')
        cum_header = (f'    {"k":>3}  {"null_99":>7}  {"null_1e-4":>9}  '
                      f'{"r=8":>5}  {"r=16":>5}  {"r=32":>5}')
        summary_lines.append(cum_header)
        summary_lines.append('    ' + '-' * (len(cum_header) - 4))
        for k in range(2, args.n_tasks + 1):
            tasks_below = [t for t in sorted(per_task_cov.keys()) if t < k]
            if not tasks_below:
                continue
            C_cum = sum(per_task_cov[t][ap_name] for t in tasks_below)
            s = torch.linalg.svdvals(C_cum.to(torch.float64))
            r99 = _energy_threshold_rank(s, energy=0.99)
            r_tol = _absolute_threshold_rank(s, tol_ratio=1e-4)
            d = C_cum.shape[0]
            null99 = d - r99
            null_tol = d - r_tol
            # Use the more conservative (smaller) null-space dim for feasibility
            feas = min(null99, null_tol)
            r8 = '✓' if feas >= 8 else '✗'
            r16 = '✓' if feas >= 16 else '✗'
            r32 = '✓' if feas >= 32 else '✗'
            summary_lines.append(
                f'    {k:>3}  {null99:>7}  {null_tol:>9}  {r8:>5}  {r16:>5}  {r32:>5}'
            )

    summary_lines.append('')
    summary_lines.append('Interpretation guide (path_b_design.md §4 Axis B/C):')
    summary_lines.append('  - All ✓ at r=16 across attach points and k=2..4: InfLoRA structural')
    summary_lines.append('    null-space init is viable. Pick attach point from D1 and proceed.')
    summary_lines.append('  - Mostly ✗ at r=16+ but ✓ at r=8: InfLoRA viable only at small rank;')
    summary_lines.append('    rank-capacity tradeoff is real, may need to fall back to soft ortho.')
    summary_lines.append('  - Mostly ✗ even at r=8: tasks share too much of the input subspace;')
    summary_lines.append('    InfLoRA infeasible -> use O-LoRA soft orthogonality regularization')
    summary_lines.append('    (Axis C option 2) instead.')

    summary = '\n'.join(summary_lines)
    summary_path = os.path.join(out_dir_root, 'null_space_viability.txt')
    with open(summary_path, 'w') as f:
        f.write(summary + '\n')
    print(summary, flush=True)
    print(f'[diag_null_space] wrote {summary_path}', flush=True)
