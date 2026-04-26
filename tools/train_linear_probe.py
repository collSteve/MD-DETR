"""Train a linear probe on hot decoder features saved by tools/extract_features.py.

Standalone — does NOT import the MD-DETR model, trainer, or DataLoader pipeline.
Purely: load features.pt → nn.Linear(d_model, n_classes) + CrossEntropy + Adam
→ save linear_probe.pt.

Meant to run on CPU (or a small GPU) in ~3-5 min on 160k samples. No cluster
submission required.

Usage:
    conda run -n MD-DETR python tools/train_linear_probe.py \\
        --features_path /ubc/.../extract_features_dp_t4/features.pt \\
        --out_path /ubc/.../train_linear_probe_dp/linear_probe.pt \\
        [--seed 43] [--lr 1e-3] [--epochs 20] [--batch_size 1024] [--weight_decay 1e-4] [--device cpu]
"""

import argparse
import os
import random

import numpy as np
import torch
import torch.nn as nn


def main():
    parser = argparse.ArgumentParser('Train linear probe on saved hot decoder features.')
    parser.add_argument('--features_path', type=str, required=True,
                        help='Path to features.pt (saved by tools/extract_features.py).')
    parser.add_argument('--out_path', type=str, required=True,
                        help='Where to save linear_probe.pt.')
    parser.add_argument('--seed', type=int, default=43,
                        help='Random seed for reproducibility (default: 43, matches codebase convention).')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--device', type=str, default='cpu',
                        help="'cpu' or 'cuda' or 'cuda:N'. Default 'cpu' — training is fast.")
    args = parser.parse_args()

    # Determinism
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Load features
    print(f'[train_linear_probe] loading features: {args.features_path}')
    data = torch.load(args.features_path, map_location='cpu', weights_only=False)
    X = data['features']  # (N, d_model)
    y = data['labels']    # (N,) int
    n_classes = int(data['n_classes'])
    d_model = int(data['d_model'])
    count = data['count'] if 'count' in data else None
    N = X.shape[0]
    print(f'[train_linear_probe] N={N}, d_model={d_model}, n_classes={n_classes}')
    if count is not None:
        print(f'[train_linear_probe] per-class count: min={int(count.min().item())}, '
              f'max={int(count.max().item())}, median={int(count.median().item())}')

    device = torch.device(args.device)
    X = X.to(device)
    y = y.long().to(device)

    # Classes observed in training (for the 'seen' mask saved into probe file).
    # With cap-based extraction, some rare classes may have 0 samples; we mark
    # those as unseen so eval-time scoring masks them to -inf.
    seen = torch.zeros(n_classes, dtype=torch.bool)
    for c in range(n_classes):
        seen[c] = (y == c).any().item()

    # Linear probe
    probe = nn.Linear(d_model, n_classes).to(device)
    opt = torch.optim.Adam(probe.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    print(f'[train_linear_probe] training: epochs={args.epochs}, batch={args.batch_size}, '
          f'lr={args.lr}, wd={args.weight_decay}, device={args.device}')

    final_loss = None
    for epoch in range(args.epochs):
        # Deterministic permutation per epoch
        gen = torch.Generator().manual_seed(args.seed + epoch)
        perm = torch.randperm(N, generator=gen)

        probe.train()
        epoch_loss = 0.0
        n_batches = 0
        for i in range(0, N, args.batch_size):
            idx = perm[i:i + args.batch_size]
            xb = X[idx]
            yb = y[idx]
            logits = probe(xb)
            loss = loss_fn(logits, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
            n_batches += 1
        avg_loss = epoch_loss / max(n_batches, 1)
        final_loss = avg_loss
        print(f'[train_linear_probe] epoch {epoch + 1}/{args.epochs}  loss={avg_loss:.4f}')

    # Training accuracy (overall + per-class)
    probe.eval()
    with torch.no_grad():
        preds = []
        for i in range(0, N, args.batch_size):
            xb = X[i:i + args.batch_size]
            preds.append(probe(xb).argmax(dim=-1))
        preds = torch.cat(preds, dim=0)
    correct = (preds == y)
    final_acc = float(correct.float().mean().item())
    per_class_acc = torch.zeros(n_classes)
    for c in range(n_classes):
        mask = (y == c)
        if mask.any():
            per_class_acc[c] = correct[mask].float().mean().item()
        else:
            per_class_acc[c] = float('nan')
    print(f'[train_linear_probe] final training accuracy: {final_acc:.4f}')
    print(f'[train_linear_probe] per-class accuracy: min={float(per_class_acc[~torch.isnan(per_class_acc)].min()):.3f}, '
          f'max={float(per_class_acc[~torch.isnan(per_class_acc)].max()):.3f}, '
          f'mean={float(per_class_acc[~torch.isnan(per_class_acc)].mean()):.3f}')

    # Save probe
    W = probe.weight.detach().cpu()  # (n_classes, d_model)
    b = probe.bias.detach().cpu()    # (n_classes,)
    os.makedirs(os.path.dirname(args.out_path) or '.', exist_ok=True)
    torch.save({
        'W': W,
        'b': b,
        'seen': seen,
        'n_classes': n_classes,
        'd_model': d_model,
        'meta': {
            'features_path': args.features_path,
            'seed': args.seed,
            'lr': args.lr,
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'weight_decay': args.weight_decay,
            'device': str(device),
            'N_train': N,
            'final_train_loss': final_loss,
            'final_train_acc': final_acc,
            'per_class_acc': per_class_acc,
            'per_class_count': count,
        },
    }, args.out_path)
    print(f'[train_linear_probe] wrote {args.out_path}')


if __name__ == '__main__':
    main()
