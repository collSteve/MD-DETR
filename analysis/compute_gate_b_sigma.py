"""Compute Gate B's ±2σ tolerance from the Pre-1b variance-baseline D1 runs.

Part of Path B Phase 1. See docs/MD-DETR/phase_1_conceptual_plan.md §3.2 and
plan file Phase 1.5.

The Pre-1b measurement runs D1 twice on the existing DP baseline checkpoint with
different seeds (via --d1_seed). This script loads the two resulting
per_layer_prototypes.pt files, computes layer-5 off-diagonal-cosine mean for
each, and reports the absolute difference as a conservative estimate of σ.
Gate B's tolerance = ±2σ around the DP baseline value (0.6138).

Usage:
    conda run -n MD-DETR python analysis/compute_gate_b_sigma.py \\
        --run_a /ubc/.../diagnostic_per_layer_separability_seed43/per_layer_prototypes.pt \\
        --run_b /ubc/.../diagnostic_per_layer_separability_seed44/per_layer_prototypes.pt \\
        [--layer 5]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


def off_diag_mean_cosine(prototypes: torch.Tensor, seen: torch.Tensor) -> float:
    """Pairwise off-diagonal cosine mean for a (n_classes, d_model) prototype matrix."""
    P = prototypes[seen]
    if P.shape[0] < 2:
        return float('nan')
    Pn = torch.nn.functional.normalize(P, dim=-1)
    M = Pn @ Pn.T
    n = P.shape[0]
    eye = torch.eye(n, dtype=torch.bool)
    return float(M[~eye].mean().item())


def load_layer_cosine(path: Path, layer: int) -> float:
    data = torch.load(path, weights_only=False, map_location='cpu')
    per_layer = data['per_layer']
    if layer not in per_layer:
        raise ValueError(f'Layer {layer} not in {path}; available: {sorted(per_layer.keys())}')
    return off_diag_mean_cosine(per_layer[layer]['prototypes'], per_layer[layer]['seen'])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run_a', type=Path, required=True,
                    help='Path to first seed run per_layer_prototypes.pt')
    ap.add_argument('--run_b', type=Path, required=True,
                    help='Path to second seed run per_layer_prototypes.pt')
    ap.add_argument('--layer', type=int, default=5,
                    help='Decoder layer index to measure (default 5 = final).')
    ap.add_argument('--dp_baseline_cosine', type=float, default=0.6138,
                    help='Reference DP baseline value at this layer (default 0.6138 for layer 5).')
    args = ap.parse_args()

    for p in (args.run_a, args.run_b):
        if not p.exists():
            print(f'ERROR: file not found: {p}', file=sys.stderr)
            return 2

    cos_a = load_layer_cosine(args.run_a, args.layer)
    cos_b = load_layer_cosine(args.run_b, args.layer)
    diff = abs(cos_a - cos_b)

    # Two independent measurements: diff ≈ √2 · σ, so σ ≈ diff / √2.
    # Conservative (and the worker's suggestion): report σ = diff and tolerance = 2σ = 2·diff.
    # That double-counts a bit but errs on the side of accepting "unchanged."
    sigma_conservative = diff
    tolerance = 2 * sigma_conservative
    baseline = args.dp_baseline_cosine

    print(f'# Gate B variance-baseline analysis (layer {args.layer})')
    print(f'# Run A (seed A): off-diag mean cosine = {cos_a:.4f}')
    print(f'# Run B (seed B): off-diag mean cosine = {cos_b:.4f}')
    print(f'# |diff| = {diff:.4f}')
    print(f'# Conservative σ = {sigma_conservative:.4f}')
    print(f'# Gate B tolerance = ±2σ = ±{tolerance:.4f}')
    print()
    print('# Gate B decision bands (applied at Phase 1.5 with the LoRA-trained D1 run):')
    print(f'#   Within [{baseline - tolerance:.4f}, {baseline + tolerance:.4f}]: features stable → ship v1 without SupCon')
    print(f'#   Above {baseline + tolerance:.4f}                             : collapse worsened → add SupCon in v2')
    print(f'#   Below {baseline - tolerance:.4f}                             : LoRA un-collapses → paper finding, no SupCon')
    return 0


if __name__ == '__main__':
    sys.exit(main())
