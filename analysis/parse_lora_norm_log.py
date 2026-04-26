"""Parse Lightning CSV training logs for LoRA-norm-ratio trajectories + Gate A check.

Part of Path B Phase 1. See docs/MD-DETR/phase_1_conceptual_plan.md §5 Gate A and
docs/MD-DETR/phase_1_implementation_plan / mellow-roaming-cocoa.md Phase 1.3.

Usage:
    conda run -n MD-DETR python analysis/parse_lora_norm_log.py \\
        --csv /ubc/.../Task_1/lightning_logs/version_0/metrics.csv \\
        [--threshold 1.5] [--n_layers 6]

Output: prints per-layer final-epoch ratios + PASS/FAIL verdict on Gate A threshold
(default 1.5 — LoRA-norm must have grown >= 1.5x its task-start value per layer).

The script reads the `lora_norm_ratio_L{i}` columns logged by engine.training_step
(at batch_idx % 100 == 0 during training). Final-epoch ratio = last non-null value
observed in the column.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Optional


def parse_csv(path: Path) -> Dict[str, List[float]]:
    """Load Lightning CSV, return dict of {column_name: [non-null float values]}."""
    with path.open() as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        cols: Dict[str, List[float]] = {name: [] for name in fieldnames}
        for row in reader:
            for name, value in row.items():
                if value is None or value == '' or name is None:
                    continue
                try:
                    cols[name].append(float(value))
                except ValueError:
                    pass
    return cols


def extract_lora_norm_cols(cols: Dict[str, List[float]]) -> Dict[int, List[float]]:
    """Return {layer_idx: [ratio trajectory]} for every lora_norm_ratio_L* column found."""
    out: Dict[int, List[float]] = {}
    for name, values in cols.items():
        if name.startswith('lora_norm_ratio_L'):
            try:
                idx = int(name[len('lora_norm_ratio_L'):])
            except ValueError:
                continue
            if values:
                out[idx] = values
    return dict(sorted(out.items()))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--csv', type=Path, required=True,
                    help='Path to Lightning metrics.csv (lightning_logs/version_N/metrics.csv)')
    ap.add_argument('--threshold', type=float, default=1.5,
                    help='Gate A LoRA-norm ratio threshold per layer (default 1.5)')
    ap.add_argument('--n_layers', type=int, default=None,
                    help='Expected number of LoRA layers (default: infer from CSV). '
                         'If set, a missing layer counts as FAIL.')
    args = ap.parse_args()

    if not args.csv.exists():
        print(f'ERROR: CSV file not found: {args.csv}', file=sys.stderr)
        return 2

    cols = parse_csv(args.csv)
    lora_cols = extract_lora_norm_cols(cols)

    if not lora_cols:
        print(f'ERROR: no lora_norm_ratio_L* columns found in {args.csv}', file=sys.stderr)
        return 2

    if args.n_layers is not None:
        missing = [i for i in range(args.n_layers) if i not in lora_cols]
        if missing:
            print(f'ERROR: expected {args.n_layers} layers but missing indices {missing}')
            return 2

    print(f'# LoRA norm-ratio trajectories ({args.csv})')
    print(f'# Gate A threshold: {args.threshold:.2f}')
    print(f'# {"layer":>5}  {"n_samples":>9}  {"first":>8}  {"final":>8}  {"max":>8}  {"verdict":>8}')
    print('# ' + '-' * 60)

    all_pass = True
    for idx, traj in lora_cols.items():
        first, final, max_ = traj[0], traj[-1], max(traj)
        verdict = 'PASS' if final >= args.threshold else 'FAIL'
        if verdict == 'FAIL':
            all_pass = False
        print(f'  {idx:>5}  {len(traj):>9}  {first:>8.3f}  {final:>8.3f}  {max_:>8.3f}  {verdict:>8}')

    print()
    print('=' * 60)
    if all_pass:
        print(f'Gate A LoRA-norm check: PASS (all layers ratio >= {args.threshold:.2f}).')
        print('LoRA actually used capacity during training — §2.8 risk not triggered.')
    else:
        print(f'Gate A LoRA-norm check: FAIL (one or more layers below {args.threshold:.2f}).')
        print('LoRA is under-trained at some depths. Escalate per §2.8 escalation ladder:')
        print('  1. Bump --lora_lr_asymmetry_factor to 5.0 and re-train.')
        print('  2. If still dead: add L2 on DP memory new slot.')
        print('  3. If still dead: bail to E1 ablation (use_prompts=0).')
    print('=' * 60)
    return 0 if all_pass else 1


if __name__ == '__main__':
    sys.exit(main())
