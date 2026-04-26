"""Parse final_stats.txt for linear-probe vs native-class_embed vs mean-prototype comparison.

Reads three MD-DETR experiment runs (DP native, DP mean-prototype, DP linear-probe)
and prints side-by-side tables — IoU=0.50:0.95 (COCO standard) and IoU=0.50
(PASCAL-style) — plus (probe − native) and (probe − prototype) deltas.

Used for the April 2026 linear-probe upper-bound diagnostic; see
docs/MD-DETR/linear_probe_diagnostic.md for the methodology + interpretation.
Structurally mirrors analysis/parse_prototype_eval.py — same parser, extra column.

Usage:
    conda run -n MD-DETR python analysis/parse_linear_probe_eval.py

To swap in different runs, edit the FILES dict below.
"""

import re
from pathlib import Path

# Default experiment paths used in the April 2026 diagnostic. Override by
# editing in place — kept as constants rather than CLI args for readability.
FILES = {
    'DP_native': '/ubc/cs/research/shield/projects/kren04/MD_DETR_runs/train_dynamic_correctness_a/final_stats.txt',
    'DP_proto':  '/ubc/cs/research/shield/projects/kren04/MD_DETR_runs/eval_prototype_dynamic/final_stats.txt',
    'DP_probe':  '/ubc/cs/research/shield/projects/kren04/MD_DETR_runs/eval_linear_probe_dp/final_stats.txt',
}


def parse(path):
    """Parse a final_stats.txt into {(task_id, 'C'|'P'|'A'): (ap5095, ap50)}.

    For training-style files (multiple "Current Task" blocks per task, one per
    epoch), keeps the last block — the final-epoch evaluation. Eval-only files
    have a single block per (task, metric) so this is a no-op for them.
    """
    content = Path(path).read_text()
    task_chunks = re.split(r'Evaluating Task (\d+)', content)
    results = {}
    for i in range(1, len(task_chunks), 2):
        tid = int(task_chunks[i])
        body = task_chunks[i + 1]
        sub = re.split(
            r'(Current Task \(mAP@C\): \d+|Previous Tasks \(mAP@P\): \d+|All seen Tasks \(mAP@A\): \d+)',
            body,
        )
        cur_blocks, prev_block, all_block = [], None, None
        for j in range(1, len(sub), 2):
            hdr, blk = sub[j], sub[j + 1]
            if 'Current Task' in hdr:
                cur_blocks.append(blk)
            elif 'Previous Tasks' in hdr:
                prev_block = blk
            elif 'All seen Tasks' in hdr:
                all_block = blk

        def extract(b):
            if b is None:
                return None
            ap5095 = re.findall(
                r'IoU=0\.50:0\.95 \| area=   all \| maxDets=100 \] = ([\d.]+)', b
            )
            ap50 = re.findall(
                r'IoU=0\.50      \| area=   all \| maxDets=100 \] = ([\d.]+)', b
            )
            if not ap5095:
                return None
            return (float(ap5095[0]), float(ap50[0]) if ap50 else None)

        if cur_blocks:
            results[(tid, 'C')] = extract(cur_blocks[-1])
        if prev_block is not None:
            results[(tid, 'P')] = extract(prev_block)
        if all_block is not None:
            results[(tid, 'A')] = extract(all_block)
    return results


def _row(name, r, idx):
    parts = []
    for tid in range(1, 5):
        c = r.get((tid, 'C'))
        p = r.get((tid, 'P'))
        cstr = f"{c[idx]*100:5.1f}" if c and c[idx] is not None else "  -  "
        pstr = f"{p[idx]*100:5.1f}" if p and p[idx] is not None else "  -  "
        parts.append(f"T{tid}C={cstr} T{tid}P={pstr}")
    print(f"  {name:11s}  " + " | ".join(parts))


def main():
    parsed = {name: parse(p) for name, p in FILES.items()}

    for iou_name, idx in [('IoU=0.50:0.95', 0), ('IoU=0.50    ', 1)]:
        print(f"\n### {iou_name}  (values x100, percent)")
        print("-" * 120)
        for k in ['DP_native', 'DP_proto', 'DP_probe']:
            _row(k, parsed[k], idx)

    print("\n### mAP@A (IoU=0.50:0.95 / IoU=0.50) where reported")
    for k in ['DP_native', 'DP_proto', 'DP_probe']:
        vals = []
        for tid in range(1, 5):
            a = parsed[k].get((tid, 'A'))
            if a and a[0] is not None:
                v50 = f"/{a[1]*100:.1f}" if a[1] is not None else ""
                vals.append(f"T{tid}A={a[0]*100:.1f}{v50}")
        if vals:
            print(f"  {k:11s}  " + " | ".join(vals))
        else:
            print(f"  {k:11s}  (none)")

    print("\n### DELTAS — (probe − native) in pp")
    for iou_name, idx in [('IoU=0.50:0.95', 0), ('IoU=0.50    ', 1)]:
        print(f"\n{iou_name}:")
        line = f"  probe-native:  "
        for tid in range(1, 5):
            for metric in ['C', 'P']:
                pv = parsed['DP_probe'].get((tid, metric))
                nv = parsed['DP_native'].get((tid, metric))
                if pv and nv and pv[idx] is not None and nv[idx] is not None:
                    d = (pv[idx] - nv[idx]) * 100
                    line += f"T{tid}{metric}={d:+5.1f}  "
        print(line)

    print("\n### DELTAS — (probe − prototype) in pp")
    for iou_name, idx in [('IoU=0.50:0.95', 0), ('IoU=0.50    ', 1)]:
        print(f"\n{iou_name}:")
        line = f"  probe-proto:   "
        for tid in range(1, 5):
            for metric in ['C', 'P']:
                pv = parsed['DP_probe'].get((tid, metric))
                qv = parsed['DP_proto'].get((tid, metric))
                if pv and qv and pv[idx] is not None and qv[idx] is not None:
                    d = (pv[idx] - qv[idx]) * 100
                    line += f"T{tid}{metric}={d:+5.1f}  "
        print(line)


if __name__ == '__main__':
    main()
