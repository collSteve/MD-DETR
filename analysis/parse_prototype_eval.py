"""Parse final_stats.txt for prototype-classifier vs native-class_embed comparison.

Reads up to four MD-DETR experiment runs (DynamicPrompt and SimpleQK, each in
native and prototype-classifier form) and prints two side-by-side tables —
IoU=0.50:0.95 (COCO standard) and IoU=0.50 (PASCAL-style) — plus prototype−native
deltas.

Used for the April 2026 prototype-classifier diagnostic; see
docs/MD-DETR/prototype_diagnostic.md for the methodology + interpretation.

Usage:
    conda run -n MD-DETR python analysis/parse_prototype_eval.py

To swap in different runs, edit the FILES dict at the top of main().
"""

import re
from pathlib import Path

# Default experiment paths used in the April 2026 diagnostic. Override by
# editing in place — kept as constants rather than CLI args for readability.
FILES = {
    'DP_proto':   '/ubc/cs/research/shield/projects/kren04/MD_DETR_runs/eval_prototype_dynamic/final_stats.txt',
    'SQK_proto':  '/ubc/cs/research/shield/projects/kren04/MD_DETR_runs/eval_prototype_simpleqk/final_stats.txt',
    'DP_native':  '/ubc/cs/research/shield/projects/kren04/MD_DETR_runs/train_dynamic_correctness_a/final_stats.txt',
    'SQK_native': '/ubc/cs/research/shield/projects/kren04/MD_DETR_runs/train_proposal_query_memory_simple_qK_mem_u_10_epoch_6/final_stats.txt',
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
        # The label patterns include task IDs (e.g. "Previous Tasks (mAP@P): 123").
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
        print("-" * 140)
        for k in ['DP_native', 'DP_proto', 'SQK_native', 'SQK_proto']:
            _row(k, parsed[k], idx)

    print("\n### mAP@A (IoU=0.50:0.95 / IoU=0.50) where reported")
    for k in ['DP_native', 'DP_proto', 'SQK_native', 'SQK_proto']:
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

    print("\n### DELTAS (prototype - native, in pp)")
    for iou_name, idx in [('IoU=0.50:0.95', 0), ('IoU=0.50    ', 1)]:
        print(f"\n{iou_name}:")
        for name, proto, native in [
            ('DP', 'DP_proto', 'DP_native'),
            ('SQK', 'SQK_proto', 'SQK_native'),
        ]:
            line = f"  {name:4s}: "
            for tid in range(1, 5):
                for metric in ['C', 'P']:
                    pv = parsed[proto].get((tid, metric))
                    nv = parsed[native].get((tid, metric))
                    if pv and nv and pv[idx] is not None and nv[idx] is not None:
                        d = (pv[idx] - nv[idx]) * 100
                        line += f"T{tid}{metric}={d:+5.1f}  "
            print(line)


if __name__ == '__main__':
    main()
