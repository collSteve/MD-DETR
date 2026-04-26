"""Inspect prototype-bank quality: pairwise cosine + most-confusable pairs.

For each `prototypes.pt` file, prints:
  - Distribution of off-diagonal pairwise cosine (mean / std / min / max).
  - Average and worst nearest-other-class similarity.
  - Top-3 most confusable class pairs (highest cosine).
  - Top-3 least confusable pairs (lowest cosine).

Used for the April 2026 prototype-classifier diagnostic to characterise the
collapsed-feature-space finding (off-diagonal cosine mean ~0.62, nearest-other
class ~0.83 — see docs/MD-DETR/prototype_diagnostic.md §"Root cause").

Usage:
    conda run -n MD-DETR python analysis/prototype_quality.py
"""

import torch
import torch.nn.functional as F


PROTOTYPES = [
    ('DP',  '/ubc/cs/research/shield/projects/kren04/MD_DETR_runs/extract_prototypes_dynamic/prototypes.pt'),
    ('SQK', '/ubc/cs/research/shield/projects/kren04/MD_DETR_runs/extract_prototypes_simpleqk/prototypes.pt'),
]

# COCO 80-class index -> name. Mirrors the task_label2name dict produced by
# datasets/coco_hug.py::task_info_coco; reproduced inline so this script has
# no project import dependencies.
TASK_LABEL2NAME = {
    0: 'airplane', 1: 'bicycle', 2: 'bird', 3: 'boat', 4: 'bus', 5: 'car',
    6: 'cat', 7: 'cow', 8: 'dog', 9: 'horse', 10: 'motorcycle', 11: 'sheep',
    12: 'train', 13: 'elephant', 14: 'bear', 15: 'zebra', 16: 'giraffe',
    17: 'truck', 18: 'person', 19: 'traffic light', 20: 'fire hydrant',
    21: 'stop sign', 22: 'parking meter', 23: 'bench', 24: 'chair',
    25: 'dining table', 26: 'potted plant', 27: 'backpack', 28: 'umbrella',
    29: 'handbag', 30: 'tie', 31: 'suitcase', 32: 'microwave', 33: 'oven',
    34: 'toaster', 35: 'sink', 36: 'refrigerator', 37: 'bed', 38: 'toilet',
    39: 'couch', 40: 'frisbee', 41: 'skis', 42: 'snowboard', 43: 'sports ball',
    44: 'kite', 45: 'baseball bat', 46: 'baseball glove', 47: 'skateboard',
    48: 'surfboard', 49: 'tennis racket', 50: 'banana', 51: 'apple',
    52: 'sandwich', 53: 'orange', 54: 'broccoli', 55: 'carrot', 56: 'hot dog',
    57: 'pizza', 58: 'donut', 59: 'cake', 60: 'laptop', 61: 'mouse',
    62: 'remote', 63: 'keyboard', 64: 'cell phone', 65: 'book', 66: 'clock',
    67: 'vase', 68: 'scissors', 69: 'teddy bear', 70: 'hair drier',
    71: 'toothbrush', 72: 'wine glass', 73: 'cup', 74: 'fork', 75: 'knife',
    76: 'spoon', 77: 'bowl', 78: 'tv', 79: 'bottle',
}


def analyze(name, path):
    d = torch.load(path, weights_only=False, map_location='cpu')
    P = d['prototypes']                       # (n_classes, d_model)
    Pn = F.normalize(P, dim=-1)
    M = Pn @ Pn.T                              # (n_classes, n_classes) cosine
    n = P.shape[0]
    eye = torch.eye(n).bool()
    off = M[~eye]
    diag = M[eye]

    print(f'\n=== {name} prototype pairwise cosine ===')
    print(f'  Diag (should be 1.0): mean={diag.mean():.4f}')
    print(f'  Off-diag cosine:  mean={off.mean():.4f}  std={off.std():.4f}'
          f'  min={off.min():.4f}  max={off.max():.4f}')

    # Each class -> nearest other class
    M_ = M.clone()
    M_[eye] = -2.0
    topk = M_.topk(3, dim=-1).values
    print(f'  Each class to top-1 other:  mean={topk[:, 0].mean():.4f}  max={topk[:, 0].max():.4f}')
    print(f'  Each class to top-3 others: mean={topk[:, 2].mean():.4f}')

    # Most/least confusable pairs (upper triangle)
    iu = torch.triu(torch.ones(n, n), diagonal=1).bool()
    pair_sim = M[iu]
    pair_idx = iu.nonzero()
    order = pair_sim.argsort(descending=True)

    print('  Top-3 most-similar pairs:')
    for o in order[:3]:
        i, j = pair_idx[o].tolist()
        print(f'    {TASK_LABEL2NAME[i]:20s} ~ {TASK_LABEL2NAME[j]:20s}  cos={pair_sim[o]:.4f}')
    print('  Bottom-3 least-similar pairs:')
    for o in order[-3:]:
        i, j = pair_idx[o].tolist()
        print(f'    {TASK_LABEL2NAME[i]:20s} ~ {TASK_LABEL2NAME[j]:20s}  cos={pair_sim[o]:.4f}')


def main():
    for name, path in PROTOTYPES:
        analyze(name, path)


if __name__ == '__main__':
    main()
