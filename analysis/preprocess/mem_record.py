
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import torch
from models.probes.memory_probe import MemoryRecord, ProcessedMemRecord

      # summarised or full (B, D) or (B, L, D)


def flatten_records(records: List[MemoryRecord]
                    ) -> Tuple[List[Dict[str, Any]],
                               List[torch.Tensor],  # weights (U,)
                               List[torch.Tensor],  # P (...) per image
                               List[torch.Tensor]]: # K_norm (U,D) per image
    meta, w_lst, P_lst= [], [], []
    for rec in records:
        B = rec.weights.shape[0]  # batch size
        for b in range(B):
            meta.append({
                "epoch":    rec.epoch,
                "img_id":   rec.img_id if B == 1 else f"{rec.img_id}_{b}",
                "task_id":  rec.task_id,
                "layer":    rec.layer,
                "true_task_id": rec.true_task_id,
            })
            w_lst.append(rec.weights[b].cpu())
            P_lst.append(rec.P[b].cpu())
    return meta, w_lst, P_lst

def flatten2processed_mem_records(
    records: List[MemoryRecord]
) -> List[ProcessedMemRecord]:
    """remove batch dimension from weights and P, and return ProcessedMemRecord"""
    processed_records = []
    for rec in records:
        B = rec.weights.shape[0]
        for b in range(B):
            processed_records.append(
                ProcessedMemRecord(
                    epoch=rec.epoch,
                    img_id=rec.img_id,
                    task_id=rec.task_id,
                    layer=rec.layer,
                    weights=rec.weights[b].cpu(),  # (U,)
                    P=rec.P[b].cpu(),              # (L, D) or (D,)
                    class_labels=rec.class_labels,
                    true_task_id=rec.true_task_id
                )
            )

    return processed_records