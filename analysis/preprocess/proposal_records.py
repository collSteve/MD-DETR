
from typing import List
from models.probes.memory_probe import ProposalMemoryRecord, ProcessedMemRecord

def flatten_proposal_records(
    records: List[ProposalMemoryRecord],
    sampling_strategy: str = 'all' # In the future, we could add 'random', 'top_k', etc.
) -> List[ProcessedMemRecord]:
    """
    Converts a list of raw ProposalMemoryRecords into a flattened list of 
    ProcessedMemRecords, making it compatible with existing analysis tools.
    
    Each proposal is treated as a separate record.
    """
    processed_records = []
    for rec in records:
        batch_size = rec.weights.shape[0]
        num_proposals = rec.weights.shape[1]

        for b in range(batch_size):
            # The image_id might be a list if the original batch size was > 1
            img_id = rec.img_id[b] if isinstance(rec.img_id, list) else rec.img_id
            class_labels = rec.class_labels[b] if isinstance(rec.class_labels, list) else rec.class_labels

            for n in range(num_proposals):
                # For now, we treat each proposal as an independent data point.
                # We can add more sophisticated sampling here later if needed.
                processed_records.append(
                    ProcessedMemRecord(
                        epoch=rec.epoch,
                        img_id=f"{img_id}_p{n}", # Append proposal index to img_id for traceability
                        task_id=rec.task_id,
                        layer=rec.layer,
                        weights=rec.weights[b, n, :].cpu(),  # Shape: (U,)
                        P=rec.P[b, n, :, :].cpu(),          # Shape: (L, D)
                        class_labels=class_labels,
                        true_task_id=rec.true_task_id
                    )
                )

    return processed_records
