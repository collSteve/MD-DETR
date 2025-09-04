
import os
import pickle
from dataclasses import dataclass
from typing import List
import torch
import pytorch_lightning as pl
import copy

@dataclass
class QueryRecord:
    """
    Stores the object queries and associated metadata for a single image
    from a single validation step.
    """
    epoch: int
    image_id: int
    task_id: int  # The task_id of the training step
    
    # Shape: [num_queries, hidden_dim]
    object_queries: torch.Tensor

    # List of ground-truth class IDs present in the image
    gt_class_ids: List[int]

class QueryProbe(pl.Callback):
    """
    A PyTorch Lightning Callback for collecting object query vectors during validation
    for later analysis.
    """
    def __init__(self, out_dir: str):
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)
        self.records: List[QueryRecord] = []
        self.tag: Optional[str] = None

    def __call__(self, record: QueryRecord):
        # Move data to CPU and detach to save memory and prevent graph issues
        record.object_queries = record.object_queries.detach().cpu()
        self.records.append(record)

    def on_validation_start(self, trainer, pl_module):
        self._epoch = trainer.current_epoch
        self.rank   = trainer.global_rank
        # The tag is now set externally by the main script

    def on_validation_end(self, trainer, pl_module):
        self._flush()

    def _flush(self):
        if not self.records:
            return

        tag_str = f"_tag_{self.tag}" if self.tag else ""
        fname = os.path.join(self.out_dir, f"query_data_epoch{self._epoch:03d}{tag_str}_rank{self.rank}.pkl")
        with open(fname, "wb") as f:
            pickle.dump(copy.deepcopy(self.records), f)
        
        print(f"QueryProbe: Saved {len(self.records)} records to {fname}")
        self.records.clear()
