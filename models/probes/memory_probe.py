# probes/memory_probe.py
from dataclasses import dataclass
import torch, pickle, pathlib, copy
import pytorch_lightning as pl

@dataclass
class MemoryRecord:
    epoch: int
    img_id: int         
    task_id: int
    layer: int
    # K_norm: torch.Tensor # (U, d)
    weights: torch.Tensor# (U) or (B, U)
    P: torch.Tensor      # summarised or full
    class_labels: list[str]
    true_task_id: str | None = None

@dataclass
class ProposalMemoryRecord:
    """A raw record for proposal-based memory, storing the full N-dimensional tensors."""
    epoch: int
    img_id: int         
    task_id: int
    layer: int
    weights: torch.Tensor # Shape: (B, N, U)
    P: torch.Tensor       # Shape: (B, N, L, D)
    class_labels: list[str]
    true_task_id: str | None = None

@dataclass
class ProcessedMemRecord(MemoryRecord):
    weights: torch.Tensor # (U) only
    P: torch.Tensor       # no batch dim


@dataclass
class DebugAttribute:
    true_task_id: str | None = None

class MemoryProbe(pl.Callback):
    def __init__(self, out_dir, max_in_ram=8):
        self.records = []
        self.out_dir = pathlib.Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.max_in_ram = max_in_ram

    # called by the prompt modules
    def __call__(self, record):
        """Accepts either a MemoryRecord or a ProposalMemoryRecord."""
        self.records.append(record)

    # Lightning hook
    def on_validation_start(self, trainer, pl_module):
        self._epoch = trainer.current_epoch
        self.rank   = trainer.global_rank
        self.tag = getattr(self, "tag", None)

    def on_validation_end(self, trainer, pl_module):
        self._flush()


    def _flush(self):
        if not self.records:
            return
        
        # Use a different filename for proposal records to easily distinguish them.
        is_proposal_data = isinstance(self.records[0], ProposalMemoryRecord)
        prefix = "prop_mem" if is_proposal_data else "mem"
        
        fname = self.out_dir / f"{prefix}_epoch{self._epoch:03d}_tag{self.tag}_rank{self.rank}.pkl"
        with fname.open("wb") as f:
            pickle.dump(copy.deepcopy(self.records), f)
        self.records.clear()