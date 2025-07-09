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

    # called by DynamicPrompt
    def __call__(self, **kwargs):
        # limit amount stored per batch to save RAM
        rec = MemoryRecord(
            epoch=self._epoch,
            img_id=kwargs.get("img_id", -1),
            task_id=kwargs.get("task_id", -1),
            layer=kwargs["layer"],
            # K_norm=kwargs["K_norm"].cpu().clone(),
            weights=kwargs["weights"].cpu().clone(),
            P=kwargs["P"].cpu().clone(),
            class_labels=kwargs.get("class_labels", []),
            true_task_id=kwargs.get("true_task_id", None),
        )
        self.records.append(rec)


        # if len(self.records) >= self.max_in_ram:
        #     self._flush()

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
        
        fname = self.out_dir / f"mem_epoch{self._epoch:03d}_tag{self.tag}_rank{self.rank}.pkl"
        with fname.open("wb") as f:
            pickle.dump(copy.deepcopy(self.records), f)
        self.records.clear()

