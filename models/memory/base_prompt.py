from abc import ABC, abstractmethod
import torch
import torch.nn as nn

class BasePromptModule(nn.Module, ABC):
    @abstractmethod
    def initialize_for_task(self, task_id: int, **kwargs):
        """Called once per new task, after module.to(device)."""
        ...

    @abstractmethod
    def forward(self, x_query, layer_idx, x_block, train=False, task_id=None, **kwargs):
        """Injects prompts into the decoder; returns ([Ek, Ev], loss, x_block)."""
        ...