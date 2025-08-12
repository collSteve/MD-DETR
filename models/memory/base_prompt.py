from abc import ABC, abstractmethod
import torch
import torch.nn as nn

class BasePromptModule(nn.Module, ABC):
    def __init__(self):
        super().__init__()
        self.image_ids = None

    @abstractmethod
    def initialize_for_task(self, task_id: int, **kwargs):
        """Called once per new task, after module.to(device)."""
        ...

    @abstractmethod
    def forward(self, x_query, layer_idx, x_block, train=False, task_id=None, **kwargs):
        """Injects prompts into the decoder; returns ([Ek, Ev], loss, x_block)."""
        ...

    @abstractmethod
    def set_activate_classes(self, object_classes_batch):
        """Sets the active classes for the current batch."""
        ...

    def set_image_ids(self, image_ids):
        """Sets the image IDs for the current batch."""
        self.image_ids = image_ids

    # @abstractmethod
    # def on_forward_end(self, layer, task_id, class_labels, K_norm, weights, P):
    #     """Called at the end of each forward pass."""
    #     ...