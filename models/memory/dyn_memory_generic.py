import abc
from typing import Dict, Sequence
import torch
import torch.nn as nn

from models.memory.base_prompt import BasePromptModule

class MemoryUnit(nn.Module, abc.ABC):
    """A set of learnable parameters *for a single slot*."""
    @property
    @abc.abstractmethod
    def output_spec(self) -> Dict[str, torch.Size]:
        """e.g. {'P': (L, D), 'K': (D,), 'A': (D,)}"""

    @abc.abstractmethod
    def forward(self) -> Dict[str, torch.Tensor]:
        """Return the tensors of this slot."""


class MemoryGroupGeneric(nn.Module):
    def __init__(self, unit_cls: type[MemoryUnit], num_units: int, **unit_kw):
        super().__init__()
        self.units = nn.ModuleList(
            unit_cls(**unit_kw) for _ in range(num_units)
        )
    
    def reset_parameters(self):
        """Re-initializes every p/k/a in this task's memory."""
        for p in self.p_list:
            if self.ortho:
                nn.init.orthogonal_(p)
            else:
                nn.init.uniform_(p)
        for k in self.k_list:
            if self.ortho:
                nn.init.orthogonal_(k.unsqueeze(0))
            else:
                nn.init.uniform_(k)
        for a in self.a_list:
            if self.ortho:
                nn.init.orthogonal_(a.unsqueeze(0))
            else:
                nn.init.uniform_(a)

    def add_units(self, num_units=1):

        for _ in range(num_units):
            p = make_memory_unit(self.e_p_length, self.emb_d, ortho=self.ortho)
            k, a = make_key_attn_unit(self.key_d, ortho=self.ortho)

            self.p_list.append(nn.Parameter(p))
            self.k_list.append(nn.Parameter(k))
            self.a_list.append(nn.Parameter(a))

    def remove_units(self, num_units=1):
        for _ in range(num_units):
            self.p_list.pop(-1)
            self.k_list.pop(-1)
            self.a_list.pop(-1)

    def forward(self):
        # stack into tensors of shape (U, length, emb_d) or (U, key_d)
        P = torch.stack(list(self.p_list), dim=0)
        K = torch.stack(list(self.k_list), dim=0)
        A = torch.stack(list(self.a_list), dim=0)
        return P, K, A
    


class DynamicMemoryGeneric(BasePromptModule):
    def __init__(self, emb_d, key_d, default_units, e_p_length, 
                 e_layers: Sequence[int] = [0,1,2,3,4,5], local_query: bool = False,
                 ortho_mu=0.0,
                 debug=False, debug_probe=None):
        super().__init__()
        self.emb_d, self.key_d, self.e_p_length = emb_d, key_d, e_p_length
        self.ortho_mu = ortho_mu
        self.default_units = default_units

        self.local_query = local_query

        # if local_query:
        self.query_tf = nn.Sequential(
                        nn.Linear(300*256, 300),
        )

        self.layer_memories = nn.ModuleDict({
            str(layer): nn.ModuleDict()
            for layer in e_layers
        })
