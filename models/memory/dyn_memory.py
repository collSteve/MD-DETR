from typing import Sequence
import torch
import torch.nn as nn

from models.memory.base_prompt import BasePromptModule
from models.prompt import Prompt, PromptParam

class TaskMemory(nn.Module):
    def __init__(self, emb_d, key_d, init_units, e_p_length, ortho=False):
        super().__init__()
        self.emb_d, self.key_d, self.e_p_length = emb_d, key_d, e_p_length
        self.ortho = ortho

        self.p_list = nn.ParameterList()
        self.k_list = nn.ParameterList()
        self.a_list = nn.ParameterList()
        self.add_units(init_units)
    
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


class DynamicPrompt(BasePromptModule):
    def __init__(self, emb_d, key_d, default_units, e_p_length, 
                 e_layers: Sequence[int] = [0,1,2,3,4,5], local_query: bool = False,
                 ortho_mu=0.0):
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


        self.task_count = 0 # not used
    
    def reset_parameters(self):
        """
        Re-initializes every TaskMemory in every layer.
        Call this once _after_ loading the pretrained weights so none of the
        meta-device placeholders remain un-initialized.
        """
        for layer_dict in self.layer_memories.values():
            for mem in layer_dict.values():
                mem.reset_parameters()

    # Not used
    def set_task_id(self, task_id=0):
        self.task_count = task_id

        print('Setting task id : ', task_id)


    def _ensure_task(self, layer: str, task_id: str):
        """Create a TaskMemory if this (layer, task) is new."""
        tid = str(task_id)

        if tid not in self.layer_memories[layer]:
            mem = TaskMemory(
                emb_d=self.emb_d,
                key_d=self.key_d,
                init_units=self.default_units,
                e_p_length=self.e_p_length,
                ortho=(self.ortho_mu > 0)
            )
            # mem = mem.to(device)
            self.layer_memories[layer][tid] = mem

    def initialize_for_subcategory(self, subcategory_name: str, num_memory_units:int = 1):
        for layer in self.layer_memories.keys():
            if subcategory_name in self.layer_memories[layer]:
                print(f"Warning: Subcategory {subcategory_name} already exists in layer {layer}.")
                continue

            self.layer_memories[layer][subcategory_name] = TaskMemory(
                emb_d=self.emb_d,
                key_d=self.key_d,
                init_units=num_memory_units,
                e_p_length=self.e_p_length,
                ortho=self.ortho_mu > 0.0
            )


    def initialize_for_task(self, task_id: int):
        """
        Public hook: create all per-layer memories for `task_id`
        and record it as the current task.
        """
        tid = str(task_id)
        # for layer in self.layer_memories.keys():
        #     self._ensure_task(layer, tid)
        self.initialize_for_subcategory(tid, self.default_units)

        self.current_task = tid


    def grow(self, layer: int, task_id: int, num_units: int = 1):
        L, T = str(layer), str(task_id)
        self._ensure_task(L, T)
        self.layer_memories[L][T].add_units(num_units)

    def shrink(self, layer: int, task_id: int, num_units: int = 1):
        L, T = str(layer), str(task_id)
        self._ensure_task(L, T)
        self.layer_memories[L][T].remove_units(num_units)

    def forward(self,
                x_query: torch.Tensor,
                l: int,
                x_block: torch.Tensor,
                train: bool = False,
                task_id: int = None):
        layer = str(l)
        if layer not in self.layer_memories:
            return None, 0, x_block
        

        if x_query.dim() != 2:
            # flatten everything except batch
            # if self.local_query:
            B = x_query.size(0)
            flat = x_query.view(B, -1)                   # (B, query_in_dim)
            qwt = self.query_tf(flat)                    # (B, query_out_dim)
            x_query = x_query * qwt.unsqueeze(-1)        # broadcast back
            x_query = x_query.sum(dim=1)


        # if task_id is not None:
        #     self._ensure_task(layer, str(task_id)) # init if not existing


        Ps, Ks, As = [], [], []
        for tid_s, mem in self.layer_memories[layer].items():
            Pk, Kk, Ak = mem.forward()
            if not train:
                # during evaluation, detach all parameters
                Pk = Pk.detach()
                Kk = Kk.detach()
                Ak = Ak.detach()
            elif task_id is not None and tid_s != str(task_id):
                # during training, detach all parameters except the current task
                Pk = Pk.detach()
                Kk = Kk.detach()
                Ak = Ak.detach()
            Ps.append(Pk)
            Ks.append(Kk)
            As.append(Ak)

        # concat across the “unit” dimension
        P = torch.cat(Ps, dim=0)  # (U_total, length, emb_d)
        K = torch.cat(Ks, dim=0)  # (U_total, key_d)
        A = torch.cat(As, dim=0)  # (U_total, key_d)

        a_q    = torch.einsum('bd,ud->bud', x_query, A)   # (B, U, key_d)
        nK     = nn.functional.normalize(K, dim=1)        # (U, key_d)
        q_norm = nn.functional.normalize(a_q, dim=2)      # (B, U, key_d)
        weights= torch.einsum('bud,ud->bu', q_norm, nK)    # (B, U)
        P_     = torch.einsum('bu,uld->bld', weights, P)   # (B, length, emb_d)

        # split prefix / suffix
        mid = self.e_p_length // 2
        Ek, Ev = P_[:, :mid, :], P_[:, mid:, :]

        return [Ek, Ev], 0, x_block




def make_memory_unit(length: int,
                     emb_d: int,
                     ortho: bool = False) -> nn.Parameter:
    
    p = nn.Parameter(torch.empty(length, emb_d))
    if ortho:
        nn.init.orthogonal_(p)
    else:
        nn.init.uniform_(p)
    return p

def make_key_attn_unit(key_d: int, ortho: bool = False) -> tuple[nn.Parameter, nn.Parameter]:
    k = nn.Parameter(torch.empty(key_d))
    a = nn.Parameter(torch.empty(key_d))
    if ortho:
        nn.init.orthogonal_(k.unsqueeze(0))
        nn.init.orthogonal_(a.unsqueeze(0))
    else:
        nn.init.uniform_(k)
        nn.init.uniform_(a)
    return k, a