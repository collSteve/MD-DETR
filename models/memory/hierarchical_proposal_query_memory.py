from typing import Sequence
import torch
import torch.nn as nn

from models.memory.base_prompt import BasePromptModule
from models.prompt import Prompt, PromptParam


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


class TaskMemory_Hierarchical_AK(nn.Module):
    def __init__(self, emb_d, key_d, init_units, e_p_length, ortho=False):
        super().__init__()
        self.emb_d, self.key_d, self.e_p_length = emb_d, key_d, e_p_length
        self.ortho = ortho

        self.p_list = nn.ParameterList()
        self.k_list = nn.ParameterList()
        self.a_list = nn.ParameterList()

        # task‑level gating parameters
        self.k_gate = nn.Parameter(torch.empty(key_d))  # (D_k,)
        self.a_gate = nn.Parameter(torch.empty(key_d))  # (D_k,)

        self.add_units(init_units)

    def reset_gate_parameters(self):
        if self.ortho:
            nn.init.orthogonal_(self.k_gate.unsqueeze(0))
            nn.init.orthogonal_(self.a_gate.unsqueeze(0))
        else:
            nn.init.uniform_(self.k_gate)
            nn.init.uniform_(self.a_gate)
    
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
        
        self.reset_gate_parameters()

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

    def gate_params(self):
        return self.k_gate, self.a_gate


class DynamicPrompt_Hierarchical_AK(BasePromptModule):
    def __init__(self, emb_d, key_d, default_units, e_p_length, 
                 e_layers: Sequence[int] = [0,1,2,3,4,5], local_query: bool = False,
                 ortho_mu=0.0,
                 debug=False, debug_probe=None):
        super().__init__()
        self.emb_d, self.key_d, self.e_p_length = emb_d, key_d, e_p_length
        self.ortho_mu = ortho_mu
        self.default_units = default_units

        self.local_query = local_query

        self.layer_memories = nn.ModuleDict({
            str(layer): nn.ModuleDict()
            for layer in e_layers
        })

        self.task_count = 0 # not used

        #debug
        self.active_classes_batch = [] # for debug
        self.debug = debug
        self.debug_probe = debug_probe   # callable | None
        self.debug_attribute = None  # for debug
    
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


    def initialize_for_subcategory(self, subcategory_name: str, num_memory_units:int = 1):
        for layer in self.layer_memories.keys():
            if subcategory_name in self.layer_memories[layer]:
                print(f"Warning: Subcategory {subcategory_name} already exists in layer {layer}.")
                continue

            self.layer_memories[layer][subcategory_name] = TaskMemory_Hierarchical_AK(
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
        self.initialize_for_subcategory(tid, self.default_units)

        self.current_task = tid

    def _unit_level_prompt(self, q_input: torch.Tensor, P: torch.Tensor, K: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        """Compute per proposal prompt inside **one** task.

        Args:
            q_input: (B,N,D_q)  - N object queries
            P: (U,L,D)
            K, A: (U,D_q)
        Returns:
            prompt_task: (B,N,L,D)
        """
        # project query through task's unit‑level A vectors
        a_q = torch.einsum("bnd,ud->bnud", q_input, A)  # (B,N,U,D_q)
        nK = nn.functional.normalize(K, dim=1)
        q_norm = nn.functional.normalize(a_q, dim=3)
        w = torch.einsum("bnud,ud->bnu", q_norm, nK)    # (B,N,U)
        prompt_task = torch.einsum("bnu,uld->bnld", w, P)
        return prompt_task


    def forward(self,
            x_query: torch.Tensor,      # (B, N, p)
            l: int,
            x_block: torch.Tensor,
            train: bool = False,
            task_id: int = None):
        layer = str(l)
        if layer not in self.layer_memories:
            return None, 0, x_block

        B, N, _ = x_query.shape
        q_input = x_query

        prompt_tasks = []  # (B,N,L,D) per task
        score_tasks = []   # (B,N)     per task

        for tid_s, mem in self.layer_memories[layer].items():
            # detach if not training this task
            freeze = (not train) or (task_id is not None and tid_s != str(task_id))

            # (1) unit‑level aggregation
            P_u, K_u, A_u = mem.forward()
            if freeze:
                P_u, K_u, A_u = P_u.detach(), K_u.detach(), A_u.detach()
            prompt_task = self._unit_level_prompt(q_input, P_u, K_u, A_u)
            prompt_tasks.append(prompt_task)

            # (2) task‑level gating score
            k_gate, a_gate = mem.gate_params()
            if freeze:
                k_gate, a_gate = k_gate.detach(), a_gate.detach()

            # mask query
            masked_q = q_input * a_gate  # broadcast (B,N,D_q) * (D_q,) → (B,N,D_q)
            # dot product with k_gate
            s_task = torch.einsum("bnd,d->bn", masked_q, k_gate)  # (B,N)
            score_tasks.append(s_task)

        # stack along task axis (T)
        P_stack = torch.stack(prompt_tasks, dim=2)   # (B,N,T,L,D)
        S_stack = torch.stack(score_tasks, dim=2)    # (B,N,T)
        g = torch.softmax(S_stack, dim=2)            # (B,N,T)

        # final prompt aggregation
        P_final = torch.einsum("bnt,bntld->bnld", g, P_stack)  # (B,N,L,D)

        # split into Ek / Ev as before
        mid = P_final.shape[2] // 2  # L//2
        Ek = P_final[:, :, :mid, :].reshape(B, N * mid, self.emb_d)
        Ev = P_final[:, :, mid:, :].reshape(B, N * mid, self.emb_d)

        return [Ek, Ev], 0, x_block
    

    def set_activate_classes(self, object_classes_batch: Sequence[Sequence[str]]):
        self.active_classes_batch = object_classes_batch




