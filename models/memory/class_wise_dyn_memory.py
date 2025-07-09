from typing import Sequence
import torch
import torch.nn as nn

from models.memory.dyn_memory import DynamicPrompt
from models.prompt import Prompt, PromptParam

class ClassWiseDynamicPrompt(DynamicPrompt):
    def __init__(self, emb_d, key_d, default_units, e_p_length, 
                 e_layers: Sequence[int] = [0,1,2,3,4,5], local_query: bool = False,
                 ortho_mu=0.0,
                 debug=False, debug_probe=None):
        super().__init__(emb_d, key_d, default_units, e_p_length, 
                         e_layers=e_layers, local_query=local_query, ortho_mu=ortho_mu)
        
        self.active_classes_batch = []

        #debug
        self.active_classes_batch = [] # for debug
        self.debug = debug
        self.debug_probe = debug_probe   # callable | None
        self.debug_attribute = None  # for debug
    
    
    def initialize_for_task(self, task_id: int, object_classes: Sequence[str]):
        for obj_class in object_classes:
            self.initialize_for_subcategory(obj_class, self.default_units)

    def set_activate_classes(self, object_classes_batch: Sequence[Sequence[str]]):
        self.active_classes_batch = object_classes_batch
        
    def forward(self,
                x_query: torch.Tensor,
                l: int,
                x_block: torch.Tensor,
                train: bool = False,
                task_id: int = None,
                object_classes_batched: Sequence[Sequence[str]] = None,):
        layer = str(l)
        if layer not in self.layer_memories:
            return None, 0, x_block
        

        if x_query.dim() != 2:
            if self.local_query:
                # flatten everything except batch
                B = x_query.size(0)
                flat = x_query.view(B, -1)                   # (B, query_in_dim)
                qwt = self.query_tf(flat)                    # (B, query_out_dim)
                x_query = x_query * qwt.unsqueeze(-1)        # broadcast back
                
            x_query = x_query.sum(dim=1)

        
        object_classes = self.active_classes_batch[0] # Only work for batchsize = 1; TODO
        if object_classes is None:
            print("Warning: object_classes_batched is None")
        if len(object_classes) == 0:
            print("Warning: object_classes_batched is empty")

        # print(f"object_classes in forward: {object_classes}")

        Ps, Ks, As = [], [], []
        for class_name, mem in self.layer_memories[layer].items():
            Pk, Kk, Ak = mem.forward()
            # only freeze non-current‐task units during training
            if (object_classes is not None) and (class_name not in object_classes):
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

        # debug

        if self.debug and self.debug_probe is not None:
            self.debug_probe(
                layer=l,
                task_id=task_id,      # what engine believes
                class_labels=object_classes,
                # K_norm=nK.detach(),   # (U, D)
                weights=weights.detach(),  # (B, U)
                P=P_.detach(),        # (B, L, D)
                true_task_id=self.debug_attribute.true_task_id if self.debug_attribute else None,
                img_id=self.image_ids[0] if self.image_ids is not None else None, # TODO: handle batch size > 1
            )

        return [Ek, Ev], 0, x_block
        