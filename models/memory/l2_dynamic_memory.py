
from typing import Sequence
import torch
import torch.nn as nn

from models.memory.dyn_memory import DynamicPrompt

class L2DynamicPrompt(DynamicPrompt):
    """
    A version of DynamicPrompt that uses L2 (Euclidean) distance for memory
    retrieval with an image-level query, instead of cosine similarity.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print("Initialized L2DynamicPrompt")

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
            # Aggregate proposal queries into a single image-level query
            B = x_query.size(0)
            flat = x_query.view(B, -1)
            qwt = self.query_tf(flat)
            x_query = x_query * qwt.unsqueeze(-1)
            x_query = x_query.sum(dim=1)

        Ps, Ks, As = [], [], []
        for tid_s, mem in self.layer_memories[layer].items():
            Pk, Kk, Ak = mem.forward()
            if not train or (task_id is not None and tid_s != str(task_id)):
                Pk, Kk, Ak = Pk.detach(), Kk.detach(), Ak.detach()
            Ps.append(Pk)
            Ks.append(Kk)
            As.append(Ak)

        P = torch.cat(Ps, dim=0)
        K = torch.cat(Ks, dim=0)
        A = torch.cat(As, dim=0)

        # --- MODIFIED LOGIC: L2 Distance-based Retrieval ---

        # Stage 1: Query Projection via A
        # q_transformed shape: (B, U, D_q)
        q_transformed = torch.einsum('bd,ud->bud', x_query, A)

        # Stage 2: L2 Distance Calculation
        # K shape: (U, D_q) -> K.unsqueeze(0) shape: (1, U, D_q)
        # distances shape: (B, U)
        distances = torch.norm(q_transformed - K.unsqueeze(0), p=2, dim=2)

        # Stage 3: Convert distance to weights
        weights = torch.exp(-distances)
        
        # --- END OF MODIFIED LOGIC ---

        P_ = torch.einsum('bu,uld->bld', weights, P)

        mid = self.e_p_length // 2
        Ek, Ev = P_[:, :mid, :], P_[:, mid:, :]

        # Debug hook (remains compatible)
        if self.debug and self.debug_probe is not None:
            object_classes = self.active_classes_batch[0]
            self.debug_probe(
                layer=l,
                task_id=task_id,
                class_labels=object_classes,
                weights=weights.detach().cpu().clone(),
                P=P_.detach().cpu().clone(),
                true_task_id=self.debug_attribute.true_task_id if self.debug_attribute else None,
                img_id=self.image_ids[0].cpu().clone().numpy()[0] if self.image_ids is not None else None,
            )

        return [Ek, Ev], 0, x_block
