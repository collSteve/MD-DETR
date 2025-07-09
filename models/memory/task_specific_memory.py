from typing import Sequence
import torch
import torch.nn as nn

from models.d_prompt import DynamicMemory


class TaskSpecificMemory(DynamicMemory):
    """
    Different than Dynamic Memory, Each memory units in this class are only used for specific task;
    Memory Units are not shared in between tasks.
    """

    def __init__(self, emb_d, key_d, default_units, e_p_length, 
                 e_layers: Sequence[int] = [0,1,2,3,4,5], local_query: bool = False,
                 ortho_mu=0.0,
                 debug=False, debug_probe=None):
        super().__init__(emb_d, key_d, default_units, e_p_length, 
                         e_layers=e_layers, local_query=local_query,
                         ortho_mu=ortho_mu,
                         debug=debug, debug_probe=debug_probe)

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
            if tid_s == str(task_id):
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
        object_classes = self.active_classes_batch[0] # Only work for batchsize = 1; TODO

        # print(f"image_ids: {self.image_ids[0]}")

        if self.debug and self.debug_probe is not None:
            self.debug_probe(
                layer=l,
                task_id=task_id,      # what engine believes
                class_labels=object_classes,
                # K_norm=nK.detach(),   # (U, D)
                weights=weights.detach().cpu().clone(),  # (B, U)
                P=P_.detach().cpu().clone(),        # (B, L, D)
                true_task_id=self.debug_attribute.true_task_id if self.debug_attribute else None,
                img_id=self.image_ids[0].cpu().clone().numpy()[0] if self.image_ids is not None else None, # TODO: handle batch size > 1
            )

        return [Ek, Ev], 0, x_block