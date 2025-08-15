
from typing import Sequence
import torch
import torch.nn as nn

from models.memory.dyn_memory import DynamicPrompt, TaskMemory

class FocusedDynamicPrompt(DynamicPrompt):
    """
    An enhanced version of DynamicPrompt that incorporates a softmax and a 'focus'
    parameter to sharpen the memory selection for image-level queries.
    """
    def __init__(self, *args, focus: float = 10.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.focus = focus
        print(f"Initialized FocusedDynamicPrompt with focus={self.focus}")

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
            # This logic for handling proposal-level queries to generate an image-level
            # query remains the same as in the base DynamicPrompt.
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

        a_q    = torch.einsum('bd,ud->bud', x_query, A)
        nK     = nn.functional.normalize(K, dim=1)
        q_norm = nn.functional.normalize(a_q, dim=2)
        raw_similarities = torch.einsum('bud,ud->bu', q_norm, nK)

        # --- NEW LOGIC: Apply Focus and Softmax ---
        # Sharpen the distribution and normalize weights to sum to 1
        focused_weights = torch.softmax(raw_similarities * self.focus, dim=-1)
        # --- END OF NEW LOGIC ---

        P_     = torch.einsum('bu,uld->bld', focused_weights, P)

        mid = self.e_p_length // 2
        Ek, Ev = P_[:, :mid, :], P_[:, mid:, :]

        # Debug hook remains the same, but now logs the focused_weights
        if self.debug and self.debug_probe is not None:
            object_classes = self.active_classes_batch[0]
            self.debug_probe(
                layer=l,
                task_id=task_id,
                class_labels=object_classes,
                weights=focused_weights.detach().cpu().clone(),
                P=P_.detach().cpu().clone(),
                true_task_id=self.debug_attribute.true_task_id if self.debug_attribute else None,
                img_id=self.image_ids[0].cpu().clone().numpy()[0] if self.image_ids is not None else None,
            )

        return [Ek, Ev], 0, x_block
