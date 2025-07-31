import torch
import torch.nn as nn
from models.memory.dyn_memory import DynamicPrompt
from models.probes.memory_probe import ProposalMemoryRecord


class ProposalQueryMemory(DynamicPrompt):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Metadata for the current batch, set externally by the engine
        self._current_img_ids = None
        self._current_class_labels = None

    def set_batch_metadata(self, img_ids: list[int], class_labels: list[list[str]]):
        """Receives metadata for the current batch from the training engine."""
        self._current_img_ids = img_ids
        self._current_class_labels = class_labels

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

        Ps, Ks, As = [], [], []
        for tid_s, mem in self.layer_memories[layer].items():
            Pk, Kk, Ak = mem.forward() 
            if not train or (task_id is not None and tid_s != str(task_id)):
                Pk, Kk, Ak = Pk.detach(), Kk.detach(), Ak.detach()
            Ps.append(Pk)
            Ks.append(Kk)
            As.append(Ak)

        P = torch.cat(Ps, dim=0)  # (U_total, length, emb_d)
        K = torch.cat(Ks, dim=0)  # (U_total, key_d)
        A = torch.cat(As, dim=0)  # (U_total, key_d)

        # a_q[b,n,u,d] = dot( x_query[b,n,:], A[u,:] )
        a_q    = torch.einsum('bnp,ud->bnud', q_input, A)  
        nK     = nn.functional.normalize(K, dim=1)             
        q_norm = nn.functional.normalize(a_q, dim=3)          
        weights= torch.einsum('bnud,ud->bnu', q_norm, nK)  # (B, N, U_total)

        # P_[b,n,l,emb_d] = sum_u weights[b,n,u] * P[u,l,emb_d]
        P_per_proposal = torch.einsum('bnu,uld->bnld', weights, P)  # (B, N, length, emb_d)

        # --- THE FIX: Add debug hook to record raw proposal data ---
        if self.debug and self.debug_probe is not None:
            # Create one record for the entire batch
            # The probe will save this single object containing the high-dimensional tensors
            record = ProposalMemoryRecord(
                epoch=self.debug_probe._epoch,
                img_id=self._current_img_ids, # This is now a list of img_ids in the batch
                task_id=task_id,
                layer=l,
                weights=weights.cpu().clone(), # Shape: (B, N, U)
                P=P_per_proposal.cpu().clone(), # Shape: (B, N, L, D)
                class_labels=self._current_class_labels,
                true_task_id=self.debug_attribute.true_task_id if self.debug_attribute else None
            )
            self.debug_probe(record)
        # --- End of fix ---

        B, N, L, D = P_per_proposal.shape
        mid    = L // 2

        # split per each proposal
        Ek = P_per_proposal[:, :, :mid, :]  # first half of L
        Ev = P_per_proposal[:, :, mid:, :]  # second half of L

        Ek = Ek.reshape(B, N * mid, D)
        Ev = Ev.reshape(B, N * mid, D)

        return [Ek, Ev], 0, x_block