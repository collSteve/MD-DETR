
import torch
import torch.nn as nn
from models.memory.dyn_memory import DynamicPrompt
from models.probes.memory_probe import ProposalMemoryRecord

class SimpleProposalMemory(DynamicPrompt):
    """
    A simplified version of ProposalQueryMemory that removes the 'A' vector.
    Memory retrieval is based on a direct cosine similarity between the object query
    and the memory keys 'K'.
    """

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

        B, N, D_q = x_query.shape
        q_input = x_query

        Ps, Ks = [], []
        for tid_s, mem in self.layer_memories[layer].items():
            # Note: We no longer retrieve or use the 'A' vector from the memory unit.
            Pk, Kk, _ = mem.forward() 
            if not train or (task_id is not None and tid_s != str(task_id)):
                Pk, Kk = Pk.detach(), Kk.detach()
            Ps.append(Pk)
            Ks.append(Kk)

        P = torch.cat(Ps, dim=0)  # (U_total, length, emb_d)
        K = torch.cat(Ks, dim=0)  # (U_total, key_d)

        # --- MODIFIED LOGIC: Direct Query-to-Key Similarity ---
        # Normalize both the input queries and the memory keys
        q_norm = nn.functional.normalize(q_input, dim=2) # Shape: (B, N, D_q)
        nK = nn.functional.normalize(K, dim=1)           # Shape: (U_total, D_q)

        # Compute direct cosine similarity
        # weights[b, n, u] = dot(q_norm[b, n, :], nK[u, :])
        weights = torch.einsum('bnd,ud->bnu', q_norm, nK)  # Shape: (B, N, U_total)
        # --- END OF MODIFIED LOGIC ---

        # P_[b,n,l,emb_d] = sum_u weights[b,n,u] * P[u,l,emb_d]
        P_per_proposal = torch.einsum('bnu,uld->bnld', weights, P)  # (B, N, length, emb_d)

        # --- Debug hook (remains the same) ---
        if self.debug and self.debug_probe is not None:
            record = ProposalMemoryRecord(
                epoch=self.debug_probe._epoch,
                img_id=self._current_img_ids,
                task_id=task_id,
                layer=l,
                weights=weights.cpu().clone(),
                P=P_per_proposal.cpu().clone(),
                class_labels=self._current_class_labels,
                true_task_id=self.debug_attribute.true_task_id if self.debug_attribute else None
            )
            self.debug_probe(record)
        # --- End of hook ---

        B, N, L, D = P_per_proposal.shape
        mid    = L // 2

        # split per each proposal
        Ek = P_per_proposal[:, :, :mid, :]  # first half of L
        Ev = P_per_proposal[:, :, mid:, :]  # second half of L

        Ek = Ek.reshape(B, N * mid, D)
        Ev = Ev.reshape(B, N * mid, D)

        return [Ek, Ev], 0, x_block
