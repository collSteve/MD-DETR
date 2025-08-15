
import torch
import torch.nn as nn
from models.memory.dyn_memory import DynamicPrompt
from models.probes.memory_probe import ProposalMemoryRecord

class L2ProposalMemory(DynamicPrompt):
    """
    A version of ProposalQueryMemory that uses L2 (Euclidean) distance for
    memory retrieval instead of cosine similarity.

    The core idea is that "closeness" in the vector space should determine
    memory selection. Weights are derived from the inverse of the distance.
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
            x_query: torch.Tensor,      # (B, N, D_q)
            l: int,
            x_block: torch.Tensor,
            train: bool = False,
            task_id: int = None):
        layer = str(l)
        if layer not in self.layer_memories:
            return None, 0, x_block

        B, N, D_q = x_query.shape
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

        # --- MODIFIED LOGIC: L2 Distance-based Retrieval ---
        
        # Stage 1: Query Projection via A (same as before)
        # q_transformed[b, n, u, d] = q_input[b, n, d] * A[u, d]
        q_transformed = torch.einsum('bnd,ud->bnud', q_input, A)

        # Stage 2: L2 Distance Calculation
        # We want to compute the distance between each transformed query and each key.
        # K_expanded shape: (1, 1, U, D_q)
        # q_transformed shape: (B, N, U, D_q)
        # distances shape: (B, N, U)
        distances = torch.norm(q_transformed - K.unsqueeze(0).unsqueeze(0), p=2, dim=3)

        # Stage 3: Convert distance to weights.
        # Using exp(-distance) is a common way to make closer vectors have higher weights.
        # The result is a soft, non-negative weighting.
        weights = torch.exp(-distances)
        
        # --- END OF MODIFIED LOGIC ---

        # P_[b,n,l,emb_d] = sum_u weights[b,n,u] * P[u,l,emb_d]
        P_per_proposal = torch.einsum('bnu,uld->bnld', weights, P)

        # --- Debug hook ---
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

        Ek = P_per_proposal[:, :, :mid, :].reshape(B, N * mid, D)
        Ev = P_per_proposal[:, :, mid:, :].reshape(B, N * mid, D)

        return [Ek, Ev], 0, x_block
