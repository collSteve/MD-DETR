
import torch
import torch.nn as nn
from models.memory.dyn_memory import DynamicPrompt
from models.probes.memory_probe import ProposalMemoryRecord

class FocusedProposalMemory(DynamicPrompt):
    """
    An enhanced version of SimpleProposalMemory that introduces two key changes:
    1.  A softmax function is applied to the similarity scores to create a true
        probability distribution over the memory units.
    2.  A 'focus' (or temperature) parameter is used to sharpen this distribution,
        forcing the model to be more selective about which memories it uses.
    """

    def __init__(self, *args, focus: float = 10.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.focus = focus
        print(f"Initialized FocusedProposalMemory with focus={self.focus}")

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
            Pk, Kk, _ = mem.forward() 
            if not train or (task_id is not None and tid_s != str(task_id)):
                Pk, Kk = Pk.detach(), Kk.detach()
            Ps.append(Pk)
            Ks.append(Kk)

        P = torch.cat(Ps, dim=0)  # (U_total, length, emb_d)
        K = torch.cat(Ks, dim=0)  # (U_total, key_d)

        # --- MODIFIED LOGIC: Direct Query-to-Key Similarity ---
        q_norm = nn.functional.normalize(q_input, dim=2)
        nK = nn.functional.normalize(K, dim=1)
        raw_similarities = torch.einsum('bnd,ud->bnu', q_norm, nK)

        # --- NEW LOGIC: Apply Focus and Softmax ---
        # Sharpen the distribution and normalize weights to sum to 1
        focused_weights = torch.softmax(raw_similarities * self.focus, dim=-1)
        # --- END OF NEW LOGIC ---

        # P_[b,n,l,emb_d] = sum_u weights[b,n,u] * P[u,l,emb_d]
        P_per_proposal = torch.einsum('bnu,uld->bnld', focused_weights, P)

        # --- Debug hook ---
        if self.debug and self.debug_probe is not None:
            record = ProposalMemoryRecord(
                epoch=self.debug_probe._epoch,
                img_id=self._current_img_ids,
                task_id=task_id,
                layer=l,
                weights=focused_weights.cpu().clone(), # Log the final weights
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
