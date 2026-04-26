
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.memory.simple_proposal_memory import SimpleProposalMemory
from models.probes.memory_probe import ProposalMemoryRecord


class SelectiveProposalMemory(SimpleProposalMemory):
    """
    Extends SimpleProposalMemory with three anti-interference mechanisms:
    1. Softmax normalization (competition between memory units)
    2. Null memory units (empty destinations for background proposals)
    3. Background suppression loss (negative training signal)
    """

    def __init__(self, *args, focus: float = 10.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.focus = focus
        self._stored_P_per_proposal = []

    def set_batch_metadata(self, img_ids: list[int], class_labels: list[list[str]]):
        super().set_batch_metadata(img_ids, class_labels)
        self._stored_P_per_proposal = []  # reset for new batch

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
            # Use forward_with_null to include null units
            Pk, Kk, _ = mem.forward_with_null()
            if not train or (task_id is not None and tid_s != str(task_id)):
                Pk, Kk = Pk.detach(), Kk.detach()
            Ps.append(Pk)
            Ks.append(Kk)

        P = torch.cat(Ps, dim=0)  # (U_total + null_total, length, emb_d)
        K = torch.cat(Ks, dim=0)  # (U_total + null_total, key_d)

        # Normalize queries and keys
        q_norm = nn.functional.normalize(q_input, dim=2)  # (B, N, D_q)
        nK = nn.functional.normalize(K, dim=1)             # (U_total, D_q)

        # Cosine similarity -> softmax with focus
        raw_similarities = torch.einsum('bnd,ud->bnu', q_norm, nK)  # (B, N, U_total)
        weights = F.softmax(raw_similarities * self.focus, dim=-1)   # (B, N, U_total)

        # Weighted sum of P values
        P_per_proposal = torch.einsum('bnu,uld->bnld', weights, P)  # (B, N, length, emb_d)

        # Store P_per_proposal for post-forward bg loss computation (only during training)
        if train:
            self._stored_P_per_proposal.append(P_per_proposal)

        # Debug hook
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

        B, N, L, D = P_per_proposal.shape
        mid = L // 2

        Ek = P_per_proposal[:, :, :mid, :].reshape(B, N * mid, D)
        Ev = P_per_proposal[:, :, mid:, :].reshape(B, N * mid, D)

        return [Ek, Ev], 0, x_block

    def compute_bg_loss(self, foreground_mask: torch.Tensor) -> torch.Tensor:
        """Compute bg suppression loss using stored P_per_proposal and Pass 2's foreground mask."""
        bg_mask = 1.0 - foreground_mask  # (B, 300)
        total_loss = torch.tensor(0.0, device=foreground_mask.device)
        for P_pp in self._stored_P_per_proposal:
            mem_norm_sq = (P_pp ** 2).sum(dim=(-1, -2))  # (B, N)
            total_loss = total_loss + (bg_mask * mem_norm_sq).mean()
        self._stored_P_per_proposal = []  # clear after use
        return total_loss
