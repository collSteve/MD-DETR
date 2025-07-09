import torch
import torch.nn as nn
from models.memory.dyn_memory import DynamicPrompt


class ProposalQueryMemory(DynamicPrompt):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

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


        # if self.local_query:
        #     # weighting each proposal
        #     q_wt     = self.query_tf(q_input)          # (B, N, key_d)
        #     q_input = q_input * q_wt.unsqueeze(-1)



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

        #    a_q[b,n,u,d] = dot( x_query[b,n,:], A[u,:] )
        a_q    = torch.einsum('bnp,ud->bnud', q_input, A)  
        nK     = nn.functional.normalize(K, dim=1)             
        q_norm = nn.functional.normalize(a_q, dim=3)          
        weights= torch.einsum('bnud,ud->bnu', q_norm, nK)  # (B, N, U_total)

        #    P_[b,n,l,emb_d] = sum_u weights[b,n,u] * P[u,l,emb_d]
        P_per_proposal = torch.einsum('bnu,uld->bnld', weights, P)  # (B, N, length, emb_d)

        B, N, L, D = P_per_proposal.shape
        mid    = L // 2

        # split per each proposal
        Ek = P_per_proposal[:, :, :mid, :]  # first half of L
        Ev = P_per_proposal[:, :, mid:, :]  # second half of L

        Ek = Ek.reshape(B, N * mid, D)
        Ev = Ev.reshape(B, N * mid, D)

        return [Ek, Ev], 0, x_block
