"""Per-task LoRA adapters for MD-DETR's decoder FFN projections.

Path B Variant A v1 — see docs/MD-DETR/phase_1_conceptual_plan.md for the spec
and docs/MD-DETR/path_b_design.md §5 Variant A for the axis-by-axis rationale.

This module provides:

    LoRALinear — an nn.Module wrapper around an existing nn.Linear. Holds per-task
        low-rank (A, B) parameter pairs and adds their contribution additively in
        forward. Supports per-task training discipline via set_current_task and
        Pass-1/Pass-2 gating via the lora_disabled flag.

    patch_decoder_with_lora(model, attach, rank, n_tasks, lora_alpha) — monkey-patches
        the 6 decoder layers' fc1 and/or fc2 modules with LoRALinear wrappers and
        returns the list of new wrappers for engine.py to iterate over.

Design notes:
  * LoRA is initialized with A ~ Kaiming-uniform and B = 0, so Δ = A·B = 0 at
    initialization — the model's behavior at step 0 is exactly its behavior without
    LoRA. This lets us claim a clean floor (DP + probe ≈ DP baseline when LoRA
    coexists and hasn't yet been trained).
  * Only the current task's (A_t, B_t) pair has requires_grad=True during task t's
    training. All other tasks' parameters are detached in forward too, for
    belt-and-suspenders semantic clarity (mirrors the existing memory module
    pattern in models/memory/simple_proposal_memory.py).
  * lora_disabled is toggled by engine.common_step at the Pass-1/Pass-2 boundary —
    Pass 1 (memory-disabled query generation) should also be LoRA-disabled so
    query addresses don't drift across tasks. See phase_1_conceptual_plan.md §2.2.
  * O-LoRA soft-orthogonality loss is computed once per training step (not per
    forward pass) by engine.training_step via get_olora_loss. See §2.5.
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """Additive per-task LoRA adapter wrapping an existing nn.Linear.

    Forward computation (when active for current task t):
        out = base(x) + (alpha/rank) * sum_{i <= t} (x @ A_i.T) @ B_i.T
    where A_i, B_i are per-task low-rank matrices. At initialization, B_i = 0 for
    all i, so the LoRA contribution is identically zero until training begins.
    """

    def __init__(
        self,
        base_linear: nn.Linear,
        rank: int,
        n_tasks: int,
        lora_alpha: Optional[float] = None,
    ):
        super().__init__()
        if rank <= 0:
            raise ValueError(f"LoRA rank must be > 0, got {rank}")
        if n_tasks <= 0:
            raise ValueError(f"n_tasks must be > 0, got {n_tasks}")

        self.base = base_linear
        for p in self.base.parameters():
            p.requires_grad = False

        in_features = base_linear.in_features
        out_features = base_linear.out_features
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.n_tasks = n_tasks
        # Standard PEFT convention: alpha = rank → scaling = 1. Overridable.
        self.lora_alpha = float(lora_alpha) if lora_alpha is not None else float(rank)
        self.scaling = self.lora_alpha / self.rank

        # Per-task A (rank, in_features) and B (out_features, rank).
        # Init: A Kaiming-uniform, B zeros → Δ = 0 at step 0.
        self.A_list = nn.ParameterList(
            [nn.Parameter(torch.zeros(rank, in_features)) for _ in range(n_tasks)]
        )
        self.B_list = nn.ParameterList(
            [nn.Parameter(torch.zeros(out_features, rank)) for _ in range(n_tasks)]
        )
        for a in self.A_list:
            nn.init.kaiming_uniform_(a, a=5 ** 0.5)
        # B_list stays at zeros (initial value).

        # Record initial Frobenius norms for the Gate A LoRA-norm-ratio check.
        # These are re-snapshotted per-task by set_current_task as we begin each
        # task's training (so the ratio is computed vs the task-start value).
        self.register_buffer(
            'A_init_norms',
            torch.tensor([float(a.data.norm('fro').item()) for a in self.A_list]),
        )

        self.current_task: int = -1
        self.lora_disabled: bool = False

    def set_current_task(self, t: int) -> None:
        """Mark task t as the currently-trainable task; freeze all other tasks.

        Also snapshots A_list[t]'s current Frobenius norm into A_init_norms[t]
        so that get_lora_norm_ratio() compares against a well-defined baseline
        (the norm at the start of this task's training).
        """
        if t < 0 or t >= self.n_tasks:
            raise ValueError(f"Task index {t} out of range [0, {self.n_tasks})")
        self.current_task = t
        for i in range(self.n_tasks):
            flag = (i == t)
            self.A_list[i].requires_grad = flag
            self.B_list[i].requires_grad = flag
        # Refresh the norm-ratio reference for this task.
        self.A_init_norms[t] = float(self.A_list[t].data.norm('fro').item())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        if self.lora_disabled or self.current_task < 0:
            return out
        # Sum current + prior tasks' contributions. Prior tasks are detached.
        # (requires_grad=False on prior slots already prevents gradients, but
        # detach() is belt-and-suspenders and matches memory-module convention.)
        for i in range(self.current_task + 1):
            A = self.A_list[i] if i == self.current_task else self.A_list[i].detach()
            B = self.B_list[i] if i == self.current_task else self.B_list[i].detach()
            # x @ A.T: (..., in) @ (in, rank) = (..., rank)
            # _ @ B.T: (..., rank) @ (rank, out) = (..., out)
            out = out + self.scaling * ((x @ A.T) @ B.T)
        return out

    def get_olora_loss(self, lambda_olora: float) -> torch.Tensor:
        """O-LoRA soft-orthogonality penalty for current task vs prior tasks.

        Returns:
            scalar tensor (on the device of the current-task A matrix) equal to
            lambda_olora * sum_{i < current_task} ||A_current @ A_i.T||_F^2

        Returns a zero tensor (still with grad to the current task's A, so
        autograd graph is consistent) when current_task <= 0 — no prior tasks.
        """
        if self.current_task <= 0:
            # No prior tasks: return a zero scalar on the right device.
            # Use A_list[0]'s device as reference.
            ref = self.A_list[0]
            return torch.zeros((), device=ref.device, dtype=ref.dtype)

        A_cur = self.A_list[self.current_task]
        loss = torch.zeros((), device=A_cur.device, dtype=A_cur.dtype)
        for i in range(self.current_task):
            A_old = self.A_list[i].detach()
            # Gram matrix between current-task rows and old-task rows.
            G = A_cur @ A_old.T  # shape (rank, rank)
            loss = loss + (G ** 2).sum()
        return lambda_olora * loss

    def get_lora_norm_ratio(self) -> float:
        """Ratio of current A's Frobenius norm to its task-start value.

        Used by Gate A (post-Task-1) to verify LoRA actually learned (i.e., the
        §2.8 gradient-attribution risk didn't leave LoRA dead at init). Healthy
        trained-LoRA: ratio > 1.5 per layer. Dead LoRA: ratio ~ 1.0.

        Returns 0.0 if no current task has been set yet.
        """
        if self.current_task < 0:
            return 0.0
        init_norm = float(self.A_init_norms[self.current_task].item())
        if init_norm <= 0.0:
            return 0.0
        cur_norm = float(self.A_list[self.current_task].data.norm('fro').item())
        return cur_norm / init_norm


def patch_decoder_with_lora(
    model,
    attach: str,
    rank: int,
    n_tasks: int,
    lora_alpha: Optional[float] = None,
) -> List[LoRALinear]:
    """Monkey-patch decoder FFN projections with LoRALinear wrappers.

    Must be called AFTER the model is loaded from HF (from_pretrained) and BEFORE
    any trainer.resume() call. See phase_1_conceptual_plan.md §2.1 for why the
    ordering matters — patching after resume() would silently drop prior-task
    LoRA weights (strict=False load).

    Args:
        model: DeformableDetrForObjectDetection instance (expects
            model.model.decoder.layers as an iterable of decoder layers, each
            with .fc1 and .fc2 nn.Linear attributes).
        attach: "fc1", "fc2", or "fc1_fc2" — which FFN projection(s) to wrap.
        rank: LoRA rank r.
        n_tasks: number of CL tasks; determines the ParameterList size.
        lora_alpha: LoRA alpha (scaling = alpha/rank). None → alpha = rank.

    Returns:
        List of the newly-created LoRALinear instances (in the order encountered
        while iterating layers — for a 6-layer decoder with fc1_fc2: 12 entries
        ordered [layer0.fc1, layer0.fc2, layer1.fc1, layer1.fc2, ...]).
    """
    valid_attach = {"fc1", "fc2", "fc1_fc2"}
    if attach not in valid_attach:
        raise ValueError(f"lora_attach must be one of {valid_attach}, got {attach!r}")
    parts = attach.split('_')

    decoder = model.model.decoder
    patched: List[LoRALinear] = []
    for layer in decoder.layers:
        if 'fc1' in parts and not isinstance(layer.fc1, LoRALinear):
            layer.fc1 = LoRALinear(layer.fc1, rank=rank, n_tasks=n_tasks, lora_alpha=lora_alpha)
            patched.append(layer.fc1)
        if 'fc2' in parts and not isinstance(layer.fc2, LoRALinear):
            layer.fc2 = LoRALinear(layer.fc2, rank=rank, n_tasks=n_tasks, lora_alpha=lora_alpha)
            patched.append(layer.fc2)
    return patched
