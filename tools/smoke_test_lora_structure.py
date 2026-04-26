"""Local structural smoke test for LoRALinear / patch_decoder_with_lora.

Verifies Phase 1.1 LoRA code structure without any cluster time. Runs in <5 s
on visionsw1; catches the most common Phase-1.1 bugs before committing to the
longer cluster smoke run (scripts/sbatch_train_lora_v1_smoke.sh) or the ~6 h
Gate A training.

Covers (per phase_1_conceptual_plan.md §2):
  1. LoRALinear construction (base frozen, A Kaiming-init, B zero-init)
  2. Δ = 0 at init (forward preserves base output when LoRA is set to a task
     before any training — standard PEFT invariant)
  3. set_current_task correctly freezes prior tasks and unfreezes current,
     and re-snapshots A_init_norms[t]
  4. lora_disabled toggle cleanly bypasses LoRA
  5. get_lora_norm_ratio at init ≈ 1.0
  6. get_olora_loss at task 0 returns 0 (no prior tasks)
  7. get_olora_loss at task 1+ returns nonzero when current A ≠ zero
  8. patch_decoder_with_lora wraps the right modules for fc1/fc2/fc1_fc2
  9. patch_decoder_with_lora is idempotent (calling twice does not double-wrap)

Run:
    conda run -n MD-DETR python tools/smoke_test_lora_structure.py
Exit 0 on pass, nonzero on any assertion failure.
"""
from __future__ import annotations

import os
import sys

# Ensure repo root is on sys.path so `from models...` imports resolve when the
# script is launched directly from tools/.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch
import torch.nn as nn

from models.adapters.lora import LoRALinear, patch_decoder_with_lora


def _assert(cond: bool, msg: str) -> None:
    if not cond:
        print(f"FAIL: {msg}")
        sys.exit(1)


def test_lora_linear_basic() -> None:
    """Test 1-2: construction, init convention, Δ=0 at init."""
    torch.manual_seed(0)
    base = nn.Linear(256, 1024)
    # Stamp specific weights so we can test Δ = 0 invariant precisely.
    base.weight.data.normal_()
    base.bias.data.normal_()
    lora = LoRALinear(base, rank=16, n_tasks=4)

    # Base frozen
    _assert(not lora.base.weight.requires_grad, "base.weight should be frozen")
    _assert(not lora.base.bias.requires_grad, "base.bias should be frozen")

    # Per-task slots structure
    _assert(len(lora.A_list) == 4, f"A_list len should be 4, got {len(lora.A_list)}")
    _assert(len(lora.B_list) == 4, f"B_list len should be 4, got {len(lora.B_list)}")
    _assert(lora.A_list[0].shape == (16, 256), f"A shape mismatch: {lora.A_list[0].shape}")
    _assert(lora.B_list[0].shape == (1024, 16), f"B shape mismatch: {lora.B_list[0].shape}")

    # B zero-init (PEFT convention)
    for i, b in enumerate(lora.B_list):
        _assert((b == 0).all().item(), f"B_list[{i}] should be zeros, found nonzero")

    # A Kaiming init (non-zero)
    for i, a in enumerate(lora.A_list):
        _assert((a.abs() > 0).any().item(), f"A_list[{i}] should be Kaiming-init, found all zeros")

    # Scaling = alpha / rank, default alpha = rank → scaling = 1
    _assert(abs(lora.scaling - 1.0) < 1e-6, f"default scaling should be 1.0, got {lora.scaling}")

    # Δ = 0 at init: enabling a task with B=0 must not change base output
    x = torch.randn(2, 300, 256)
    lora.current_task = -1
    out_bypass = lora(x)
    lora.set_current_task(0)
    out_t0 = lora(x)
    maxd = (out_bypass - out_t0).abs().max().item()
    _assert(maxd < 1e-6, f"Δ should be 0 at init with B zero-init, max abs diff = {maxd:.2e}")

    print("  [1-2] construction + Δ=0 at init: PASS")


def test_set_current_task() -> None:
    """Test 3: set_current_task freezes right slots, snapshots A_init_norms."""
    torch.manual_seed(1)
    base = nn.Linear(256, 1024)
    lora = LoRALinear(base, rank=8, n_tasks=4)

    # Before set_current_task, no slot is trainable in a meaningful sense
    # (init has requires_grad=True on all via nn.Parameter default, but that's fine).
    lora.set_current_task(2)
    _assert(lora.current_task == 2, f"current_task should be 2, got {lora.current_task}")
    for i in range(4):
        expected = (i == 2)
        _assert(
            lora.A_list[i].requires_grad == expected,
            f"A_list[{i}].requires_grad should be {expected} after set_current_task(2), got {lora.A_list[i].requires_grad}",
        )
        _assert(
            lora.B_list[i].requires_grad == expected,
            f"B_list[{i}].requires_grad should be {expected} after set_current_task(2), got {lora.B_list[i].requires_grad}",
        )

    # A_init_norms[2] should be snapshotted (positive, matches current A_list[2] norm)
    expected_norm = float(lora.A_list[2].data.norm('fro').item())
    got_norm = float(lora.A_init_norms[2].item())
    _assert(
        abs(got_norm - expected_norm) < 1e-6,
        f"A_init_norms[2] should match current norm {expected_norm:.4f}, got {got_norm:.4f}",
    )

    # Out-of-range task should raise
    try:
        lora.set_current_task(10)
    except ValueError:
        pass
    else:
        _assert(False, "set_current_task(10) should raise ValueError")

    print("  [3] set_current_task freezing + A_init_norms: PASS")


def test_lora_disabled() -> None:
    """Test 4: lora_disabled toggle bypasses LoRA cleanly."""
    torch.manual_seed(2)
    base = nn.Linear(256, 1024)
    lora = LoRALinear(base, rank=8, n_tasks=2)
    lora.set_current_task(0)

    # Simulate trained LoRA: put non-zero values in B so the contribution is nonzero
    with torch.no_grad():
        lora.B_list[0].data.normal_()

    x = torch.randn(2, 300, 256)
    out_enabled = lora(x)
    lora.lora_disabled = True
    out_disabled = lora(x)
    _assert(
        not torch.allclose(out_enabled, out_disabled),
        "With trained B, enabled vs disabled should differ — either B was re-zeroed or gating is broken",
    )
    # Disabled path should equal bare base(x)
    bare = lora.base(x)
    _assert(torch.allclose(out_disabled, bare), "lora_disabled should return base(x)")

    lora.lora_disabled = False
    print("  [4] lora_disabled bypass: PASS")


def test_norm_ratio() -> None:
    """Test 5: get_lora_norm_ratio is 1.0 at init, > 1 after (simulated) training."""
    torch.manual_seed(3)
    base = nn.Linear(256, 1024)
    lora = LoRALinear(base, rank=16, n_tasks=4)
    lora.set_current_task(0)

    r0 = lora.get_lora_norm_ratio()
    _assert(abs(r0 - 1.0) < 1e-6, f"norm ratio at init should be 1.0, got {r0:.6f}")

    # Simulate training: scale A_list[0] up by 2x. Ratio should become 2.0.
    with torch.no_grad():
        lora.A_list[0].data *= 2.0
    r1 = lora.get_lora_norm_ratio()
    _assert(abs(r1 - 2.0) < 1e-6, f"norm ratio after 2x scale should be 2.0, got {r1:.6f}")

    print("  [5] get_lora_norm_ratio: PASS")


def test_olora_loss() -> None:
    """Test 6-7: O-LoRA loss is 0 at task 0, nonzero at task 1+ with A's differing."""
    torch.manual_seed(4)
    base = nn.Linear(256, 1024)
    lora = LoRALinear(base, rank=16, n_tasks=4)

    # Task 0: no prior tasks → loss should be 0
    lora.set_current_task(0)
    loss0 = lora.get_olora_loss(0.5)
    _assert(loss0.item() == 0.0, f"olora loss at task 0 should be 0, got {loss0.item()}")

    # Task 1: one prior task. Since both A's are Kaiming-init with different seeds,
    # the Gram matrix A_cur @ A_old.T should have nonzero entries.
    lora.set_current_task(1)
    loss1 = lora.get_olora_loss(0.5)
    _assert(loss1.item() > 0.0, f"olora loss at task 1 should be > 0, got {loss1.item()}")

    # Gradient flows only to current task's A (not prior)
    loss1.backward()
    _assert(
        lora.A_list[0].grad is None or (lora.A_list[0].grad == 0).all().item(),
        "A_list[0].grad should be None or zero after olora backward (prior task)",
    )
    _assert(
        lora.A_list[1].grad is not None and lora.A_list[1].grad.abs().sum() > 0,
        "A_list[1].grad should be nonzero after olora backward (current task)",
    )

    print("  [6-7] O-LoRA loss: PASS")


class _FakeDecoderLayer(nn.Module):
    def __init__(self, d_model: int = 256, d_ff: int = 1024):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)


class _FakeDecoder(nn.Module):
    def __init__(self, n_layers: int = 6):
        super().__init__()
        self.layers = nn.ModuleList([_FakeDecoderLayer() for _ in range(n_layers)])


class _FakeDetrInner(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = _FakeDecoder()


class _FakeModel(nn.Module):
    """Stand-in for DeformableDetrForObjectDetection with the nested .model.decoder structure."""

    def __init__(self):
        super().__init__()
        self.model = _FakeDetrInner()


def test_patch_decoder() -> None:
    """Test 8-9: patch_decoder_with_lora wraps the right modules, is idempotent."""
    # fc1-only
    m = _FakeModel()
    patched = patch_decoder_with_lora(m, attach='fc1', rank=16, n_tasks=4)
    _assert(len(patched) == 6, f"fc1 attach should wrap 6 layers, got {len(patched)}")
    for i, layer in enumerate(m.model.decoder.layers):
        _assert(isinstance(layer.fc1, LoRALinear), f"layer {i}.fc1 should be LoRALinear")
        _assert(not isinstance(layer.fc2, LoRALinear), f"layer {i}.fc2 should be plain Linear")
        _assert(layer.fc1.in_features == 256, f"layer {i}.fc1 in_features = {layer.fc1.in_features}")
        _assert(layer.fc1.out_features == 1024, f"layer {i}.fc1 out_features = {layer.fc1.out_features}")

    # Idempotency — wrapping again should be a no-op (same list length, same wrapped instances)
    patched_again = patch_decoder_with_lora(m, attach='fc1', rank=16, n_tasks=4)
    _assert(len(patched_again) == 0, f"second patch should wrap nothing new, got {len(patched_again)}")
    for i, layer in enumerate(m.model.decoder.layers):
        _assert(isinstance(layer.fc1, LoRALinear), f"layer {i}.fc1 should still be LoRALinear")

    # fc2-only
    m2 = _FakeModel()
    patched2 = patch_decoder_with_lora(m2, attach='fc2', rank=16, n_tasks=4)
    _assert(len(patched2) == 6, f"fc2 attach should wrap 6 layers, got {len(patched2)}")
    for i, layer in enumerate(m2.model.decoder.layers):
        _assert(not isinstance(layer.fc1, LoRALinear), f"layer {i}.fc1 should be plain")
        _assert(isinstance(layer.fc2, LoRALinear), f"layer {i}.fc2 should be LoRALinear")
        _assert(layer.fc2.in_features == 1024, f"layer {i}.fc2 in_features = {layer.fc2.in_features}")
        _assert(layer.fc2.out_features == 256, f"layer {i}.fc2 out_features = {layer.fc2.out_features}")

    # fc1_fc2
    m3 = _FakeModel()
    patched3 = patch_decoder_with_lora(m3, attach='fc1_fc2', rank=8, n_tasks=4)
    _assert(len(patched3) == 12, f"fc1_fc2 attach should wrap 12 modules, got {len(patched3)}")
    for i, layer in enumerate(m3.model.decoder.layers):
        _assert(isinstance(layer.fc1, LoRALinear), f"layer {i}.fc1 should be LoRALinear")
        _assert(isinstance(layer.fc2, LoRALinear), f"layer {i}.fc2 should be LoRALinear")

    # Invalid attach value
    m4 = _FakeModel()
    try:
        patch_decoder_with_lora(m4, attach='bogus', rank=16, n_tasks=4)
    except ValueError:
        pass
    else:
        _assert(False, "patch_decoder_with_lora with invalid attach should raise ValueError")

    print("  [8-9] patch_decoder_with_lora fc1/fc2/fc1_fc2 + idempotency: PASS")


def test_freeze_integration() -> None:
    """Test 10: LoRA requires_grad survives engine.py::resume()'s freeze loop.

    Replicates the exact sequence engine.Trainer.__init__ + engine.Trainer.resume()
    executes when `use_lora_adapter=True`:

      1. Build a model with `layer.fc1` nested under a path containing 'decoder'.
      2. patch_decoder_with_lora + set_current_task(0).
      3. Apply the freeze-substring loop from engine.py::resume() (splits each
         param name by '.', sets requires_grad=False if any segment matches
         freeze={'backbone','encoder','decoder',''}).
      4. Re-apply set_current_task(0) — the post-freeze restoration fix.
      5. Assert current-task A and B have requires_grad=True.
      6. Assert a configure_optimizers-style 4-way split places LoRA in group 0.

    Without the post-freeze restoration (step 4), step 5 fails because the
    freeze loop matches the 'decoder' segment in LoRA param names. The April
    21 2026 smoke test was stuck at LoRA-norm ratio = 1.0 for 6000 steps
    because of this exact interaction; this test is the canary for it.
    """
    torch.manual_seed(42)

    # Build a model whose LoRA params will have 'decoder' in their path.
    m = _FakeModel()
    patched = patch_decoder_with_lora(m, attach='fc1', rank=8, n_tasks=4)
    for lora_mod in patched:
        lora_mod.set_current_task(0)

    # Pre-freeze sanity: current-task A and B trainable.
    for i, layer in enumerate(m.model.decoder.layers):
        _assert(
            layer.fc1.A_list[0].requires_grad,
            f"L{i}: A_list[0].requires_grad should be True pre-freeze",
        )
        _assert(
            layer.fc1.B_list[0].requires_grad,
            f"L{i}: B_list[0].requires_grad should be True pre-freeze",
        )

    # Step 3: replicate engine.py::resume()'s freeze loop verbatim.
    freeze = 'backbone,encoder,decoder,'.split(',')
    for name, params in m.named_parameters():
        params.requires_grad = True
        for n in name.split('.'):
            if n in freeze:
                params.requires_grad = False

    # Without the fix, current-task A and B are now False. Assert that ACTUALLY
    # happens in the raw freeze output (sanity check on the bug).
    raw_frozen_A = not m.model.decoder.layers[0].fc1.A_list[0].requires_grad
    raw_frozen_B = not m.model.decoder.layers[0].fc1.B_list[0].requires_grad
    _assert(
        raw_frozen_A and raw_frozen_B,
        "freeze loop should (buggily) clobber LoRA requires_grad — if this "
        "assertion fires, either the freeze logic changed or LoRA naming "
        "escaped the 'decoder' match, and this test may no longer be "
        "defending against the original bug.",
    )

    # Step 4: the fix — re-apply set_current_task post-freeze.
    for lora_mod in patched:
        lora_mod.set_current_task(0)

    # Step 5: current-task A and B are back to True.
    for i, layer in enumerate(m.model.decoder.layers):
        _assert(
            layer.fc1.A_list[0].requires_grad,
            f"L{i}: A_list[0].requires_grad should be True post-fix",
        )
        _assert(
            layer.fc1.B_list[0].requires_grad,
            f"L{i}: B_list[0].requires_grad should be True post-fix",
        )
        # Non-current-task slots stay frozen.
        for t in range(1, 4):
            _assert(
                not layer.fc1.A_list[t].requires_grad,
                f"L{i}: A_list[{t}].requires_grad should be False (not current)",
            )

    # Step 6: simulate engine.py's 4-way optimizer split and verify LoRA
    # params land in the LoRA group (not filtered out by requires_grad).
    def _match(n, kws):
        return any(kw in n for kw in kws)

    lora_kws = ['A_list', 'B_list']
    dp_kws = ['prompts']
    ce_kws = ['class_embed']
    named = list(m.named_parameters())
    group_lora = [p for n, p in named if _match(n, lora_kws) and p.requires_grad]
    group_other = [p for n, p in named
                   if not _match(n, lora_kws)
                   and not _match(n, ce_kws)
                   and not _match(n, dp_kws)
                   and p.requires_grad]

    # Expect 6 layers × 2 params (A_list[0] + B_list[0] trainable; slots 1..3 frozen) = 12.
    _assert(
        len(group_lora) == 12,
        f"LoRA group should have 12 params (6 layers × 2 current-task), got {len(group_lora)}",
    )
    # The 'other' group should be ≈ 0 (no class_embed in this fake model, no DP).
    # base.weight/bias of each fc1 are inside LoRALinear.base, which LoRALinear's __init__
    # sets requires_grad=False. Then freeze runs and also sets them False. Good.
    _assert(
        len(group_other) == 0,
        f"'other' group should have 0 trainable params in this fake model, got {len(group_other)}",
    )

    print("  [10] freeze-loop + post-freeze restoration integration: PASS")


def main() -> None:
    print("LoRA structural smoke test — running 10 checks ...")
    test_lora_linear_basic()
    test_set_current_task()
    test_lora_disabled()
    test_norm_ratio()
    test_olora_loss()
    test_patch_decoder()
    test_freeze_integration()
    print("\nALL CHECKS PASSED. Phase 1.1 LoRA code is structurally sound.")
    print("Next: submit scripts/sbatch_train_lora_v1_smoke.sh to verify end-to-end training plumbing.")


if __name__ == '__main__':
    main()
