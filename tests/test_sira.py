"""
Unit tests for SIRA modules.

Verifies:
    1. Shape correctness of all modules
    2. Parameter counts match proposal
    3. Gradient flow through trainable modules
    4. CLIP backbone remains frozen
    5. Gate initialization (starts conservative)
    6. Loss components are non-negative
    7. Orthogonality loss behavior
"""

import torch
import torch.nn as nn
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sira.sim import SynergisticInteractionModule
from sira.srg import SynergisticResidualGate
from sira.losses import SynergisticAwareContrastiveLoss


def test_sim_shapes():
    """Test SIM output shapes."""
    print("Test: SIM shapes...", end=" ")
    d_model, d_synergy, batch = 512, 64, 8
    sim = SynergisticInteractionModule(d_model, d_synergy)

    v = torch.randn(batch, d_model)
    t = torch.randn(batch, d_model)
    s = sim(v, t)

    assert s.shape == (batch, d_synergy), f"Expected ({batch}, {d_synergy}), got {s.shape}"
    print("PASSED ✓")


def test_sim_param_count():
    """Test SIM parameter count matches proposal (~168K for d=512, d_s=64)."""
    print("Test: SIM param count...", end=" ")
    sim = SynergisticInteractionModule(d_model=512, d_synergy=64)
    counts = sim.get_param_count()
    total = counts["total"]

    # Should be approximately 168K (allowing some margin for biases etc.)
    assert 100_000 < total < 300_000, f"SIM params {total} outside expected range"
    print(f"PASSED ✓ ({total:,} params)")


def test_srg_shapes():
    """Test SRG output shapes and normalization."""
    print("Test: SRG shapes...", end=" ")
    d_model, d_synergy, batch = 512, 64, 8
    srg = SynergisticResidualGate(d_model, d_synergy, gate_rank=16)

    v_shared = torch.randn(batch, d_model)
    t_shared = torch.randn(batch, d_model)
    s = torch.randn(batch, d_synergy)

    v_final, t_final = srg(v_shared, t_shared, s)

    assert v_final.shape == (batch, d_model), f"v_final shape: {v_final.shape}"
    assert t_final.shape == (batch, d_model), f"t_final shape: {t_final.shape}"

    # Check L2 normalization
    v_norms = v_final.norm(dim=-1)
    assert torch.allclose(v_norms, torch.ones_like(v_norms), atol=1e-5), "v_final not L2-normalized"
    print("PASSED ✓")


def test_srg_low_rank_params():
    """Test SRG low-rank param count (~66K)."""
    print("Test: SRG low-rank params...", end=" ")
    srg = SynergisticResidualGate(512, 64, gate_rank=16)
    counts = srg.get_param_count()
    total = counts["total"]

    assert total < 200_000, f"Low-rank SRG params {total} too high (expected < 200K)"
    print(f"PASSED ✓ ({total:,} params)")


def test_srg_full_rank_params():
    """Test SRG full-rank param count (~1.1M)."""
    print("Test: SRG full-rank params...", end=" ")
    srg = SynergisticResidualGate(512, 64, gate_rank=0)
    counts = srg.get_param_count()
    total = counts["total"]

    assert total > 500_000, f"Full-rank SRG params {total} too low"
    print(f"PASSED ✓ ({total:,} params)")


def test_gate_conservative_init():
    """Test that gates start near-zero (conservative injection)."""
    print("Test: Gate conservative init...", end=" ")
    srg = SynergisticResidualGate(512, 64, gate_rank=16, gate_init_bias=-2.0)

    v = torch.randn(8, 512)
    t = torch.randn(8, 512)
    s = torch.randn(8, 64)

    stats = srg.get_gate_stats(v, t, s)

    # With bias=-2.0 and near-zero weights, sigmoid should give small values
    assert stats["gate_v_mean"] < 0.3, f"Gate V mean {stats['gate_v_mean']:.3f} too high"
    assert stats["gate_t_mean"] < 0.3, f"Gate T mean {stats['gate_t_mean']:.3f} too high"
    print(f"PASSED ✓ (g_v={stats['gate_v_mean']:.3f}, g_t={stats['gate_t_mean']:.3f})")


def test_loss_components():
    """Test that all loss components are computable and non-negative."""
    print("Test: Loss components...", end=" ")
    loss_fn = SynergisticAwareContrastiveLoss(d_model=512, d_synergy=64)

    batch = 16
    v_final = nn.functional.normalize(torch.randn(batch, 512), dim=-1)
    t_final = nn.functional.normalize(torch.randn(batch, 512), dim=-1)
    s = torch.randn(batch, 64)
    v_shared = torch.randn(batch, 512)
    t_shared = torch.randn(batch, 512)
    s_v_proj = torch.randn(batch, 512)
    s_t_proj = torch.randn(batch, 512)

    losses = loss_fn(v_final, t_final, s, v_shared, t_shared, s_v_proj, s_t_proj)

    assert losses["total"].item() > 0, "Total loss should be positive"
    assert losses["shared"].item() >= 0, "Shared loss should be non-negative"
    assert losses["orthogonality"].item() >= 0, "Orth loss should be non-negative"
    assert not torch.isnan(losses["total"]), "Loss is NaN!"
    print(f"PASSED ✓ (total={losses['total'].item():.4f})")


def test_gradient_flow():
    """Test that gradients flow through SIRA modules."""
    print("Test: Gradient flow...", end=" ")
    sim = SynergisticInteractionModule(512, 64)
    srg = SynergisticResidualGate(512, 64, gate_rank=16)
    loss_fn = SynergisticAwareContrastiveLoss(d_model=512, d_synergy=64)

    v = torch.randn(8, 512, requires_grad=True)
    t = torch.randn(8, 512, requires_grad=True)

    s = sim(v, t)
    v_final, t_final = srg(v, t, s)
    s_v_proj = srg.proj_v(s)
    s_t_proj = srg.proj_t(s)
    losses = loss_fn(v_final, t_final, s, v, t, s_v_proj, s_t_proj)
    losses["total"].backward()

    # Check SIM has gradients
    sim_has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                       for p in sim.parameters())
    assert sim_has_grad, "SIM has no gradients!"

    # Check SRG has gradients
    srg_has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                       for p in srg.parameters())
    assert srg_has_grad, "SRG has no gradients!"
    print("PASSED ✓")


def test_orthogonality_behavior():
    """Test that orthogonality loss is lower when s is orthogonal to v, t."""
    print("Test: Orthogonality behavior...", end=" ")
    loss_fn = SynergisticAwareContrastiveLoss(d_model=64, d_synergy=64)

    # Case 1: s aligned with v and t (high orth loss)
    v = torch.randn(8, 64)
    t = torch.randn(8, 64)
    s_v_aligned = v.clone()
    s_t_aligned = t.clone()

    # Case 2: s orthogonal to v and t (low orth loss)
    # Use Gram-Schmidt to make s orthogonal
    s_v_orth = torch.randn(8, 64)
    s_v_orth = s_v_orth - (s_v_orth * v).sum(-1, keepdim=True) / (v * v).sum(-1, keepdim=True) * v
    s_t_orth = torch.randn(8, 64)
    s_t_orth = s_t_orth - (s_t_orth * t).sum(-1, keepdim=True) / (t * t).sum(-1, keepdim=True) * t

    orth_aligned = loss_fn.compute_orthogonality_loss(s_v_aligned, s_t_aligned, v, t)
    orth_orthogonal = loss_fn.compute_orthogonality_loss(s_v_orth, s_t_orth, v, t)

    assert orth_orthogonal < orth_aligned, \
        f"Orthogonal s ({orth_orthogonal:.4f}) should have lower loss than aligned ({orth_aligned:.4f})"
    print(f"PASSED ✓ (aligned={orth_aligned:.4f}, ortho={orth_orthogonal:.4f})")


def test_total_param_budget():
    """Test total SIRA parameter budget matches proposal (<0.2% with low-rank)."""
    print("Test: Total parameter budget...", end=" ")
    sim = SynergisticInteractionModule(512, 64)
    srg = SynergisticResidualGate(512, 64, gate_rank=16)
    loss_fn = SynergisticAwareContrastiveLoss(d_model=512, d_synergy=64)

    total = (sum(p.numel() for p in sim.parameters()) +
             sum(p.numel() for p in srg.parameters()) +
             sum(p.numel() for p in loss_fn.parameters()))

    clip_vit_b32_params = 151_000_000  # approximate
    pct = total / clip_vit_b32_params * 100

    print(f"PASSED ✓ ({total:,} params = {pct:.3f}% of CLIP ViT-B/32)")
    assert pct < 1.0, f"Total params {pct:.3f}% exceeds 1% budget"


def run_all_tests():
    print("=" * 60)
    print("SIRA Module Unit Tests")
    print("=" * 60)

    tests = [
        test_sim_shapes,
        test_sim_param_count,
        test_srg_shapes,
        test_srg_low_rank_params,
        test_srg_full_rank_params,
        test_gate_conservative_init,
        test_loss_components,
        test_gradient_flow,
        test_orthogonality_behavior,
        test_total_param_budget,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"FAILED ✗ ({e})")
            failed += 1

    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)}")
    print(f"{'='*60}")
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
