"""
Synergistic-Aware Contrastive Loss (SACL)

A composite training objective that explicitly encourages the model to learn
synergistic representations, beyond what standard InfoNCE captures.

Components:
    L_SIRA = L_shared + λ_orth * L_orth

    1. L_shared: Standard InfoNCE on final (synergy-enhanced) representations
       → Preserves pre-trained alignment quality

    2. L_orth: Orthogonality regularizer
       → Prevents synergy from collapsing into redundancy by enforcing
         orthogonality between synergistic and unimodal representations.
         Uses the SRG projections (the ones actually fused into v_final/t_final)
         so the constraint acts on the real injection path.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SynergisticAwareContrastiveLoss(nn.Module):
    """
    Synergistic-Aware Contrastive Loss (SACL).

    Combines two loss terms:
        - Standard contrastive (InfoNCE) on final representations
        - Orthogonality regularizer between synergy (via SRG projections)
          and unimodal features

    Args:
        lambda_orth: Weight for the orthogonality regularizer. Default: 0.1.
        learnable_temperature: Whether temperature is learnable. Default: True.
    """

    def __init__(
        self,
        lambda_orth: float = 0.1,
        learnable_temperature: bool = True,
        # Accept but ignore legacy kwargs for backward compat during transition
        **kwargs,
    ):
        super().__init__()
        self.lambda_orth = lambda_orth

        # Temperature (optionally learnable)
        if learnable_temperature:
            self.log_temp = nn.Parameter(torch.log(torch.tensor(0.07)))
        else:
            self.register_buffer("log_temp", torch.log(torch.tensor(0.07)))

    def compute_orthogonality_loss(
        self,
        s_v_proj: torch.Tensor,
        s_t_proj: torch.Tensor,
        v: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute orthogonality loss between SRG-projected synergy and unimodal features.

        Args:
            s_v_proj: SRG-projected synergy for vision [B, d_model] (from srg.proj_v(s)).
            s_t_proj: SRG-projected synergy for text [B, d_model] (from srg.proj_t(s)).
            v: Shared vision features [B, d_model].
            t: Shared text features [B, d_model].

        Returns:
            Scalar loss: mean of cos²(s_v_proj, v) + cos²(s_t_proj, t).
        """
        s_v_norm = F.normalize(s_v_proj, p=2, dim=-1)
        s_t_norm = F.normalize(s_t_proj, p=2, dim=-1)
        v_norm = F.normalize(v, p=2, dim=-1)
        t_norm = F.normalize(t, p=2, dim=-1)

        cos_v = (s_v_norm * v_norm).sum(dim=-1)
        cos_t = (s_t_norm * t_norm).sum(dim=-1)

        loss_orth = (cos_v ** 2 + cos_t ** 2).mean()
        return loss_orth

    def forward(
        self,
        v_final: torch.Tensor,
        t_final: torch.Tensor,
        s: torch.Tensor,
        v_shared: torch.Tensor,
        t_shared: torch.Tensor,
        s_v_proj: torch.Tensor,
        s_t_proj: torch.Tensor,
    ) -> dict:
        """
        Compute the synergistic-aware contrastive loss (SACL).

        Args:
            v_final: Synergy-enhanced vision features [B, d_model]
            t_final: Synergy-enhanced text features [B, d_model]
            s: Synergy vectors [B, d_synergy]
            v_shared: Shared vision features [B, d_model]
            t_shared: Shared text features [B, d_model]
            s_v_proj: SRG-projected synergy for vision [B, d_model]
            s_t_proj: SRG-projected synergy for text [B, d_model]

        Returns:
            Dictionary containing losses.
        """
        temperature = torch.exp(self.log_temp).clamp(min=1e-4, max=100.0)

        # 1. L_shared: Standard InfoNCE on final (synergy-enhanced) representations
        logits_shared = (v_final @ t_final.t()) / temperature
        labels = torch.arange(logits_shared.size(0), device=logits_shared.device)
        l_shared = (
            F.cross_entropy(logits_shared, labels)
            + F.cross_entropy(logits_shared.t(), labels)
        ) / 2.0

        # 2. L_orth: Orthogonality regularizer (using SRG projections)
        l_orth = self.compute_orthogonality_loss(s_v_proj, s_t_proj, v_shared, t_shared)

        # Total Loss
        total = l_shared + self.lambda_orth * l_orth

        return {
            "total": total,
            "shared": l_shared.detach(),
            "base": l_shared.detach(),  # matching train_sira.py baseline logging
            "orthogonality": l_orth.detach(),
            "temperature": temperature.detach(),
        }
