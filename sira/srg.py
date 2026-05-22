"""
Synergistic Residual Gate (SRG)

Dynamically fuses synergistic information back into the unimodal representations
for downstream retrieval. Uses a learned gate that controls how much synergy to
inject per-sample.

Design rationale:
    Not all queries require synergistic reasoning.
    - "A photo of a dog" → purely shared/redundant → gate ≈ 0
    - "The dog scared by thunder" → requires synergy → gate ≈ 1
    The gate learns to adaptively weight synergistic contributions.

Low-rank variant:
    To keep parameter count minimal (<0.2% of backbone), the gate uses
    low-rank factorized projections instead of full dense layers.
"""

import torch
import torch.nn as nn


class LowRankLinear(nn.Module):
    """
    Low-rank factorized linear layer: W = W_down @ W_up
    Reduces parameters from d_in × d_out to d_in × rank + rank × d_out.
    """

    def __init__(self, d_in: int, d_out: int, rank: int = 16):
        super().__init__()
        self.down = nn.Linear(d_in, rank, bias=False)
        self.up = nn.Linear(rank, d_out, bias=False)

        nn.init.xavier_uniform_(self.down.weight)
        nn.init.xavier_uniform_(self.up.weight, gain=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(self.down(x))


class SynergisticResidualGate(nn.Module):
    """
    Synergistic Residual Gate (SRG).

    Adaptively fuses synergistic representations back into unimodal features:
        v_final = v_shared + g_v ⊙ s_v
        t_final = t_shared + g_t ⊙ s_t

    where g_v, g_t are sigmoid gates that control per-dimension injection of
    synergistic information.

    Args:
        d_model: Dimension of the backbone features (e.g., 512 for CLIP ViT-B/32).
        d_synergy: Dimension of the synergistic representation from SIM.
        gate_rank: Rank of the low-rank gate factorization. Set to 0 for full-rank.
        gate_init_bias: Initial bias for gate sigmoid. Default: 0.0
                        (σ(0) = 0.5, allowing moderate synergy injection from
                        the start; magnitude clamp provides the safety bound).
    """

    def __init__(
        self,
        d_model: int,
        d_synergy: int = 64,
        gate_rank: int = 16,
        gate_init_bias: float = -2.0,
        max_residual_ratio: float = 0.3,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_synergy = d_synergy
        self.max_residual_ratio = max_residual_ratio

        # ------------------------------------------------------------------
        # Synergy-to-modality projections: ℝ^{d_synergy} → ℝ^{d_model}
        # Projects the synergistic representation to match each modality's
        # feature dimension for residual addition.
        # ------------------------------------------------------------------
        self.proj_v = nn.Linear(d_synergy, d_model, bias=True)
        self.proj_t = nn.Linear(d_synergy, d_model, bias=True)

        # ------------------------------------------------------------------
        # Gating networks: sigmoid gates controlling per-dimension injection
        # Input: concatenation of [modality_features; projected_synergy]
        # Output: gate values ∈ (0, 1)^{d_model}
        # ------------------------------------------------------------------
        gate_input_dim = d_model + d_model  # [shared_feat; projected_synergy]

        if gate_rank > 0:
            # Low-rank gates (recommended: ~49K params vs ~1.1M full-rank)
            self.gate_v = LowRankLinear(gate_input_dim, d_model, rank=gate_rank)
            self.gate_t = LowRankLinear(gate_input_dim, d_model, rank=gate_rank)
        else:
            # Full-rank gates
            self.gate_v = nn.Linear(gate_input_dim, d_model)
            self.gate_t = nn.Linear(gate_input_dim, d_model)

        # ------------------------------------------------------------------
        # Gate bias: the magnitude clamp (max_residual_ratio) provides the
        # hard safety bound; the bias just sets the initial operating point.
        # With bias=0.0, σ(0)=0.5, so the gate starts at moderate injection
        # and the clamp prevents overshooting.
        # ------------------------------------------------------------------
        self.gate_bias_v = nn.Parameter(torch.full((d_model,), gate_init_bias))
        self.gate_bias_t = nn.Parameter(torch.full((d_model,), gate_init_bias))

        self._init_weights()

    def _init_weights(self):
        """Initialize projection weights with small values for stability."""
        nn.init.xavier_uniform_(self.proj_v.weight, gain=0.1)
        nn.init.zeros_(self.proj_v.bias)
        nn.init.xavier_uniform_(self.proj_t.weight, gain=0.1)
        nn.init.zeros_(self.proj_t.bias)

        # Ensure full-rank gates start conservative (near zero output before bias)
        if isinstance(self.gate_v, nn.Linear):
            nn.init.zeros_(self.gate_v.weight)
            if self.gate_v.bias is not None:
                nn.init.zeros_(self.gate_v.bias)
        if isinstance(self.gate_t, nn.Linear):
            nn.init.zeros_(self.gate_t.weight)
            if self.gate_t.bias is not None:
                nn.init.zeros_(self.gate_t.bias)

    def _clamp_residual(self, residual, original):
        """Clamp residual norm to be at most max_residual_ratio of original norm."""
        res_norm = residual.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        orig_norm = original.norm(dim=-1, keepdim=True)
        max_norm = self.max_residual_ratio * orig_norm
        # Scale down if residual is too large, leave unchanged if within budget
        scale = torch.clamp(max_norm / res_norm, max=1.0)
        return residual * scale

    def forward(
        self,
        v_shared: torch.Tensor,
        t_shared: torch.Tensor,
        s: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Fuse synergistic information into modality representations.

        Args:
            v_shared: Shared vision features [B, d_model] from DsRA-adapted encoder.
            t_shared: Shared text features [B, d_model] from DsRA-adapted encoder.
            s: Synergistic representation [B, d_synergy] from SIM.

        Returns:
            v_final: Enhanced vision features [B, d_model], L2-normalized.
            t_final: Enhanced text features [B, d_model], L2-normalized.
        """
        # Step 1: Project synergy to modality dimensions
        s_v = self.proj_v(s)  # [B, d_model]
        s_t = self.proj_t(s)  # [B, d_model]

        # Step 2: Compute gating scores
        gate_input_v = torch.cat([v_shared, s_v], dim=-1)  # [B, 2*d_model]
        gate_input_t = torch.cat([t_shared, s_t], dim=-1)  # [B, 2*d_model]

        g_v = torch.sigmoid(self.gate_v(gate_input_v) + self.gate_bias_v)  # [B, d_model]
        g_t = torch.sigmoid(self.gate_t(gate_input_t) + self.gate_bias_t)  # [B, d_model]

        # Step 3: Gated residual fusion with magnitude clamping
        residual_v = self._clamp_residual(g_v * s_v, v_shared)
        residual_t = self._clamp_residual(g_t * s_t, t_shared)

        v_final = v_shared + residual_v
        t_final = t_shared + residual_t

        # Step 4: L2 normalize for retrieval
        v_final = nn.functional.normalize(v_final, p=2, dim=-1)
        t_final = nn.functional.normalize(t_final, p=2, dim=-1)

        return v_final, t_final

    def get_gate_stats(
        self,
        v_shared: torch.Tensor,
        t_shared: torch.Tensor,
        s: torch.Tensor,
    ) -> dict:
        """
        Compute gate activation statistics for analysis/visualization.

        Returns:
            Dictionary with mean, std, min, max of gate activations for both
            vision and text branches.
        """
        s_v = self.proj_v(s)
        s_t = self.proj_t(s)

        gate_input_v = torch.cat([v_shared, s_v], dim=-1)
        gate_input_t = torch.cat([t_shared, s_t], dim=-1)

        g_v = torch.sigmoid(self.gate_v(gate_input_v) + self.gate_bias_v)
        g_t = torch.sigmoid(self.gate_t(gate_input_t) + self.gate_bias_t)

        return {
            "gate_v_mean": g_v.mean().item(),
            "gate_v_std": g_v.std().item(),
            "gate_v_min": g_v.min().item(),
            "gate_v_max": g_v.max().item(),
            "gate_t_mean": g_t.mean().item(),
            "gate_t_std": g_t.std().item(),
            "gate_t_min": g_t.min().item(),
            "gate_t_max": g_t.max().item(),
        }

    def get_param_count(self) -> dict:
        """Return parameter count breakdown for this module."""
        counts = {
            "projections": sum(
                p.numel() for p in list(self.proj_v.parameters()) + list(self.proj_t.parameters())
            ),
            "gates": sum(
                p.numel() for p in list(self.gate_v.parameters()) + list(self.gate_t.parameters())
            ),
            "gate_biases": self.gate_bias_v.numel() + self.gate_bias_t.numel(),
        }
        counts["total"] = sum(counts.values())
        return counts
