"""
Synergistic Interaction Module (SIM)

Generates a synergistic representation that captures information emerging only
from the *interaction* of both modalities — information that neither modality
encodes alone.

Architecture:
    1. Bilinear interaction: captures non-linear cross-modal dependencies
    2. Residual subtraction: removes unimodal contributions (PID approximation)
    3. Projection: maps to synergy space

Theoretical grounding:
    This is a neural approximation of the Partial Information Decomposition (PID).
    By training h_interact to capture all joint information and subtracting
    unimodal predictions, what remains is approximately the synergistic component.
"""

import torch
import torch.nn as nn


class UnimodalPredictor(nn.Module):
    """
    Single-modality predictor: estimates what information a single modality
    can contribute independently. Used to subtract unimodal contributions
    from the joint interaction, isolating the synergistic residual.
    """

    def __init__(self, d_in: int, d_out: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_out),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_out, d_out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SynergisticInteractionModule(nn.Module):
    """
    Synergistic Interaction Module (SIM).

    Computes a synergistic representation s ∈ ℝ^{B×d_s} by:
        1. Computing a bilinear interaction of vision and text features
        2. Subtracting unimodal-only predictions (residual subtraction)
        3. Projecting the residual into a normalized synergy space

    Args:
        d_model: Dimension of the input features from the backbone (e.g., 512 for CLIP ViT-B/32).
        d_synergy: Dimension of the synergistic bottleneck (d_s). Controls capacity
                   allocated to synergistic representations. Default: 64.
        dropout: Dropout rate for regularization. Default: 0.1.
    """

    def __init__(self, d_model: int, d_synergy: int = 64, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.d_synergy = d_synergy

        # ------------------------------------------------------------------
        # Step 1: Bilinear Interaction
        # Three projection matrices capture: vision-only, text-only, and
        # the cross-modal interaction contributions.
        #
        # FIX: Raw v*t between L2-normalized vectors is ~0.002 per element,
        # making W_vt(v*t) contribute only ~4%. We project v and t to a
        # richer interaction space first, then compute the Hadamard product.
        # ------------------------------------------------------------------
        self.W_v = nn.Linear(d_model, d_synergy, bias=False)
        self.W_t = nn.Linear(d_model, d_synergy, bias=False)

        # Cross-modal interaction: project to interaction space, Hadamard, project out
        d_interact = d_synergy * 2  # richer interaction space
        self.proj_v_interact = nn.Linear(d_model, d_interact, bias=False)
        self.proj_t_interact = nn.Linear(d_model, d_interact, bias=False)
        self.W_vt = nn.Linear(d_interact, d_synergy, bias=False)
        # Learnable scale so the cross-modal term can compete with unimodal terms
        self.interaction_scale = nn.Parameter(torch.tensor(1.0))

        self.interact_bias = nn.Parameter(torch.zeros(d_synergy))

        # ------------------------------------------------------------------
        # Step 2: Unimodal Predictors
        # Each predicts what a single modality alone can contribute.
        # The synergistic residual = interaction - (vision_only + text_only)
        # ------------------------------------------------------------------
        self.predictor_v = UnimodalPredictor(d_model, d_synergy, dropout)
        self.predictor_t = UnimodalPredictor(d_model, d_synergy, dropout)

        # ------------------------------------------------------------------
        # Step 3: Projection to synergy space
        # ------------------------------------------------------------------
        self.projection = nn.Sequential(
            nn.Linear(d_synergy, d_synergy),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_synergy, d_synergy),
        )
        self.layer_norm = nn.LayerNorm(d_synergy)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights for stable training."""
        for module in [self.W_v, self.W_t, self.W_vt,
                       self.proj_v_interact, self.proj_t_interact]:
            nn.init.xavier_uniform_(module.weight)

        for predictor in [self.predictor_v, self.predictor_t]:
            for m in predictor.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

        for m in self.projection.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self, v: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the synergistic representation.

        Args:
            v: Vision features [B, d_model] (pooled, from DsRA-adapted encoder).
            t: Text features [B, d_model] (pooled, from DsRA-adapted encoder).

        Returns:
            s: Synergistic representation [B, d_synergy].
        """
        # Step 1: Bilinear interaction
        # h_interact captures ALL joint information (shared + unique + synergistic)
        # Cross-modal: project to interaction space, Hadamard product, project out
        v_inter = self.proj_v_interact(v)  # [B, d_interact]
        t_inter = self.proj_t_interact(t)  # [B, d_interact]
        cross_modal = self.W_vt(v_inter * t_inter) * self.interaction_scale

        h_interact = (
            self.W_v(v)
            + self.W_t(t)
            + cross_modal
            + self.interact_bias
        )
        # Note: no nonlinearity here — keeping this linear preserves the
        # PID decomposition (ReLU(a+b) ≠ ReLU(a)+ReLU(b)). The projection
        # MLP in step 3 provides sufficient nonlinearity.

        # Step 2: Residual subtraction (PID approximation)
        # Subtract what each modality can predict independently
        h_v_only = self.predictor_v(v)  # Vision-alone contribution
        h_t_only = self.predictor_t(t)  # Text-alone contribution
        s_raw = h_interact - (h_v_only + h_t_only)

        # Step 3: Project to synergy space and normalize
        s = self.projection(s_raw)
        s = self.layer_norm(s)

        return s

    def forward_pairwise(
        self, v: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the synergistic representation for all N x M pairs.

        Args:
            v: Vision features [N, d_model].
            t: Text features [M, d_model].

        Returns:
            s: Synergistic representation [N, M, d_synergy].
        """
        # Step 1: Bilinear interaction
        v_inter = self.proj_v_interact(v)  # [N, d_interact]
        t_inter = self.proj_t_interact(t)  # [M, d_interact]
        
        # Broadcasting [N, 1, d_interact] * [1, M, d_interact] -> [N, M, d_interact]
        cross_modal = self.W_vt(v_inter.unsqueeze(1) * t_inter.unsqueeze(0)) * self.interaction_scale

        h_interact = (
            self.W_v(v).unsqueeze(1)
            + self.W_t(t).unsqueeze(0)
            + cross_modal
            + self.interact_bias
        )
        # Note: no nonlinearity — same reasoning as forward()

        # Step 2: Residual subtraction
        h_v_only = self.predictor_v(v)  # [N, d_synergy]
        h_t_only = self.predictor_t(t)  # [M, d_synergy]
        s_raw = h_interact - (h_v_only.unsqueeze(1) + h_t_only.unsqueeze(0))

        # Step 3: Project to synergy space and normalize
        s = self.projection(s_raw)
        s = self.layer_norm(s)

        return s

    def get_param_count(self) -> dict:
        """Return parameter count breakdown for this module."""
        counts = {
            "interaction_weights": sum(
                p.numel() for p in [self.W_v.weight, self.W_t.weight, self.W_vt.weight, self.interact_bias]
            ),
            "unimodal_predictors": sum(
                p.numel() for p in list(self.predictor_v.parameters()) + list(self.predictor_t.parameters())
            ),
            "projection": sum(p.numel() for p in self.projection.parameters()) + sum(p.numel() for p in self.layer_norm.parameters()),
        }
        counts["total"] = sum(counts.values())
        return counts
