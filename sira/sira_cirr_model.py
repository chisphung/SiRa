"""
SIRA-CIRR Model — Synergistic Information-Aware Retrieval Adaptation
                   for Composed Image Retrieval (CIRR).

v3: Lightweight Text-Residual with Synergy-Only Delta

    v2 diagnostics showed:
    - Gates nearly closed (0.06) → v_final ≈ v_ref, t_final ≈ t_cap
    - Combiner MLP with 1M params overfitting on 27K samples (train 0.75 vs val 2.71)
    - R@1 still -1.48% below text-only

    v3 fixes:
    1. REMOVE the heavy MLP combiner (3*512 → 512 → 512 was 1M params)
    2. Use synergy directly as the correction: delta = proj(s)
       Since s = SIM(v_ref, t_cap) already captures the interaction,
       we don't need another MLP on top of v_final/t_final
    3. Much fewer params → less overfitting
    4. Learnable alpha with tighter clamp (max 0.5)

Architecture:
    v_ref, t_cap → SIM(v_ref, t_cap) → s
                 → delta = proj(s)                  # synergy IS the correction
                 → z_query = normalize(t_cap + α * delta)

    Total trainable params: ~350K (vs v2's 1.5M)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .sim import SynergisticInteractionModule
from .srg import SynergisticResidualGate


class SynergyDeltaCombiner(nn.Module):
    """
    Lightweight combiner: synergy → delta → text-residual query.

    Key insight: the synergy vector s already captures the non-decomposable
    interaction between v_ref and t_cap. We only need to PROJECT it into
    the CLIP embedding space as a correction to the text anchor.

    No need for a heavy MLP on [v_final; t_final] — that was overfitting.

    z_query = normalize(t_cap + α * proj(s))

    This design:
    - Has only ~65K params (vs 1M in v2's TextResidualCombiner)
    - Directly uses synergy as the correction signal
    - Prevents overfitting: no way to memorize train samples
    - α starts at 0.1 and is clamped to [0.01, 0.5]
    """

    def __init__(self, d_model=512, d_synergy=64, dropout=0.1):
        super().__init__()
        # Project synergy (d_synergy) → CLIP space (d_model) as correction
        self.synergy_to_delta = nn.Sequential(
            nn.Linear(d_synergy, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
        )
        # Learnable residual scale
        self.alpha = nn.Parameter(torch.tensor(0.1))

        self._init_weights()

    def _init_weights(self):
        for m in self.synergy_to_delta:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, s, t_cap):
        """
        Args:
            s:     (B, d_synergy) — synergy vector from SIM
            t_cap: (B, d_model) — original text embedding (anchor)

        Returns:
            z_query: (B, d_model) — L2-normalized composed query
            alpha:   scalar — current residual scale
        """
        delta = self.synergy_to_delta(s)  # (B, d_model)
        alpha_clamped = self.alpha.clamp(min=0.01, max=0.5)
        z_query = t_cap + alpha_clamped * delta
        z_query = F.normalize(z_query, p=2, dim=-1)
        return z_query, alpha_clamped


class SIRACIRRModel(nn.Module):
    """
    SIRA adapted for Composed Image Retrieval (CIRR).

    Supports v1, v2, and v3 checkpoints dynamically.

    v3: Lightweight design — synergy IS the correction.
    v2: TextResidualCombiner with s_proj and delta_net.
    v1: MLP query_combiner on top of srg outputs.

    Args:
        d_model:        Hidden dimension of CLIP (512 for ViT-B/32).
        d_synergy:      Synergistic bottleneck dimension. Default: 64.
        gate_rank:      Low-rank gate factorization rank. Default: 16.
        gate_init_bias: Gate init bias. Default: 0.0.
        dropout:        Dropout rate. Default: 0.1.
        version:        Model version ('v1', 'v2', or 'v3'). Default: 'v3'.
    """

    def __init__(self, d_model=512, d_synergy=64,
                 gate_rank=16, gate_init_bias=0.0, dropout=0.1, version="v3"):
        super().__init__()
        self.d_model = d_model
        self.d_synergy = d_synergy
        self.version = version

        # Core SIRA modules
        self.sim = SynergisticInteractionModule(d_model, d_synergy, dropout)
        self.srg = SynergisticResidualGate(
            d_model=d_model, d_synergy=d_synergy, gate_rank=gate_rank,
            gate_init_bias=gate_init_bias)

        if version == "v1":
            self.query_combiner = nn.Module()
            self.query_combiner.net = nn.Sequential(
                nn.Linear(d_model * 2, d_model * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model * 2, d_model),
                nn.LayerNorm(d_model),
            )
        elif version == "v2":
            self.combiner = nn.Module()
            self.combiner.s_proj = nn.Linear(d_synergy, d_model, bias=False)
            self.combiner.delta_net = nn.Sequential(
                nn.Linear(d_model * 3, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, d_model),
            )
            self.combiner.alpha = nn.Parameter(torch.tensor(0.1))
        else:  # v3
            self.combiner = SynergyDeltaCombiner(d_model, d_synergy, dropout)

    def forward(self, v_ref, t_cap):
        """
        Produce a composed query embedding from reference image + caption.

        Args:
            v_ref: (B, d_model) — CLIP CLS embedding of reference image (L2-normed)
            t_cap: (B, d_model) — CLIP text embedding of relative caption (L2-normed)

        Returns:
            z_query: (B, d_model) — L2-normalized composed query
            s:       (B, d_synergy) — synergy vector (for auxiliary losses)
        """
        # Step 1: Compute synergy
        s_raw = self.sim(v_ref, t_cap)

        # Step 2: L2-normalize synergy to unit scale, then restore capacity
        s = F.normalize(s_raw, p=2, dim=-1) * (self.d_synergy ** 0.5)

        # Step 3: SRG for gated features (used by L_orth loss)
        v_final, t_final = self.srg(v_ref, t_cap, s)

        # Step 4: Compose query
        if self.version == "v1":
            z_query = self.query_combiner.net(torch.cat([v_final, t_final], dim=-1))
            z_query = F.normalize(z_query, p=2, dim=-1)
        elif self.version == "v2":
            s_proj_out = self.combiner.s_proj(s)
            delta = self.combiner.delta_net(torch.cat([v_final, t_final, s_proj_out], dim=-1))
            alpha_clamped = self.combiner.alpha.clamp(min=0.01, max=0.5)
            z_query = t_cap + alpha_clamped * delta
            z_query = F.normalize(z_query, p=2, dim=-1)
        else:  # v3
            z_query, alpha = self.combiner(s, t_cap)

        return z_query, s

    @torch.no_grad()
    def encode_query(self, v_ref, t_cap):
        """Encode query for retrieval (inference mode)."""
        z_query, s = self.forward(v_ref, t_cap)
        return z_query

    def get_trainable_params(self):
        """Return all trainable parameters."""
        params = []
        combiner = self.query_combiner if self.version == "v1" else self.combiner
        for module in [self.sim, self.srg, combiner]:
            params.extend(module.parameters())
        return params

    def get_combiner_alpha(self):
        """Return current combiner residual scale for monitoring."""
        if self.version == "v1":
            return 0.0
        return self.combiner.alpha.item()

    def get_param_summary(self):
        """Return parameter count breakdown."""
        sim_p = sum(p.numel() for p in self.sim.parameters())
        srg_p = sum(p.numel() for p in self.srg.parameters())
        combiner = self.query_combiner if self.version == "v1" else self.combiner
        combiner_p = sum(p.numel() for p in combiner.parameters())
        total = sim_p + srg_p + combiner_p
        return {
            "sim": sim_p,
            "srg": srg_p,
            "combiner": combiner_p,
            "total_trainable": total,
            "sim_breakdown": self.sim.get_param_count(),
        }

    @classmethod
    def from_config(cls, d_model=512, d_synergy=64, device="cuda", **kwargs):
        """Factory: create SIRA-CIRR model from config."""
        model = cls(d_model=d_model, d_synergy=d_synergy, **kwargs)
        model = model.to(device)
        return model
