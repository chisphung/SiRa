"""
CIRR-Specific Loss Functions for SIRA.

Composed Image Retrieval requires different loss objectives than
standard image-text matching:

    1. L_retrieval:  InfoNCE aligning z_query with z_target
                     (the main task loss — can we retrieve the right image?)

    2. L_syn_align:  Synergy vector s should predict the MODIFICATION DELTA
                     (z_target - v_ref), not z_target directly.
                     This captures "what changes" rather than "what the target is".

    3. L_orth:       Orthogonality regularizer from SIRA — prevents synergy
                     from collapsing into redundant unimodal information.

Total: L_total = λ_ret * L_ret + λ_syn * L_syn_align + λ_orth * L_orth
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CIRRLoss(nn.Module):
    """
    Combined loss for SIRA-CIRR training.

    Supports phased introduction of loss terms:
        Phase 1: L_retrieval only
        Phase 2: + L_syn_align
        Phase 3: + L_orth

    Args:
        lambda_ret:   Weight for retrieval loss. Default: 1.0.
        lambda_syn:   Weight for synergy alignment loss. Default: 0.3.
        lambda_orth:  Weight for orthogonality loss. Default: 0.1.
        temperature:  InfoNCE temperature. Default: 0.07.
        learnable_temperature: Whether temperature is learnable. Default: True.
        syn_align_epoch: Epoch to start L_syn_align. Default: 5.
        orth_epoch:      Epoch to start L_orth. Default: 10.
    """

    def __init__(
        self,
        lambda_ret=1.0,
        lambda_syn=0.3,
        lambda_orth=0.1,
        temperature=0.07,
        learnable_temperature=True,
        syn_align_epoch=5,
        orth_epoch=10,
    ):
        super().__init__()
        self.lambda_ret = lambda_ret
        self.lambda_syn = lambda_syn
        self.lambda_orth = lambda_orth
        self.syn_align_epoch = syn_align_epoch
        self.orth_epoch = orth_epoch

        if learnable_temperature:
            self.log_temp = nn.Parameter(torch.log(torch.tensor(temperature)))
        else:
            self.register_buffer("log_temp", torch.log(torch.tensor(temperature)))

    def _get_temperature(self):
        return torch.exp(self.log_temp).clamp(min=1e-4, max=100.0)

    def l_retrieval(self, z_query, z_target, temperature=None):
        """
        Standard InfoNCE: align composed query with target image embedding.

        Args:
            z_query:  (B, D) — composed query from SIM+SRG+QueryCombiner
            z_target: (B, D) — CLIP embedding of target image
        """
        if temperature is None:
            temperature = self._get_temperature()

        B = z_query.shape[0]
        q = F.normalize(z_query, dim=-1)
        k = F.normalize(z_target, dim=-1)
        logits = q @ k.T / temperature
        labels = torch.arange(B, device=logits.device)

        # Symmetric InfoNCE (query→target + target→query)
        loss = (
            F.cross_entropy(logits, labels) +
            F.cross_entropy(logits.T, labels)
        ) / 2.0
        return loss

    def l_syn_align(self, s, v_ref, z_target, temperature=None):
        """
        Synergy alignment: s should predict the MODIFICATION DELTA.

        The modification delta = z_target - v_ref represents the direction
        of change from reference to target. Synergy (the non-decomposable
        interaction between image and text) should capture exactly this —
        what the modification text means in the context of this specific image.

        Args:
            s:        (B, d_synergy) — synergy vector from SIM
            v_ref:    (B, D) — reference image CLIP embedding
            z_target: (B, D) — target image CLIP embedding
        """
        if temperature is None:
            temperature = self._get_temperature()

        # Modification delta: what changes between reference and target
        delta = F.normalize(z_target - v_ref, dim=-1)  # (B, D)

        # Project synergy to same space as delta for alignment
        # Note: s is d_synergy dim, delta is d_model dim
        # We use cosine similarity which is scale-invariant,
        # but we need matching dimensions. The SRG proj_v already does
        # d_synergy → d_model. We'll use a simpler approach:
        # align s with itself across the batch (synergy should be
        # similar for samples that have similar deltas).

        # InfoNCE between normalized synergy vectors and normalized deltas
        # requires matching dimensions. Since d_synergy ≠ d_model,
        # we use the delta projected down to d_synergy space.
        # Alternatively, we use a batch-level alignment approach.

        B = s.shape[0]
        s_norm = F.normalize(s, dim=-1)        # (B, d_synergy)

        # Cross-sample alignment: synergy vectors that correspond to
        # similar deltas should be close. Use delta-delta similarity
        # as soft labels.
        delta_sim = delta @ delta.T              # (B, B) — soft labels
        s_sim = s_norm @ s_norm.T                # (B, B) — predictions

        # KL divergence between distributions
        delta_probs = F.softmax(delta_sim / temperature, dim=-1)
        s_log_probs = F.log_softmax(s_sim / temperature, dim=-1)
        loss = F.kl_div(s_log_probs, delta_probs, reduction="batchmean")

        return loss

    def l_orth(self, s, v_ref, t_cap, srg):
        """
        Orthogonality regularizer: synergy projections should be orthogonal
        to unimodal features, preventing synergy from collapsing into
        redundant information.

        Uses SRG's learned projections (the actual injection path).

        Args:
            s:     (B, d_synergy) — synergy vector
            v_ref: (B, D) — reference image embedding
            t_cap: (B, D) — caption text embedding
            srg:   SynergisticResidualGate module (for proj_v, proj_t)
        """
        s_v_proj = srg.proj_v(s)  # (B, D) — what gets injected into vision
        s_t_proj = srg.proj_t(s)  # (B, D) — what gets injected into text

        s_v_norm = F.normalize(s_v_proj, p=2, dim=-1)
        s_t_norm = F.normalize(s_t_proj, p=2, dim=-1)
        v_norm = F.normalize(v_ref, p=2, dim=-1)
        t_norm = F.normalize(t_cap, p=2, dim=-1)

        cos_v = (s_v_norm * v_norm).sum(dim=-1)
        cos_t = (s_t_norm * t_norm).sum(dim=-1)

        loss = (cos_v ** 2 + cos_t ** 2).mean()
        return loss

    def forward(self, z_query, z_target, s, v_ref, t_cap, srg, epoch=0):
        """
        Compute combined CIRR loss with phase-aware weighting.

        Args:
            z_query:  (B, D) — composed query embedding
            z_target: (B, D) — target image embedding
            s:        (B, d_synergy) — synergy vector
            v_ref:    (B, D) — reference image embedding
            t_cap:    (B, D) — caption text embedding
            srg:      SynergisticResidualGate module
            epoch:    Current epoch (for phased loss introduction)

        Returns:
            Dictionary of losses including 'total'.
        """
        temperature = self._get_temperature()
        device = z_query.device

        # Phase 1: Always on
        loss_ret = self.l_retrieval(z_query, z_target, temperature)

        # Phase 2: Synergy alignment
        if epoch >= self.syn_align_epoch and self.lambda_syn > 0:
            loss_syn = self.l_syn_align(s, v_ref, z_target, temperature)
        else:
            loss_syn = torch.tensor(0.0, device=device)

        # Phase 3: Orthogonality
        if epoch >= self.orth_epoch and self.lambda_orth > 0:
            loss_orth = self.l_orth(s, v_ref, t_cap, srg)
        else:
            loss_orth = torch.tensor(0.0, device=device)

        # Total
        total = (
            self.lambda_ret * loss_ret +
            self.lambda_syn * loss_syn +
            self.lambda_orth * loss_orth
        )

        return {
            "total": total,
            "l_ret": loss_ret.detach(),
            "l_syn": loss_syn.detach() if isinstance(loss_syn, torch.Tensor) else loss_syn,
            "l_orth": loss_orth.detach() if isinstance(loss_orth, torch.Tensor) else loss_orth,
            "temperature": temperature.detach(),
            "synergy_norm": s.norm(dim=-1).mean().detach(),
        }
