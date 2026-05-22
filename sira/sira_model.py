"""
SIRA Model — Unified wrapper integrating frozen CLIP with Synergistic modules.

Usage:
    model, preprocess = SIRAModel.from_clip("ViT-B/32", device="cuda")
    losses = model(images, texts)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .sim import SynergisticInteractionModule
from .srg import SynergisticResidualGate
from .losses import SynergisticAwareContrastiveLoss


class SIRAModel(nn.Module):
    """
    Synergistic Information-Aware Retrieval Adaptation (SIRA).

    Args:
        clip_model: A pre-loaded CLIP model (will be frozen).
        d_model: Hidden dimension of CLIP (512 for ViT-B/32).
        d_synergy: Synergistic bottleneck dimension. Default: 64.
        gate_rank: Low-rank gate factorization rank. Default: 16.
        lambda_orth: Orthogonality loss weight. Default: 0.1.
        gate_init_bias: Gate init bias. Default: 0.0.
        dropout: Dropout rate. Default: 0.1.
    """

    def __init__(self, clip_model, d_model=512, d_synergy=64,
                 gate_rank=16, lambda_orth=0.1,
                 gate_init_bias=0.0, dropout=0.1, **kwargs):
        super().__init__()
        self.clip = clip_model
        for param in self.clip.parameters():
            param.requires_grad = False
        self.clip.eval()  # M6: keep CLIP in eval (no training-time dropout/noise)
        self.d_model = d_model
        self.d_synergy = d_synergy
        self._unfrozen_backbone_params = []  # track which backbone params are trainable

        self.sim = SynergisticInteractionModule(d_model, d_synergy, dropout)
        self.srg = SynergisticResidualGate(
            d_model=d_model, d_synergy=d_synergy, gate_rank=gate_rank,
            gate_init_bias=gate_init_bias)
        # C1: scorer removed — it was never trained (zero gradient signal)
        self.loss_fn = SynergisticAwareContrastiveLoss(
            lambda_orth=lambda_orth)

    def train(self, mode: bool = True):
        """Override train() to keep CLIP backbone in eval mode always."""
        super().train(mode)
        self.clip.eval()  # M6: CLIP stays in eval regardless
        return self

    def _encode_image(self, images):
        """Encode images through CLIP visual encoder.

        When projections are unfrozen, only the projection runs with grad;
        the frozen transformer runs under no_grad to save memory (H4).
        """
        if self._unfrozen_backbone_params:
            # H4: Run the frozen transformer under no_grad, then let the
            # projection (which has requires_grad=True) run with grad.
            # We check if visual.proj is among the unfrozen params.
            visual = self.clip.visual
            has_unfrozen_proj = (
                hasattr(visual, 'proj') and visual.proj is not None
                and any(p is visual.proj for p in self._unfrozen_backbone_params)
            )
            if has_unfrozen_proj:
                with torch.no_grad():
                    x = visual.conv1(images.type(visual.conv1.weight.dtype))
                    x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
                    x = torch.cat([
                        visual.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
                        x
                    ], dim=1)
                    x = x + visual.positional_embedding.to(x.dtype)
                    x = visual.ln_pre(x)
                    x = x.permute(1, 0, 2)  # NLD -> LND
                    x = visual.transformer(x)
                    x = x.permute(1, 0, 2)  # LND -> NLD
                    x = visual.ln_post(x[:, 0, :])
                # Projection runs WITH grad
                if visual.proj is not None:
                    x = x @ visual.proj
                features = x
            else:
                # Other unfreezing strategy (e.g., last-layer, full)
                features = self.clip.encode_image(images)
        else:
            with torch.no_grad():
                features = self.clip.encode_image(images)
        # H3: removed .float() — let AMP control dtype
        return F.normalize(features, p=2, dim=-1)

    def _encode_text(self, texts):
        """Encode text through CLIP text encoder.

        When projections are unfrozen, only the projection runs with grad;
        the frozen transformer runs under no_grad to save memory (H4).
        """
        if self._unfrozen_backbone_params:
            clip_model = self.clip
            has_unfrozen_text_proj = (
                hasattr(clip_model, 'text_projection')
                and clip_model.text_projection is not None
                and any(p is clip_model.text_projection for p in self._unfrozen_backbone_params)
            )
            if has_unfrozen_text_proj:
                with torch.no_grad():
                    x = clip_model.token_embedding(texts).type(clip_model.dtype)
                    x = x + clip_model.positional_embedding.type(clip_model.dtype)
                    x = x.permute(1, 0, 2)  # NLD -> LND
                    x = clip_model.transformer(x)
                    x = x.permute(1, 0, 2)  # LND -> NLD
                    x = clip_model.ln_final(x).type(clip_model.dtype)
                # Projection runs WITH grad
                x = x[torch.arange(x.shape[0]), texts.argmax(dim=-1)] @ clip_model.text_projection
                features = x
            else:
                features = self.clip.encode_text(texts)
        else:
            with torch.no_grad():
                features = self.clip.encode_text(texts)
        # H3: removed .float() — let AMP control dtype
        return F.normalize(features, p=2, dim=-1)

    def forward(self, images, texts):
        """Forward: encode -> SIM -> SRG -> loss. Returns dict of losses."""
        v_shared = self._encode_image(images)
        t_shared = self._encode_text(texts)
        # M3: removed redundant detach().requires_grad_(True)
        # — tensors from no_grad() are already leaf tensors not on any graph

        # Compute synergy vector for matched pairs
        s = self.sim(v_shared, t_shared)

        # Fuse synergy back into unimodal features using the gate
        v_final, t_final = self.srg(v_shared, t_shared, s)

        # C2: extract the SRG projections used for actual fusion,
        # pass them to L_orth so orthogonality acts on the real injection path
        s_v_proj = self.srg.proj_v(s)
        s_t_proj = self.srg.proj_t(s)

        losses = self.loss_fn(v_final, t_final, s, v_shared, t_shared,
                              s_v_proj=s_v_proj, s_t_proj=s_t_proj)
        losses["synergy_norm"] = s.norm(dim=-1).mean().detach()
        return losses

    @torch.no_grad()
    def encode_for_retrieval(self, images=None, texts=None):
        """Encode for retrieval."""
        result = {}
        if images is not None and texts is not None:
            v_shared = self._encode_image(images)
            t_shared = self._encode_text(texts)
            s = self.sim(v_shared, t_shared)
            v_final, t_final = self.srg(v_shared, t_shared, s)
            result["image_features"] = v_final
            result["text_features"] = t_final
            result["synergy"] = s
        elif images is not None:
            result["image_features"] = self._encode_image(images)
        elif texts is not None:
            result["text_features"] = self._encode_text(texts)
        return result

    def get_trainable_params(self):
        """Return list of trainable parameters (SIRA modules + unfrozen backbone)."""
        params = []
        for module in [self.sim, self.srg, self.loss_fn]:
            params.extend(module.parameters())
        params.extend(self._unfrozen_backbone_params)
        return params

    def get_trainable_param_groups(self, lr=5e-4, backbone_lr_scale=0.1):
        """Return param groups with different LRs for backbone vs adapter."""
        adapter_params = []
        for module in [self.sim, self.srg, self.loss_fn]:
            adapter_params.extend(module.parameters())

        groups = [{"params": adapter_params, "lr": lr}]
        if self._unfrozen_backbone_params:
            groups.append({
                "params": self._unfrozen_backbone_params,
                "lr": lr * backbone_lr_scale,
                "weight_decay": 0.001,  # lighter regularization for pretrained weights
            })
        return groups

    def unfreeze_layers(self, strategy="projection"):
        """
        Unfreeze CLIP backbone using specified strategy:
        'projection': Unfreeze only visual and text projection heads
        'last-layer': Unfreeze projections + last transformer block
        'full': Unfreeze entire model
        """
        unfrozen = []

        if strategy == "full":
            for p in self.clip.parameters():
                p.requires_grad = True
                unfrozen.append(p)
            self._unfrozen_backbone_params = unfrozen
            n_unfrozen = sum(p.numel() for p in unfrozen)
            print(f"  Unfroze {len(unfrozen)} param tensors ({n_unfrozen:,} params) "
                  f"using strategy: {strategy}")
            return unfrozen

        # 1. Unfreeze Projections
        if hasattr(self.clip, "text_projection") and self.clip.text_projection is not None:
            if isinstance(self.clip.text_projection, nn.Parameter):
                self.clip.text_projection.requires_grad = True
                unfrozen.append(self.clip.text_projection)
            else:
                for p in self.clip.text_projection.parameters():
                    p.requires_grad = True
                    unfrozen.append(p)

        if hasattr(self.clip, "visual") and hasattr(self.clip.visual, "proj"):
            if self.clip.visual.proj is not None:
                if isinstance(self.clip.visual.proj, nn.Parameter):
                    self.clip.visual.proj.requires_grad = True
                    unfrozen.append(self.clip.visual.proj)
                else:
                    for p in self.clip.visual.proj.parameters():
                        p.requires_grad = True
                        unfrozen.append(p)

        if hasattr(self.clip, "visual") and hasattr(self.clip.visual, "head"):
            for p in self.clip.visual.head.parameters():
                p.requires_grad = True
                unfrozen.append(p)

        # 2. Unfreeze Last Layer if requested
        if strategy == "last-layer":
            if hasattr(self.clip, "visual"):
                visual = self.clip.visual
                if hasattr(visual, "transformer") and hasattr(visual.transformer, "resblocks"):
                    for p in visual.transformer.resblocks[-1].parameters():
                        p.requires_grad = True
                        unfrozen.append(p)
                elif hasattr(visual, "blocks"):
                    for p in visual.blocks[-1].parameters():
                        p.requires_grad = True
                        unfrozen.append(p)
                if hasattr(visual, "ln_post"):
                    for p in visual.ln_post.parameters():
                        p.requires_grad = True
                        unfrozen.append(p)

            if hasattr(self.clip, "transformer") and hasattr(self.clip.transformer, "resblocks"):
                for p in self.clip.transformer.resblocks[-1].parameters():
                    p.requires_grad = True
                    unfrozen.append(p)
            if hasattr(self.clip, "ln_final"):
                for p in self.clip.ln_final.parameters():
                    p.requires_grad = True
                    unfrozen.append(p)

        self._unfrozen_backbone_params = unfrozen
        n_unfrozen = sum(p.numel() for p in unfrozen)
        print(f"  Unfroze {len(unfrozen)} param tensors ({n_unfrozen:,} params) "
              f"using strategy: {strategy}")
        return unfrozen

    def get_param_summary(self):
        """Return parameter count summary."""
        frozen = sum(p.numel() for p in self.clip.parameters() if not p.requires_grad)
        backbone_trainable = sum(p.numel() for p in self._unfrozen_backbone_params)
        sim_p = sum(p.numel() for p in self.sim.parameters())
        srg_p = sum(p.numel() for p in self.srg.parameters())
        loss_p = sum(p.numel() for p in self.loss_fn.parameters())
        adapter_trainable = sim_p + srg_p + loss_p
        total_trainable = adapter_trainable + backbone_trainable
        total_all = frozen + total_trainable
        return {
            "frozen": frozen,
            "backbone_trainable": backbone_trainable,
            "sim": sim_p, "srg": srg_p, "loss_fn": loss_p,
            "total_trainable": total_trainable,
            "total": total_all,
            "trainable_pct": f"{total_trainable / total_all * 100:.3f}%",
            "sim_breakdown": self.sim.get_param_count(),
        }

    @classmethod
    def from_clip(cls, clip_model_name="ViT-B/32", device="cuda",
                  d_synergy=64, **kwargs):
        """Factory: create SIRA from a CLIP model name."""
        try:
            import clip
            clip_model, preprocess = clip.load(clip_model_name, device=device)
            clip_model = clip_model.float()
        except ImportError:
            import open_clip
            clip_model, _, preprocess = open_clip.create_model_and_transforms(
                clip_model_name.replace("/", "-"), pretrained="openai")
            clip_model = clip_model.to(device).float()

        if hasattr(clip_model, "text_projection"):
            d_model = clip_model.text_projection.shape[1]
        elif hasattr(clip_model, "embed_dim"):
            d_model = clip_model.embed_dim
        else:
            d_model = {"ViT-B/32": 512, "ViT-B/16": 512, "ViT-L/14": 768}.get(clip_model_name, 512)

        model = cls(clip_model, d_model, d_synergy, **kwargs)
        model = model.to(device)
        return model, preprocess
