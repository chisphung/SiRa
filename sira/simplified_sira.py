import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleSIM(nn.Module):
    """
    Simplified Synergistic Interaction Module.
    Directly extracts cross-modal synergy from v and t.
    """
    def __init__(self, d_model=512, d_synergy=64):
        super().__init__()
        self.proj_v = nn.Linear(d_model, d_synergy, bias=False)
        self.proj_t = nn.Linear(d_model, d_synergy, bias=False)
        self.mlp = nn.Sequential(
            nn.Linear(d_synergy, d_synergy),
            nn.GELU(),
            nn.Linear(d_synergy, d_synergy)
        )
        
    def forward(self, v, t):
        # v: (B, 1, d_model), t: (1, B, d_model) or similar broadcastable shapes
        inter = self.proj_v(v) * self.proj_t(t)
        return self.mlp(inter)


class SimplifiedSIRA(nn.Module):
    """
    Simplified SIRA Architecture.
    Computes direct scalar scores for pairwise inputs.
    """
    def __init__(self, clip_model, d_model=512, d_synergy=64):
        super().__init__()
        self.clip = clip_model
        
        # Freeze CLIP
        for p in self.clip.parameters():
            p.requires_grad = False
        self.clip.eval()
            
        self.d_model = d_model
        
        # SIRA components
        self.sim = SimpleSIM(d_model, d_synergy)
        self.synergy_scorer = nn.Linear(d_synergy, 1)
        
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.gamma = nn.Parameter(torch.tensor(1.0))
        
        # InfoNCE temperature
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07)) if 'np' in globals() else nn.Parameter(torch.tensor(2.6592))

    def train(self, mode=True):
        super().train(mode)
        # Always keep CLIP in eval mode
        self.clip.eval()
        return self

    def _encode_image(self, images):
        with torch.no_grad():
            features = self.clip.encode_image(images)
        return F.normalize(features, p=2, dim=-1)

    def _encode_text(self, texts):
        with torch.no_grad():
            features = self.clip.encode_text(texts)
        return F.normalize(features, p=2, dim=-1)

    def forward(self, images, texts):
        """
        Computes pairwise scores for a batch of images and texts.
        Returns: (B, B) matrix of scores.
        """
        # 1. Base features
        v = self._encode_image(images) # (B, d_model)
        t = self._encode_text(texts)   # (B, d_model)
        
        # 2. Pairwise computation
        # base_scores: (B, B)
        base_scores = v @ t.T
        
        # synergy computation
        v_exp = v.unsqueeze(1) # (B, 1, d_model)
        t_exp = t.unsqueeze(0) # (1, B, d_model)
        s_matrix = self.sim(v_exp, t_exp) # (B, B, d_synergy)
        
        syn_scores = self.synergy_scorer(s_matrix).squeeze(-1) # (B, B)
        
        # 3. Final pairwise scores
        final_scores = self.alpha * base_scores + self.gamma * syn_scores
        
        # Scale for contrastive loss
        logit_scale = self.logit_scale.exp()
        return final_scores * logit_scale
    
    @classmethod
    def from_clip(cls, clip_model_name="ViT-B/32", device="cuda", d_synergy=64, **kwargs):
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

        model = cls(clip_model, d_model, d_synergy)
        model = model.to(device)
        return model, preprocess
