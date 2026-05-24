"""Symmetric InfoNCE retrieval loss (CLIP-style learnable temperature)."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SymInfoNCE(nn.Module):
    def __init__(self, init_temperature: float = 0.07, max_scale: float = 100.0):
        super().__init__()
        self.log_temp = nn.Parameter(torch.tensor(0.0))
        self.init_temperature = init_temperature
        self.max_scale = max_scale

    def forward(self, q: torch.Tensor, z: torch.Tensor):
        scale = (1.0 / self.init_temperature) * self.log_temp.exp()
        scale = scale.clamp(max=self.max_scale)
        logits = scale * (q @ z.t())
        labels = torch.arange(q.size(0), device=q.device)
        loss = 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))
        with torch.no_grad():
            acc = (logits.argmax(dim=1) == labels).float().mean()
        return loss, acc
