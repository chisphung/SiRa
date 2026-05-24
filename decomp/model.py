"""
DecompModel — minimal three-branch additively decomposable composed-query model.

Designed for the CLS-level CIRR cache (cache/cirr_cls/):
    v_ref ∈ ℝ^512   (reference image CLS)
    t_cap ∈ ℝ^512   (caption CLS)

    e_V = MLP_v(v_ref)                       ← V-only path
    e_T = MLP_t(t_cap)                       ← T-only path
    e_S = MLP_s( v_ref ⊙ t_cap )             ← synergy path (Hadamard mix)

    q   = α · e_V + β · e_T + γ · e_S        (3 learnable scalars)
    q   ← L2-normalize(q)

Why this is decomposable:
    - e_V is a function of v_ref only      → V-private channel.
    - e_T is a function of t_cap only      → T-private channel.
    - e_S requires BOTH (Hadamard product zeros out if either is zero, and
      the function v→f(v⊙t) is constant in v when t≡0).
    - Composition is a weighted sum of unit vectors, so the contribution
      of each branch to q is exactly the corresponding γ-scaled vector.

No transformer, no attention, no cross-modal interaction beyond the elementwise
product fed into a small MLP. ~2 M params at d_model=512.
"""

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _mlp(d_in: int, d_hidden: int, d_out: int, dropout: float = 0.0) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(d_in, d_hidden),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(d_hidden, d_out),
    )


class DecompModel(nn.Module):
    def __init__(
        self,
        d_in: int = 512,
        d_hidden: int = 512,
        d_out: int = 512,
        dropout: float = 0.1,
        synergy_op: str = "hadamard",   # "hadamard" | "lowrank_bilinear"
        bilinear_rank: int = 64,
    ):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.synergy_op = synergy_op

        # Two unimodal MLP branches (LN at input for stability)
        self.ln_v = nn.LayerNorm(d_in)
        self.ln_t = nn.LayerNorm(d_in)
        self.branch_v = _mlp(d_in, d_hidden, d_out, dropout)
        self.branch_t = _mlp(d_in, d_hidden, d_out, dropout)

        # Synergy branch
        if synergy_op == "hadamard":
            self.ln_s = nn.LayerNorm(d_in)
            self.branch_s = _mlp(d_in, d_hidden, d_out, dropout)
        elif synergy_op == "lowrank_bilinear":
            # f_s = MLP_s( (W_v v) ⊙ (W_t t) ),  W_*: ℝ^{d_in × rank}
            self.proj_vs = nn.Linear(d_in, bilinear_rank, bias=False)
            self.proj_ts = nn.Linear(d_in, bilinear_rank, bias=False)
            self.ln_s = nn.LayerNorm(bilinear_rank)
            self.branch_s = _mlp(bilinear_rank, d_hidden, d_out, dropout)
        else:
            raise ValueError(f"Unknown synergy_op={synergy_op}")

        # Fusion gates (3 scalars), init=1 so every branch contributes equally at start
        self.alpha = nn.Parameter(torch.ones(1))
        self.beta  = nn.Parameter(torch.ones(1))
        self.gamma = nn.Parameter(torch.ones(1))

    # ------------------------------------------------------------------
    # Branch encoders
    # ------------------------------------------------------------------
    def encode_v(self, v: torch.Tensor) -> torch.Tensor:
        e = self.branch_v(self.ln_v(v))
        return F.normalize(e, dim=-1)

    def encode_t(self, t: torch.Tensor) -> torch.Tensor:
        e = self.branch_t(self.ln_t(t))
        return F.normalize(e, dim=-1)

    def encode_s(self, v: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if self.synergy_op == "hadamard":
            x = v * t
        else:  # lowrank_bilinear
            x = self.proj_vs(v) * self.proj_ts(t)
        e = self.branch_s(self.ln_s(x))
        return F.normalize(e, dim=-1)

    # ------------------------------------------------------------------
    # Composition
    # ------------------------------------------------------------------
    def compose(self, e_V: torch.Tensor, e_T: torch.Tensor, e_S: torch.Tensor) -> torch.Tensor:
        q_raw = self.alpha * e_V + self.beta * e_T + self.gamma * e_S
        return F.normalize(q_raw, dim=-1)

    def forward(self, v: torch.Tensor, t: torch.Tensor,
                return_branches: bool = False):
        e_V = self.encode_v(v)
        e_T = self.encode_t(t)
        e_S = self.encode_s(v, t)
        q = self.compose(e_V, e_T, e_S)
        if return_branches:
            return q, {
                "e_V": e_V, "e_T": e_T, "e_S": e_S,
                "alpha": self.alpha.detach().clone(),
                "beta":  self.beta.detach().clone(),
                "gamma": self.gamma.detach().clone(),
            }
        return q

    # ------------------------------------------------------------------
    def get_param_count(self) -> Dict[str, int]:
        n_total = sum(p.numel() for p in self.parameters())
        n_train = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": n_total, "trainable": n_train}
