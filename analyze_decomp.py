"""
Decomposition diagnostics for Decomp checkpoints.

  A. Branch geometry          : cos(e_V,e_T), cos(e_V,e_S), cos(e_T,e_S), gates.
  B. Linear PID on composed q : R²(e_V→q), R²(e_T→q), R²(e_S→q),
                                R²((e_V,e_T)→q), R²((all)→q).
  C. Synergy V/T sensitivity  : cos(e_S, e_S | v_shuf), cos(... | t_shuf).
  D. Synergy noise robustness : cos(e_S, e_S | v+ε), cos(... | t+ε)  at σ ∈ {0.05, 0.1, 0.2}.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from decomp.model   import DecompModel
from decomp.dataset import CIRRClsQuery, query_collate
from eval_decomp import build_model


def linear_r2(X_tr, Y_tr, X_te, Y_te, ridge=1e-3):
    XTX = X_tr.t() @ X_tr
    reg = ridge * torch.eye(X_tr.size(1), device=X_tr.device)
    W = torch.linalg.solve(XTX + reg, X_tr.t() @ Y_tr)
    Y_pred = X_te @ W
    ss_res = ((Y_te - Y_pred) ** 2).sum()
    ss_tot = ((Y_te - Y_te.mean(dim=0, keepdim=True)) ** 2).sum()
    return float(1 - ss_res / ss_tot.clamp(min=1e-9))


@torch.no_grad()
def collect_branches(model, loader, device):
    eVs, eTs, eSs = [], [], []
    for batch in loader:
        v = batch["v"].to(device); t = batch["t"].to(device)
        eVs.append(model.encode_v(v).cpu())
        eTs.append(model.encode_t(t).cpu())
        eSs.append(model.encode_s(v, t).cpu())
    return torch.cat(eVs), torch.cat(eTs), torch.cat(eSs)


def branch_geometry(e_V, e_T, e_S, alpha, beta, gamma):
    def cos(a, b):
        return float(F.cosine_similarity(a, b, dim=-1).mean())
    return {
        "cos(e_V,e_T)": cos(e_V, e_T),
        "cos(e_V,e_S)": cos(e_V, e_S),
        "cos(e_T,e_S)": cos(e_T, e_S),
        "alpha": alpha, "beta": beta, "gamma": gamma,
    }


def pid_on_q(e_V, e_T, e_S, alpha, beta, gamma, seed=0, train_frac=0.8):
    q = alpha * e_V + beta * e_T + gamma * e_S
    g = torch.Generator().manual_seed(seed)
    n = q.size(0)
    perm = torch.randperm(n, generator=g)
    n_tr = int(train_frac * n)
    tr, te = perm[:n_tr], perm[n_tr:]
    def C(X): return X - X.mean(dim=0, keepdim=True)
    Vc, Tc, Sc, qc = C(e_V), C(e_T), C(e_S), C(q)
    R2_V   = linear_r2(Vc[tr], qc[tr], Vc[te], qc[te])
    R2_T   = linear_r2(Tc[tr], qc[tr], Tc[te], qc[te])
    R2_S   = linear_r2(Sc[tr], qc[tr], Sc[te], qc[te])
    R2_VT  = linear_r2(torch.cat([Vc, Tc],    -1)[tr], qc[tr],
                       torch.cat([Vc, Tc],    -1)[te], qc[te])
    R2_VTS = linear_r2(torch.cat([Vc, Tc, Sc],-1)[tr], qc[tr],
                       torch.cat([Vc, Tc, Sc],-1)[te], qc[te])
    return {
        "R2(e_V->q)":             R2_V,
        "R2(e_T->q)":             R2_T,
        "R2(e_S->q)":             R2_S,
        "R2((e_V,e_T)->q)":       R2_VT,
        "R2((e_V,e_T,e_S)->q)":   R2_VTS,
        "synergy_share (R2_all - R2_VT)": R2_VTS - R2_VT,
    }


@torch.no_grad()
def synergy_shuffle_sensitivity(model, loader, device, n_perturb=4, seed=0):
    g = torch.Generator(device=device).manual_seed(seed)
    cs_V, cs_T, cs_B = [], [], []
    for batch in loader:
        v = batch["v"].to(device); t = batch["t"].to(device)
        e_S = model.encode_s(v, t)
        for _ in range(n_perturb):
            pV = torch.randperm(v.size(0), generator=g, device=device)
            pT = torch.randperm(v.size(0), generator=g, device=device)
            cs_V.append(F.cosine_similarity(e_S, model.encode_s(v[pV], t),     dim=-1).cpu())
            cs_T.append(F.cosine_similarity(e_S, model.encode_s(v,     t[pT]), dim=-1).cpu())
            cs_B.append(F.cosine_similarity(e_S, model.encode_s(v[pV], t[pT]), dim=-1).cpu())
    return {
        "cos(e_S,e_S | v_shuf)":   float(torch.cat(cs_V).mean()),
        "cos(e_S,e_S | t_shuf)":   float(torch.cat(cs_T).mean()),
        "cos(e_S,e_S | v&t_shuf)": float(torch.cat(cs_B).mean()),
    }


@torch.no_grad()
def synergy_noise_robustness(model, loader, device, sigmas=(0.05, 0.1, 0.2),
                              n_perturb=4, seed=0):
    torch.manual_seed(seed)
    out = {}
    for sigma in sigmas:
        cs = []
        for batch in loader:
            v = batch["v"].to(device); t = batch["t"].to(device)
            e_S = model.encode_s(v, t)
            for _ in range(n_perturb):
                v_n = F.normalize(v + sigma * torch.randn_like(v), dim=-1)
                t_n = F.normalize(t + sigma * torch.randn_like(t), dim=-1)
                cs.append(F.cosine_similarity(e_S, model.encode_s(v_n, t_n),
                                              dim=-1).cpu())
        out[f"cos_noise_sigma_{sigma:.2f}"] = float(torch.cat(cs).mean())
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--cache-dir", default="./cache/cirr_cls")
    p.add_argument("--split", default="val")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output", default=None)
    args = p.parse_args()

    device = torch.device(args.device)
    model = build_model(args.checkpoint, device)
    cache_split = Path(args.cache_dir) / args.split

    qds = CIRRClsQuery(cache_split)
    loader = DataLoader(qds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, collate_fn=query_collate)

    e_V, e_T, e_S = collect_branches(model, loader, device)
    alpha = model.alpha.item(); beta = model.beta.item(); gamma = model.gamma.item()

    out = {"gates": {"alpha": alpha, "beta": beta, "gamma": gamma}}

    print("\n=== A. Branch geometry ===")
    geo = branch_geometry(e_V, e_T, e_S, alpha, beta, gamma)
    out["branch_geometry"] = geo
    for k, v in geo.items(): print(f"  {k:<28s} {v:.4f}")

    print("\n=== B. Linear PID on q (composed) ===")
    pid = pid_on_q(e_V, e_T, e_S, alpha, beta, gamma)
    out["pid"] = pid
    for k, v in pid.items(): print(f"  {k:<40s} {v:.4f}")

    print("\n=== C. Synergy V/T-shuffle sensitivity ===")
    sens = synergy_shuffle_sensitivity(model, loader, device)
    out["sensitivity"] = sens
    for k, v in sens.items(): print(f"  {k:<32s} {v:.4f}")

    print("\n=== D. Synergy input-noise robustness ===")
    nr = synergy_noise_robustness(model, loader, device)
    out["noise_robustness"] = nr
    for k, v in nr.items(): print(f"  {k:<32s} {v:.4f}")

    print("\n  (Branch-ablation Recall lives in eval_decomp.py.)")

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        json.dump(out, open(args.output, "w"), indent=2)
        print(f"\n  Saved -> {args.output}")


if __name__ == "__main__":
    main()
