"""
Evaluation for Decomp checkpoints.

Reports retrieval for all 7 branch-ablation variants of q:
    q_full, q_no_S, q_no_V, q_no_T, q_only_V, q_only_T, q_only_S.

Also reports CIRR subset R@1/2/3 from img_set_members.
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
from decomp.dataset import CIRRClsQuery, query_collate, load_gallery


def build_model(ckpt_path: str, device: torch.device) -> DecompModel:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = DecompModel(
        d_in=cfg["d_in"], d_out=cfg["d_out"],
        synergy_op=cfg.get("synergy_op", "hadamard"),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"  Loaded {ckpt_path} (epoch {ckpt.get('epoch','?')})")
    return model


def _recall_from_sims(sims, refs, tgts, members, gallery_ids, id_to_idx):
    sims = sims.clone()
    for i, ref in enumerate(refs):
        if ref in id_to_idx:
            sims[i, id_to_idx[ref]] = -float("inf")
    sorted_idx = torch.argsort(sims, dim=-1, descending=True).numpy()
    gids_np = np.array(gallery_ids)
    Ks = [1, 5, 10, 50]; sub_Ks = [1, 2, 3]
    rec = {k: 0 for k in Ks}; sub = {k: 0 for k in sub_Ks}
    ranks = []; n = 0; n_sub = 0
    for i, tgt in enumerate(tgts):
        if not tgt or tgt not in id_to_idx:
            continue
        n += 1
        ranked = gids_np[sorted_idx[i]]
        pos = np.where(ranked == tgt)[0]
        r = int(pos[0]) if len(pos) else len(ranked)
        ranks.append(r)
        for k in Ks:
            if r < k: rec[k] += 1
        member_set = [m for m in members[i] if m != refs[i] and m in id_to_idx]
        if member_set:
            n_sub += 1
            mask = np.isin(ranked, member_set)
            sranked = ranked[mask]
            sp = np.where(sranked == tgt)[0]
            if len(sp):
                rs = int(sp[0])
                for k in sub_Ks:
                    if rs < k: sub[k] += 1
    return {
        "n_eval": n,
        "median_rank": float(np.median(ranks)) if ranks else float("inf"),
        "recall":        {f"R@{k}": rec[k] / max(n, 1) * 100 for k in Ks},
        "subset_recall": {f"R_sub@{k}": sub[k] / max(n_sub, 1) * 100 for k in sub_Ks},
    }


@torch.no_grad()
def evaluate(args):
    device = torch.device(args.device)
    model = build_model(args.checkpoint, device)
    cache_split = Path(args.cache_dir) / args.split

    gallery_emb_cpu, gallery_ids = load_gallery(cache_split)
    gallery_emb = F.normalize(gallery_emb_cpu, dim=-1).to(device)
    id_to_idx = {iid: i for i, iid in enumerate(gallery_ids)}
    print(f"  Gallery: {len(gallery_ids)} images")

    qds = CIRRClsQuery(cache_split)
    loader = DataLoader(qds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, collate_fn=query_collate)

    eVs, eTs, eSs = [], [], []
    refs, tgts, members = [], [], []
    for batch in loader:
        v = batch["v"].to(device); t = batch["t"].to(device)
        eVs.append(model.encode_v(v).cpu())
        eTs.append(model.encode_t(t).cpu())
        eSs.append(model.encode_s(v, t).cpu())
        refs.extend(batch["reference"]); tgts.extend(batch["target"])
        members.extend(batch["group_members"])
    e_V = torch.cat(eVs); e_T = torch.cat(eTs); e_S = torch.cat(eSs)
    alpha = model.alpha.item(); beta = model.beta.item(); gamma = model.gamma.item()
    print(f"  Gates: α={alpha:.3f}  β={beta:.3f}  γ={gamma:.3f}")

    def sims_for(q):
        qn = F.normalize(q, dim=-1).to(device)
        chunks = []
        for i in range(0, qn.size(0), 512):
            chunks.append((qn[i:i + 512] @ gallery_emb.t()).cpu())
        return torch.cat(chunks, dim=0)

    variants = {
        "q_full":   alpha * e_V + beta * e_T + gamma * e_S,
        "q_no_S":   alpha * e_V + beta * e_T,
        "q_no_V":                 beta * e_T + gamma * e_S,
        "q_no_T":   alpha * e_V               + gamma * e_S,
        "q_only_V": alpha * e_V,
        "q_only_T": beta  * e_T,
        "q_only_S": gamma * e_S,
    }

    out = {
        "split": args.split,
        "gates": {"alpha": alpha, "beta": beta, "gamma": gamma},
        "variants": {},
    }

    print(f"\n{'=' * 78}")
    print(f"  Decomp Eval [{args.split}]")
    print(f"{'=' * 78}")
    print(f"  {'variant':<10}  R@1     R@5     R@10    R@50   medR | sR@1  sR@2  sR@3")
    print(f"  {'-'*10}  ------  ------  ------  ------  ---- + ----  ----  ----")
    for name, q in variants.items():
        res = _recall_from_sims(sims_for(q), refs, tgts, members, gallery_ids, id_to_idx)
        out["variants"][name] = res
        r = res["recall"]; s = res["subset_recall"]
        print(
            f"  {name:<10}  {r['R@1']:6.2f}  {r['R@5']:6.2f}  {r['R@10']:6.2f}  "
            f"{r['R@50']:6.2f}  {res['median_rank']:4.0f} | "
            f"{s['R_sub@1']:4.1f}  {s['R_sub@2']:4.1f}  {s['R_sub@3']:4.1f}"
        )

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        json.dump(out, open(args.output, "w"), indent=2)
        print(f"\n  Saved -> {args.output}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--cache-dir", default="./cache/cirr_cls")
    p.add_argument("--split", default="val", choices=["train", "val"])
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output", default=None)
    args = p.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
