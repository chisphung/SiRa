"""
Evaluate CLIP vs SIRA vs CLIP-Baseline on HL test set.

Protocols:
    1. Cross-Modal Retrieval (R@1, R@5, R@10)
    2. Caption Swap Accuracy
    3. Hard Negative Discrimination
    4. HL Axis-Specific Retrieval (scene/action/rationale/object)

Usage:
    python eval/eval_hl.py --sira-ckpt checkpoints/sira/sira_best.pt
    python eval/eval_hl.py --sira-ckpt checkpoints/sira/sira_best.pt --baseline-ckpt checkpoints/clip_baseline/clip_baseline_best.pt
"""

import os, sys, json, random, argparse
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sira.sira_model import SIRAModel


# ======================================================================
# Data Loading
# ======================================================================
def load_hl_test(hl_root, max_samples=None, seed=42):
    """Load HL test metadata."""
    jsonl_path = Path(hl_root) / "data" / "test" / "metadata.jsonl"
    entries = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            caps = entry.get("captions", {})
            if all(caps.get(k) for k in ["scene", "action", "rationale", "object"]):
                entries.append(entry)

    random.seed(seed)
    if max_samples and len(entries) > max_samples:
        entries = random.sample(entries, max_samples)

    print(f"  HL test: {len(entries)} entries")
    return entries


def build_captions(entries, mode="fused"):
    """Build caption strings from HL entries."""
    captions = []
    for e in entries:
        c = e["captions"]
        conf = e.get("confidence", {})
        pick = lambda lst, cf=None: lst[0] if not cf or len(cf) != len(lst) else lst[max(range(len(cf)), key=lambda i: cf[i])]

        scene = pick(c["scene"], conf.get("scene")).strip().rstrip(".")
        action = pick(c["action"], conf.get("action")).strip().rstrip(".")
        rationale = pick(c["rationale"], conf.get("rationale")).strip().rstrip(".")
        obj = pick(c["object"]).strip()

        if mode == "fused":
            s, a, r = scene.lower(), action.lower(), rationale.lower()
            captions.append(f"{s}, {a} because {r}.")
        elif mode == "unified":
            captions.append(f"Scene: {scene}. Action: {action}. Rationale: {rationale}. Description: {obj}")
        elif mode == "object":
            captions.append(obj)
        elif mode == "scene":
            captions.append(scene)
        elif mode == "action":
            captions.append(action)
        elif mode == "rationale":
            captions.append(rationale)
    return captions


# ======================================================================
# Model Loading
# ======================================================================
def load_clip(model_name, device):
    """Load vanilla CLIP."""
    try:
        import clip
        # Standardize name for OpenAI CLIP
        name = model_name
        if "/" not in name and "-" in name:
            parts = name.split("-")
            if len(parts) >= 2:
                name = parts[0] + "-" + parts[1] + "/" + "-".join(parts[2:])
        model, preprocess = clip.load(name, device=device)
        model.eval()
        tokenizer = clip.tokenize
    except Exception:
        import open_clip
        # Standardize name for OpenCLIP
        name = model_name.replace("/", "-")
        model, _, preprocess = open_clip.create_model_and_transforms(
            name, pretrained="openai")
        model = model.to(device).eval()
        tokenizer = open_clip.get_tokenizer(name)
    return model, preprocess, tokenizer


def load_sira(model_name, ckpt_path, device, base_ckpt_path=None):
    """Load SIRA model from checkpoint."""
    sira_model, preprocess = SIRAModel.from_clip(model_name, device)
    if base_ckpt_path:
        base_ckpt = torch.load(base_ckpt_path, map_location=device, weights_only=True)
        if "model_state_dict" in base_ckpt:
            clip_state = {k: v for k, v in base_ckpt["model_state_dict"].items()}
            sira_model.clip.load_state_dict(clip_state, strict=False)
            print(f"  Loaded base CLIP backbone from {base_ckpt_path}")
            
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    sira_model.sim.load_state_dict(ckpt["sim_state_dict"])
    if "srg_state_dict" in ckpt:
        incompatible = sira_model.srg.load_state_dict(ckpt["srg_state_dict"], strict=False)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            print(f"  SRG load (strict=False): missing={incompatible.missing_keys}, "
                  f"unexpected={incompatible.unexpected_keys}")
    if "loss_fn_state_dict" in ckpt:
        sira_model.loss_fn.load_state_dict(ckpt["loss_fn_state_dict"], strict=False)
    
    # Load backbone state if it was unfrozen during training
    if "clip_state_dict" in ckpt:
        sira_model.clip.load_state_dict(ckpt["clip_state_dict"])
        print("  Loaded fine-tuned CLIP backbone from SIRA checkpoint.")
        
    sira_model.eval()
    print(f"  SIRA loaded from {ckpt_path} (epoch {ckpt.get('epoch', '?')})")
    return sira_model, preprocess


def load_baseline(model_name, ckpt_path, device, strategy="projection"):
    """Load fine-tuned CLIP baseline from checkpoint."""
    from train_clip_baseline import CLIPBaseline
    model, preprocess = CLIPBaseline.from_clip(model_name, device, strategy=strategy)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    # Load only the saved (trainable) state dict
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()
    print(f"  Baseline loaded from {ckpt_path} (epoch {ckpt.get('epoch', '?')})")
    return model, preprocess


# ======================================================================
# Encoding Helpers
# ======================================================================
@torch.no_grad()
def encode_images_clip(model, preprocess, image_dir, entries, device, batch_size=64):
    """Encode images with CLIP. Returns [N, d] tensor."""
    all_feats = []
    for i in range(0, len(entries), batch_size):
        batch = entries[i:i+batch_size]
        imgs = torch.stack([
            preprocess(Image.open(image_dir / e["file_name"]).convert("RGB"))
            for e in batch
        ]).to(device)
        feats = model.encode_image(imgs)
        all_feats.append(F.normalize(feats.float(), dim=-1).cpu())
    return torch.cat(all_feats)


@torch.no_grad()
def encode_texts_clip(model, tokenizer, captions, device, batch_size=256):
    """Encode texts with CLIP. Returns [N, d] tensor."""
    all_feats = []
    for i in range(0, len(captions), batch_size):
        batch = captions[i:i+batch_size]
        try:
            tokens = tokenizer(batch, truncate=True).to(device)
        except TypeError:
            tokens = tokenizer(batch).to(device)
        feats = model.encode_text(tokens)
        all_feats.append(F.normalize(feats.float(), dim=-1).cpu())
    return torch.cat(all_feats)


@torch.no_grad()
def encode_sira_independent(sira_model, preprocess, image_dir, entries, device, batch_size=64):
    """Encode images independently with SIRA's frozen CLIP backbone (no SIM/SRG).
    This is the fair baseline — no cross-modal cheating."""
    all_feats = []
    for i in range(0, len(entries), batch_size):
        batch = entries[i:i+batch_size]
        imgs = torch.stack([
            preprocess(Image.open(image_dir / e["file_name"]).convert("RGB"))
            for e in batch
        ]).to(device)
        feats = sira_model._encode_image(imgs)
        all_feats.append(feats.cpu())
    return torch.cat(all_feats)


@torch.no_grad()
def encode_sira_texts_independent(sira_model, tokenizer, captions, device, batch_size=256):
    """Encode texts independently with SIRA's frozen CLIP backbone (no SIM/SRG)."""
    all_feats = []
    for i in range(0, len(captions), batch_size):
        batch = captions[i:i+batch_size]
        try:
            tokens = tokenizer(batch, truncate=True).to(device)
        except TypeError:
            tokens = tokenizer(batch).to(device)
        feats = sira_model._encode_text(tokens)
        all_feats.append(feats.cpu())
    return torch.cat(all_feats)


@torch.no_grad()
def sira_rerank(sira_model, preprocess, tokenizer, image_dir, entries,
                captions, img_feats, txt_feats, device, top_k=50, batch_size=64):
    """
    SIRA re-ranking: use CLIP for initial retrieval, then apply SIM+SRG
    on the top-K shortlist to re-score candidates.

    Returns re-ranked similarity matrix [N, N].
    """
    N = len(entries)
    clip_sim = img_feats @ txt_feats.t()  # [N, N]
    _, top_text_idx = clip_sim.topk(min(top_k, N), dim=1)

    reranked_sim = clip_sim.clone()

    # For each image, re-score its top-K text candidates via SIM+SRG
    for i in range(N):
        cand_indices = top_text_idx[i].tolist()
        
        # Encode through SIRA jointly
        v_shared = img_feats[i].unsqueeze(0).expand(len(cand_indices), -1).to(device)  # [top_k, d_model]
        t_shared = txt_feats[cand_indices].to(device)  # [top_k, d_model]

        # Compute synergy and fuse it back using the gate
        s = sira_model.sim(v_shared, t_shared)
        v_final, t_final = sira_model.srg(v_shared, t_shared, s)
        
        # Fused cosine similarity (dot product of L2-normalized features)
        syn_scores = (v_final * t_final).sum(dim=-1)

        # Final score is the gated synergistic similarity score
        for k, j in enumerate(cand_indices):
            reranked_sim[i, j] = syn_scores[k].cpu()

    return reranked_sim


# ======================================================================
# Evaluation Protocols
# ======================================================================
def retrieval_metrics(img_feats, txt_feats):
    """Compute R@1, R@5, R@10 for image-text and text-image retrieval."""
    sim = img_feats @ txt_feats.t()  # [N, N]
    N = sim.size(0)

    results = {}
    for direction, matrix in [("i2t", sim), ("t2i", sim.t())]:
        _, sorted_idx = matrix.sort(dim=1, descending=True)
        ranks = []
        for i in range(N):
            rank = (sorted_idx[i] == i).nonzero(as_tuple=True)[0].item()
            ranks.append(rank)
        ranks = np.array(ranks)
        results[f"{direction}_R@1"] = float(np.mean(ranks < 1))
        results[f"{direction}_R@5"] = float(np.mean(ranks < 5))
        results[f"{direction}_R@10"] = float(np.mean(ranks < 10))
        results[f"{direction}_median_rank"] = float(np.median(ranks) + 1)

    results["rsum"] = sum(results[k] for k in results if "R@" in k)
    return results


def caption_swap_accuracy(img_feats, txt_feats, n_pairs=500):
    """Test if model prefers matched (img, txt) over swapped pairs."""
    N = min(img_feats.size(0), n_pairs)
    correct = 0
    margins = []
    for i in range(N):
        j = (i + 1) % img_feats.size(0)
        sim_match = (img_feats[i] @ txt_feats[i]).item()
        sim_swap = (img_feats[i] @ txt_feats[j]).item()
        if sim_match > sim_swap:
            correct += 1
        margins.append(sim_match - sim_swap)
    return {
        "swap_accuracy": correct / N,
        "avg_margin": float(np.mean(margins)),
        "pct_confused": float(np.mean([m < 0 for m in margins])),
    }


def hard_negative_test(img_feats, txt_feats, top_k=5, n_queries=500):
    """Test discrimination against visually similar hard negatives."""
    N = min(img_feats.size(0), n_queries)
    img_sim = img_feats[:N] @ img_feats.t()
    img_sim.fill_diagonal_(-float("inf"))
    _, hard_idx = img_sim.topk(top_k, dim=1)

    correct = 0
    total = 0
    margins = []
    for i in range(N):
        matched = (img_feats[i] @ txt_feats[i]).item()
        for j in hard_idx[i].tolist():
            neg = (img_feats[i] @ txt_feats[j]).item()
            if matched > neg:
                correct += 1
            margins.append(matched - neg)
            total += 1
    return {
        "hard_neg_accuracy": correct / max(total, 1),
        "avg_hard_margin": float(np.mean(margins)),
        "pct_confused": float(np.mean([m < 0 for m in margins])),
    }


def caption_swap_accuracy_sira(sira_model, img_feats, txt_feats, device, n_pairs=500):
    """Test if SIRA interactive scoring prefers matched (img, txt) over swapped pairs."""
    N = min(img_feats.size(0), n_pairs)
    correct = 0
    margins = []
    
    for i in range(N):
        j = (i + 1) % img_feats.size(0)
        
        # Match
        v_match = img_feats[i].unsqueeze(0).to(device)
        t_match = txt_feats[i].unsqueeze(0).to(device)
        with torch.no_grad():
            s_match = sira_model.sim(v_match, t_match)
            v_f_m, t_f_m = sira_model.srg(v_match, t_match, s_match)
            sim_match = (v_f_m * t_f_m).sum(dim=-1).item()
        
        # Swap
        v_swap = img_feats[i].unsqueeze(0).to(device)
        t_swap = txt_feats[j].unsqueeze(0).to(device)
        with torch.no_grad():
            s_swap = sira_model.sim(v_swap, t_swap)
            v_f_s, t_f_s = sira_model.srg(v_swap, t_swap, s_swap)
            sim_swap = (v_f_s * t_f_s).sum(dim=-1).item()
            
        if sim_match > sim_swap:
            correct += 1
        margins.append(sim_match - sim_swap)
        
    return {
        "swap_accuracy": correct / N,
        "avg_margin": float(np.mean(margins)),
        "pct_confused": float(np.mean([m < 0 for m in margins])),
    }


def hard_negative_test_sira(sira_model, img_feats, txt_feats, device, top_k=5, n_queries=500):
    """Test discrimination of SIRA interactive scoring against visually similar hard negatives."""
    N = min(img_feats.size(0), n_queries)
    img_sim = img_feats[:N] @ img_feats.t()
    img_sim.fill_diagonal_(-float("inf"))
    _, hard_idx = img_sim.topk(top_k, dim=1)

    correct = 0
    total = 0
    margins = []
    
    for i in range(N):
        # Match
        v_match = img_feats[i].unsqueeze(0).to(device)
        t_match = txt_feats[i].unsqueeze(0).to(device)
        with torch.no_grad():
            s_match = sira_model.sim(v_match, t_match)
            v_f_m, t_f_m = sira_model.srg(v_match, t_match, s_match)
            sim_match = (v_f_m * t_f_m).sum(dim=-1).item()
            
        # Hard Negatives
        for j in hard_idx[i].tolist():
            v_neg = img_feats[i].unsqueeze(0).to(device)
            t_neg = txt_feats[j].unsqueeze(0).to(device)
            with torch.no_grad():
                s_neg = sira_model.sim(v_neg, t_neg)
                v_f_n, t_f_n = sira_model.srg(v_neg, t_neg, s_neg)
                sim_neg = (v_f_n * t_f_n).sum(dim=-1).item()
                
            if sim_match > sim_neg:
                correct += 1
            margins.append(sim_match - sim_neg)
            total += 1
            
    return {
        "hard_neg_accuracy": correct / max(total, 1),
        "avg_hard_margin": float(np.mean(margins)),
        "pct_confused": float(np.mean([m < 0 for m in margins])),
    }


# ======================================================================
# Run Full Eval for One Model
# ======================================================================
def evaluate_model(name, img_feats, txt_feats, axis_feats=None):
    """Run all protocols and return results dict."""
    print(f"\n{'─'*60}")
    print(f"  {name}")
    print(f"{'─'*60}")

    r = retrieval_metrics(img_feats, txt_feats)
    print(f"  I→T  R@1={r['i2t_R@1']*100:.1f}%  R@5={r['i2t_R@5']*100:.1f}%  R@10={r['i2t_R@10']*100:.1f}%")
    print(f"  T→I  R@1={r['t2i_R@1']*100:.1f}%  R@5={r['t2i_R@5']*100:.1f}%  R@10={r['t2i_R@10']*100:.1f}%")
    print(f"  Rsum={r['rsum']*100:.1f}")

    sw = caption_swap_accuracy(img_feats, txt_feats)
    print(f"  Swap accuracy:     {sw['swap_accuracy']*100:.1f}%  (margin={sw['avg_margin']:.4f})")

    hn = hard_negative_test(img_feats, txt_feats)
    print(f"  Hard neg accuracy: {hn['hard_neg_accuracy']*100:.1f}%  (margin={hn['avg_hard_margin']:.4f})")

    result = {"retrieval": r, "swap": sw, "hard_neg": hn}

    # Axis-specific retrieval if available
    if axis_feats:
        print(f"  --- Axis-Specific Retrieval ---")
        for axis_name, a_feats in axis_feats.items():
            ar = retrieval_metrics(img_feats, a_feats)
            print(f"  [{axis_name:10s}] I→T R@1={ar['i2t_R@1']*100:.1f}%  R@5={ar['i2t_R@5']*100:.1f}%")
            result[f"axis_{axis_name}"] = ar

    return result


# ======================================================================
# Main
# ======================================================================
def main():
    parser = argparse.ArgumentParser(description="Evaluate CLIP vs SIRA on HL test set")
    parser.add_argument("--clip-model", default="ViT-B-32")
    parser.add_argument("--sira-ckpt", default=None, help="SIRA checkpoint")
    parser.add_argument("--sira-base-ckpt", default=None, help="Base CLIP checkpoint loaded before SIRA (if SIRA was trained on top of it)")
    parser.add_argument("--baseline-ckpt", default=None, help="Fine-tuned CLIP checkpoint")
    parser.add_argument("--baseline-strategy", default="projection")
    parser.add_argument("--hl-root", default="/home/otw/chisphung/hl")
    parser.add_argument("--hl-mode", default="fused", choices=["fused", "unified"])
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default="results/hl_eval_comparison.json")
    args = parser.parse_args()

    image_dir = Path(args.hl_root) / "data" / "test"

    print(f"\n{'='*60}")
    print(f"  HL Evaluation: CLIP vs SIRA")
    print(f"{'='*60}")

    # Load data
    entries = load_hl_test(args.hl_root, args.max_samples)
    captions = build_captions(entries, mode=args.hl_mode)

    # Build axis-specific captions for fine-grained analysis
    axis_caps = {}
    for axis in ["scene", "action", "rationale", "object"]:
        axis_caps[axis] = build_captions(entries, mode=axis)

    # ---- 1. Vanilla CLIP ----
    print("\n[1/3] Loading vanilla CLIP...")
    clip_model, preprocess, tokenizer = load_clip(args.clip_model, args.device)

    print("  Encoding images...")
    clip_img = encode_images_clip(clip_model, preprocess, image_dir, entries, args.device)
    print("  Encoding texts...")
    clip_txt = encode_texts_clip(clip_model, tokenizer, captions, args.device)

    # Encode axis-specific texts
    clip_axis = {}
    for axis, caps in axis_caps.items():
        clip_axis[axis] = encode_texts_clip(clip_model, tokenizer, caps, args.device)

    clip_results = evaluate_model("CLIP (Pretrained)", clip_img, clip_txt, clip_axis)

    # ---- 2. SIRA ----
    all_results = {"clip_pretrained": clip_results}

    # 2. Evaluate SIRA
    if args.sira_ckpt:
        print("\n[2/3] Loading SIRA...")
        sira_model, sira_preprocess = load_sira(args.clip_model, args.sira_ckpt, args.device, args.sira_base_ckpt)

        # Get tokenizer
        try:
            import open_clip
            tokenizer_sira = open_clip.get_tokenizer(args.clip_model)
        except Exception:
            import clip as clip_lib
            tokenizer_sira = clip_lib.tokenize

        # Independent encoding (fair — no cross-modal cheating)
        print("  Encoding images independently...")
        sira_img = encode_sira_independent(
            sira_model, sira_preprocess, image_dir, entries, args.device)
        print("  Encoding texts independently...")
        sira_txt = encode_sira_texts_independent(
            sira_model, tokenizer_sira, captions, args.device)

        # SIRA re-ranking: CLIP shortlist → SIM+SRG re-score top-K
        print("  Re-ranking top-50 candidates with SIM+SRG...")
        reranked_sim = sira_rerank(
            sira_model, sira_preprocess, tokenizer_sira,
            image_dir, entries, captions,
            sira_img, sira_txt, args.device, top_k=50)

        # Compute retrieval from re-ranked sim matrix
        def retrieval_from_sim(sim_matrix):
            N = sim_matrix.size(0)
            results = {}
            for direction, matrix in [("i2t", sim_matrix), ("t2i", sim_matrix.t())]:
                _, sorted_idx = matrix.sort(dim=1, descending=True)
                ranks = []
                for i in range(N):
                    rank = (sorted_idx[i] == i).nonzero(as_tuple=True)[0].item()
                    ranks.append(rank)
                ranks = np.array(ranks)
                results[f"{direction}_R@1"] = float(np.mean(ranks < 1))
                results[f"{direction}_R@5"] = float(np.mean(ranks < 5))
                results[f"{direction}_R@10"] = float(np.mean(ranks < 10))
                results[f"{direction}_median_rank"] = float(np.median(ranks) + 1)
            results["rsum"] = sum(results[k] for k in results if "R@" in k)
            return results

        r = retrieval_from_sim(reranked_sim)
        sw = caption_swap_accuracy_sira(sira_model, sira_img, sira_txt, args.device)
        hn = hard_negative_test_sira(sira_model, sira_img, sira_txt, args.device)

        print(f"\n{'─'*60}")
        print(f"  SIRA (Re-ranked, top-50)")
        print(f"{'─'*60}")
        print(f"  I→T  R@1={r['i2t_R@1']*100:.1f}%  R@5={r['i2t_R@5']*100:.1f}%  R@10={r['i2t_R@10']*100:.1f}%")
        print(f"  T→I  R@1={r['t2i_R@1']*100:.1f}%  R@5={r['t2i_R@5']*100:.1f}%  R@10={r['t2i_R@10']*100:.1f}%")
        print(f"  Rsum={r['rsum']*100:.1f}")
        print(f"  Swap accuracy:     {sw['swap_accuracy']*100:.1f}%  (margin={sw['avg_margin']:.4f})")
        print(f"  Hard neg accuracy: {hn['hard_neg_accuracy']*100:.1f}%  (margin={hn['avg_hard_margin']:.4f})")

        sira_results = {"retrieval": r, "swap": sw, "hard_neg": hn}
        all_results["sira"] = sira_results
    else:
        print(f"\n[2/3] SIRA checkpoint not found: {args.sira_ckpt}, skipping.")

    # ---- 3. Baseline (optional) ----
    if args.baseline_ckpt and Path(args.baseline_ckpt).exists():
        print("\n[3/3] Loading CLIP Baseline...")
        bl_model, bl_preprocess = load_baseline(
            args.clip_model, args.baseline_ckpt, args.device, args.baseline_strategy)

        bl_img = encode_images_clip(bl_model, bl_preprocess, image_dir, entries, args.device)
        bl_txt = encode_texts_clip(bl_model, tokenizer, captions, args.device)

        bl_axis = {}
        for axis, caps in axis_caps.items():
            bl_axis[axis] = encode_texts_clip(bl_model, tokenizer, caps, args.device)

        bl_results = evaluate_model("CLIP (Fine-tuned)", bl_img, bl_txt, bl_axis)
        all_results["clip_finetuned"] = bl_results
    else:
        print(f"\n[3/3] No baseline checkpoint, skipping.")

    # ---- Summary Table ----
    print(f"\n{'='*60}")
    print(f"  COMPARISON SUMMARY")
    print(f"{'='*60}")
    header = f"{'Metric':<25}"
    for name in all_results:
        header += f" {name:>15}"
    print(header)
    print("─" * len(header))

    metrics = [
        ("I→T R@1", lambda r: r["retrieval"]["i2t_R@1"]),
        ("I→T R@5", lambda r: r["retrieval"]["i2t_R@5"]),
        ("T→I R@1", lambda r: r["retrieval"]["t2i_R@1"]),
        ("T→I R@5", lambda r: r["retrieval"]["t2i_R@5"]),
        ("Rsum", lambda r: r["retrieval"]["rsum"]),
        ("Swap Accuracy", lambda r: r["swap"]["swap_accuracy"]),
        ("Hard Neg Accuracy", lambda r: r["hard_neg"]["hard_neg_accuracy"]),
    ]
    for label, fn in metrics:
        row = f"{label:<25}"
        for name, res in all_results.items():
            val = fn(res)
            row += f" {val*100:>14.1f}%"
        print(row)
    print(f"{'='*60}\n")

    # Save
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
