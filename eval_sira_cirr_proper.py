import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path
import torch
import torch.nn.functional as F
from sira.sira_cirr_model import SIRACIRRModel

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate SIRA-CIRR in the exact CIRR way")
    parser.add_argument('--checkpoint', type=str, default='./checkpoints/sira_cirr_v3/sira_cirr_best.pt',
                        help='Path to model checkpoint')
    parser.add_argument('--cache-dir', type=str, default='./cache/cirr_cls',
                        help='Path to precomputed embeddings directory')
    parser.add_argument('--split', type=str, default='val', choices=['val'],
                        help='Split to evaluate on')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to run evaluation on')
    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device(args.device)
    cache_split_dir = Path(args.cache_dir) / args.split

    # 1. Load gallery embeddings and info
    print("LOADING GALLERY EMBEDDINGS")
    gallery_emb_path = cache_split_dir / "gallery_val_embeddings.pt"
    gallery_info_path = cache_split_dir / "gallery_val_info.json"
    
    if not gallery_emb_path.exists() or not gallery_info_path.exists():
        raise FileNotFoundError(
            f"Gallery precomputations not found in {cache_split_dir}. "
            "Please run cirr/precompute_val_gallery.py first."
        )
        
    gallery_embeddings = torch.load(gallery_emb_path, map_location=device).float()
    gallery_embeddings = F.normalize(gallery_embeddings, dim=-1)
    
    with open(gallery_info_path, "r") as f:
        gallery_info = json.load(f)
    gallery_paths = np.array(gallery_info["img_ids"])
    
    print(f"Loaded {len(gallery_paths)} gallery images.")
    assert len(gallery_paths) == len(list(set(gallery_paths))), "Gallery contains duplicate paths/IDs!"

    # Create mapping from gallery image ID to index for fast lookup
    gallery_id_to_idx = {img_id: idx for idx, img_id in enumerate(gallery_paths)}

    # 2. Load model
    print(f"LOADING MODEL FROM {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device)
    config = ckpt.get("config", {})
    d_model = ckpt.get("d_model", 512)
    
    model = SIRACIRRModel(
        d_model=d_model,
        d_synergy=config.get("d_synergy", 64),
        gate_rank=config.get("gate_rank", 16),
        gate_init_bias=config.get("gate_init_bias", 0.0),
        dropout=config.get("dropout", 0.1),
    ).to(device)
    
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print("Model loaded successfully.")

    # 3. Load precomputed query embeddings and entries
    print("LOADING QUERY CLS EMBEDDINGS")
    v_ref_all = torch.load(cache_split_dir / "v_ref.pt", map_location=device).float()
    t_cap_all = torch.load(cache_split_dir / "t_cap.pt", map_location=device).float()
    
    with open(cache_split_dir / "entries.json", "r") as f:
        entries = json.load(f)
        
    assert len(entries) == v_ref_all.shape[0] == t_cap_all.shape[0], "Mismatch in query sizes!"
    print(f"Loaded {len(entries)} query samples.")

    target_paths = np.array([entry["target_hard"] for entry in entries])

    # 4. Extract composed query embeddings
    print("COMPUTING SIRA-CIRR EMBEDDINGS")
    vl_embeddings = []
    synergies = []
    
    # Process in batches to avoid GPU OOM
    batch_size = 256
    with torch.no_grad():
        for i in range(0, len(entries), batch_size):
            v_ref_batch = v_ref_all[i : i + batch_size].to(device)
            t_cap_batch = t_cap_all[i : i + batch_size].to(device)
            
            z_query, s = model(v_ref_batch, t_cap_batch)
            vl_embeddings.append(z_query)
            synergies.append(s)
            
    vl_embeddings = torch.cat(vl_embeddings, dim=0)
    vl_embeddings = F.normalize(vl_embeddings, dim=-1)
    
    synergies = torch.cat(synergies, dim=0)
    print(f"Synergy norm: mean={synergies.norm(dim=-1).mean().item():.4f} std={synergies.norm(dim=-1).std().item():.4f}")

    # Helper function to compute recall & rank metrics
    def compute_recall_metrics(queries_normalized, label_name, exclude_ref=False):
        # Compute similarities
        sims = torch.matmul(queries_normalized, gallery_embeddings.T).cpu().numpy()
        
        # Mask out the query's reference image if exclude_ref is True
        if exclude_ref:
            for i in range(len(entries)):
                ref = entries[i]["reference"]
                if ref in gallery_id_to_idx:
                    sims[i, gallery_id_to_idx[ref]] = -9999.0
                    
        suffix = " (Excluding Reference Image)" if exclude_ref else ""
        print(f"\nEvaluating {label_name}{suffix}:")
        recalls = [1, 5, 10, 50, 100]
        sorted_indices = np.argsort(-sims, axis=1)  # Sort in descending order
        
        # Diagnostics: how often is the reference image retrieved at rank 1?
        if not exclude_ref:
            ref_at_1 = 0
            ref_in_top_5 = 0
            for i in range(len(entries)):
                ref = entries[i]["reference"]
                if ref in gallery_id_to_idx:
                    ref_idx = gallery_id_to_idx[ref]
                    rank_of_ref = np.where(sorted_indices[i] == ref_idx)[0][0]
                    if rank_of_ref == 0:
                        ref_at_1 += 1
                    if rank_of_ref < 5:
                        ref_in_top_5 += 1
            print(f"  Reference image at Rank 1: {ref_at_1 / len(entries) * 100.0:.2f}%")
            print(f"  Reference image in Top 5:  {ref_in_top_5 / len(entries) * 100.0:.2f}%")

        results = {}
        # 1. Full Gallery Recalls
        for recall in recalls:
            top_k_indices = sorted_indices[:, :recall]
            correct = sum(target_paths[i] in gallery_paths[top_k_indices[i]] for i in range(len(target_paths)))
            score = correct / len(target_paths)
            results[f"R@{recall}"] = score * 100.0
            print(f'  Top-{recall} recall: {score:.4f} ({score * 100.0:.2f}%)')
            
        # 2. Median Rank
        ranks = np.array([np.where(gallery_paths[sorted_indices[i]] == target_paths[i])[0][0] for i in range(len(target_paths))])
        med_rank = np.median(ranks)
        results["Median_Rank"] = float(med_rank)
        print(f'  Median rank: {med_rank:.1f}')
        
        # 3. Subset Recalls (within img_set_members)
        subset_recalls = {1: 0.0, 2: 0.0, 3: 0.0}
        valid_subset_queries = 0
        for i in range(len(target_paths)):
            target = target_paths[i]
            members = entries[i]["img_set_members"]
            if target not in members:
                continue
                
            member_idxs = []
            member_ids_filtered = []
            for m in members:
                if m in gallery_id_to_idx:
                    # Mask out reference in subset too if exclude_ref is True
                    if exclude_ref and m == entries[i]["reference"]:
                        continue
                    member_idxs.append(gallery_id_to_idx[m])
                    member_ids_filtered.append(m)
                    
            if not member_idxs or target not in member_ids_filtered:
                continue
                
            valid_subset_queries += 1
            
            sim_sub = sims[i, member_idxs]
            sorted_sub_indices = np.argsort(-sim_sub)
            
            tgt_idx_in_filtered = member_ids_filtered.index(target)
            rank = np.where(sorted_sub_indices == tgt_idx_in_filtered)[0][0]
            
            for k in [1, 2, 3]:
                if rank < k:
                    subset_recalls[k] += 1.0
                    
        if valid_subset_queries > 0:
            for k in [1, 2, 3]:
                score_sub = subset_recalls[k] / valid_subset_queries
                results[f"R_sub@{k}"] = score_sub * 100.0
                print(f'  Top-{k} subset recall: {score_sub:.4f} ({score_sub * 100.0:.2f}%)')
                
        return results

    # Run SIRA-CIRR evaluation
    sira_results = compute_recall_metrics(vl_embeddings, "SIRA-CIRR")
    sira_results_ex = compute_recall_metrics(vl_embeddings, "SIRA-CIRR", exclude_ref=True)

    # Run Baselines for comparison
    print("\n" + "#"*40 + "\nCOMPUTING BASELINES\n" + "#"*40)
    
    # Image-only baseline
    img_only_results = compute_recall_metrics(F.normalize(v_ref_all, dim=-1), "Image-Only Baseline")
    img_only_results_ex = compute_recall_metrics(F.normalize(v_ref_all, dim=-1), "Image-Only Baseline", exclude_ref=True)
    
    # Text-only baseline
    txt_only_results = compute_recall_metrics(F.normalize(t_cap_all, dim=-1), "Text-Only Baseline")
    txt_only_results_ex = compute_recall_metrics(F.normalize(t_cap_all, dim=-1), "Text-Only Baseline", exclude_ref=True)
    
    # Sum baseline (v + t)
    sum_normalized = F.normalize(v_ref_all + t_cap_all, dim=-1)
    sum_results = compute_recall_metrics(sum_normalized, "Sum (v+t) Baseline")
    sum_results_ex = compute_recall_metrics(sum_normalized, "Sum (v+t) Baseline", exclude_ref=True)
    
    # Best linear combination (alpha-sweep)
    best_r1 = -1.0
    best_alpha = 0.0
    best_linear_results = None
    for alpha in np.linspace(0.0, 1.0, 11):
        lin_comb = F.normalize(alpha * v_ref_all + (1 - alpha) * t_cap_all, dim=-1)
        # Compute just R@1 quickly
        sims_lin = torch.matmul(lin_comb, gallery_embeddings.T).cpu().numpy()
        sorted_indices_lin = np.argsort(-sims_lin, axis=1)
        top_1_indices = sorted_indices_lin[:, :1]
        correct = sum(target_paths[i] in gallery_paths[top_1_indices[i]] for i in range(len(target_paths)))
        r1 = correct / len(target_paths)
        if r1 > best_r1:
            best_r1 = r1
            best_alpha = alpha
            
    print(f"\nSweep found best alpha = {best_alpha:.1f} with R@1 = {best_r1*100.0:.2f}%")
    best_lin_comb = F.normalize(best_alpha * v_ref_all + (1 - best_alpha) * t_cap_all, dim=-1)
    best_linear_results = compute_recall_metrics(best_lin_comb, f"Best Linear Baseline (alpha={best_alpha:.1f})")
    best_linear_results_ex = compute_recall_metrics(best_lin_comb, f"Best Linear Baseline (alpha={best_alpha:.1f})", exclude_ref=True)

    # Output consolidated results in a format comparable to the training logs / other scripts
    output_dir = Path("./results")
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "sira_cirr_proper_eval.json"
    
    summary = {
        "sira_cirr": sira_results,
        "sira_cirr_exclude_ref": sira_results_ex,
        "baselines": {
            "image_only": img_only_results,
            "image_only_exclude_ref": img_only_results_ex,
            "text_only": txt_only_results,
            "text_only_exclude_ref": txt_only_results_ex,
            "sum": sum_results,
            "sum_exclude_ref": sum_results_ex,
            "best_linear": best_linear_results,
            "best_linear_exclude_ref": best_linear_results_ex,
            "best_alpha": float(best_alpha)
        }
    }
    
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved proper evaluation results to {summary_path}")

if __name__ == '__main__':
    main()
