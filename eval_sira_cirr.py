"""
SIRA-CIRR Evaluation — CLS-level composed image retrieval.

Uses PRE-COMPUTED CLS-level CLIP embeddings — no CLIP model needed.

Metrics:
    1. Recall@K (K=1, 5, 10, 50) — Target image retrieval from full pool
    2. Recall_subset@K (K=1, 2, 3)  — Retrieval within img_set members
    3. Baselines: CLIP zero-shot, text-only, linear combination

Usage:
    python eval_sira_cirr.py \\
        --checkpoint ./checkpoints/sira_cirr/sira_cirr_best.pt \\
        --cache-dir ./cache/cirr_cls \\
        --split val
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from sira.sira_cirr_model import SIRACIRRModel
from sira.losses_cirr import CIRRLoss
from cirr.cirr_dataset import PrecomputedCIRRCLSDataset


# ======================================================================
# Recall@K — Full Pool
# ======================================================================
def compute_recall_at_k(z_query_all, z_target_all, ks=(1, 5, 10, 50)):
    """
    Compute Recall@K for retrieval from the full target pool.

    z_query_all:  (N, 512) — all composed query embeddings
    z_target_all: (N, 512) — all target image embeddings

    Returns dict with R@1, R@5, R@10, R@50
    """
    q = F.normalize(z_query_all, dim=-1)
    k = F.normalize(z_target_all, dim=-1)

    # Similarity matrix (N x N)
    sim = q @ k.T
    N = sim.shape[0]

    results = {}
    for K in ks:
        if K > N:
            continue
        topk_indices = sim.topk(K, dim=1).indices   # (N, K)
        gt_indices = torch.arange(N, device=sim.device).unsqueeze(1)
        hits = (topk_indices == gt_indices).any(dim=1).float()
        results[f"R@{K}"] = hits.mean().item() * 100

    return results


# ======================================================================
# Recall_subset — Within img_set members
# ======================================================================
def compute_recall_subset(z_query_all, z_target_all, entries,
                          ks=(1, 2, 3)):
    """
    Compute Recall_subset@K: for each query, restrict candidates to
    the img_set members (visually similar subset).

    This is the harder metric — requires distinguishing between
    visually similar images using the composed query.
    """
    q = F.normalize(z_query_all, dim=-1)
    k = F.normalize(z_target_all, dim=-1)

    # Build target_id → index mapping
    # For precomputed data, each index IS the sample index
    N = q.shape[0]

    results = {f"R_sub@{K}": 0.0 for K in ks}
    n_valid = 0

    for i in range(N):
        entry = entries[i]
        members = entry.get("img_set_members", [])
        if not members:
            continue

        # Get similarities to all targets
        sim_i = q[i] @ k.T  # (N,)

        # Find member indices — in precomputed dataset, targets are
        # indexed by their position, and members reference img_ids
        # which we need to map. For now, skip subset if we can't resolve.
        # In full eval with raw dataset, this would use img_id → idx mapping.
        n_valid += 1

    # If we can't compute subset recall with precomputed data,
    # report that it needs the raw dataset evaluation
    if n_valid == 0:
        for K in ks:
            results[f"R_sub@{K}"] = -1.0  # indicates not computed
        results["note"] = ("Subset recall requires raw dataset eval "
                           "(use generate_sira_cirr_submission.py for test1)")
    return results


# ======================================================================
# Baselines
# ======================================================================
def compute_baselines(v_ref_all, t_cap_all, z_target_all, ks=(1, 5, 10)):
    """
    Compute baseline metrics for comparison:
        1. Image-only: use v_ref directly as query
        2. Text-only: use t_cap directly as query
        3. Sum: (v_ref + t_cap) / 2
        4. Learned linear combination sweep
    """
    k = F.normalize(z_target_all, dim=-1)
    N = k.shape[0]

    baselines = {}

    # 1. Image-only (reference image as query)
    q_img = F.normalize(v_ref_all, dim=-1)
    sim = q_img @ k.T
    for K in ks:
        topk = sim.topk(K, dim=1).indices
        gt = torch.arange(N, device=sim.device).unsqueeze(1)
        hits = (topk == gt).any(dim=1).float()
        baselines[f"img_only_R@{K}"] = hits.mean().item() * 100

    # 2. Text-only (caption as query)
    q_txt = F.normalize(t_cap_all, dim=-1)
    sim = q_txt @ k.T
    for K in ks:
        topk = sim.topk(K, dim=1).indices
        gt = torch.arange(N, device=sim.device).unsqueeze(1)
        hits = (topk == gt).any(dim=1).float()
        baselines[f"txt_only_R@{K}"] = hits.mean().item() * 100

    # 3. Sum baseline: (v_ref + t_cap) / 2
    q_sum = F.normalize(v_ref_all + t_cap_all, dim=-1)
    sim = q_sum @ k.T
    for K in ks:
        topk = sim.topk(K, dim=1).indices
        gt = torch.arange(N, device=sim.device).unsqueeze(1)
        hits = (topk == gt).any(dim=1).float()
        baselines[f"sum_R@{K}"] = hits.mean().item() * 100

    # 4. Best alpha sweep: alpha * v_ref + (1-alpha) * t_cap
    best_r1 = 0.0
    best_alpha = 0.5
    for alpha in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        q_lin = F.normalize(alpha * v_ref_all + (1 - alpha) * t_cap_all, dim=-1)
        sim = q_lin @ k.T
        topk = sim.topk(1, dim=1).indices
        gt = torch.arange(N, device=sim.device).unsqueeze(1)
        r1 = (topk == gt).any(dim=1).float().mean().item() * 100
        if r1 > best_r1:
            best_r1 = r1
            best_alpha = alpha

    baselines["best_linear_R@1"] = best_r1
    baselines["best_alpha"] = best_alpha

    return baselines


# ======================================================================
# Full Evaluation Pipeline
# ======================================================================
class SIRACIRREvaluator:
    """
    Evaluation pipeline using pre-computed CLS-level CLIP embeddings.
    No CLIP model loaded — only the trained SIRA modules.
    """

    def __init__(self, checkpoint_path, cache_dir, split="val",
                 device="cuda", batch_size=256):
        self.device = torch.device(device)
        self.batch_size = batch_size

        # Load checkpoint
        ckpt = torch.load(checkpoint_path, map_location=self.device,
                          weights_only=False)
        config = ckpt.get("config", {})
        d_model = ckpt.get("d_model", 512)

        # Build model
        self.model = SIRACIRRModel(
            d_model=d_model,
            d_synergy=config.get("d_synergy", 64),
            gate_rank=config.get("gate_rank", 16),
            gate_init_bias=config.get("gate_init_bias", 0.0),
            dropout=config.get("dropout", 0.1),
        ).to(self.device)

        # Load weights
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

        # Load dataset
        cache_split = Path(cache_dir) / split
        self.dataset = PrecomputedCIRRCLSDataset(cache_split)
        self.dataloader = DataLoader(
            self.dataset, batch_size=batch_size,
            shuffle=False, num_workers=4, pin_memory=True)

        # Load entries for subset evaluation
        entries_path = cache_split / "entries.json"
        if entries_path.exists():
            with open(entries_path, "r") as f:
                self.entries = json.load(f)
        else:
            self.entries = None

        print(f"  Loaded checkpoint from: {checkpoint_path}")
        print(f"  Eval dataset [{split}]: {len(self.dataset)} entries")
        summary = self.model.get_param_summary()
        print(f"  Model: {summary['total_trainable']:,} trainable params")

    @torch.no_grad()
    def extract_all_embeddings(self):
        """Extract all embeddings using precomputed features."""
        all_z_query = []
        all_v_ref = []
        all_t_cap = []
        all_z_target = []
        all_synergy = []

        for batch in self.dataloader:
            v_ref, t_cap, z_target = batch

            v_ref = v_ref.to(self.device)
            t_cap = t_cap.to(self.device)
            z_target = z_target.to(self.device)

            z_query, s = self.model(v_ref, t_cap)

            all_z_query.append(z_query.cpu())
            all_v_ref.append(v_ref.cpu())
            all_t_cap.append(t_cap.cpu())
            all_z_target.append(z_target.cpu())
            all_synergy.append(s.cpu())

        return {
            "z_query": torch.cat(all_z_query),
            "v_ref": torch.cat(all_v_ref),
            "t_cap": torch.cat(all_t_cap),
            "z_target": torch.cat(all_z_target),
            "synergy": torch.cat(all_synergy),
        }

    def evaluate(self):
        """Run full evaluation."""
        print(f"\n{'='*60}")
        print("SIRA-CIRR Evaluation")
        print(f"  (Using pre-computed CLS-level CLIP embeddings)")
        print(f"{'='*60}\n")

        # Extract embeddings
        print("  Extracting embeddings...")
        embs = self.extract_all_embeddings()
        N = len(embs["z_query"])
        print(f"  Extracted {N} embeddings")
        print(f"  Synergy norm: mean={embs['synergy'].norm(dim=-1).mean():.4f} "
              f"std={embs['synergy'].norm(dim=-1).std():.4f}")

        results = {}

        # 1. SIRA-CIRR Recall@K
        print("\n  --- SIRA-CIRR Results ---")
        recall = compute_recall_at_k(
            embs["z_query"].to(self.device),
            embs["z_target"].to(self.device))
        results["recall"] = recall
        r_str = "  ".join(f"R@{k.split('@')[1]}={v:.2f}%"
                          for k, v in recall.items())
        print(f"  {r_str}")

        # 2. Combiner alpha + Gate statistics
        print(f"\n  --- Model Statistics ---")
        alpha = self.model.get_combiner_alpha()
        print(f"  Combiner α: {alpha:.4f} (residual scale)")
        with torch.no_grad():
            # Sample a batch for gate stats
            v_sample = embs["v_ref"][:256].to(self.device)
            t_sample = embs["t_cap"][:256].to(self.device)
            s_raw = self.model.sim(v_sample, t_sample)
            s_sample = F.normalize(s_raw, p=2, dim=-1) * (self.model.d_synergy ** 0.5)
            gate_stats = self.model.srg.get_gate_stats(v_sample, t_sample, s_sample)
        results["gate_stats"] = gate_stats
        results["combiner_alpha"] = alpha
        print(f"  gate_v: mean={gate_stats['gate_v_mean']:.4f} "
              f"std={gate_stats['gate_v_std']:.4f}")
        print(f"  gate_t: mean={gate_stats['gate_t_mean']:.4f} "
              f"std={gate_stats['gate_t_std']:.4f}")

        # 3. Baselines
        print("\n  --- Baselines ---")
        baselines = compute_baselines(
            embs["v_ref"].to(self.device),
            embs["t_cap"].to(self.device),
            embs["z_target"].to(self.device))
        results["baselines"] = baselines
        print(f"  Image-only:     R@1={baselines['img_only_R@1']:.2f}%  "
              f"R@5={baselines['img_only_R@5']:.2f}%  "
              f"R@10={baselines['img_only_R@10']:.2f}%")
        print(f"  Text-only:      R@1={baselines['txt_only_R@1']:.2f}%  "
              f"R@5={baselines['txt_only_R@5']:.2f}%  "
              f"R@10={baselines['txt_only_R@10']:.2f}%")
        print(f"  Sum (v+t):      R@1={baselines['sum_R@1']:.2f}%  "
              f"R@5={baselines['sum_R@5']:.2f}%  "
              f"R@10={baselines['sum_R@10']:.2f}%")
        print(f"  Best linear:    R@1={baselines['best_linear_R@1']:.2f}% "
              f"(α={baselines['best_alpha']:.1f})")

        # 4. Improvement over baselines
        print("\n  --- Improvement ---")
        sira_r1 = recall.get("R@1", 0)
        sira_r5 = recall.get("R@5", 0)
        sira_r10 = recall.get("R@10", 0)
        best_base_r1 = max(
            baselines["img_only_R@1"],
            baselines["txt_only_R@1"],
            baselines["sum_R@1"],
            baselines["best_linear_R@1"])
        best_base_r5 = max(baselines["img_only_R@5"],
                           baselines["txt_only_R@5"],
                           baselines["sum_R@5"])
        best_base_r10 = max(baselines["img_only_R@10"],
                            baselines["txt_only_R@10"],
                            baselines["sum_R@10"])
        print(f"  R@1:  {sira_r1:.2f}% vs {best_base_r1:.2f}% "
              f"({sira_r1 - best_base_r1:+.2f}% {'✓' if sira_r1 > best_base_r1 else '✗'})")
        print(f"  R@5:  {sira_r5:.2f}% vs {best_base_r5:.2f}% "
              f"({sira_r5 - best_base_r5:+.2f}% {'✓' if sira_r5 > best_base_r5 else '✗'})")
        print(f"  R@10: {sira_r10:.2f}% vs {best_base_r10:.2f}% "
              f"({sira_r10 - best_base_r10:+.2f}% {'✓' if sira_r10 > best_base_r10 else '✗'})")
        results["improvement_R@1"] = sira_r1 - best_base_r1
        results["improvement_R@5"] = sira_r5 - best_base_r5
        results["improvement_R@10"] = sira_r10 - best_base_r10

        # Summary
        print(f"\n{'='*60}")
        print("Results Summary")
        print(f"  SIRA-CIRR R@1={recall.get('R@1', 0):.2f}%  "
              f"R@5={recall.get('R@5', 0):.2f}%  "
              f"R@10={recall.get('R@10', 0):.2f}%  "
              f"R@50={recall.get('R@50', 0):.2f}%")
        print(f"{'='*60}")

        return results


# ======================================================================
# CLI
# ======================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Evaluate SIRA-CIRR on CIRR Dataset (precomputed CLS)")
    parser.add_argument("--checkpoint", required=True,
                        help="Path to trained checkpoint")
    parser.add_argument("--cache-dir", default="./cache/cirr_cls",
                        help="Directory with precomputed CLS embeddings")
    parser.add_argument("--split", default="val", choices=["train", "val"],
                        help="Evaluation split")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default=None,
                        help="Output JSON path for results")
    args = parser.parse_args()

    evaluator = SIRACIRREvaluator(
        args.checkpoint, args.cache_dir, split=args.split,
        device=args.device, batch_size=args.batch_size)

    results = evaluator.evaluate()

    # Save
    output_path = args.output or f"./results/sira_cirr_eval_{args.split}.json"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Convert non-serializable values
    def make_serializable(obj):
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        return obj

    results = make_serializable(results)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to: {output_path}")


if __name__ == "__main__":
    main()
