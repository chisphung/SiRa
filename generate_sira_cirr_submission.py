"""
Generate CIRR Test1 Submission Files for Official Evaluation Server — SIRA Version.

This script:
    1. Loads CLIP + trained SIRA-CIRR checkpoint
    2. Encodes ALL test1 images as retrieval candidates (CLIP image embeddings)
    3. For each query (reference_image + caption):
       - Extracts CLIP CLS embeddings for reference + caption
       - Runs SIRA SIM + SRG + QueryCombiner → z_query
       - Ranks all candidates by cosine similarity
    4. Outputs JSON files for CIRR evaluation server

Output format:
    {
        "version": "rc2",
        "metric": "recall",        // or "recall_subset"
        "<pair_id>": ["img_id_1", "img_id_2", ...],
        ...
    }

Usage:
    python generate_sira_cirr_submission.py \\
        --checkpoint ./checkpoints/sira_cirr/sira_cirr_best.pt \\
        --cirr-root /home/otw/chiennhm/data/CIRR \\
        --output-dir ./submissions/sira_cirr
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from PIL import Image

# Disable PIL decompression bomb warning for large CIRR images
Image.MAX_IMAGE_PIXELS = None

from sira.sira_cirr_model import SIRACIRRModel


# ======================================================================
# Dataset for encoding ALL test1 images (candidates)
# ======================================================================
class CIRRCandidateDataset(Dataset):
    """Load all images in a CIRR split as retrieval candidates."""

    def __init__(self, cirr_root, split="test1", preprocess=None):
        self.preprocess = preprocess
        cirr_root = Path(cirr_root)

        split_path = cirr_root / "cirr" / "image_splits" / f"split.rc2.{split}.json"
        with open(split_path, "r") as f:
            self.img_paths = json.load(f)

        self.cirr_root = cirr_root
        self.img_ids = list(self.img_paths.keys())
        print(f"  CIRRCandidateDataset [{split}]: {len(self.img_ids)} images")

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        rel_path = self.img_paths[img_id]
        if rel_path.startswith("./"):
            rel_path = rel_path[2:]
        img_path = self.cirr_root / rel_path
        image = self.preprocess(Image.open(img_path).convert("RGB"))
        return image, img_id


# ======================================================================
# Dataset for test1 queries (reference image + caption)
# ======================================================================
class CIRRQueryDataset(Dataset):
    """Load test1 queries: (reference_image, caption, pair_id, img_set)."""

    def __init__(self, cirr_root, split="test1", preprocess=None, tokenizer=None):
        self.preprocess = preprocess
        self.tokenizer = tokenizer
        cirr_root = Path(cirr_root)

        cap_path = cirr_root / "cirr" / "captions" / f"cap.rc2.{split}.json"
        with open(cap_path, "r") as f:
            self.annotations = json.load(f)

        split_path = cirr_root / "cirr" / "image_splits" / f"split.rc2.{split}.json"
        with open(split_path, "r") as f:
            self.img_paths = json.load(f)

        self.cirr_root = cirr_root
        print(f"  CIRRQueryDataset [{split}]: {len(self.annotations)} queries")

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        entry = self.annotations[idx]

        ref_id = entry["reference"]
        rel_path = self.img_paths[ref_id]
        if rel_path.startswith("./"):
            rel_path = rel_path[2:]
        ref_path = self.cirr_root / rel_path
        ref_image = self.preprocess(Image.open(ref_path).convert("RGB"))

        caption = entry["caption"]
        try:
            text_tokens = self.tokenizer(caption, truncate=True).squeeze(0)
        except TypeError:
            text_tokens = self.tokenizer(caption).squeeze(0)

        pair_id = entry["pairid"]
        img_set_members = entry.get("img_set", {}).get("members", [])

        return ref_image, text_tokens, pair_id, img_set_members


def collate_queries(batch):
    """Custom collate to handle variable-length img_set_members lists."""
    images = torch.stack([b[0] for b in batch])
    tokens = torch.stack([b[1] for b in batch])
    pair_ids = [b[2] for b in batch]
    img_sets = [b[3] for b in batch]
    return images, tokens, pair_ids, img_sets


# ======================================================================
# Main submission generation
# ======================================================================
def generate_submission(args):
    device = torch.device(args.device)

    print(f"\n{'='*60}")
    print(f"SIRA-CIRR Test1 Submission Generation")
    print(f"{'='*60}")

    # ---- Load CLIP ----
    try:
        import clip
        clip_model, preprocess = clip.load(args.clip_model, device=device)
        clip_model = clip_model.float()
        tokenizer = clip.tokenize
    except ImportError:
        import open_clip
        clip_model, _, preprocess = open_clip.create_model_and_transforms(
            args.clip_model.replace("/", "-"), pretrained="openai")
        clip_model = clip_model.to(device).float()
        tokenizer = open_clip.get_tokenizer(args.clip_model.replace("/", "-"))
    clip_model.eval()

    # ---- Load trained SIRA-CIRR checkpoint ----
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = ckpt.get("config", {})
    d_model = ckpt.get("d_model", 512)

    sira_model = SIRACIRRModel(
        d_model=d_model,
        d_synergy=config.get("d_synergy", 64),
        gate_rank=config.get("gate_rank", 16),
        gate_init_bias=config.get("gate_init_bias", 0.0),
        dropout=0.0,  # no dropout at inference
    ).to(device)
    sira_model.load_state_dict(ckpt["model_state_dict"])
    sira_model.eval()

    print(f"  Loaded SIRA-CIRR checkpoint: {args.checkpoint}")

    # ================================================================
    # Step 1: Encode ALL test1 candidate images
    # ================================================================
    print(f"\n  [1/3] Encoding all test1 candidate images...")
    candidate_ds = CIRRCandidateDataset(
        args.cirr_root, split="test1", preprocess=preprocess)
    candidate_loader = DataLoader(
        candidate_ds, batch_size=args.batch_size,
        shuffle=False, num_workers=args.workers, pin_memory=True)

    all_candidate_embeddings = []
    all_candidate_ids = []

    start = time.time()
    with torch.no_grad():
        for images, img_ids in candidate_loader:
            images = images.to(device)
            z_img = clip_model.encode_image(images)
            z_img = F.normalize(z_img.float(), dim=-1)
            all_candidate_embeddings.append(z_img.cpu())
            all_candidate_ids.extend(img_ids)

    candidate_embeddings = torch.cat(all_candidate_embeddings, dim=0)
    id_to_idx = {img_id: i for i, img_id in enumerate(all_candidate_ids)}

    elapsed = time.time() - start
    print(f"    Encoded {len(all_candidate_ids)} candidates in {elapsed:.0f}s")

    # ================================================================
    # Step 2: Compute z_query for each test1 query via SIRA
    # ================================================================
    print(f"\n  [2/3] Computing SIRA fusion embeddings for queries...")
    query_ds = CIRRQueryDataset(
        args.cirr_root, split="test1",
        preprocess=preprocess, tokenizer=tokenizer)
    query_loader = DataLoader(
        query_ds, batch_size=args.batch_size,
        shuffle=False, num_workers=args.workers,
        pin_memory=True, collate_fn=collate_queries)

    recall_results = {"version": "rc2", "metric": "recall"}
    recall_subset_results = {"version": "rc2", "metric": "recall_subset"}

    n_queries = 0
    start = time.time()

    with torch.no_grad():
        for ref_images, text_tokens, pair_ids, img_sets in query_loader:
            ref_images = ref_images.to(device)
            text_tokens = text_tokens.to(device)

            # CLIP encode: reference image + caption
            v_ref = clip_model.encode_image(ref_images)
            v_ref = F.normalize(v_ref.float(), dim=-1)

            t_cap = clip_model.encode_text(text_tokens)
            t_cap = F.normalize(t_cap.float(), dim=-1)

            # SIRA compose: SIM + SRG + QueryCombiner → z_query
            z_query = sira_model.encode_query(v_ref, t_cap)  # (B, 512)

            # Compute similarity against ALL candidates
            sim = z_query.cpu() @ candidate_embeddings.T  # (B, N_cand)

            for i in range(z_query.shape[0]):
                pid = str(pair_ids[i])
                scores = sim[i]

                # --- Recall (top-50 from all candidates) ---
                topk_50 = scores.topk(50).indices.tolist()
                recall_results[pid] = [all_candidate_ids[j] for j in topk_50]

                # --- Recall_subset (top-3 from img_set members only) ---
                members = img_sets[i]
                if members:
                    member_indices = [id_to_idx[m] for m in members
                                      if m in id_to_idx]
                    if member_indices:
                        member_scores = scores[member_indices]
                        topk_3_local = member_scores.topk(
                            min(3, len(member_indices))).indices.tolist()
                        recall_subset_results[pid] = [
                            all_candidate_ids[member_indices[j]]
                            for j in topk_3_local
                        ]
                    else:
                        topk_3 = scores.topk(3).indices.tolist()
                        recall_subset_results[pid] = [
                            all_candidate_ids[j] for j in topk_3]
                else:
                    topk_3 = scores.topk(3).indices.tolist()
                    recall_subset_results[pid] = [
                        all_candidate_ids[j] for j in topk_3]

                n_queries += 1

    elapsed = time.time() - start
    print(f"    Processed {n_queries} queries in {elapsed:.0f}s")

    # ================================================================
    # Step 3: Save submission files
    # ================================================================
    print(f"\n  [3/3] Saving submission files...")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    recall_path = out_dir / "submission_recall.json"
    recall_subset_path = out_dir / "submission_recall_subset.json"

    with open(recall_path, "w") as f:
        json.dump(recall_results, f)
    with open(recall_subset_path, "w") as f:
        json.dump(recall_subset_results, f)

    recall_size = recall_path.stat().st_size / 1e6
    subset_size = recall_subset_path.stat().st_size / 1e6

    print(f"    {recall_path} ({recall_size:.1f} MB)")
    print(f"    {recall_subset_path} ({subset_size:.1f} MB)")

    if recall_size > 5.0:
        print(f"    ⚠ WARNING: recall file exceeds 5MB limit!")
    if subset_size > 5.0:
        print(f"    ⚠ WARNING: recall_subset file exceeds 5MB limit!")

    print(f"\n{'='*60}")
    print(f"  Submission files generated!")
    print(f"  Upload to: https://cirr.cecs.anu.edu.au/test_process/")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Generate CIRR test1 submission files (SIRA-CIRR)")
    parser.add_argument("--checkpoint",
                        default="./checkpoints/sira_cirr/sira_cirr_best.pt",
                        help="Path to trained SIRA-CIRR checkpoint")
    parser.add_argument("--cirr-root",
                        default="/home/otw/chiennhm/data/CIRR")
    parser.add_argument("--clip-model", default="ViT-B/32")
    parser.add_argument("--output-dir", default="./submissions/sira_cirr")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    generate_submission(args)


if __name__ == "__main__":
    main()
