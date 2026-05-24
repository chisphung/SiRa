import sys
import os
import json
import argparse
from pathlib import Path
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader, Dataset

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sira.simplified_sira import SimplifiedSIRA


class SCPPDataset(Dataset):
    """
    SugarCrepe++ evaluation dataset.
    Loads all JSON files in the SCPP data directory.
    """
    def __init__(self, scpp_root, image_root, preprocess, tokenizer):
        self.scpp_root = Path(scpp_root)
        self.image_root = Path(image_root)
        self.preprocess = preprocess
        self.tokenizer = tokenizer
        
        self.data_dir = self.scpp_root / "data"
        self.entries = []
        
        # Load all json files
        for json_file in self.data_dir.glob("*.json"):
            cat_name = json_file.stem
            with open(json_file, "r") as f:
                data = json.load(f)
                for item in data:
                    item["category"] = cat_name
                    self.entries.append(item)
                    
        print(f"Loaded {len(self.entries)} entries from SCPP.")

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        from PIL import Image
        entry = self.entries[idx]
        
        img_path = self.image_root / entry["filename"]
        try:
            image = self.preprocess(Image.open(img_path).convert("RGB"))
        except Exception as e:
            print(f"Warning: Failed to load {img_path}: {e}")
            from PIL import Image as PILImage
            image = self.preprocess(PILImage.new("RGB", (224, 224)))
            
        pos_text_raw = entry["caption"]
        neg_text_raw = entry["negative_caption"]
        category = entry["category"]
        
        try:
            pos_text = self.tokenizer(pos_text_raw, truncate=True).squeeze(0)
            neg_text = self.tokenizer(neg_text_raw, truncate=True).squeeze(0)
        except TypeError:
            pos_text = self.tokenizer(pos_text_raw).squeeze(0)
            neg_text = self.tokenizer(neg_text_raw).squeeze(0)
            
        return image, pos_text, neg_text, category


def evaluate():
    parser = argparse.ArgumentParser(description="Evaluate Simplified SIRA on SCPP")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--clip-model", default="ViT-B/32", help="CLIP backbone used for training")
    parser.add_argument("--scpp-root", required=True, help="Path to SugarCrepe++ root dir")
    parser.add_argument("--scpp-image-root", required=True, help="Path to COCO val2017 images")
    parser.add_argument("--d-synergy", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    print(f"Initializing SimplifiedSIRA with {args.clip_model}...")
    model, preprocess = SimplifiedSIRA.from_clip(args.clip_model, device=args.device, d_synergy=args.d_synergy)
    
    print(f"Loading checkpoint {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    try:
        import clip
        tokenizer = clip.tokenize
    except ImportError:
        import open_clip
        tokenizer = open_clip.get_tokenizer(args.clip_model.replace("/", "-"))

    dataset = SCPPDataset(args.scpp_root, args.scpp_image_root, preprocess, tokenizer)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True
    )

    print("Evaluating...")
    category_correct = {}
    category_total = {}
    
    with torch.no_grad():
        for images, pos_texts, neg_texts, categories in tqdm(loader):
            images = images.to(args.device)
            pos_texts = pos_texts.to(args.device)
            neg_texts = neg_texts.to(args.device)
            
            # SimplifiedSIRA takes a batch of images and texts and computes all pairs
            # But here we want aligned scores: score(images[i], pos_texts[i])
            # We can just extract features and do it manually to save computation.
            v = model._encode_image(images)
            t_pos = model._encode_text(pos_texts)
            t_neg = model._encode_text(neg_texts)
            
            # Base scores
            base_pos = (v * t_pos).sum(dim=-1)
            base_neg = (v * t_neg).sum(dim=-1)
            
            # Synergy scores
            # Shape is (B, D). We want B matched pairs, so we don't need (B, B)
            # We just do element-wise Hadamard and pass to MLP
            # For SimplifiedSIRA.sim:
            # sim expects (B, 1, D) and (1, B, D) to produce (B, B, D).
            # We can bypass it or reshape.
            # actually, sim takes v and t and does self.proj_v(v) * self.proj_t(t).
            # If we pass (B, D) it will return (B, D). Let's check sim.
            # Yes, inter = self.proj_v(v) * self.proj_t(t).
            s_pos = model.sim(v, t_pos)
            s_neg = model.sim(v, t_neg)
            
            syn_pos = model.synergy_scorer(s_pos).squeeze(-1)
            syn_neg = model.synergy_scorer(s_neg).squeeze(-1)
            
            final_pos = model.alpha * base_pos + model.gamma * syn_pos
            final_neg = model.alpha * base_neg + model.gamma * syn_neg
            
            correct = (final_pos > final_neg).cpu().numpy()
            
            for c, is_corr in zip(categories, correct):
                if c not in category_correct:
                    category_correct[c] = 0
                    category_total[c] = 0
                category_correct[c] += is_corr
                category_total[c] += 1

    print("\n" + "="*40)
    print("SCPP Evaluation Results")
    print("="*40)
    
    total_corr = 0
    total_count = 0
    for cat in sorted(category_correct.keys()):
        corr = category_correct[cat]
        cnt = category_total[cat]
        acc = corr / cnt * 100
        print(f"  {cat:<15s}: {acc:.2f}% ({corr}/{cnt})")
        total_corr += corr
        total_count += cnt
        
    overall_acc = total_corr / total_count * 100
    print("-" * 40)
    print(f"  {'Overall':<15s}: {overall_acc:.2f}% ({total_corr}/{total_count})")
    print("="*40)


if __name__ == "__main__":
    evaluate()
