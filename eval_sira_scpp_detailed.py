import sys
import os
import json
import argparse
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sira.simplified_sira import SimplifiedSIRA

class SCPP_Detailed_Dataset(Dataset):
    """
    SugarCrepe++ evaluation dataset matching test_NegCLIP.py format.
    Loads a single JSON file.
    """
    def __init__(self, json_path, image_root, preprocess, tokenizer):
        self.image_root = Path(image_root)
        self.preprocess = preprocess
        self.tokenizer = tokenizer
        
        with open(json_path, "r") as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        from PIL import Image
        entry = self.data[idx]
        
        img_fname = entry["filename"]
        ipath = self.image_root / img_fname
        
        try:
            image = self.preprocess(Image.open(ipath).convert("RGB"))
        except Exception as e:
            from PIL import Image as PILImage
            image = self.preprocess(PILImage.new("RGB", (224, 224)))
            
        p1_raw = entry["caption"]
        p2_raw = entry["caption2"]
        neg_raw = entry["negative_caption"]
        
        try:
            p1 = self.tokenizer(p1_raw, truncate=True).squeeze(0)
            p2 = self.tokenizer(p2_raw, truncate=True).squeeze(0)
            neg = self.tokenizer(neg_raw, truncate=True).squeeze(0)
        except TypeError:
            p1 = self.tokenizer(p1_raw).squeeze(0)
            p2 = self.tokenizer(p2_raw).squeeze(0)
            neg = self.tokenizer(neg_raw).squeeze(0)
            
        return image, p1, p2, neg, img_fname

def test(args, device, model, test_dataloader):
    total = 0
    correct_img_p1 = 0
    correct_img_p2 = 0
    correct_full = 0  # main task: P1 and P2 closer to Image than Negative
    correct_text = 0
    
    model.eval()

    cos = nn.CosineSimilarity(dim=1, eps=1e-6)

    with torch.no_grad():
        for batch in test_dataloader:
            images, tokens_p1, tokens_p2, tokens_neg, img_name = batch
            images = images.to(device)
            tokens_p1 = tokens_p1.to(device)
            tokens_p2 = tokens_p2.to(device)
            tokens_neg = tokens_neg.to(device)

            # Extract base representations
            v = model._encode_image(images)
            t_p1 = model._encode_text(tokens_p1)
            t_p2 = model._encode_text(tokens_p2)
            t_neg = model._encode_text(tokens_neg)

            # Compute SIRA final scores for Image-Text
            def get_score(v, t):
                base = (v * t).sum(dim=-1)
                s = model.sim(v, t)
                syn = model.synergy_scorer(s).squeeze(-1)
                return model.alpha * base + model.gamma * syn
                
            score_p1 = get_score(v, t_p1)
            score_p2 = get_score(v, t_p2)
            score_neg = get_score(v, t_neg)
            
            # For Text-Text similarities, SimplifiedSIRA doesn't have a text-text synergy module.
            # We will use the base CLIP cosine similarity, as done in NegCLIP.
            cos_p1p2 = cos(t_p1, t_p2)
            cos_p1_neg = cos(t_p1, t_neg)
            cos_p2_neg = cos(t_p2, t_neg)

            # Convert to scalars for comparison
            score_p1 = score_p1.item()
            score_p2 = score_p2.item()
            score_neg = score_neg.item()
            cos_p1p2 = cos_p1p2.item()
            cos_p1_neg = cos_p1_neg.item()
            cos_p2_neg = cos_p2_neg.item()

            total += 1

            if score_p1 > score_neg and score_p2 > score_neg:
                correct_full += 1
            if score_p1 > score_neg:
                correct_img_p1 += 1
            if score_p2 > score_neg:
                correct_img_p2 += 1
            if cos_p1p2 > cos_p1_neg and cos_p1p2 > cos_p2_neg:
                correct_text += 1

    print(f"====== evaluation results ======", flush=True)
    ave_score = float(correct_full) / float(total) if total > 0 else 0
    print(f"Accuracy image-to-text task: {ave_score}", flush=True)
    
    ave_score_orig_p1 = float(correct_img_p1) / float(total) if total > 0 else 0
    print(f"Accuracy Image-P1-Neg: {ave_score_orig_p1}", flush=True)
    
    ave_score_orig_p2 = float(correct_img_p2) / float(total) if total > 0 else 0
    print(f"Accuracy Image-P2-Neg: {ave_score_orig_p2}", flush=True)

    ave_score_txt = float(correct_text) / float(total) if total > 0 else 0
    print(f"Accuracy text-only task: {ave_score_txt}", flush=True)

def evaluate():
    parser = argparse.ArgumentParser(description="Detailed SCPP Evaluation")
    parser.add_argument("--checkpoint", required=True, help="Path to SimplifiedSIRA checkpoint")
    parser.add_argument("--clip-model", default="ViT-B/32", help="CLIP backbone")
    parser.add_argument("--scpp-root", required=True, help="Path to SugarCrepe++ root dir")
    parser.add_argument("--scpp-image-root", required=True, help="Path to COCO images")
    parser.add_argument("--d-synergy", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=1) # Same as NegCLIP
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

    data_path = os.path.join(args.scpp_root, 'data')
    fnames = os.listdir(data_path)

    for fname in sorted(fnames):
        if not fname.endswith(".json"):
            continue
            
        print('=======================================================================')
        print('=======================================', fname, '=====================')
        print('=======================================================================')

        test_json_path = os.path.join(data_path, fname)
        test_dataset = SCPP_Detailed_Dataset(test_json_path, args.scpp_image_root, preprocess, tokenizer)
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
        
        test(args, args.device, model, test_dataloader)

if __name__ == "__main__":
    evaluate()
