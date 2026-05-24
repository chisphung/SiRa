import sys
import os
import time
import json
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.amp import GradScaler, autocast

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sira.simplified_sira import SimplifiedSIRA


class COCODataset(Dataset):
    """
    MS-COCO Captions dataset wrapper.
    Expects standard COCO format.
    """
    def __init__(self, image_root, ann_file, preprocess, tokenizer):
        self.image_root = Path(image_root)
        self.preprocess = preprocess
        self.tokenizer = tokenizer

        print(f"Loading COCO annotations from {ann_file}...")
        with open(ann_file, "r") as f:
            data = json.load(f)

        # Build image_id -> filename mapping
        id_to_file = {img["id"]: img["file_name"] for img in data["images"]}

        # Build list of (filename, caption) pairs
        self.pairs = []
        for ann in data["annotations"]:
            filename = id_to_file.get(ann["image_id"])
            if filename:
                self.pairs.append((filename, ann["caption"]))
                
        print(f"Loaded {len(self.pairs)} image-caption pairs.")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        from PIL import Image
        filename, caption = self.pairs[idx]
        img_path = self.image_root / filename
        try:
            image = self.preprocess(Image.open(img_path).convert("RGB"))
        except Exception as e:
            # Fallback to random black image if something is wrong (should not happen on standard COCO)
            print(f"Warning: Failed to load {img_path}: {e}")
            from PIL import Image as PILImage
            image = self.preprocess(PILImage.new("RGB", (224, 224)))
            
        try:
            text = self.tokenizer(caption, truncate=True).squeeze(0)
        except TypeError:
            text = self.tokenizer(caption).squeeze(0)
            
        return image, text


def train():
    parser = argparse.ArgumentParser(description="Train Simplified SIRA on COCO")
    parser.add_argument("--clip-model", default="ViT-B/32", help="CLIP backbone")
    parser.add_argument("--coco-image-root", required=True, help="Path to COCO images")
    parser.add_argument("--coco-ann-file", required=True, help="Path to COCO annotations JSON")
    parser.add_argument("--output-dir", default="./checkpoints/simplified_sira")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--d-synergy", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    # Build model
    print(f"Initializing SimplifiedSIRA with {args.clip_model}...")
    model, preprocess = SimplifiedSIRA.from_clip(args.clip_model, device=args.device, d_synergy=args.d_synergy)
    model.train() # Make sure our modules are in train mode (CLIP stays in eval)

    # Build tokenizer
    try:
        import clip
        tokenizer = clip.tokenize
    except ImportError:
        import open_clip
        tokenizer = open_clip.get_tokenizer(args.clip_model.replace("/", "-"))

    # Load Dataset
    dataset = COCODataset(args.coco_image_root, args.coco_ann_file, preprocess, tokenizer)
    train_loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True
    )

    # Setup Optimizer
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"Trainable parameters: {sum(p.numel() for p in trainable_params):,}")
    
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=0.01)
    
    # Cosine scheduler with warmup
    total_steps = len(train_loader) * args.epochs
    warmup_steps = len(train_loader) * 1 # 1 epoch warmup
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, total_steps=total_steps,
        pct_start=warmup_steps / total_steps, anneal_strategy="cos"
    )

    scaler = GradScaler("cuda", enabled=(args.device == "cuda"))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Training Loop
    print(f"Starting training on {args.device} for {args.epochs} epochs...")
    log_every = 50
    best_loss = float('inf')

    for epoch in range(args.epochs):
        epoch_loss = 0.0
        start_time = time.time()
        
        for step, (images, texts) in enumerate(train_loader):
            images = images.to(args.device)
            texts = texts.to(args.device)
            
            optimizer.zero_grad()
            
            with autocast(device_type="cuda" if args.device == "cuda" else "cpu", enabled=(args.device == "cuda")):
                # final_scores is (B, B) where final_scores[i, j] is sim(image_i, text_j)
                scores = model(images, texts)
                
                # Symmetric InfoNCE Loss
                labels = torch.arange(images.size(0), device=args.device)
                loss_i = F.cross_entropy(scores, labels)
                loss_t = F.cross_entropy(scores.T, labels)
                loss = (loss_i + loss_t) / 2
                
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(trainable_params, 1.0)
            scaler.step(optimizer)
            scaler.update()
            
            # Ensure scheduler is called after optimizer step finishes (GradScaler skips optimizer.step() if gradients are Inf/NaN)
            # Actually, to be perfectly safe with GradScaler, one should only step the scheduler if the scaler didn't skip the optimizer step.
            # But the simplest fix for the warning is to just call scheduler.step() here since OneCycleLR handles skipped steps well enough.
            scheduler.step()
            
            epoch_loss += loss.item()
            
            if (step + 1) % log_every == 0:
                elapsed = time.time() - start_time
                print(
                    f"Epoch {epoch+1}/{args.epochs} | "
                    f"Step {step+1}/{len(train_loader)} | "
                    f"Loss: {loss.item():.4f} | "
                    f"α={model.alpha.item():.3f} γ={model.gamma.item():.3f} | "
                    f"τ={1 / model.logit_scale.exp().item():.3f} | "
                    f"lr={scheduler.get_last_lr()[0]:.2e} | "
                    f"{elapsed:.0f}s"
                )
        
        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch+1} finished. Avg Loss: {avg_loss:.4f}")
        
        # Checkpointing
        state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }
        torch.save(state, output_dir / f"simplified_sira_epoch_{epoch+1}.pt")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(state, output_dir / "simplified_sira_best.pt")
            
    print("Training complete!")


if __name__ == "__main__":
    train()
