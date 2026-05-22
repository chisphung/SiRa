"""
Training pipeline for SIRA.

Two-phase training:
    Phase 1 (optional): Warm-up DsRA adapters with standard InfoNCE
    Phase 2: Joint SIRA training with L_shared + L_syn + L_orth

Supports COCO and Flickr30K with hard negative mining.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import time
import json
import random
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import GradScaler, autocast
from pathlib import Path

from sira.sira_model import SIRAModel


# ======================================================================
# Dataset
# ======================================================================
class ImageTextDataset(Dataset):
    """
    Generic image-text pair dataset. Expects:
        - image_dir: directory of images
        - annotations: JSON file with list of {"image": filename, "caption": text}
    """

    def __init__(self, image_dir, annotations_file, preprocess, tokenizer, max_len=77):
        self.image_dir = Path(image_dir)
        self.preprocess = preprocess
        self.tokenizer = tokenizer
        self.max_len = max_len

        with open(annotations_file, "r") as f:
            self.annotations = json.load(f)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        from PIL import Image
        ann = self.annotations[idx]
        img_path = self.image_dir / ann["image"]
        image = self.preprocess(Image.open(img_path).convert("RGB"))
        text = self.tokenizer(ann["caption"], truncate=True).squeeze(0)
        return image, text


class COCODataset(Dataset):
    """
    MS-COCO Captions dataset wrapper.

    Expects standard COCO format:
        - image_root/train2014/*.jpg
        - annotations/captions_train2014.json
    """

    def __init__(self, image_root, ann_file, preprocess, tokenizer):
        self.image_root = Path(image_root)
        self.preprocess = preprocess
        self.tokenizer = tokenizer

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

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        from PIL import Image
        filename, caption = self.pairs[idx]
        img_path = self.image_root / filename
        image = self.preprocess(Image.open(img_path).convert("RGB"))
        text = self.tokenizer(caption, truncate=True).squeeze(0)
        return image, text


class HLDataset(Dataset):
    """
    High-Level (HL) Dataset — scene/action/rationale/object captions.

    Reads metadata.jsonl directly from the HL dataset and fuses the
    multi-axis annotations into a single synergistic text description
    for SIRA training.

    Caption format:
        "[scene], [action] because [rationale]."

    This captures the cross-modal synergistic information (causality,
    intent, context) that standard object-centric captions miss.

    Args:
        hl_root: Path to HL dataset root (e.g., /home/otw/chisphung/hl).
        split: 'train' or 'test'.
        preprocess: CLIP image preprocessing function.
        tokenizer: CLIP text tokenizer.
        mode: 'fused' (scene+action+rationale) or 'unified' (all 4 axes).
    """

    FUSION_TEMPLATES = [
        "{scene}, {action} because {rationale}.",
        "{action} {scene}, {rationale}.",
        "In a scene {scene}, {action}. The reason is that {rationale}.",
        "{action} {scene} — {rationale}.",
        "This image shows someone {action} {scene} because {rationale}.",
    ]

    def __init__(self, hl_root, split="train", preprocess=None, tokenizer=None,
                 mode="fused"):
        self.preprocess = preprocess
        self.tokenizer = tokenizer
        self.mode = mode

        data_dir = Path(hl_root) / "data" / split
        jsonl_path = data_dir / "metadata.jsonl"

        if not jsonl_path.exists():
            raise FileNotFoundError(f"HL metadata not found: {jsonl_path}")

        self.image_dir = data_dir
        self.entries = []

        with open(jsonl_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    entry = json.loads(line)
                    # Only include entries with all caption axes
                    caps = entry.get("captions", {})
                    if all(caps.get(k) for k in ["scene", "action", "rationale", "object"]):
                        self.entries.append(entry)

        print(f"  HLDataset [{split}]: {len(self.entries)} entries loaded")

    def _select_best(self, captions, confidences=None):
        """Pick the caption with highest confidence score."""
        if not captions:
            return ""
        if confidences and len(confidences) == len(captions):
            best_idx = max(range(len(confidences)), key=lambda i: confidences[i])
            return captions[best_idx]
        return captions[0]

    def _clean(self, text):
        """Normalize caption text."""
        text = text.strip().rstrip('.')
        if text and text[0].isupper():
            text = text[0].lower() + text[1:]
        return text

    def _fuse_caption(self, entry, idx):
        """Fuse scene + action + rationale into a synergistic caption."""
        caps = entry["captions"]
        conf = entry.get("confidence", {})

        scene = self._clean(self._select_best(caps["scene"], conf.get("scene")))
        action = self._clean(self._select_best(caps["action"], conf.get("action")))
        rationale = self._clean(self._select_best(caps["rationale"], conf.get("rationale")))

        template = random.choice(self.FUSION_TEMPLATES)
        caption = template.format(scene=scene, action=action, rationale=rationale)
        caption = caption[0].upper() + caption[1:] if caption else caption
        return caption

    def _unified_caption(self, entry):
        """Create a unified caption with all four axes."""
        caps = entry["captions"]
        conf = entry.get("confidence", {})

        scene = self._select_best(caps["scene"], conf.get("scene"))
        action = self._select_best(caps["action"], conf.get("action"))
        rationale = self._select_best(caps["rationale"], conf.get("rationale"))
        obj = self._select_best(caps["object"])

        return (
            f"Scene: {scene.strip().rstrip('.')}. "
            f"Action: {action.strip().rstrip('.')}. "
            f"Rationale: {rationale.strip().rstrip('.')}. "
            f"Description: {obj.strip().rstrip('.')}."
        )

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        from PIL import Image
        entry = self.entries[idx]

        img_path = self.image_dir / entry["file_name"]
        image = self.preprocess(Image.open(img_path).convert("RGB"))

        if self.mode == "fused":
            caption = self._fuse_caption(entry, idx)
        else:
            caption = self._unified_caption(entry)

        try:
            text = self.tokenizer(caption, truncate=True).squeeze(0)
        except TypeError:
            text = self.tokenizer(caption).squeeze(0)
        return image, text


# ======================================================================
# Trainer
# ======================================================================
class SIRATrainer:
    """
    SIRA training loop with logging, checkpointing, and mixed precision.

    Args:
        model: SIRAModel instance.
        train_dataset: Training dataset.
        val_dataset: Optional validation dataset.
        lr: Learning rate for SIRA modules. Default: 5e-4.
        batch_size: Training batch size. Default: 256.
        num_epochs: Total training epochs. Default: 15.
        warmup_epochs: LR warmup epochs. Default: 1.
        output_dir: Directory for checkpoints and logs.
        use_amp: Use automatic mixed precision. Default: True.
        grad_clip: Max gradient norm. Default: 1.0.
        num_workers: DataLoader workers. Default: 4.
        log_every: Log every N steps. Default: 50.
    """

    def __init__(self, model, train_dataset, val_dataset=None,
                 lr=5e-4, batch_size=256, num_epochs=15, warmup_epochs=1,
                 output_dir="./checkpoints", use_amp=True, grad_clip=1.0,
                 num_workers=4, log_every=50):
        self.model = model
        self.device = next(model.sim.parameters()).device
        self.num_epochs = num_epochs
        self.warmup_epochs = warmup_epochs
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.use_amp = use_amp and torch.cuda.is_available()
        self.grad_clip = grad_clip
        self.log_every = log_every

        # DataLoaders
        self.train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True, drop_last=True)
        self.val_loader = None
        if val_dataset is not None:
            self.val_loader = DataLoader(
                val_dataset, batch_size=batch_size, shuffle=False,
                num_workers=num_workers, pin_memory=True)

        # Optimizer: separate param groups for adapter vs backbone
        param_groups = model.get_trainable_param_groups(lr=lr, backbone_lr_scale=0.1)
        self.optimizer = torch.optim.AdamW(param_groups, weight_decay=0.01)

        # Cosine scheduler with warmup
        total_steps = len(self.train_loader) * num_epochs
        warmup_steps = len(self.train_loader) * warmup_epochs
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer, max_lr=lr, total_steps=total_steps,
            pct_start=warmup_steps / total_steps, anneal_strategy="cos")

        # Mixed precision
        self.scaler = GradScaler(enabled=self.use_amp)

        # Logging
        self.train_log = []

    def train(self):
        """Run full training loop."""
        print(f"\n{'='*60}")
        print(f"SIRA Training")
        print(f"{'='*60}")
        summary = self.model.get_param_summary()
        frozen = summary['frozen']
        adapter_trainable = summary['sim'] + summary['srg'] + summary['loss_fn']
        backbone_trainable = summary['backbone_trainable']

        print(f"\n{'='*60}")
        print(f"SIRA Training")
        print(f"{'='*60}")
        print(f"Total params:    {summary['total']:,}")
        print(f"Frozen backbone: {frozen:,} params")
        print(f"Trainable SIRA:  {adapter_trainable:,} params ({summary['trainable_pct']})")
        print(f"  SIM: {summary['sim']:,}  |  SRG: {summary['srg']:,}  |  Loss: {summary['loss_fn']:,}")
        if backbone_trainable > 0:
            print(f"Unfrozen backbone: {backbone_trainable:,} params")
        print(f"Epochs: {self.num_epochs}  |  Batch: {self.train_loader.batch_size}")
        print(f"AMP: {self.use_amp}  |  Device: {self.device}")
        print(f"{'='*60}\n")

        best_loss = float("inf")

        for epoch in range(self.num_epochs):
            epoch_loss = self._train_epoch(epoch)

            # Validation
            if self.val_loader is not None:
                val_loss = self._validate(epoch)
                print(f"  Val loss: {val_loss:.4f}")
                track_loss = val_loss
            else:
                track_loss = epoch_loss

            # Save checkpoint
            if track_loss < best_loss:
                best_loss = track_loss
                self._save_checkpoint(epoch, is_best=True)

            if (epoch + 1) % 5 == 0:
                self._save_checkpoint(epoch)

        # Save final
        self._save_checkpoint(self.num_epochs - 1, is_final=True)
        self._save_log()
        print(f"\nTraining complete. Best loss: {best_loss:.4f}")

    def _train_epoch(self, epoch):
        self.model.train()  # M4: model-level train(); CLIP stays eval via override

        total_loss = 0.0
        total_base = 0.0
        total_orth = 0.0
        num_batches = 0
        start_time = time.time()

        for step, (images, texts) in enumerate(self.train_loader):
            images = images.to(self.device)
            texts = texts.to(self.device)

            self.optimizer.zero_grad()

            with autocast(enabled=self.use_amp):
                losses = self.model(images, texts)

            self.scaler.scale(losses["total"]).backward()

            if self.grad_clip > 0:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.get_trainable_params(), self.grad_clip)

            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()

            total_loss += losses["total"].item()
            total_base += losses["base"].item()
            total_orth += losses["orthogonality"].item()
            num_batches += 1

            if (step + 1) % self.log_every == 0:
                elapsed = time.time() - start_time
                print(
                    f"  Epoch {epoch+1}/{self.num_epochs} | "
                    f"Step {step+1}/{len(self.train_loader)} | "
                    f"Loss: {losses['total'].item():.4f} "
                    f"(base={losses['base'].item():.3f} orth={losses['orthogonality'].item():.3f}) | "
                    f"s_norm={losses['synergy_norm'].item():.3f} | "
                    f"τ={losses['temperature'].item():.4f} | "
                    f"lr={self.scheduler.get_last_lr()[0]:.2e} | "
                    f"{elapsed:.0f}s"
                )

                # Compute gate stats for logging
                with torch.no_grad():
                    gate_stats = self.model.srg.get_gate_stats(
                        self.model._encode_image(images),
                        self.model._encode_text(texts),
                        self.model.sim(
                            self.model._encode_image(images),
                            self.model._encode_text(texts),
                        )
                    )

                self.train_log.append({
                    "epoch": epoch, "step": step,
                    "loss": losses["total"].item(),
                    "base": losses["base"].item(),
                    "orthogonality": losses["orthogonality"].item(),
                    "temperature": losses["temperature"].item(),
                    "synergy_norm": losses["synergy_norm"].item(),
                    "gate_v_mean": gate_stats["gate_v_mean"],
                    "gate_t_mean": gate_stats["gate_t_mean"],
                })

        avg_loss = total_loss / num_batches
        elapsed = time.time() - start_time
        print(
            f"Epoch {epoch+1}/{self.num_epochs} | "
            f"Avg Loss: {avg_loss:.4f} "
            f"(base={total_base/num_batches:.3f}) | "
            f"Time: {elapsed:.0f}s"
        )
        return avg_loss

    @torch.no_grad()
    def _validate(self, epoch):
        self.model.eval()  # M4: model-level eval()

        total_loss = 0.0
        num_batches = 0
        for images, texts in self.val_loader:
            images = images.to(self.device)
            texts = texts.to(self.device)
            losses = self.model(images, texts)
            total_loss += losses["total"].item()
            num_batches += 1

        return total_loss / num_batches

    def _save_checkpoint(self, epoch, is_best=False, is_final=False):
        state = {
            "epoch": epoch,
            "sim_state_dict": self.model.sim.state_dict(),
            "srg_state_dict": self.model.srg.state_dict(),
            "loss_fn_state_dict": self.model.loss_fn.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }
        # Save backbone state if layers were unfrozen
        if self.model._unfrozen_backbone_params:
            state["clip_state_dict"] = self.model.clip.state_dict()
        if is_best:
            path = self.output_dir / "sira_best.pt"
        elif is_final:
            path = self.output_dir / "sira_final.pt"
        else:
            path = self.output_dir / f"sira_epoch{epoch+1}.pt"
        torch.save(state, path)

    def _save_log(self):
        with open(self.output_dir / "train_log.json", "w") as f:
            json.dump(self.train_log, f, indent=2)


# ======================================================================
# CLI
# ======================================================================
def main():
    parser = argparse.ArgumentParser(description="Train SIRA")
    parser.add_argument("--clip-model", default="ViT-B/32", help="CLIP backbone")
    parser.add_argument("--dataset-type", default="coco", choices=["coco", "hl", "json"],
                        help="Dataset type: 'coco' (COCO format), 'hl' (HL dataset), 'json' (generic)")
    parser.add_argument("--image-root", default=None, help="Path to images (required for coco/json)")
    parser.add_argument("--ann-file", default=None, help="COCO annotation JSON (required for coco)")
    parser.add_argument("--hl-root", default="/home/otw/chisphung/hl",
                        help="HL dataset root (for --dataset-type hl)")
    parser.add_argument("--hl-split", default="train", choices=["train", "test"],
                        help="HL dataset split")
    parser.add_argument("--hl-mode", default="fused", choices=["fused", "unified"],
                        help="HL caption mode: 'fused' (scene+action+rationale) or 'unified' (all 4 axes)")
    parser.add_argument("--output-dir", default="./checkpoints/sira")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--d-synergy", type=int, default=64)
    parser.add_argument("--lambda-orth", type=float, default=0.1,
                        help="Weight for orthogonality loss")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--unfreeze-strategy", default="none",
                        choices=["none", "projection", "last-layer", "full"],
                        help="Fine-tuning strategy for the CLIP backbone")
    parser.add_argument("--load-backbone", default=None,
                        help="Path to fine-tuned CLIP baseline checkpoint to load before freezing")
    parser.add_argument("--backbone-lr-scale", type=float, default=0.1,
                        help="LR multiplier for unfrozen backbone params (default: 0.1x)")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    # Build model
    model, preprocess = SIRAModel.from_clip(
        args.clip_model, args.device, args.d_synergy, lambda_orth=args.lambda_orth)

    # Build tokenizer
    try:
        import clip
        tokenizer = clip.tokenize
    except ImportError:
        import open_clip
        tokenizer = open_clip.get_tokenizer(args.clip_model.replace("/", "-"))

    # Build dataset based on type
    if args.dataset_type == "hl":
        dataset = HLDataset(
            args.hl_root, split=args.hl_split, preprocess=preprocess,
            tokenizer=tokenizer, mode=args.hl_mode)
    elif args.dataset_type == "coco":
        if not args.image_root or not args.ann_file:
            parser.error("--image-root and --ann-file required for COCO dataset")
        dataset = COCODataset(args.image_root, args.ann_file, preprocess, tokenizer)
    else:  # json
        if not args.image_root or not args.ann_file:
            parser.error("--image-root and --ann-file required for JSON dataset")
        dataset = ImageTextDataset(args.image_root, args.ann_file, preprocess, tokenizer)
    print(f"Dataset: {len(dataset)} pairs")

    # Load pre-fine-tuned backbone if provided
    if args.load_backbone:
        print(f"Loading fine-tuned backbone from {args.load_backbone}")
        ckpt = torch.load(args.load_backbone, map_location=args.device, weights_only=True)
        if "model_state_dict" in ckpt:
            # Baseline script saves as model_state_dict with 'clip.' prefix stripped
            clip_state = {}
            for k, v in ckpt["model_state_dict"].items():
                clip_state[k] = v
            model.clip.load_state_dict(clip_state, strict=False)
        elif "clip_state_dict" in ckpt:
            model.clip.load_state_dict(ckpt["clip_state_dict"])
            
    # Unfreeze backbone layers if requested
    if args.unfreeze_strategy != "none":
        model.unfreeze_layers(args.unfreeze_strategy)

    # Train
    trainer = SIRATrainer(
        model, dataset, lr=args.lr, batch_size=args.batch_size,
        num_epochs=args.epochs, output_dir=args.output_dir,
        use_amp=not args.no_amp, num_workers=args.workers)
    trainer.train()


if __name__ == "__main__":
    main()
