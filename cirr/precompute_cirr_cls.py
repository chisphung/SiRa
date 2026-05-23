"""
Pre-compute CLS-level CLIP embeddings for the CIRR dataset.

Runs CLIP encoding ONCE and saves CLS-level embeddings to disk.
Subsequent training runs load from cache — no CLIP forward pass needed.

Unlike the Fusion pipeline's token-level precomputation (49 patches × 768),
this produces compact CLS-level vectors suitable for SIRA's bilinear
interaction module.

Cached tensors per split:
    - v_ref:     (N, 512)  — reference image CLIP CLS embedding
    - t_cap:     (N, 512)  — relative caption CLIP text embedding
    - z_target:  (N, 512)  — target image CLIP CLS embedding
    - entries.json         — per-sample metadata (pairid, caption, etc.)
    - metadata.json        — split info, shapes, model config

Usage:
    python cirr/precompute_cirr_cls.py --split train
    python cirr/precompute_cirr_cls.py --split val
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Allow importing from parent directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cirr.cirr_dataset import CIRRCLSDataset


def precompute(args):
    device = torch.device(args.device)

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

    # ---- Load dataset ----
    dataset = CIRRCLSDataset(
        args.cirr_root, split=args.split,
        preprocess=preprocess, tokenizer=tokenizer)

    loader = DataLoader(
        dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=args.workers,
        pin_memory=True)

    # ---- Output directory ----
    out_dir = Path(args.output_dir) / args.split
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Pre-compute ----
    all_v_ref = []
    all_t_cap = []
    all_z_target = []

    print(f"\n{'='*60}")
    print(f"Pre-computing CLS-level embeddings for CIRR [{args.split}]")
    print(f"  Samples:      {len(dataset)}")
    print(f"  Batch:        {args.batch_size}")
    print(f"  CLIP model:   {args.clip_model}")
    print(f"  Output:       {out_dir}")
    print(f"{'='*60}\n")

    start = time.time()

    with torch.no_grad():
        for step, (ref_images, text_tokens, target_images) in enumerate(loader):
            ref_images = ref_images.to(device)
            text_tokens = text_tokens.to(device)
            target_images = target_images.to(device)

            # Reference image → CLIP CLS embedding
            v_ref = clip_model.encode_image(ref_images)     # (B, 512)
            v_ref = F.normalize(v_ref.float(), dim=-1)

            # Relative caption → CLIP text embedding
            t_cap = clip_model.encode_text(text_tokens)     # (B, 512)
            t_cap = F.normalize(t_cap.float(), dim=-1)

            # Target image → CLIP CLS embedding
            z_target = clip_model.encode_image(target_images)  # (B, 512)
            z_target = F.normalize(z_target.float(), dim=-1)

            # Store as float16 to save disk space
            all_v_ref.append(v_ref.cpu().half())
            all_t_cap.append(t_cap.cpu().half())
            all_z_target.append(z_target.cpu().half())

            if (step + 1) % 10 == 0:
                elapsed = time.time() - start
                pct = (step + 1) / len(loader) * 100
                print(f"  [{pct:5.1f}%] Step {step+1}/{len(loader)} | {elapsed:.0f}s")

    # Concatenate
    all_v_ref = torch.cat(all_v_ref, dim=0)
    all_t_cap = torch.cat(all_t_cap, dim=0)
    all_z_target = torch.cat(all_z_target, dim=0)

    N = all_v_ref.shape[0]
    d_model = all_v_ref.shape[-1]
    elapsed = time.time() - start

    print(f"\n  Encoded {N} samples in {elapsed:.0f}s")
    print(f"  Feature dim: {d_model}")

    # ---- Save tensors ----
    print(f"  Saving to {out_dir}...")

    torch.save(all_v_ref, out_dir / "v_ref.pt")
    torch.save(all_t_cap, out_dir / "t_cap.pt")
    torch.save(all_z_target, out_dir / "z_target.pt")

    # ---- Save per-sample metadata (for evaluation) ----
    entries = []
    for entry in dataset.entries:
        entries.append({
            "pairid": entry["pairid"],
            "reference": entry["reference"],
            "target_hard": entry["target_hard"],
            "caption": entry["caption"],
            "img_set_members": entry.get("img_set", {}).get("members", []),
        })
    with open(out_dir / "entries.json", "w") as f:
        json.dump(entries, f)

    # ---- Save split metadata ----
    meta = {
        "n_samples": N,
        "split": args.split,
        "clip_model": args.clip_model,
        "d_model": d_model,
        "dataset": "cirr",
        "encoding": "cls_level",
        "normalized": True,
        "shapes": {
            "v_ref": list(all_v_ref.shape),
            "t_cap": list(all_t_cap.shape),
            "z_target": list(all_z_target.shape),
        },
        "dtype": "float16",
    }
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    # Size report
    total_bytes = sum(
        (out_dir / f).stat().st_size
        for f in ["v_ref.pt", "t_cap.pt", "z_target.pt"]
    )
    print(f"  Total cache size: {total_bytes / 1e6:.1f} MB")

    print(f"\n{'='*60}")
    print(f"  Pre-computation complete!")
    print(f"  {N} samples saved to {out_dir}")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Pre-compute CLS-level CLIP embeddings for CIRR dataset")
    parser.add_argument("--cirr-root", default="/home/otw/chiennhm/data/CIRR")
    parser.add_argument("--output-dir", default="./cache/cirr_cls")
    parser.add_argument("--split", default="train", choices=["train", "val"])
    parser.add_argument("--clip-model", default="ViT-B/32")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    precompute(args)


if __name__ == "__main__":
    main()
