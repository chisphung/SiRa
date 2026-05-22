#!/usr/bin/env python3
"""
Evaluate SIRA (Synergistic Information-Aware Retrieval Adaptation) on the Winoground benchmark.

Winoground tests compositional reasoning. It contains 400 examples.
Each example has:
    - Image 0 (I0)
    - Image 1 (I1)
    - Caption 0 (C0)
    - Caption 1 (C1)

We compute the 4 pairwise similarities:
    - S(I0, C0), S(I0, C1)
    - S(I1, C0), S(I1, C1)

Scoring:
    - Text Score: S(I0,C0) > S(I0,C1) AND S(I1,C1) > S(I1,C0)
    - Image Score: S(I0,C0) > S(I1,C0) AND S(I1,C1) > S(I0,C1)
    - Group Score: Both Text and Image Score are correct.

Methods:
    - CLIP Baseline: Cosine similarity of shared (base) features.
    - SIRA Gated Interaction: (v_final * t_final).sum(dim=-1) via SIM + SRG.
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm

# Add paths to load SIRA
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sira.sira_model import SIRAModel


def load_sira(checkpoint_path, clip_model_name="ViT-B/32", device="cuda"):
    """Load SIRA model from checkpoint."""
    sira_model, preprocess = SIRAModel.from_clip(clip_model_name, device)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)

    # Load SIRA submodules
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
    else:
        print("  Using standard frozen CLIP backbone.")

    sira_model = sira_model.to(device).eval()
    print(f"Loaded SIRA from {checkpoint_path}")
    return sira_model, preprocess


def get_tokenizer(clip_model_name="ViT-B/32"):
    """L2: Get the tokenizer matching the CLIP backend."""
    try:
        import clip
        return clip.tokenize
    except ImportError:
        import open_clip
        return open_clip.get_tokenizer(clip_model_name.replace("/", "-"))


@torch.no_grad()
def evaluate_winoground(sira_model, preprocess, tokenizer, data_dir, device="cuda"):
    jsonl_path = os.path.join(data_dir, "examples.jsonl")
    images_dir = os.path.join(data_dir, "images")

    with open(jsonl_path, "r") as f:
        examples = [json.loads(line) for line in f]

    # Counters
    clip_text, clip_img, clip_group = 0, 0, 0
    sira_text, sira_img, sira_group = 0, 0, 0

    for example in tqdm(examples, desc="Evaluating Winoground"):
        # Load images and texts
        img0_path = os.path.join(images_dir, example["image_0"] + ".png")
        img1_path = os.path.join(images_dir, example["image_1"] + ".png")

        img0 = preprocess(Image.open(img0_path).convert("RGB")).unsqueeze(0).to(device)
        img1 = preprocess(Image.open(img1_path).convert("RGB")).unsqueeze(0).to(device)

        txt0 = tokenizer([example["caption_0"]]).to(device)
        txt1 = tokenizer([example["caption_1"]]).to(device)

        # 1. Base Shared Embeddings (L2-normalized)
        v0 = sira_model._encode_image(img0)
        v1 = sira_model._encode_image(img1)
        t0 = sira_model._encode_text(txt0)
        t1 = sira_model._encode_text(txt1)

        # ── CLIP Baseline: Cosine Similarity ──
        c_00 = (v0 @ t0.T).item()
        c_01 = (v0 @ t1.T).item()
        c_10 = (v1 @ t0.T).item()
        c_11 = (v1 @ t1.T).item()

        c_t = (c_00 > c_01) and (c_11 > c_10)
        c_i = (c_00 > c_10) and (c_11 > c_01)
        if c_t: clip_text += 1
        if c_i: clip_img += 1
        if c_t and c_i: clip_group += 1

        # ── SIRA Gated Interaction ──
        # 00
        s_00 = sira_model.sim(v0, t0)
        v_final_00, t_final_00 = sira_model.srg(v0, t0, s_00)
        a_00 = (v_final_00 * t_final_00).sum(dim=-1).item()
        # 01
        s_01 = sira_model.sim(v0, t1)
        v_final_01, t_final_01 = sira_model.srg(v0, t1, s_01)
        a_01 = (v_final_01 * t_final_01).sum(dim=-1).item()
        # 10
        s_10 = sira_model.sim(v1, t0)
        v_final_10, t_final_10 = sira_model.srg(v1, t0, s_10)
        a_10 = (v_final_10 * t_final_10).sum(dim=-1).item()
        # 11
        s_11 = sira_model.sim(v1, t1)
        v_final_11, t_final_11 = sira_model.srg(v1, t1, s_11)
        a_11 = (v_final_11 * t_final_11).sum(dim=-1).item()

        a_t = (a_00 > a_01) and (a_11 > a_10)
        a_i = (a_00 > a_10) and (a_11 > a_01)
        if a_t: sira_text += 1
        if a_i: sira_img += 1
        if a_t and a_i: sira_group += 1

    total = len(examples)

    print("\n" + "="*72)
    print("  Winoground Results (Zero-Shot)")
    print("="*72)
    print(f"{'Metric':<15} {'CLIP Baseline':>15} {'SIRA Gated Sim':>20}")
    print("-" * 72)
    print(f"{'Text Score':<15} {clip_text/total*100:>14.1f}% {sira_text/total*100:>19.1f}%")
    print(f"{'Image Score':<15} {clip_img/total*100:>14.1f}% {sira_img/total*100:>19.1f}%")
    print(f"{'Group Score':<15} {clip_group/total*100:>14.1f}% {sira_group/total*100:>19.1f}%")
    print("="*72)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/sira_noprj/sira_best.pt")
    parser.add_argument("--clip-model", default="ViT-B/32")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--data-dir", default="/home/otw/chisphung/Synergistic/winoground/data")
    args = parser.parse_args()

    print("Loading SIRA model and processor...")
    sira_model, preprocess = load_sira(args.checkpoint, args.clip_model, args.device)

    tokenizer = get_tokenizer(args.clip_model)

    print("Loading Winoground Dataset locally...")
    evaluate_winoground(
        sira_model,
        preprocess,
        tokenizer,
        args.data_dir,
        args.device,
    )


if __name__ == "__main__":
    main()
