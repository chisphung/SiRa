"""
CIRR (Composed Image Retrieval on Real-life images) Dataset — CLS-Level.

Adapted for SIRA framework: uses CLS-level CLIP embeddings (512-dim)
instead of token-level patch/text features.

CIRR task mapping to SIRA:
    Reference Image (V):     v_ref — CLIP image CLS embedding (512)
    Relative Caption (L):    t_cap — CLIP text CLS embedding (512)
    Target Image (Y):        z_target — CLIP image CLS embedding (512)

    SIM(v_ref, t_cap) → s   — synergy of "what to change" in context of "this image"
    SRG(v_ref, t_cap, s) → v_final, t_final
    z_query = combine(v_final, t_final)
    retrieve: argmax sim(z_query, z_target_pool)
"""

import json
from pathlib import Path
from PIL import Image

# Disable PIL decompression bomb warning for large CIRR images
# (some NLVR2 source images exceed the 89M pixel default limit)
Image.MAX_IMAGE_PIXELS = None

import torch
from torch.utils.data import Dataset


class CIRRCLSDataset(Dataset):
    """
    CIRR dataset returning raw images + text for online CLIP encoding.

    Each sample returns:
        - reference_image: preprocessed reference image tensor
        - text_tokens:     tokenized relative caption (77,) int tensor
        - target_image:    preprocessed target image tensor

    Args:
        cirr_root:   Path to CIRR dataset root (e.g., /home/otw/chiennhm/data/CIRR)
        split:       'train' or 'val'
        preprocess:  CLIP image preprocessing function
        tokenizer:   CLIP tokenizer function (e.g., clip.tokenize)
    """

    def __init__(self, cirr_root, split="train", preprocess=None, tokenizer=None):
        self.preprocess = preprocess
        self.tokenizer = tokenizer
        self.split = split

        cirr_root = Path(cirr_root)
        annotation_dir = cirr_root / "cirr"

        # Load captions (query-target pairs)
        cap_path = annotation_dir / "captions" / f"cap.rc2.{split}.json"
        if not cap_path.exists():
            raise FileNotFoundError(f"CIRR captions not found: {cap_path}")

        with open(cap_path, "r") as f:
            self.annotations = json.load(f)

        # Load image splits (img_id → relative path)
        split_path = annotation_dir / "image_splits" / f"split.rc2.{split}.json"
        if not split_path.exists():
            raise FileNotFoundError(f"CIRR image splits not found: {split_path}")

        with open(split_path, "r") as f:
            self.img_paths = json.load(f)

        self.cirr_root = cirr_root

        # Filter out entries where images are missing from the split file
        valid_entries = []
        for entry in self.annotations:
            ref_id = entry["reference"]
            target_id = entry["target_hard"]
            if ref_id in self.img_paths and target_id in self.img_paths:
                valid_entries.append(entry)

        self.entries = valid_entries
        print(f"  CIRRCLSDataset [{split}]: {len(self.entries)} entries loaded "
              f"(from {len(self.annotations)} total)")

    def _get_image_path(self, img_id):
        """Resolve image ID to absolute path."""
        rel_path = self.img_paths[img_id]
        if rel_path.startswith("./"):
            rel_path = rel_path[2:]
        return self.cirr_root / rel_path

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]

        # Load reference image
        ref_path = self._get_image_path(entry["reference"])
        ref_image = self.preprocess(Image.open(ref_path).convert("RGB"))

        # Load target image
        target_path = self._get_image_path(entry["target_hard"])
        target_image = self.preprocess(Image.open(target_path).convert("RGB"))

        # Tokenize relative caption
        caption = entry["caption"]
        try:
            text_tokens = self.tokenizer(caption, truncate=True).squeeze(0)
        except TypeError:
            text_tokens = self.tokenizer(caption).squeeze(0)

        return ref_image, text_tokens, target_image


class CIRRCLSTestDataset(CIRRCLSDataset):
    """
    Test-time dataset that also returns metadata for evaluation.

    Returns:
        ref_image, text_tokens, target_image, metadata_dict
    """

    def __getitem__(self, idx):
        entry = self.entries[idx]

        ref_path = self._get_image_path(entry["reference"])
        ref_image = self.preprocess(Image.open(ref_path).convert("RGB"))

        target_path = self._get_image_path(entry["target_hard"])
        target_image = self.preprocess(Image.open(target_path).convert("RGB"))

        caption = entry["caption"]
        try:
            text_tokens = self.tokenizer(caption, truncate=True).squeeze(0)
        except TypeError:
            text_tokens = self.tokenizer(caption).squeeze(0)

        metadata = {
            "pairid": entry["pairid"],
            "reference": entry["reference"],
            "target_hard": entry["target_hard"],
            "caption": caption,
            "img_set_members": entry.get("img_set", {}).get("members", []),
        }

        return ref_image, text_tokens, target_image, metadata


class PrecomputedCIRRCLSDataset(Dataset):
    """
    Dataset that loads pre-computed CLS-level CLIP embeddings from disk.

    Created by precompute_cirr_cls.py. No CLIP forward pass needed
    during training — just loads cached tensors.

    Each sample returns:
        v_ref:     (512,)  — reference image CLIP CLS embedding (float32)
        t_cap:     (512,)  — relative caption CLIP text embedding (float32)
        z_target:  (512,)  — target image CLIP CLS embedding (float32)

    Args:
        cache_dir: Path to cached embeddings (e.g., ./cache/cirr_cls/train)
    """

    def __init__(self, cache_dir):
        cache_dir = Path(cache_dir)

        if not (cache_dir / "v_ref.pt").exists():
            raise FileNotFoundError(
                f"Cache not found at {cache_dir}. "
                f"Run cirr/precompute_cirr_cls.py first.")

        print(f"  Loading precomputed CIRR CLS embeddings from {cache_dir}...")

        # Load all tensors
        self.v_ref = torch.load(cache_dir / "v_ref.pt", weights_only=True)
        self.t_cap = torch.load(cache_dir / "t_cap.pt", weights_only=True)
        self.z_target = torch.load(cache_dir / "z_target.pt", weights_only=True)

        # Load metadata
        with open(cache_dir / "metadata.json", "r") as f:
            self.metadata = json.load(f)

        N = self.v_ref.shape[0]
        print(f"  PrecomputedCIRRCLSDataset: {N} samples loaded "
              f"(dtype={self.v_ref.dtype})")

    def __len__(self):
        return self.v_ref.shape[0]

    def __getitem__(self, idx):
        return (
            self.v_ref[idx].float(),      # (512,)
            self.t_cap[idx].float(),       # (512,)
            self.z_target[idx].float(),    # (512,)
        )


class PrecomputedCIRRCLSTestDataset(Dataset):
    """
    Precomputed dataset for evaluation that also loads per-sample metadata.

    Each sample returns:
        v_ref, t_cap, z_target, metadata_dict

    Metadata includes pairid, reference, target_hard, caption, img_set_members.
    """

    def __init__(self, cache_dir):
        cache_dir = Path(cache_dir)

        if not (cache_dir / "v_ref.pt").exists():
            raise FileNotFoundError(
                f"Cache not found at {cache_dir}. "
                f"Run cirr/precompute_cirr_cls.py first.")

        print(f"  Loading precomputed CIRR CLS embeddings from {cache_dir}...")

        self.v_ref = torch.load(cache_dir / "v_ref.pt", weights_only=True)
        self.t_cap = torch.load(cache_dir / "t_cap.pt", weights_only=True)
        self.z_target = torch.load(cache_dir / "z_target.pt", weights_only=True)

        # Load per-sample metadata
        with open(cache_dir / "entries.json", "r") as f:
            self.entries = json.load(f)

        with open(cache_dir / "metadata.json", "r") as f:
            self.metadata = json.load(f)

        N = self.v_ref.shape[0]
        print(f"  PrecomputedCIRRCLSTestDataset: {N} samples loaded")

    def __len__(self):
        return self.v_ref.shape[0]

    def __getitem__(self, idx):
        meta = self.entries[idx]
        return (
            self.v_ref[idx].float(),
            self.t_cap[idx].float(),
            self.z_target[idx].float(),
            meta,
        )
