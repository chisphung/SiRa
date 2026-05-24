"""Datasets for the CLS-level CIRR cache (cache/cirr_cls/).

Each split directory contains:
    v_ref.pt                (N, 512) fp16 — reference image CLS
    t_cap.pt                (N, 512) fp16 — caption CLS
    z_target.pt             (N, 512) fp16 — target image CLS
    entries.json            list of N dicts with reference, target_hard, caption, img_set_members
    gallery_<split>_embeddings.pt  (G, 512) fp16
    gallery_<split>_info.json      {img_ids: [...], img_paths: [...]}
"""

import json
from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import Dataset


def _load_fp32(path: Path) -> torch.Tensor:
    t = torch.load(path, weights_only=False)
    return t.to(torch.float32)


class CIRRClsTriplet(Dataset):
    """For training: returns (v_ref, t_cap, z_target)."""

    def __init__(self, split_dir: str | Path):
        split_dir = Path(split_dir)
        self.v = _load_fp32(split_dir / "v_ref.pt")
        self.t = _load_fp32(split_dir / "t_cap.pt")
        self.z = _load_fp32(split_dir / "z_target.pt")
        assert self.v.size(0) == self.t.size(0) == self.z.size(0), "Length mismatch"
        print(f"  CIRRClsTriplet[{split_dir.name}]: {self.v.size(0)} samples")

    def __len__(self) -> int:
        return self.v.size(0)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.v[i], self.t[i], self.z[i]


class CIRRClsQuery(Dataset):
    """For evaluation: returns (v_ref, t_cap, reference_id, target_id, group_members)."""

    def __init__(self, split_dir: str | Path):
        split_dir = Path(split_dir)
        self.v = _load_fp32(split_dir / "v_ref.pt")
        self.t = _load_fp32(split_dir / "t_cap.pt")
        self.entries = json.load(open(split_dir / "entries.json"))
        assert self.v.size(0) == len(self.entries), "v_ref/entries length mismatch"
        print(f"  CIRRClsQuery[{split_dir.name}]: {len(self.entries)} queries")

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, i: int) -> dict:
        e = self.entries[i]
        return {
            "v": self.v[i],
            "t": self.t[i],
            "reference":     e.get("reference", ""),
            "target":        e.get("target_hard", "") or "",
            "group_members": e.get("img_set_members", []) or [],
        }


def query_collate(batch):
    return {
        "v": torch.stack([b["v"] for b in batch], dim=0),
        "t": torch.stack([b["t"] for b in batch], dim=0),
        "reference":     [b["reference"]     for b in batch],
        "target":        [b["target"]        for b in batch],
        "group_members": [b["group_members"] for b in batch],
    }


def load_gallery(split_dir: str | Path) -> Tuple[torch.Tensor, list]:
    """Load (gallery_emb, gallery_ids) for a split."""
    split_dir = Path(split_dir)
    split_name = split_dir.name  # "val" or "train"
    emb_path = split_dir / f"gallery_{split_name}_embeddings.pt"
    info_path = split_dir / f"gallery_{split_name}_info.json"
    emb = torch.load(emb_path, weights_only=False).to(torch.float32)
    info = json.load(open(info_path))
    ids = info["img_ids"]
    assert emb.size(0) == len(ids), "Gallery emb / id count mismatch"
    return emb, ids
