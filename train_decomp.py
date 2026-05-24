"""
Training driver for the Decomp model (three-branch CLS-level composed retrieval).

Validation: full-gallery R@1 each epoch. Best-on-R@1 checkpointing.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from decomp.model   import DecompModel
from decomp.losses  import SymInfoNCE
from decomp.dataset import CIRRClsTriplet, CIRRClsQuery, query_collate, load_gallery


class Trainer:
    def __init__(
        self,
        model: DecompModel,
        retrieval_loss: SymInfoNCE,
        train_ds,
        val_split_dir: Path,
        *,
        lr: float = 1e-3,
        weight_decay: float = 0.05,
        batch_size: int = 256,
        num_epochs: int = 30,
        warmup_epochs: int = 2,
        output_dir: str = "./checkpoints/decomp",
        device: str = "cuda",
        num_workers: int = 4,
    ):
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.retrieval_loss = retrieval_loss.to(self.device)
        self.num_epochs = num_epochs

        self.train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True, drop_last=True,
        )

        # Validation: full-gallery
        self.val_query_ds = CIRRClsQuery(val_split_dir)
        self.val_query_loader = DataLoader(
            self.val_query_ds, batch_size=256, shuffle=False,
            num_workers=num_workers, collate_fn=query_collate, pin_memory=True,
        )
        gallery_emb, gallery_ids = load_gallery(val_split_dir)
        self.val_gallery_emb = F.normalize(gallery_emb, dim=-1).to(self.device)
        self.val_gallery_ids = gallery_ids
        self.val_id_to_idx = {iid: i for i, iid in enumerate(gallery_ids)}

        params = list(self.model.parameters()) + list(self.retrieval_loss.parameters())
        self.optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
        steps_per_epoch = max(len(self.train_loader), 1)
        pct_start = min(0.5, max(0.05, warmup_epochs / max(num_epochs, 1)))
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer, max_lr=lr,
            total_steps=steps_per_epoch * num_epochs,
            pct_start=pct_start, anneal_strategy="cos",
            div_factor=25.0, final_div_factor=1e3,
        )

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.best_r1 = -1.0
        self.history = []

    # --------------------------------------------------------------
    @torch.no_grad()
    def _eval_recall(self):
        self.model.eval()
        all_q = []; refs = []; tgts = []
        for batch in self.val_query_loader:
            v = batch["v"].to(self.device, non_blocking=True)
            t = batch["t"].to(self.device, non_blocking=True)
            q = self.model(v, t)
            all_q.append(q.cpu())
            refs.extend(batch["reference"]); tgts.extend(batch["target"])
        Q = torch.cat(all_q, dim=0).to(self.device)
        sims = []
        for i in range(0, Q.size(0), 512):
            sims.append((Q[i:i + 512] @ self.val_gallery_emb.t()).cpu())
        sims = torch.cat(sims, dim=0)
        for i, ref in enumerate(refs):
            if ref in self.val_id_to_idx:
                sims[i, self.val_id_to_idx[ref]] = -float("inf")
        sorted_idx = torch.argsort(sims, dim=-1, descending=True).numpy()
        gids_np = np.array(self.val_gallery_ids)
        Ks = (1, 5, 10)
        hits = {k: 0 for k in Ks}; ranks = []; n = 0
        for i, tgt in enumerate(tgts):
            if not tgt or tgt not in self.val_id_to_idx:
                continue
            n += 1
            ranked = gids_np[sorted_idx[i]]
            pos = np.where(ranked == tgt)[0]
            r = int(pos[0]) if len(pos) else len(ranked)
            ranks.append(r)
            for k in Ks:
                if r < k: hits[k] += 1
        return {
            "R@1":  hits[1]  / max(n, 1) * 100,
            "R@5":  hits[5]  / max(n, 1) * 100,
            "R@10": hits[10] / max(n, 1) * 100,
            "med_rank": float(np.median(ranks)) if ranks else float("inf"),
            "n_eval": n,
        }

    # --------------------------------------------------------------
    def train(self):
        for epoch in range(self.num_epochs):
            self.model.train()
            t0 = time.time()
            losses, accs = [], []
            for step, (v, t, z) in enumerate(self.train_loader):
                v = v.to(self.device, non_blocking=True)
                t = t.to(self.device, non_blocking=True)
                z = F.normalize(z.to(self.device, non_blocking=True), dim=-1)

                q = self.model(v, t)
                L, acc = self.retrieval_loss(q, z)

                self.optimizer.zero_grad(set_to_none=True)
                L.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                self.scheduler.step()

                losses.append(L.item()); accs.append(acc.item())

                if step % 20 == 0:
                    print(
                        f"  Ep {epoch + 1:2d}/{self.num_epochs} "
                        f"Step {step:4d}/{len(self.train_loader):4d} "
                        f"loss={L.item():.4f} acc={acc.item()*100:.1f}% "
                        f"αβγ=({self.model.alpha.item():.2f},"
                        f"{self.model.beta.item():.2f},"
                        f"{self.model.gamma.item():.2f}) "
                        f"lr={self.scheduler.get_last_lr()[0]:.2e}",
                        flush=True,
                    )

            dt = time.time() - t0
            mean_loss = float(np.mean(losses)); mean_acc = float(np.mean(accs)) * 100
            print(
                f"Epoch {epoch + 1} avg loss={mean_loss:.4f} acc={mean_acc:.2f}% "
                f"αβγ=({self.model.alpha.item():.3f},"
                f"{self.model.beta.item():.3f},"
                f"{self.model.gamma.item():.3f}) ({dt:.0f}s)",
                flush=True,
            )

            metrics = self._eval_recall()
            print(
                f"  Val full-gallery: R@1={metrics['R@1']:.2f}  "
                f"R@5={metrics['R@5']:.2f}  R@10={metrics['R@10']:.2f}  "
                f"med_rank={metrics['med_rank']:.0f}",
                flush=True,
            )
            self.history.append({
                "epoch": epoch + 1, "loss": mean_loss, "acc": mean_acc,
                "alpha": self.model.alpha.item(),
                "beta":  self.model.beta.item(),
                "gamma": self.model.gamma.item(),
                **metrics,
            })

            self._save("final", epoch + 1)
            if metrics["R@1"] > self.best_r1:
                self.best_r1 = metrics["R@1"]
                self._save("best", epoch + 1)
                print(f"  -> new best R@1 = {self.best_r1:.2f}%", flush=True)

            json.dump(self.history, open(self.output_dir / "train_log.json", "w"), indent=2)

        print(f"Done. Best R@1={self.best_r1:.2f}%")

    # --------------------------------------------------------------
    def _save(self, tag: str, epoch: int):
        torch.save({
            "epoch": epoch,
            "arch": "decomp",
            "model_state_dict": self.model.state_dict(),
            "retrieval_loss_state_dict": self.retrieval_loss.state_dict(),
            "config": {
                "d_in":  self.model.d_in,
                "d_out": self.model.d_out,
                "synergy_op": self.model.synergy_op,
            },
        }, self.output_dir / f"decomp_{tag}.pt")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache-dir",  default="./cache/cirr_cls")
    p.add_argument("--train-split", default="train")
    p.add_argument("--val-split",   default="val")
    p.add_argument("--output-dir", default="./checkpoints/decomp")
    p.add_argument("--epochs",  type=int, default=30)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr",      type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--warmup-epochs", type=int, default=2)
    p.add_argument("--workers", type=int, default=4)
    # Arch
    p.add_argument("--d-hidden", type=int, default=512)
    p.add_argument("--d-out",    type=int, default=512)
    p.add_argument("--dropout",  type=float, default=0.1)
    p.add_argument("--synergy-op", default="hadamard", choices=["hadamard", "lowrank_bilinear"])
    p.add_argument("--bilinear-rank", type=int, default=64)
    p.add_argument("--retrieval-temperature", type=float, default=0.07)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    cache_dir = Path(args.cache_dir).resolve()
    print("=" * 60)
    print(f"  Decomp training (CLS-level, three-branch additive)")
    print(f"  cache dir  : {cache_dir}")
    print(f"  output dir : {args.output_dir}")
    print(f"  synergy op : {args.synergy_op}")
    print(f"  device     : {args.device}")
    print("=" * 60)

    train_ds = CIRRClsTriplet(cache_dir / args.train_split)
    d_in = train_ds.v.size(-1)

    model = DecompModel(
        d_in=d_in, d_hidden=args.d_hidden, d_out=args.d_out,
        dropout=args.dropout,
        synergy_op=args.synergy_op, bilinear_rank=args.bilinear_rank,
    )
    print(f"  params: {model.get_param_count()}")

    retrieval_loss = SymInfoNCE(init_temperature=args.retrieval_temperature)

    trainer = Trainer(
        model, retrieval_loss, train_ds,
        val_split_dir=cache_dir / args.val_split,
        lr=args.lr, weight_decay=args.weight_decay,
        batch_size=args.batch_size, num_epochs=args.epochs,
        warmup_epochs=args.warmup_epochs,
        output_dir=args.output_dir, device=args.device,
        num_workers=args.workers,
    )
    trainer.train()


if __name__ == "__main__":
    main()
