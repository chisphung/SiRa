"""
Training pipeline for SIRA-CIRR.

Trains SIRA modules (SIM + SRG + TextResidualCombiner) on the CIRR dataset
using pre-computed CLS-level CLIP embeddings. No CLIP model needed
during training — only the lightweight trainable modules run.

CIRR task: Given (reference_image, relative_caption), retrieve target image.

Phased loss introduction:
    Phase 1 (epoch 0–4):   L_retrieval only — warm-up
    Phase 2 (epoch 5–9):   + L_syn_align — teach synergy the modification direction
    Phase 3 (epoch 10+):   + L_orth — prevent synergy collapse

Usage:
    # Step 1: Pre-compute CLS embeddings (run ONCE)
    python cirr/precompute_cirr_cls.py --split train
    python cirr/precompute_cirr_cls.py --split val

    # Step 2: Train
    python train_sira_cirr.py --cache-dir ./cache/cirr_cls --epochs 20
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import time
import json
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.cuda.amp import GradScaler, autocast

from sira.sira_cirr_model import SIRACIRRModel
from sira.losses_cirr import CIRRLoss
from cirr.cirr_dataset import PrecomputedCIRRCLSDataset


class SIRACIRRTrainer:
    """
    Training pipeline for SIRA-CIRR using pre-computed CLIP embeddings.

    No CLIP model loaded during training — only the trainable SIRA modules
    (SIM, SRG, TextResidualCombiner, CIRRLoss temperature).
    """

    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.device)
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # ---- Load precomputed embeddings ----
        train_cache = Path(config.cache_dir) / "train"
        val_cache = Path(config.cache_dir) / "val"

        self.train_dataset = PrecomputedCIRRCLSDataset(train_cache)

        # ---- Detect feature dimension ----
        self.d_model = self.train_dataset.v_ref.shape[-1]
        print(f"  Feature dim: {self.d_model}")

        # ---- Use actual CIRR val split (not random split from train) ----
        if val_cache.exists() and (val_cache / "v_ref.pt").exists():
            self.val_dataset = PrecomputedCIRRCLSDataset(val_cache)
            print(f"  Using actual CIRR val split for validation")
        else:
            # Fallback: split from train if val cache not available
            n_val = min(500, len(self.train_dataset) // 10)
            n_train = len(self.train_dataset) - n_val
            self.train_dataset, self.val_dataset = random_split(
                self.train_dataset, [n_train, n_val],
                generator=torch.Generator().manual_seed(42))
            print(f"  WARNING: val cache not found, splitting from train")

        self.train_loader = DataLoader(
            self.train_dataset, batch_size=config.batch_size,
            shuffle=True, num_workers=config.workers,
            pin_memory=True, drop_last=True)
        self.val_loader = DataLoader(
            self.val_dataset, batch_size=config.batch_size,
            shuffle=False, num_workers=config.workers,
            pin_memory=True)

        print(f"  Train: {len(self.train_dataset)} | Val: {len(self.val_dataset)}")

        # ---- Build model ----
        self.model = SIRACIRRModel(
            d_model=self.d_model,
            d_synergy=config.d_synergy,
            gate_rank=config.gate_rank,
            gate_init_bias=config.gate_init_bias,
            dropout=config.dropout,
        ).to(self.device)

        # ---- Build loss ----
        self.loss_fn = CIRRLoss(
            lambda_ret=config.lambda_ret,
            lambda_syn=config.lambda_syn,
            lambda_orth=config.lambda_orth,
            syn_align_epoch=config.syn_epoch,
            orth_epoch=config.orth_epoch,
        ).to(self.device)

        # ---- Optimizer ----
        all_params = (
            list(self.model.parameters()) +
            list(self.loss_fn.parameters())
        )
        self.optimizer = torch.optim.AdamW(
            all_params, lr=config.lr, weight_decay=config.weight_decay)

        # ---- Scheduler ----
        total_steps = len(self.train_loader) * config.epochs
        warmup_steps = len(self.train_loader) * config.warmup_epochs
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer, max_lr=config.lr,
            total_steps=total_steps,
            pct_start=warmup_steps / total_steps if total_steps > 0 else 0.1,
            anneal_strategy="cos")

        # ---- AMP ----
        self.scaler = GradScaler(enabled=config.use_amp)

        # Logging
        self.train_log = []
        self.best_loss = float("inf")

    def train(self):
        """Run full training loop."""
        print(f"\n{'='*60}")
        print(f"SIRA-CIRR Training")
        print(f"  (Using pre-computed CLS-level CLIP embeddings)")
        print(f"{'='*60}")

        summary = self.model.get_param_summary()
        print(f"  SIM:            {summary['sim']:,} params")
        print(f"  SRG:            {summary['srg']:,} params")
        print(f"  Combiner:       {summary['combiner']:,} params")
        print(f"  Total trainable: {summary['total_trainable']:,} params")
        print(f"  Epochs: {self.config.epochs} | Batch: {self.config.batch_size}")
        print(f"  Loss phases: 1(ret) → 2(+syn@{self.config.syn_epoch}) "
              f"→ 3(+orth@{self.config.orth_epoch})")
        print(f"{'='*60}\n")

        for epoch in range(self.config.epochs):
            # Determine active phase
            phase = 1
            if epoch >= self.config.orth_epoch:
                phase = 3
            elif epoch >= self.config.syn_epoch:
                phase = 2

            print(f"\n--- Epoch {epoch+1}/{self.config.epochs} [Phase {phase}] ---")

            train_metrics = self._train_epoch(epoch)
            val_metrics = self._validate(epoch)

            log_entry = {
                "epoch": epoch + 1,
                "phase": phase,
                **{f"train_{k}": v for k, v in train_metrics.items()},
                **{f"val_{k}": v for k, v in val_metrics.items()},
                "lr": self.scheduler.get_last_lr()[0],
            }
            self.train_log.append(log_entry)

            print(f"  Train: loss={train_metrics['total']:.4f} "
                  f"(ret={train_metrics['l_ret']:.3f} "
                  f"syn={train_metrics['l_syn']:.3f} "
                  f"orth={train_metrics['l_orth']:.3f}) "
                  f"s_norm={train_metrics['synergy_norm']:.3f}")
            print(f"  Val:   loss={val_metrics['total']:.4f}")

            # Save best
            if val_metrics["total"] < self.best_loss:
                self.best_loss = val_metrics["total"]
                self._save_checkpoint(epoch, is_best=True)
                print(f"  ★ New best model (val_loss={self.best_loss:.4f})")

            # Periodic save
            if (epoch + 1) % 5 == 0:
                self._save_checkpoint(epoch)

        # Final save
        self._save_checkpoint(self.config.epochs - 1, is_final=True)
        self._save_log()
        print(f"\n{'='*60}")
        print(f"Training complete. Best val loss: {self.best_loss:.4f}")
        print(f"{'='*60}\n")

    def _train_epoch(self, epoch):
        """Train one epoch."""
        self.model.train()
        self.loss_fn.train()

        accum = {}
        n_batches = 0
        start = time.time()

        for step, (v_ref, t_cap, z_target) in enumerate(self.train_loader):
            v_ref = v_ref.to(self.device)
            t_cap = t_cap.to(self.device)
            z_target = z_target.to(self.device)

            self.optimizer.zero_grad()

            with autocast(enabled=self.config.use_amp):
                z_query, s = self.model(v_ref, t_cap)
                losses = self.loss_fn(
                    z_query, z_target, s, v_ref, t_cap,
                    srg=self.model.srg, epoch=epoch)

            self.scaler.scale(losses["total"]).backward()

            # Gradient clipping
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.get_trainable_params(), 1.0)

            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()

            # Accumulate metrics
            for k, v in losses.items():
                if isinstance(v, torch.Tensor):
                    v = v.item()
                accum[k] = accum.get(k, 0.0) + v
            n_batches += 1

            if (step + 1) % self.config.log_every == 0:
                elapsed = time.time() - start
                gate_stats = self._get_gate_stats(v_ref, t_cap)
                alpha = self.model.get_combiner_alpha()
                print(f"    Step {step+1}/{len(self.train_loader)} | "
                      f"loss={losses['total'].item():.4f} | "
                      f"s_norm={losses['synergy_norm'].item():.3f} | "
                      f"gate_v={gate_stats['gate_v_mean']:.3f} "
                      f"gate_t={gate_stats['gate_t_mean']:.3f} | "
                      f"α={alpha:.3f} | "
                      f"τ={losses['temperature'].item():.4f} | "
                      f"lr={self.scheduler.get_last_lr()[0]:.2e} | "
                      f"{elapsed:.0f}s")

        return {k: v / max(n_batches, 1) for k, v in accum.items()}

    @torch.no_grad()
    def _validate(self, epoch):
        """Validate on held-out set."""
        self.model.eval()
        self.loss_fn.eval()

        accum = {}
        n_batches = 0

        for v_ref, t_cap, z_target in self.val_loader:
            v_ref = v_ref.to(self.device)
            t_cap = t_cap.to(self.device)
            z_target = z_target.to(self.device)

            z_query, s = self.model(v_ref, t_cap)
            losses = self.loss_fn(
                z_query, z_target, s, v_ref, t_cap,
                srg=self.model.srg, epoch=epoch)

            for k, v in losses.items():
                if isinstance(v, torch.Tensor):
                    v = v.item()
                accum[k] = accum.get(k, 0.0) + v
            n_batches += 1

        return {k: v / max(n_batches, 1) for k, v in accum.items()}

    @torch.no_grad()
    def _get_gate_stats(self, v_ref, t_cap):
        """Get gate activation statistics."""
        s_raw = self.model.sim(v_ref, t_cap)
        s = torch.nn.functional.normalize(s_raw, p=2, dim=-1) * (self.model.d_synergy ** 0.5)
        return self.model.srg.get_gate_stats(v_ref, t_cap, s)

    def _save_checkpoint(self, epoch, is_best=False, is_final=False):
        state = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "loss_fn_state_dict": self.loss_fn.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_loss": self.best_loss,
            "config": vars(self.config),
            "d_model": self.d_model,
            "dataset": "cirr",
            "architecture": "sira_cirr_cls",
        }
        if is_best:
            path = self.output_dir / "sira_cirr_best.pt"
        elif is_final:
            path = self.output_dir / "sira_cirr_final.pt"
        else:
            path = self.output_dir / f"sira_cirr_epoch{epoch+1}.pt"
        torch.save(state, path)
        print(f"  Saved: {path}")

    def _save_log(self):
        log_path = self.output_dir / "train_log.json"
        with open(log_path, "w") as f:
            json.dump(self.train_log, f, indent=2)
        print(f"  Log saved: {log_path}")


# ======================================================================
# CLI
# ======================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Train SIRA-CIRR on CIRR Dataset (precomputed CLS)")

    # Data
    parser.add_argument("--cache-dir", default="./cache/cirr_cls",
                        help="Directory with precomputed CLS embeddings")

    # Architecture
    parser.add_argument("--d-synergy", type=int, default=64,
                        help="Synergistic bottleneck dimension")
    parser.add_argument("--gate-rank", type=int, default=16,
                        help="Low-rank gate factorization rank")
    parser.add_argument("--gate-init-bias", type=float, default=0.0,
                        help="Gate initialization bias")
    parser.add_argument("--dropout", type=float, default=0.1)

    # Training
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--warmup-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--use-amp", action="store_true", default=True)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=10)

    # Loss weights
    parser.add_argument("--lambda-ret", type=float, default=1.0,
                        help="Weight for L_retrieval")
    parser.add_argument("--lambda-syn", type=float, default=0.3,
                        help="Weight for L_syn_align")
    parser.add_argument("--lambda-orth", type=float, default=0.1,
                        help="Weight for L_orth")

    # Loss phase schedule
    parser.add_argument("--syn-epoch", type=int, default=5,
                        help="Epoch to start L_syn_align")
    parser.add_argument("--orth-epoch", type=int, default=10,
                        help="Epoch to start L_orth")

    # Output
    parser.add_argument("--output-dir", default="./checkpoints/sira_cirr")
    parser.add_argument("--device", default="cuda")

    args = parser.parse_args()
    if args.no_amp:
        args.use_amp = False

    trainer = SIRACIRRTrainer(args)
    trainer.train()


if __name__ == "__main__":
    main()
