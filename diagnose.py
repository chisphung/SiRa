#!/usr/bin/env python3
"""
SIRA gate-stuck diagnostic.

Run this from the repo root (so `sira/...` is importable):

    cd /home/otw/chisphung/Synergistic
    python diagnose_gate.py \
        --checkpoint checkpoints/sira_noprj/sira_best.pt \
        --winoground-data /home/otw/chisphung/Synergistic/winoground/data

Output is printed to stdout. Paste it back to the assistant.

Three sections:
    A. Static checkpoint inspection  (no data needed)
    B. Live SRG behavior on Winoground inputs
    C. Scorer head noise check

This script does NOT modify any code or any checkpoint.
"""
import argparse
import json
import math
import os
import statistics
import sys
from collections import OrderedDict

import torch
import torch.nn.functional as F

# Make the repo importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from sira.sira_model import SIRAModel
except Exception as e:
    print(f"FATAL: cannot import sira.sira_model: {e}")
    print(f"Run this script from the repo root (where the `sira/` package lives).")
    sys.exit(1)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _stat(t: torch.Tensor):
    """Quick stats on a tensor."""
    t = t.detach().float()
    return {
        "shape": tuple(t.shape),
        "mean": t.mean().item(),
        "std": t.std(unbiased=False).item() if t.numel() > 1 else 0.0,
        "min": t.min().item(),
        "max": t.max().item(),
        "abs_mean": t.abs().mean().item(),
        "frobenius_norm": t.norm().item(),
    }


def _print_stat(name: str, s: dict, indent: int = 2):
    pad = " " * indent
    print(f"{pad}{name}: shape={s['shape']}")
    print(f"{pad}  mean={s['mean']:.5f}  std={s['std']:.5f}  "
          f"min={s['min']:.5f}  max={s['max']:.5f}  "
          f"abs_mean={s['abs_mean']:.5f}  ||F||={s['frobenius_norm']:.4f}")


# ----------------------------------------------------------------------
# A. Static checkpoint inspection
# ----------------------------------------------------------------------
def static_inspect(checkpoint_path: str, device: str = "cpu"):
    print("=" * 90)
    print(" A. STATIC CHECKPOINT INSPECTION")
    print("=" * 90)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    print(f"\nCheckpoint keys: {list(ckpt.keys())}")

    epoch = ckpt.get("epoch", "?")
    print(f"Epoch saved: {epoch}")

    # -------- SRG --------
    print("\n--- SRG ---")
    srg = ckpt.get("srg_state_dict", {})
    if not srg:
        print("  (no srg_state_dict in checkpoint!)")
    else:
        # gate biases — init was -2.0
        for k in ["gate_bias_v", "gate_bias_t"]:
            if k in srg:
                t = srg[k]
                s = _stat(t)
                _print_stat(k, s)
                # how far has it moved from init -2.0
                drift = (t - (-2.0)).abs()
                print(f"    drift_from_-2.0: mean={drift.mean():.5f}  "
                      f"max={drift.max():.5f}  "
                      f"frac_within_0.1={(drift < 0.1).float().mean():.3f}  "
                      f"frac_within_0.5={(drift < 0.5).float().mean():.3f}")

        # gate down/up weights — init was up=0, down=xavier
        for branch in ["gate_v", "gate_t"]:
            for sub in ["down.weight", "up.weight", "up.bias"]:
                key = f"{branch}.{sub}"
                if key in srg:
                    s = _stat(srg[key])
                    _print_stat(key, s)

        # proj_v / proj_t — init was xavier_uniform(gain=0.1), bias=0
        for branch in ["proj_v", "proj_t"]:
            for sub in ["weight", "bias"]:
                key = f"{branch}.{sub}"
                if key in srg:
                    s = _stat(srg[key])
                    _print_stat(key, s)

    # -------- SIM --------
    print("\n--- SIM ---")
    sim = ckpt.get("sim_state_dict", {})
    if "interaction_scale" in sim:
        v = sim["interaction_scale"]
        print(f"  interaction_scale: {v.item():.5f}  (init=1.0)")
    if "interact_bias" in sim:
        s = _stat(sim["interact_bias"])
        _print_stat("interact_bias", s)
    for k in ["W_v.weight", "W_t.weight", "W_vt.weight",
              "proj_v_interact.weight", "proj_t_interact.weight"]:
        if k in sim:
            s = _stat(sim[k])
            _print_stat(k, s)

    # -------- Loss-fn temperatures & projections --------
    print("\n--- Loss-fn ---")
    lf = ckpt.get("loss_fn_state_dict", {})
    for k in ["log_temp", "log_temp_syn"]:
        if k in lf:
            v = lf[k].item()
            print(f"  {k}: log={v:.4f}  temp={math.exp(v):.5f}  (init temp={'0.07' if k=='log_temp' else '0.10'})")
    for k in ["proj_v.weight", "proj_t.weight"]:
        if k in lf:
            s = _stat(lf[k])
            _print_stat(f"loss_fn.{k}", s)

    # -------- Scorer head --------
    print("\n--- Scorer head ---")
    sc = ckpt.get("scorer_state_dict", {})
    if not sc:
        print("  (no scorer_state_dict in checkpoint)")
    else:
        for k, v in sc.items():
            s = _stat(v)
            _print_stat(k, s)

    # -------- Compare scorer to fresh-init scorer --------
    # Build a fresh SIRAModel just to grab the init values
    print("\n--- Fresh-init reference (to compare scorer drift) ---")
    try:
        import clip as _clip
        fresh_model = SIRAModel.from_clip("ViT-B/32", device=device)[0]
        if hasattr(fresh_model, "scorer"):
            fresh_sc = fresh_model.scorer.state_dict()
            max_drift_keys = []
            for k in sc:
                if k in fresh_sc:
                    fresh_t = fresh_sc[k].detach().cpu().float()
                    trained_t = sc[k].detach().cpu().float()
                    if fresh_t.shape == trained_t.shape:
                        drift = (trained_t - fresh_t).abs().mean().item()
                        max_drift_keys.append((k, drift, fresh_t.abs().mean().item()))
            for k, drift, init_mag in max_drift_keys:
                ratio = drift / (init_mag + 1e-9)
                print(f"  scorer.{k}: mean|trained-init|={drift:.6f}  "
                      f"vs init|mag|={init_mag:.6f}  drift/init={ratio:.4f}")
        else:
            print("  scorer head removed from current model; drift check skipped.")
    except Exception as e:
        print(f"  (skipped fresh-init comparison: {e})")


# ----------------------------------------------------------------------
# B. Live SRG behavior on Winoground inputs
# ----------------------------------------------------------------------
@torch.no_grad()
def live_inspect(checkpoint_path: str, winoground_data: str,
                 clip_model_name: str = "ViT-B/32",
                 device: str = "cuda",
                 max_examples: int = 400):
    print()
    print("=" * 90)
    print(" B. LIVE SRG BEHAVIOR ON WINOGROUND INPUTS")
    print("=" * 90)

    try:
        import clip
    except ImportError:
        print("  Need openai-clip installed (`pip install git+https://github.com/openai/CLIP.git`)")
        return

    from PIL import Image

    sira_model, preprocess = SIRAModel.from_clip(clip_model_name, device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    sira_model.sim.load_state_dict(ckpt["sim_state_dict"])
    if "srg_state_dict" in ckpt:
        incompatible = sira_model.srg.load_state_dict(ckpt["srg_state_dict"], strict=False)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            print(f"  SRG load (strict=False): missing={incompatible.missing_keys}, "
                  f"unexpected={incompatible.unexpected_keys}")
    if "loss_fn_state_dict" in ckpt:
        sira_model.loss_fn.load_state_dict(ckpt["loss_fn_state_dict"], strict=False)
    if "scorer_state_dict" in ckpt and hasattr(sira_model, "scorer"):
        sira_model.scorer.load_state_dict(ckpt["scorer_state_dict"])
    if "clip_state_dict" in ckpt:
        sira_model.clip.load_state_dict(ckpt["clip_state_dict"])
    sira_model.eval()

    jsonl = os.path.join(winoground_data, "examples.jsonl")
    images_dir = os.path.join(winoground_data, "images")
    if not os.path.exists(jsonl):
        print(f"  examples.jsonl not found at {jsonl}; skipping live section.")
        return

    with open(jsonl) as f:
        examples = [json.loads(line) for line in f]
    examples = examples[:max_examples]
    print(f"  Using {len(examples)} Winoground examples")

    # Collect features
    g_v_all, g_t_all = [], []
    s_norm_all = []
    res_v_ratio_all, res_t_ratio_all = [], []
    scorer_out_all = []
    v_final_minus_v_shared_cos = []

    # Reference: what the gate WOULD output at fresh init (sigmoid(-2.0))
    fresh_gate_baseline = 1.0 / (1.0 + math.exp(2.0))   # ≈ 0.1192

    for ex in examples:
        img0 = preprocess(Image.open(os.path.join(images_dir, ex["image_0"] + ".png")).convert("RGB")).unsqueeze(0).to(device)
        img1 = preprocess(Image.open(os.path.join(images_dir, ex["image_1"] + ".png")).convert("RGB")).unsqueeze(0).to(device)
        txt0 = clip.tokenize([ex["caption_0"]]).to(device)
        txt1 = clip.tokenize([ex["caption_1"]]).to(device)

        # 4 pairs per example
        v0 = sira_model._encode_image(img0)
        v1 = sira_model._encode_image(img1)
        t0 = sira_model._encode_text(txt0)
        t1 = sira_model._encode_text(txt1)

        for v_s, t_s in [(v0, t0), (v0, t1), (v1, t0), (v1, t1)]:
            s = sira_model.sim(v_s, t_s)
            s_v = sira_model.srg.proj_v(s)
            s_t = sira_model.srg.proj_t(s)
            gi_v = torch.cat([v_s, s_v], dim=-1)
            gi_t = torch.cat([t_s, s_t], dim=-1)
            g_v = torch.sigmoid(sira_model.srg.gate_v(gi_v) + sira_model.srg.gate_bias_v)
            g_t = torch.sigmoid(sira_model.srg.gate_t(gi_t) + sira_model.srg.gate_bias_t)
            g_v_all.append(g_v.detach().cpu().flatten())
            g_t_all.append(g_t.detach().cpu().flatten())

            # residual ratios
            residual_v = sira_model.srg._clamp_residual(g_v * s_v, v_s)
            residual_t = sira_model.srg._clamp_residual(g_t * s_t, t_s)
            res_v_ratio_all.append((residual_v.norm(dim=-1) / v_s.norm(dim=-1)).item())
            res_t_ratio_all.append((residual_t.norm(dim=-1) / t_s.norm(dim=-1)).item())

            # synergy norm
            s_norm_all.append(s.norm(dim=-1).item())

            # how different is v_final from v_shared in *angle*
            v_final = F.normalize(v_s + residual_v, dim=-1)
            v_final_minus_v_shared_cos.append(F.cosine_similarity(v_final, v_s, dim=-1).item())

            # scorer output
            if hasattr(sira_model, "scorer"):
                scorer_out_all.append(sira_model.scorer(s).item())

    g_v_all = torch.cat(g_v_all)
    g_t_all = torch.cat(g_t_all)

    print(f"\n  gate_v (per-dim, sigmoid output)  N={g_v_all.numel():,}")
    print(f"    mean={g_v_all.mean():.4f}  std={g_v_all.std():.4f}  "
          f"min={g_v_all.min():.4f}  max={g_v_all.max():.4f}")
    print(f"    frac>0.5={(g_v_all > 0.5).float().mean():.4f}  "
          f"frac>0.3={(g_v_all > 0.3).float().mean():.4f}  "
          f"frac<0.15={(g_v_all < 0.15).float().mean():.4f}")
    print(f"    delta_from_init({fresh_gate_baseline:.4f}): "
          f"mean={(g_v_all - fresh_gate_baseline).mean():.5f}  "
          f"abs_mean={(g_v_all - fresh_gate_baseline).abs().mean():.5f}")

    print(f"\n  gate_t (per-dim, sigmoid output)  N={g_t_all.numel():,}")
    print(f"    mean={g_t_all.mean():.4f}  std={g_t_all.std():.4f}  "
          f"min={g_t_all.min():.4f}  max={g_t_all.max():.4f}")
    print(f"    frac>0.5={(g_t_all > 0.5).float().mean():.4f}  "
          f"frac>0.3={(g_t_all > 0.3).float().mean():.4f}  "
          f"frac<0.15={(g_t_all < 0.15).float().mean():.4f}")
    print(f"    delta_from_init({fresh_gate_baseline:.4f}): "
          f"mean={(g_t_all - fresh_gate_baseline).mean():.5f}  "
          f"abs_mean={(g_t_all - fresh_gate_baseline).abs().mean():.5f}")

    print(f"\n  synergy norm  ||s||")
    print(f"    mean={statistics.mean(s_norm_all):.4f}  "
          f"std={statistics.stdev(s_norm_all):.4f}  "
          f"min={min(s_norm_all):.4f}  max={max(s_norm_all):.4f}")

    print(f"\n  residual_v / v_shared  norm ratio")
    print(f"    mean={statistics.mean(res_v_ratio_all):.4f}  "
          f"std={statistics.stdev(res_v_ratio_all):.4f}  "
          f"min={min(res_v_ratio_all):.4f}  max={max(res_v_ratio_all):.4f}  "
          f"frac_at_clamp(>=0.299)={sum(1 for r in res_v_ratio_all if r >= 0.299)/len(res_v_ratio_all):.4f}")

    print(f"\n  residual_t / t_shared  norm ratio")
    print(f"    mean={statistics.mean(res_t_ratio_all):.4f}  "
          f"std={statistics.stdev(res_t_ratio_all):.4f}  "
          f"min={min(res_t_ratio_all):.4f}  max={max(res_t_ratio_all):.4f}  "
          f"frac_at_clamp(>=0.299)={sum(1 for r in res_t_ratio_all if r >= 0.299)/len(res_t_ratio_all):.4f}")

    print(f"\n  cos(v_final, v_shared)  (1.0 = SRG had no effect on angle)")
    print(f"    mean={statistics.mean(v_final_minus_v_shared_cos):.5f}  "
          f"std={statistics.stdev(v_final_minus_v_shared_cos):.5f}  "
          f"min={min(v_final_minus_v_shared_cos):.5f}")

    if hasattr(sira_model, "scorer"):
        print(f"\n  C. SCORER HEAD output on synergy vectors  (untrained-or-not check)")
        print(f"    mean={statistics.mean(scorer_out_all):.5f}  "
              f"std={statistics.stdev(scorer_out_all):.5f}  "
              f"min={min(scorer_out_all):.5f}  max={max(scorer_out_all):.5f}")
        print(f"    (If scorer is at-init, mean ≈ small constant and std reflects only")
        print(f"     the noise from variation in s. If trained, you'd expect a clear")
        print(f"     monotonic relationship between |s| and scorer(s) and a wider range.)")

        # Pearson correlation between ||s|| and scorer(s)
        n = len(s_norm_all)
        mx = sum(s_norm_all) / n
        my = sum(scorer_out_all) / n
        num = sum((s_norm_all[i] - mx) * (scorer_out_all[i] - my) for i in range(n))
        dx = math.sqrt(sum((s_norm_all[i] - mx) ** 2 for i in range(n)))
        dy = math.sqrt(sum((scorer_out_all[i] - my) ** 2 for i in range(n)))
        rho = num / (dx * dy + 1e-12)
        print(f"    Pearson(||s||, scorer(s)) = {rho:.4f}   "
              f"(near 0 ⇒ scorer is not using s meaningfully)")
    else:
        print("\n  C. SCORER HEAD skipped (removed from model)")


# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/sira_noprj/sira_best.pt")
    ap.add_argument("--winoground-data",
                    default="/home/otw/chisphung/Synergistic/winoground/data")
    ap.add_argument("--clip-model", default="ViT-B/32")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--max-examples", type=int, default=400)
    ap.add_argument("--skip-live", action="store_true",
                    help="Skip section B (no GPU / no Winoground data needed)")
    args = ap.parse_args()

    static_inspect(args.checkpoint, device="cpu")

    if not args.skip_live:
        live_inspect(args.checkpoint, args.winoground_data,
                     clip_model_name=args.clip_model,
                     device=args.device,
                     max_examples=args.max_examples)
    else:
        print("\n[skipping live section per --skip-live]")


if __name__ == "__main__":
    main()
