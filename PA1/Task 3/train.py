"""
train.py
========
Training script for Task 3: Domain Generalization.

Usage
-----
    # Core models:
    python train.py --method dan_dg --data_root ... --splits_path ... --checkpoints_dir ... --results_dir ...
    python train.py --method sam    --data_root ... --splits_path ... --checkpoints_dir ... --results_dir ...

    # Controlled study (DAN-DG, vary lambda_dg):
    python train.py --method dan_dg --lambda_dg 0.1 ...
    python train.py --method dan_dg --lambda_dg 1.0 ...
    python train.py --method dan_dg --lambda_dg 10.0 ...

Notes
-----
* ERM reuses the Task 2 source_only checkpoint — no training required.
* Target (Sketch) labels are NEVER consulted during training.
* Checkpoint selection: mean source-val macro-F1 (seed 6304).
* No target loader is used in Task 3 (Domain Generalization).
"""
from __future__ import annotations

import warnings
warnings.filterwarnings("ignore", message=".*DataLoader.*worker.*")

import argparse
import json
import os
import random
import sys

import numpy as np
import torch
import yaml

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── local imports ─────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

from shared.pacs_protocol import (
    build_source_loaders,
    build_source_datasets,
    DomainBalancedIterator,
    get_or_create_splits,
    get_train_transform,
    get_eval_transform,
    download_pacs,
)
from models.backbone import BNFrozenResNet18
from models.classifier_head import ClassifierHead
from selection.source_validation import evaluate_source_domains


# ─── seed ────────────────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ─── config loading ──────────────────────────────────────────────────────────

def load_config(method: str) -> dict:
    base_path = "configs/base.yaml"
    method_path = f"configs/{method}.yaml"

    cfg = {}
    if os.path.exists(base_path):
        with open(base_path, "r") as f:
            cfg.update(yaml.safe_load(f) or {})

    if os.path.exists(method_path):
        with open(method_path, "r") as f:
            cfg.update(yaml.safe_load(f) or {})

    # Hardcode some defaults if base.yaml missing
    cfg.setdefault("epochs", 30)
    cfg.setdefault("lr", 1e-4)
    cfg.setdefault("weight_decay", 1e-4)
    cfg.setdefault("patience", 5)
    cfg.setdefault("num_classes", 7)
    cfg.setdefault("batch_size", 24)
    return cfg


# ─── method factory ──────────────────────────────────────────────────────────

def build_method(args, cfg, backbone, head, device):
    if cfg["method"] == "erm":
        return None
    elif cfg["method"] == "dan_dg":
        from methods.dan_dg import DANDGMethod
        l_dg = args.lambda_dg if args.lambda_dg is not None else cfg.get("lambda_dg", 1.0)
        return DANDGMethod(backbone, head, device, cfg["lr"], cfg["weight_decay"],
                           lambda_dg=l_dg,
                           kernel_muls=cfg.get("mmd_kernel_muls", [0.5, 1.0, 2.0]))
    elif cfg["method"] == "sam":
        from methods.sam import SAMMethod
        rho = args.rho if args.rho is not None else cfg.get("rho", 0.05)
        return SAMMethod(backbone, head, device, cfg["lr"], cfg["weight_decay"], rho=rho)
    else:
        raise ValueError(f"Unknown method {cfg['method']}")


# ─── training curves ─────────────────────────────────────────────────────────

def save_training_curves(epoch_losses, val_history, out_dir, run_name):
    """Save loss curves as PNG and JSON."""
    os.makedirs(out_dir, exist_ok=True)

    # JSON
    curve_data = {"losses": epoch_losses, "val_macro_f1": val_history}
    json_path = os.path.join(out_dir, f"{run_name}_curves.json")
    with open(json_path, "w") as f:
        json.dump(curve_data, f, indent=2)
        f.flush()
        os.fsync(f.fileno())

    # Plot
    n_epochs = len(val_history)
    epochs = list(range(1, n_epochs + 1))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for key, values in epoch_losses.items():
        if values:
            axes[0].plot(epochs[:len(values)], values, label=key)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title(f"{run_name} – Training Losses")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, val_history, "o-", color="steelblue", label="Mean src val macro-F1")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Macro-F1")
    axes[1].set_title(f"{run_name} – Source Validation Macro-F1")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    png_path = os.path.join(out_dir, f"{run_name}_curves.png")
    plt.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Force flush the PNG
    with open(png_path, "rb") as f:
        os.fsync(f.fileno())
    print(f"  Training curves saved → {json_path}, {png_path}")
    sys.stdout.flush()


# ─── pretty print source val ─────────────────────────────────────────────────

def print_source_val_table(results):
    """Pretty-print source validation results."""
    header = f"{'Domain':<18} {'Accuracy':>10} {'Macro-F1':>10}"
    print(header)
    print("-" * len(header))
    for domain, r in results.items():
        if domain in ("mean", "worst"):
            continue
        print(f"  {domain:<16} {r['accuracy']:>10.4f} {r['macro_f1']:>10.4f}")
    m = results["mean"]
    print("-" * len(header))
    print(f"  {'Mean':<16} {m['accuracy']:>10.4f} {m['macro_f1']:>10.4f}")
    sys.stdout.flush()


# ─── main training loop ──────────────────────────────────────────────────────

def train(args) -> None:
    # ── Seed ─────────────────────────────────────────────────────────────
    set_seed(6304)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] Method: {args.method}  |  Device: {device}")
    sys.stdout.flush()

    # ── Config ───────────────────────────────────────────────────────────
    cfg = load_config(args.method)
    cfg["checkpoints_dir"] = args.checkpoints_dir
    cfg["results_dir"] = args.results_dir

    os.makedirs(cfg["checkpoints_dir"], exist_ok=True)
    os.makedirs(cfg["results_dir"], exist_ok=True)

    if args.method == "erm":
        print("[train] ERM method: reusing Task 2 checkpoint. No training required.")
        sys.stdout.flush()
        return

    # ── Run name ─────────────────────────────────────────────────────────
    run_name = args.method
    if args.method == "dan_dg" and args.lambda_dg is not None:
        run_name = f"dan_dg_lambda_dg{args.lambda_dg}"
    elif args.method == "sam" and args.rho is not None:
        run_name = f"sam_rho{args.rho}"
    print(f"[train] Run name: {run_name}")
    sys.stdout.flush()

    # ── Data ─────────────────────────────────────────────────────────────
    pacs_root = download_pacs(args.data_root)
    splits = get_or_create_splits(args.splits_path, pacs_root)

    train_tf = get_train_transform()
    eval_tf = get_eval_transform()
    source_datasets = build_source_datasets(pacs_root, splits, train_tf, eval_tf)
    source_loaders = build_source_loaders(source_datasets, batch_size_per_source=8, num_workers=4)

    train_loaders = {d: source_loaders[d]["train"] for d in source_loaders}
    val_loaders = {d: source_loaders[d]["val"] for d in source_loaders}

    # ── Model ────────────────────────────────────────────────────────────
    backbone = BNFrozenResNet18()
    head = ClassifierHead(feat_dim=512, num_classes=cfg["num_classes"])
    method = build_method(args, cfg, backbone, head, device)

    # ── Warm-start from ERM checkpoint if provided ────────────────────────

    if args.method in ("dan_dg", "sam") and args.erm_ckpt:
        if os.path.isfile(args.erm_ckpt):
            erm_ckpt = torch.load(args.erm_ckpt, map_location=device)
            sd = erm_ckpt["state_dict"]
            method.backbone.load_state_dict(sd["backbone"])
            method.head.load_state_dict(sd["head"])
            print(f"[train] Warm-started backbone+head from ERM: {args.erm_ckpt}")
        else:
            print(f"[train] WARNING: --erm_ckpt path not found: {args.erm_ckpt}")
            print(f"         Training will proceed from ImageNet init (risk of collapse).")

    # ── Training state ───────────────────────────────────────────────────
    steps_per_epoch = min(len(ldr) for ldr in train_loaders.values())
    total_steps = cfg["epochs"] * steps_per_epoch
    print(f"[train] Steps/epoch: {steps_per_epoch}  |  Total steps: {total_steps}")
    sys.stdout.flush()

    best_val_f1 = -1.0
    patience_cnt = 0
    epoch_losses_history = {}
    val_f1_history = []

    # ── Training loop ────────────────────────────────────────────────────
    for epoch in range(cfg["epochs"]):
        
        # ── Lambda warmup: linearly scale alignment weight for first 5 epochs ──
        WARMUP_EPOCHS = 5
        if hasattr(method, 'lambda_dg'):
            warmup_scale = min(1.0, (epoch + 1) / WARMUP_EPOCHS)
            # Use command-line arg if provided, otherwise fallback to yaml config
            target_lambda = args.lambda_dg if args.lambda_dg is not None else cfg.get("lambda_dg", 1.0)
            method.lambda_dg = warmup_scale * target_lambda
            
        domain_iter_gen = iter(DomainBalancedIterator(train_loaders, None, steps_per_epoch=steps_per_epoch))
        step_losses = {}

        for local_step, batch in enumerate(domain_iter_gen):
            if isinstance(batch, tuple) and len(batch) == 2 and isinstance(batch[1], tuple):
                src_batches, tgt_batch = batch
            else:
                src_batches = batch

            loss_dict = method.train_step(src_batches)

            for k, v in loss_dict.items():
                step_losses.setdefault(k, []).append(v)
                method.history.setdefault(k, []).append(v)

        # ── Epoch mean losses ─────────────────────────────────────────────
        for k, vs in step_losses.items():
            epoch_losses_history.setdefault(k, []).append(float(np.mean(vs)))

        # ── Source-validation evaluation ──────────────────────────────────
        val_results = evaluate_source_domains(method.backbone, method.head, val_loaders, device)
        mean_f1 = val_results["mean"]["macro_f1"]
        val_f1_history.append(mean_f1)

        # Guard: if grads exploded, F1 is NaN
        if np.isnan(mean_f1) or np.isinf(mean_f1):
            print(f"  [WARNING] mean_f1 is NaN/Inf at epoch {epoch+1}. "
                  f"Gradient explosion likely — check clip_grad_norm.")
            sys.stdout.flush()
            patience_cnt += 1
            if patience_cnt >= cfg["patience"]:
                print(f"\n[train] Early stopping (NaN F1 for {cfg['patience']} epochs).")
                sys.stdout.flush()
                break
            continue

        # ── Print epoch summary (matches Task 2 format) ──────────────────
        loss_str = "  ".join(
            f"{k}={float(np.mean(v)):.4f}" for k, v in step_losses.items()
        )
        print(
            f"Epoch [{epoch+1:3d}/{cfg['epochs']}]  {loss_str}  "
            f"| mean_src_F1={mean_f1:.4f}"
            + ("  ← best" if mean_f1 > best_val_f1 else "")
        )
        sys.stdout.flush()

        # ── Checkpoint selection ──────────────────────────────────────────
        if mean_f1 > best_val_f1:
            best_val_f1 = mean_f1
            patience_cnt = 0
            ckpt_path = os.path.join(cfg["checkpoints_dir"], f"{run_name}_best.pth")
            ckpt_data = {
                "epoch":      epoch + 1,
                "val_f1":     best_val_f1,
                "patience_cnt": patience_cnt,
                "state_dict": method.state_dict(),
                "val_results": val_results,
                "run_name":   run_name,
                "method":     args.method,
                "cfg":        cfg,
            }
            torch.save(ckpt_data, ckpt_path)
            # Force flush to disk (important for Google Drive FUSE mount)
            with open(ckpt_path, "rb") as _f:
                os.fsync(_f.fileno())
            size_mb = os.path.getsize(ckpt_path) / 1_048_576
            print(f"  [ckpt] Saved → {ckpt_path}  ({size_mb:.1f} MB)")
            sys.stdout.flush()

            # Save source validation results
            sv_path = os.path.join(cfg["results_dir"], f"{run_name}_source_val.json")
            with open(sv_path, "w") as f:
                json.dump(val_results, f, indent=2)
                f.flush()
                os.fsync(f.fileno())
        else:
            patience_cnt += 1

        # ── Early stopping ────────────────────────────────────────────────
        if patience_cnt >= cfg["patience"]:
            print(
                f"\n[train] Early stopping at epoch {epoch+1} "
                f"(no improvement for {cfg['patience']} epochs)."
            )
            sys.stdout.flush()
            break

    # ── Per-domain val results from best checkpoint ───────────────────────
    ckpt_path = os.path.join(cfg["checkpoints_dir"], f"{run_name}_best.pth")
    if os.path.isfile(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        print(f"\n[train] Best epoch: {ckpt['epoch']}  |  Best mean src F1: {best_val_f1:.4f}")
        print("\nSource validation results (best checkpoint):")
        print_source_val_table(ckpt["val_results"])

    # ── Save training curves ─────────────────────────────────────────────
    save_training_curves(epoch_losses_history, val_f1_history, cfg["results_dir"], run_name)

    print(f"\n[train] Done. Checkpoint: {ckpt_path}")
    sys.stdout.flush()


# ─── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Task 3 – Domain Generalization training script")
    parser.add_argument("--method", type=str, required=True)
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--checkpoints_dir", type=str, required=True)
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--splits_path", type=str, required=True)
    parser.add_argument("--lambda_dg", type=float, default=None)
    parser.add_argument("--rho", type=float, default=None)
    parser.add_argument("--erm_ckpt", type=str, default=None,
                    help="Path to source_only_best.pth for warm initialization of DAN-DG and SAM")
    args = parser.parse_args()
    train(args)
