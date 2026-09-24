"""
train.py
========
Unified training script for Task 2: Unsupervised Domain Adaptation.

Usage
-----
    # Main comparison (Steps 1-4):
    python train.py --method source_only
    python train.py --method dan
    python train.py --method dann
    python train.py --method cdan

    # Step 6 controlled study (DAN, vary lambda_mmd):
    python train.py --method dan  --study lambda_mmd  --lambda_mmd 0.1
    python train.py --method dan  --study lambda_mmd  --lambda_mmd 1.0  # (already done above)
    python train.py --method dan  --study lambda_mmd  --lambda_mmd 10.0

    # Step 6 controlled study (DANN, vary grl_max_alpha):
    python train.py --method dann --study grl_alpha   --grl_max_alpha 0.25
    python train.py --method dann --study grl_alpha   --grl_max_alpha 0.5
    python train.py --method dann --study grl_alpha   --grl_max_alpha 1.0  # (already done above)

    # Override data root:
    python train.py --method source_only --data_root /path/to/pacs_parent

Checkpoint selection
--------------------
Checkpoints are selected by the MEAN MACRO-F1 across the THREE source
validation domains.  Target labels are NEVER consulted during training.

BatchNorm freeze policy
-----------------------
Running means and variances are kept at their ImageNet pretrained values.
Affine params (gamma, beta) remain trainable.  Implemented inside BNFrozenResNet18
and also enforced explicitly at every train_step call.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# ── local imports ─────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

from models.backbone       import BNFrozenResNet18, FEAT_DIM
from models.classifier_head import ClassifierHead
from shared.pacs_protocol   import (
    DomainBalancedIterator,
    SOURCE_DOMAINS,
    TARGET_DOMAIN,
    build_source_datasets,
    build_source_loaders,
    build_target_dataset,
    build_target_loader,
    get_eval_transform,
    get_or_create_splits,
    get_train_transform,
)
from shared.pacs             import download_pacs
from evaluation.metrics      import evaluate_source_domains, print_source_val_table

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ─── seed ────────────────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ─── config loading ───────────────────────────────────────────────────────────

def load_config(method: str, config_dir: str = "configs") -> dict:
    """Load base.yaml then merge method-specific YAML on top."""
    try:
        import yaml
    except ImportError:
        import subprocess
        subprocess.run([sys.executable, "-m", "pip", "install", "pyyaml", "-q"], check=True)
        import yaml

    base_path   = os.path.join(config_dir, "base.yaml")
    method_path = os.path.join(config_dir, f"{method}.yaml")

    with open(base_path, "r") as f:
        cfg = yaml.safe_load(f)
    if os.path.isfile(method_path):
        with open(method_path, "r") as f:
            method_cfg = yaml.safe_load(f) or {}
        cfg.update(method_cfg)
    return cfg


# ─── method factory ──────────────────────────────────────────────────────────

def build_method(args, cfg, backbone, head, device):
    """Instantiate the appropriate method class."""
    from methods.source_only import SourceOnly
    from methods.dan          import DAN
    from methods.dann         import DANN
    from methods.cdan         import CDAN

    kw = dict(
        backbone=backbone, head=head, device=device,
        lr=cfg["lr"], weight_decay=cfg["weight_decay"],
    )

    if cfg["method"] == "source_only":
        return SourceOnly(**kw)

    elif cfg["method"] == "dan":
        lmmd = args.lambda_mmd if args.lambda_mmd is not None else cfg.get("lambda_mmd", 1.0)
        return DAN(**kw, lambda_mmd=lmmd,
                   kernel_muls=cfg.get("mmd_kernel_muls", [0.5, 1.0, 2.0]))

    elif cfg["method"] == "dann":
        alpha = args.grl_max_alpha if args.grl_max_alpha is not None else cfg.get("grl_max_alpha", 1.0)
        return DANN(**kw, lambda_domain=cfg.get("lambda_domain", 1.0),
                    grl_max_alpha=alpha,
                    disc_hidden=cfg.get("disc_hidden", 256),
                    disc_dropout=cfg.get("disc_dropout", 0.5),
                    feat_dim=FEAT_DIM)

    elif cfg["method"] == "cdan":
        alpha = args.grl_max_alpha if args.grl_max_alpha is not None else cfg.get("grl_max_alpha", 1.0)
        return CDAN(**kw, lambda_domain=cfg.get("lambda_domain", 1.0),
                    grl_max_alpha=alpha,
                    disc_hidden=cfg.get("disc_hidden", 256),
                    disc_dropout=cfg.get("disc_dropout", 0.5),
                    feat_dim=FEAT_DIM,
                    num_classes=cfg["num_classes"])
    else:
        raise ValueError(f"Unknown method: {cfg['method']}")


# ─── checkpoint name helper ───────────────────────────────────────────────────

def checkpoint_name(method: str, study: str | None, **kwargs) -> str:
    """Generate a unique checkpoint filename for the run."""
    name = method
    if study:
        for k, v in kwargs.items():
            if v is not None:
                name += f"_{k}{v}"
    return name


# ─── training curves ─────────────────────────────────────────────────────────

def save_training_curves(
    epoch_losses:  dict,
    val_history:   list,
    out_dir:       str,
    run_name:      str,
) -> None:
    """Save loss curves as PNG and JSON."""
    os.makedirs(out_dir, exist_ok=True)

    # ── JSON ─────────────────────────────────────────────────────────────
    curve_data = {"losses": epoch_losses, "val_macro_f1": val_history}
    json_path  = os.path.join(out_dir, f"{run_name}_curves.json")
    with open(json_path, "w") as f:
        json.dump(curve_data, f, indent=2)
        f.flush()
        os.fsync(f.fileno())

    # ── Plot ─────────────────────────────────────────────────────────────
    n_epochs = len(val_history)
    epochs   = list(range(1, n_epochs + 1))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Loss curves
    for key, values in epoch_losses.items():
        if values:
            axes[0].plot(epochs[:len(values)], values, label=key)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title(f"{run_name} – Training Losses")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Val macro-F1
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
    with open(png_path, "ab") as f:
        f.flush()
        os.fsync(f.fileno())
    print(f"  Training curves saved → {json_path}, {png_path}")


# ─── main training loop ───────────────────────────────────────────────────────

def train(args) -> None:
    # ── Config ───────────────────────────────────────────────────────────
    cfg = load_config(args.method)
    if args.data_root:       cfg["data_root"]       = args.data_root
    if args.checkpoints_dir: cfg["checkpoints_dir"] = args.checkpoints_dir
    if args.results_dir:     cfg["results_dir"]     = args.results_dir
    if args.splits_path:     cfg["splits_path"]     = args.splits_path

    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] Method: {args.method}  |  Device: {device}")

    # ── Run name (for checkpoints and results) ────────────────────────────
    study_kwargs = {}
    if args.lambda_mmd   is not None: study_kwargs["lambda_mmd"]   = args.lambda_mmd
    if args.grl_max_alpha is not None: study_kwargs["grl_max_alpha"] = args.grl_max_alpha
    run_name = checkpoint_name(args.method, args.study, **study_kwargs)
    print(f"[train] Run name: {run_name}")

    os.makedirs(cfg["checkpoints_dir"], exist_ok=True)
    os.makedirs(cfg["results_dir"],     exist_ok=True)

    # ── Data ─────────────────────────────────────────────────────────────
    pacs_root = download_pacs(cfg["data_root"])
    splits    = get_or_create_splits(cfg["splits_path"], pacs_root)

    train_tfm = get_train_transform(cfg["img_size"], cfg["resize_size"])
    eval_tfm  = get_eval_transform(cfg["img_size"], cfg["resize_size"])

    source_datasets = build_source_datasets(pacs_root, splits, train_tfm, eval_tfm)
    source_loaders  = build_source_loaders(
        source_datasets,
        batch_size_per_source=cfg["batch_size_per_source"],
        num_workers=cfg["num_workers"],
    )
    target_train_ds = build_target_dataset(pacs_root, train_tfm, return_labels=False)
    target_loader   = build_target_loader(
        target_train_ds,
        batch_size=cfg["batch_size_target"],
        shuffle=True,
        num_workers=cfg["num_workers"],
    )

    # Validation loaders (one per source domain)
    val_loaders = {d: source_loaders[d]["val"] for d in SOURCE_DOMAINS}
    # Train loaders (one per source domain)
    train_loaders = {d: source_loaders[d]["train"] for d in SOURCE_DOMAINS}

    # Domain-balanced iterator
    domain_iter = DomainBalancedIterator(train_loaders, target_loader)
    steps_per_epoch = len(domain_iter)
    total_steps = cfg["epochs"] * steps_per_epoch
    print(f"[train] Steps/epoch: {steps_per_epoch}  |  Total steps: {total_steps}")

    # ── Model ────────────────────────────────────────────────────────────
    backbone = BNFrozenResNet18()
    head     = ClassifierHead(feat_dim=FEAT_DIM, num_classes=cfg["num_classes"])
    method   = build_method(args, cfg, backbone, head, device)

    # ── Training state ────────────────────────────────────────────────────
    best_val_f1   = -1.0
    start_epoch = 0
    patience_cnt  = 0
    val_f1_history: list = []
    epoch_losses: dict = {}

    # ── Resume from existing checkpoint if present ────────────────────────
    ckpt_path = os.path.join(cfg["checkpoints_dir"], f"{run_name}_best.pth")

    if os.path.isfile(ckpt_path):
        print(f"[train] Resuming from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device)
        method.load_state_dict(ckpt["state_dict"])
        best_val_f1  = ckpt.get("val_f1", -1.0)
        start_epoch  = ckpt.get("epoch", 0)
        patience_cnt = ckpt.get("patience_cnt", 0)
        print(f"  Loaded epoch {start_epoch}, best F1 so far: {best_val_f1:.4f}")
    else:
        print(f"[train] No checkpoint found — starting from scratch.")

    for epoch in range(start_epoch, cfg["epochs"]):
        # ── Reset iterators for new epoch ────────────────────────────────
        domain_iter_gen = iter(DomainBalancedIterator(train_loaders, target_loader))

        step_losses: dict = {}

        for local_step, (src_batches, tgt_batch) in enumerate(domain_iter_gen):
            global_step = epoch * steps_per_epoch + local_step
            p = global_step / max(total_steps - 1, 1)

            losses = method.train_step(src_batches, tgt_batch, p)

            for k, v in losses.items():
                step_losses.setdefault(k, []).append(v)

        # ── Epoch mean losses ─────────────────────────────────────────────
        for k, vs in step_losses.items():
            epoch_losses.setdefault(k, []).append(float(np.mean(vs)))

        # ── Source-validation evaluation ──────────────────────────────────
        val_results = evaluate_source_domains(
            method.backbone, method.head, val_loaders, device
        )
        mean_f1 = val_results["mean"]["macro_f1"]
        val_f1_history.append(mean_f1)

        # Guard: if grads exploded, F1 is NaN — warn and skip checkpoint
        if np.isnan(mean_f1) or np.isinf(mean_f1):
            print(f"  [WARNING] mean_f1 is NaN/Inf at epoch {epoch+1}. "
                  f"Gradient explosion likely — check clip_grad_norm.")
            patience_cnt += 1
            if patience_cnt >= cfg["patience"]:
                print(f"\n[train] Early stopping (NaN F1 for {cfg['patience']} epochs).")
                break
            continue

        loss_str = "  ".join(
            f"{k}={float(np.mean(v)):.4f}" for k, v in step_losses.items()
            if k != "alpha"
        )
        print(
            f"Epoch [{epoch+1:3d}/{cfg['epochs']}]  {loss_str}  "
            f"| mean_src_F1={mean_f1:.4f}"
            + ("  ← best" if mean_f1 > best_val_f1 else "")
        )

        # ── Checkpoint selection: only by source-val macro-F1 ─────────────
        if mean_f1 > best_val_f1:
            best_val_f1 = mean_f1
            patience_cnt = 0
            ckpt_path = os.path.join(
                cfg["checkpoints_dir"], f"{run_name}_best.pth"
            )
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
        else:
            patience_cnt += 1

        # ── Early stopping ────────────────────────────────────────────────
        if patience_cnt >= cfg["patience"]:
            print(
                f"\n[train] Early stopping at epoch {epoch+1} "
                f"(no improvement for {cfg['patience']} epochs)."
            )
            break

    # ── Per-domain val results from best checkpoint ───────────────────────
    ckpt = torch.load(
        os.path.join(cfg["checkpoints_dir"], f"{run_name}_best.pth"),
        map_location=device,
    )
    print(f"\n[train] Best epoch: {ckpt['epoch']}  |  Best mean src F1: {best_val_f1:.4f}")
    print("\nSource validation results (best checkpoint):")
    print_source_val_table(ckpt["val_results"])

    # Save source-val results JSON
    sv_path = os.path.join(cfg["results_dir"], f"{run_name}_source_val.json")
    with open(sv_path, "w") as f:
        json.dump(ckpt["val_results"], f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    print(f"\n  Source-val results → {sv_path}")

    # ── Save training curves ──────────────────────────────────────────────
    save_training_curves(epoch_losses, val_f1_history, cfg["results_dir"], run_name)

    print(f"\n[train] Done. Checkpoint: {os.path.join(cfg['checkpoints_dir'], run_name + '_best.pth')}")


# ─── CLI ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Task 2 – UDA training script")
    p.add_argument("--method",           required=True,
                   choices=["source_only", "dan", "dann", "cdan"],
                   help="Adaptation method")
    # ── Path overrides (for Google Drive / Colab) ─────────────────────────
    p.add_argument("--data_root",        default=None,
                   help="Parent directory for PACS dataset download")
    p.add_argument("--checkpoints_dir",  default=None,
                   help="Directory for saving checkpoints (e.g. Drive/task2/checkpoints)")
    p.add_argument("--results_dir",      default=None,
                   help="Directory for saving result files and plots")
    p.add_argument("--splits_path",      default=None,
                   help="Full path for the splits JSON file")
    # ── Controlled study ─────────────────────────────────────────────────
    p.add_argument("--study",            default=None,
                   choices=["lambda_mmd", "grl_alpha"],
                   help="Step 6 controlled design study (optional)")
    p.add_argument("--lambda_mmd",       type=float, default=None,
                   help="Override lambda_mmd for DAN (study only)")
    p.add_argument("--grl_max_alpha",    type=float, default=None,
                   help="Override grl_max_alpha for DANN/CDAN (study only)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.study == "lambda_mmd":
        # Controlled study: vary lambda_mmd for DAN
        assert args.method == "dan", "--study lambda_mmd requires --method dan"
        if args.lambda_mmd is None:
            print("[study] Running all lambda_mmd values: 0.1, 1.0, 10.0")
            for lmmd in [0.1, 1.0, 10.0]:
                args.lambda_mmd = lmmd
                train(args)
        else:
            train(args)

    elif args.study == "grl_alpha":
        # Controlled study: vary grl_max_alpha for DANN
        assert args.method in ("dann", "cdan"), "--study grl_alpha requires --method dann or cdan"
        if args.grl_max_alpha is None:
            print("[study] Running all grl_max_alpha values: 0.25, 0.5, 1.0")
            for alpha in [0.25, 0.5, 1.0]:
                args.grl_max_alpha = alpha
                train(args)
        else:
            train(args)

    else:
        train(args)
