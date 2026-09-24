"""
train.py — Task 4: Open-Set Recognition training script.
Trains Vanilla, GCSC, or PROSER ResNet-18 on CIFAR-10.
"""
from __future__ import annotations

import warnings
warnings.filterwarnings("ignore", message=".*DataLoader.*worker.*")

import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from data.cifar10 import get_cifar10_datasets
from methods.vanilla import VanillaMethod
from methods.gcsc import GCSCMethod
from methods.proser import PROSERMethod


# ── seed ─────────────────────────────────────────────────────────────────────

def set_seed(seed: int = 6304) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ── evaluation ───────────────────────────────────────────────────────────────

def evaluate(model, loader, device, num_known=10):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)[:, :num_known]
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total


# ── training curves ──────────────────────────────────────────────────────────

def save_training_curves(epoch_loss_history, val_acc_history, out_dir, run_name):
    """Save training loss + val accuracy curves as PNG and JSON."""
    os.makedirs(out_dir, exist_ok=True)

    # JSON
    curve_data = {"losses": epoch_loss_history, "val_acc": val_acc_history}
    json_path = os.path.join(out_dir, f"{run_name}_curves.json")
    with open(json_path, "w") as f:
        json.dump(curve_data, f, indent=2)
        f.flush()

    # PNG
    epochs = list(range(1, len(val_acc_history) + 1))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Loss subplot
    for key, values in epoch_loss_history.items():
        axes[0].plot(epochs[:len(values)], values, label=key)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title(f"{run_name} — Training Losses")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Val accuracy subplot
    axes[1].plot(epochs, val_acc_history, "o-", color="steelblue", label="Val Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title(f"{run_name} — Validation Accuracy")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    png_path = os.path.join(out_dir, f"{run_name}_curves.png")
    plt.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[train] Curves saved → {png_path}")
    sys.stdout.flush()


# ── main training loop ───────────────────────────────────────────────────────

def train(args):
    set_seed(6304)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] Task 4 Method: {args.method}  |  Device: {device}")
    sys.stdout.flush()

    os.makedirs(args.checkpoints_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    # ── Data ──────────────────────────────────────────────────────────────
    train_set, val_set, _ = get_cifar10_datasets(
        args.data_root, seed=6304, gcsc_aug=(args.method == "gcsc")
    )
    g = torch.Generator()
    g.manual_seed(6304)
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=128, shuffle=True, num_workers=8, generator=g, pin_memory=True, drop_last=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=128, shuffle=False, num_workers=8, pin_memory=True
    )

    # ── Method ────────────────────────────────────────────────────────────
    epochs = 100
    if args.method == "vanilla":
        method = VanillaMethod(device, epochs=100)
    elif args.method == "gcsc":
        method = GCSCMethod(device, epochs=100)
    elif args.method == "proser":
        epochs = 50
        print(f"[train] Loading Vanilla checkpoint from {args.vanilla_ckpt}")
        sys.stdout.flush()
        vanilla_sd = torch.load(args.vanilla_ckpt, map_location=device)
        method = PROSERMethod(device, vanilla_state_dict=vanilla_sd, epochs=50)
    else:
        raise ValueError(f"Unknown method {args.method}")

    # ── Training state ────────────────────────────────────────────────────
    best_val_acc = -1.0
    epoch_losses_history = {}
    val_acc_history = []

    print(f"[train] Training for {epochs} epochs...")
    sys.stdout.flush()

    for epoch in range(epochs):
        t0 = time.time()
        step_losses = {}

        for imgs, labels in train_loader:
            res = method.train_step(imgs, labels)
            if isinstance(res, dict):
                for k, v in res.items():
                    step_losses.setdefault(k, []).append(v)
            else:
                step_losses.setdefault("loss", []).append(res)

        method.step_scheduler()

        # Epoch-level mean losses
        epoch_means = {}
        for k, v in step_losses.items():
            mean_val = float(np.mean(v))
            epoch_means[k] = mean_val
            epoch_losses_history.setdefault(k, []).append(mean_val)

        # Validation
        val_acc = evaluate(method.model, val_loader, device)
        val_acc_history.append(val_acc)

        dt = time.time() - t0
        loss_str = "  ".join([f"{k}={v:.4f}" for k, v in epoch_means.items()])
        marker = ""

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            marker = "  ← best"
            ckpt_path = os.path.join(args.checkpoints_dir, f"{args.method}_best.pth")
            torch.save({
                "epoch": epoch + 1,
                "val_acc": best_val_acc,
                "model": method.state_dict()["model"]
            }, ckpt_path)

        print(
            f"Epoch [{epoch+1:3d}/{epochs}]  {loss_str}  "
            f"| val_acc={val_acc:.4f}{marker}  ({dt:.1f}s)"
        )
        sys.stdout.flush()

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n[train] Best val accuracy: {best_val_acc:.4f}")
    sys.stdout.flush()

    # ── Save curves ───────────────────────────────────────────────────────
    save_training_curves(epoch_losses_history, val_acc_history,
                         args.results_dir, args.method)

    print(f"[train] Done. Checkpoint: {args.checkpoints_dir}/{args.method}_best.pth")
    sys.stdout.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Task 4 — Open-Set Recognition training")
    parser.add_argument("--method", type=str, required=True,
                        choices=["vanilla", "gcsc", "proser"])
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--checkpoints_dir", type=str, required=True)
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--vanilla_ckpt", type=str, default=None)
    args = parser.parse_args()
    train(args)
