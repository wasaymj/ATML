"""
evaluate_final.py
=================
Step 5 and 6 final evaluation script for Task 2.

This script is run AFTER all training is complete and all checkpoints are
fixed.  It is the ONLY place where target (Sketch) class labels are consumed.

What it produces
----------------
1. Source-val + target metric table for all four methods.
2. Domain separability score for each method.
3. Per-class target accuracy, top gains/losses vs Source-only, dominant confusions.
4. Controlled design study comparison table/plot.
5. Everything saved to results/final_evaluation.json and results/*.png.

Usage
-----
    python evaluate_final.py
    python evaluate_final.py --data_root /path/to/pacs_parent
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(__file__))

from models.backbone        import BNFrozenResNet18, FEAT_DIM
from models.classifier_head import ClassifierHead
from shared.pacs             import download_pacs, PACS_CLASSES
from shared.pacs_protocol    import (
    SOURCE_DOMAINS,
    build_source_datasets,
    build_source_loaders,
    build_target_dataset,
    build_target_loader,
    get_eval_transform,
    get_or_create_splits,
    get_train_transform,
)
from evaluation.metrics            import evaluate_loader, evaluate_source_domains
from evaluation.domain_separability import compute_domain_separability
from evaluation.class_analysis      import (
    full_class_analysis,
    print_class_analysis,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

METHODS = ["source_only", "dan", "dann", "cdan"]
METHOD_LABELS = {
    "source_only": "Source-only",
    "dan":         "DAN",
    "dann":        "DANN",
    "cdan":        "CDAN",
}


# ─── helpers ──────────────────────────────────────────────────────────────────

def load_config(config_dir: str = "configs") -> dict:
    try:
        import yaml
    except ImportError:
        import subprocess
        subprocess.run([sys.executable, "-m", "pip", "install", "pyyaml", "-q"], check=True)
        import yaml
    with open(os.path.join(config_dir, "base.yaml"), "r") as f:
        return yaml.safe_load(f)


def load_method_checkpoint(
    method:    str,
    ckpt_dir:  str,
    device:    torch.device,
    num_classes: int = 7,
) -> tuple:
    """Load backbone + head from the best checkpoint for *method*."""
    ckpt_path = os.path.join(ckpt_dir, f"{method}_best.pth")
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}\n"
            f"Run: python train.py --method {method}"
        )

    backbone = BNFrozenResNet18()
    head     = ClassifierHead(feat_dim=FEAT_DIM, num_classes=num_classes)

    ckpt = torch.load(ckpt_path, map_location=device)
    sd   = ckpt["state_dict"]
    backbone.load_state_dict(sd["backbone"])
    head.load_state_dict(sd["head"])

    backbone = backbone.to(device).eval()
    head     = head.to(device).eval()

    best_epoch = ckpt.get("epoch", "?")
    best_f1    = ckpt.get("val_f1", float("nan"))
    print(f"  [{method}] loaded from epoch {best_epoch}  (src val F1={best_f1:.4f})")

    return backbone, head

def _evaluate_study_checkpoints(cfg, device, target_loader, val_loaders):
    """Evaluate controlled study variants on target for analysis."""
    study_results = {}
    
    # DAN lambda_mmd variants
    for lmmd in [0.1, 1.0, 10.0]:
        ckpt_path = os.path.join(cfg["checkpoints_dir"], 
                                  f"dan_lambda_mmd{lmmd}_best.pth")
        if not os.path.isfile(ckpt_path):
            continue
        backbone = BNFrozenResNet18().to(device).eval()
        head     = ClassifierHead(FEAT_DIM, cfg["num_classes"]).to(device).eval()
        sd = torch.load(ckpt_path, map_location=device)["state_dict"]
        backbone.load_state_dict(sd["backbone"])
        head.load_state_dict(sd["head"])
        tgt = evaluate_loader(backbone, head, target_loader, device)
        src = evaluate_source_domains(backbone, head, val_loaders, device)
        study_results[f"dan_lmmd{lmmd}"] = {"target": tgt, "source_val": src}
        print(f"  DAN λ={lmmd}: src_F1={src['mean']['macro_f1']:.4f}  "
              f"tgt_acc={tgt['accuracy']:.4f}")
    
    # DANN grl_alpha variants
    for alpha in [0.25, 0.5, 1.0]:
        ckpt_path = os.path.join(cfg["checkpoints_dir"],
                                  f"dann_grl_max_alpha{alpha}_best.pth")
        if not os.path.isfile(ckpt_path):
            continue
        backbone = BNFrozenResNet18().to(device).eval()
        head     = ClassifierHead(FEAT_DIM, cfg["num_classes"]).to(device).eval()
        sd = torch.load(ckpt_path, map_location=device)["state_dict"]
        backbone.load_state_dict(sd["backbone"])
        head.load_state_dict(sd["head"])
        tgt = evaluate_loader(backbone, head, target_loader, device)
        src = evaluate_source_domains(backbone, head, val_loaders, device)
        study_results[f"dann_alpha{alpha}"] = {"target": tgt, "source_val": src}
        print(f"  DANN α={alpha}: src_F1={src['mean']['macro_f1']:.4f}  "
              f"tgt_acc={tgt['accuracy']:.4f}")
    
    return study_results


# ─── main evaluation ──────────────────────────────────────────────────────────

def run_evaluation(args) -> None:
    cfg    = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[eval] Device: {device}")

    if args.data_root:       cfg["data_root"]       = args.data_root
    if args.checkpoints_dir: cfg["checkpoints_dir"] = args.checkpoints_dir
    if args.results_dir:     cfg["results_dir"]     = args.results_dir
    if args.splits_path:     cfg["splits_path"]     = args.splits_path

    os.makedirs(cfg["results_dir"], exist_ok=True)

    # ── Data setup ────────────────────────────────────────────────────────
    pacs_root = download_pacs(cfg["data_root"])
    splits    = get_or_create_splits(cfg["splits_path"], pacs_root)

    eval_tfm  = get_eval_transform(cfg["img_size"], cfg["resize_size"])
    train_tfm = get_train_transform(cfg["img_size"], cfg["resize_size"])

    source_datasets = build_source_datasets(pacs_root, splits, train_tfm, eval_tfm)
    source_loaders  = build_source_loaders(
        source_datasets, batch_size_per_source=64, num_workers=cfg["num_workers"]
    )
    val_loaders = {d: source_loaders[d]["val"] for d in SOURCE_DOMAINS}

    # Target evaluation dataset (labels used here for the FIRST and ONLY time)
    target_eval_ds  = build_target_dataset(pacs_root, eval_tfm, return_labels=True)
    target_loader   = build_target_loader(
        target_eval_ds, batch_size=64, shuffle=False, num_workers=cfg["num_workers"], drop_last=False
    )
    # Target dataset also used for domain separability (unlabeled)
    target_adapt_ds = build_target_dataset(pacs_root, eval_tfm, return_labels=False)
    target_adapt_ldr = build_target_loader(
        target_adapt_ds, batch_size=64, shuffle=False, num_workers=cfg["num_workers"], drop_last=False
    )

    # ── Evaluate each method ──────────────────────────────────────────────
    all_results: Dict[str, dict] = {}
    source_only_per_class: Optional[dict] = None

    for method in METHODS:
        print(f"\n{'='*60}")
        print(f"  Evaluating: {METHOD_LABELS[method]}")
        print(f"{'='*60}")

        try:
            backbone, head = load_method_checkpoint(
                method, cfg["checkpoints_dir"], device, cfg["num_classes"]
            )
        except FileNotFoundError as e:
            print(f"  SKIPPING (checkpoint not found): {e}")
            continue

        # Source validation
        src_val = evaluate_source_domains(backbone, head, val_loaders, device)

        # Target (Sketch) accuracy and F1
        tgt_metrics = evaluate_loader(backbone, head, target_loader, device)

        # Domain separability
        dom_sep = compute_domain_separability(
            backbone, val_loaders, target_adapt_ldr, device
        )

        # Per-class target analysis
        cls_analysis = full_class_analysis(
            backbone, head, target_loader, device,
            baseline_per_class=source_only_per_class,
        )

        all_results[method] = {
            "source_val":           src_val,
            "target":               tgt_metrics,
            "domain_separability":  dom_sep,
            "class_analysis":       {
                "per_class_accuracy":  cls_analysis["per_class_accuracy"],
                "dominant_confusions": cls_analysis["dominant_confusions"],
                "comparison":          cls_analysis.get("comparison"),
            },
        }

        # Store source_only per-class for delta comparison
        if method == "source_only":
            source_only_per_class = cls_analysis["per_class_accuracy"]

        # Print summary
        print(f"\n  Source-val results:")
        for d in SOURCE_DOMAINS:
            r = src_val[d]
            print(f"    {d:<18}  acc={r['accuracy']:.4f}  F1={r['macro_f1']:.4f}")
        m = src_val["mean"]
        print(f"    {'Mean':<18}  acc={m['accuracy']:.4f}  F1={m['macro_f1']:.4f}")
        print(f"\n  Target (Sketch):   acc={tgt_metrics['accuracy']:.4f}  "
              f"F1={tgt_metrics['macro_f1']:.4f}")
        print(f"  Domain separability: {dom_sep['domain_separability']:.4f}  "
              f"(0.50 = chance)")
        print("\n  Per-class target accuracy:")
        print_class_analysis(cls_analysis)

    # ── Evaluate controlled study checkpoints ─────────────────────────────

    print("\n" + "="*60)
    print("  CONTROLLED STUDY — Target Evaluation (analysis only)")
    print("="*60)
    study_results = _evaluate_study_checkpoints(
        cfg, device, target_loader, val_loaders
    )
    study_out = os.path.join(cfg["results_dir"], "controlled_study_target.json")
    with open(study_out, "w") as f:
        json.dump(study_results, f, indent=2)
    print(f"\n  Study results saved → {study_out}")

    # ── Compute target Δ vs source-only ──────────────────────────────────
    baseline_tgt = all_results.get("source_only", {}).get("target", {})
    for method, r in all_results.items():
        tgt = r["target"]
        tgt["delta_vs_source_only"] = (
            tgt["accuracy"] - baseline_tgt.get("accuracy", 0.0)
            if baseline_tgt else 0.0
        )

    # ── Save all results ──────────────────────────────────────────────────
    out_path = os.path.join(cfg["results_dir"], "final_evaluation.json")
    # Make class_analysis JSON-serializable (replace ndarray etc.)
    results_serializable = json.loads(json.dumps(all_results, default=str))
    with open(out_path, "w") as f:
        json.dump(results_serializable, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    print(f"\n[eval] All results saved → {out_path}")

    # ── Print comparison table ────────────────────────────────────────────
    _print_comparison_table(all_results)

    # ── Plots ────────────────────────────────────────────────────────────
    _plot_target_accuracy(all_results, cfg["results_dir"])
    _plot_domain_separability(all_results, cfg["results_dir"])
    _plot_per_class_deltas(all_results, cfg["results_dir"])
    _plot_controlled_study(cfg["results_dir"])

    print("\n[eval] Evaluation complete.")


def _print_comparison_table(results: Dict[str, dict]) -> None:
    print("\n" + "=" * 90)
    print("COMPARISON TABLE — Source-only / DAN / DANN / CDAN")
    print("=" * 90)
    header = (
        f"{'Method':<14} "
        f"{'Photo':>8} {'ArtPnt':>8} {'Cartoon':>8} "
        f"{'MeanAcc':>8} {'MeanF1':>8} "
        f"{'TgtAcc':>8} {'TgtF1':>8} {'ΔTgt':>8} "
        f"{'DomSep':>8}"
    )
    print(header)
    print("-" * 90)
    for method, r in results.items():
        sv  = r["source_val"]
        tgt = r["target"]
        dom = r["domain_separability"]
        print(
            f"{METHOD_LABELS.get(method, method):<14} "
            f"{sv.get('photo',{}).get('accuracy',float('nan')):>8.4f} "
            f"{sv.get('art_painting',{}).get('accuracy',float('nan')):>8.4f} "
            f"{sv.get('cartoon',{}).get('accuracy',float('nan')):>8.4f} "
            f"{sv.get('mean',{}).get('accuracy',float('nan')):>8.4f} "
            f"{sv.get('mean',{}).get('macro_f1',float('nan')):>8.4f} "
            f"{tgt.get('accuracy',float('nan')):>8.4f} "
            f"{tgt.get('macro_f1',float('nan')):>8.4f} "
            f"{tgt.get('delta_vs_source_only',0.0):>+8.4f} "
            f"{dom.get('domain_separability',float('nan')):>8.4f}"
        )
    print("=" * 90)


def _plot_target_accuracy(results: Dict, out_dir: str) -> None:
    names  = [METHOD_LABELS[m] for m in METHODS if m in results]
    accs   = [results[m]["target"]["accuracy"] for m in METHODS if m in results]
    f1s    = [results[m]["target"]["macro_f1"] for m in METHODS if m in results]

    x = np.arange(len(names))
    w = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - w/2, accs, w, label="Target Accuracy", color="#2980b9")
    ax.bar(x + w/2, f1s,  w, label="Target Macro-F1", color="#27ae60")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Target (Sketch) Accuracy and Macro-F1 by Method")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    for bar in ax.patches:
        ax.annotate(f"{bar.get_height():.3f}",
                    (bar.get_x() + bar.get_width()/2, bar.get_height()),
                    ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    path = os.path.join(out_dir, "target_accuracy_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot → {path}")


def _plot_domain_separability(results: Dict, out_dir: str) -> None:
    names = [METHOD_LABELS[m] for m in METHODS if m in results]
    seps  = [results[m]["domain_separability"]["domain_separability"]
             for m in METHODS if m in results]
    tgt_accs = [results[m]["target"]["accuracy"] for m in METHODS if m in results]

    fig, ax1 = plt.subplots(figsize=(8, 5))
    color1 = "#e74c3c"
    color2 = "#2980b9"
    x = np.arange(len(names))
    ax1.bar(x - 0.2, seps, 0.4, color=color1, alpha=0.8, label="Domain Separability")
    ax1.axhline(0.5, color="gray", linestyle="--", alpha=0.5, label="Chance (0.50)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(names)
    ax1.set_ylabel("Domain Separability", color=color1)
    ax1.set_ylim(0, 1.05)

    ax2 = ax1.twinx()
    ax2.bar(x + 0.2, tgt_accs, 0.4, color=color2, alpha=0.8, label="Target Accuracy")
    ax2.set_ylabel("Target Accuracy", color=color2)
    ax2.set_ylim(0, 1.05)

    ax1.set_title("Domain Separability vs Target Accuracy by Method")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
    ax1.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(out_dir, "domain_separability_vs_target.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot → {path}")


def _plot_per_class_deltas(results: Dict, out_dir: str) -> None:
    """Bar chart of per-class accuracy change vs source-only for each adaptation method."""
    methods_to_plot = [m for m in ["dan", "dann", "cdan"] if m in results]
    if not methods_to_plot:
        return

    class_names = PACS_CLASSES
    x = np.arange(len(class_names))
    w = 0.25

    fig, ax = plt.subplots(figsize=(12, 5))
    for i, method in enumerate(methods_to_plot):
        comp = results[method]["class_analysis"].get("comparison")
        if comp is None:
            continue
        deltas = [comp["deltas"].get(c, 0.0) or 0.0 for c in class_names]
        ax.bar(x + (i - 1) * w, deltas, w, label=METHOD_LABELS[method])

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=15, ha="right")
    ax.set_ylabel("Accuracy Δ vs Source-only")
    ax.set_title("Per-class Target Accuracy Change Relative to Source-only")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(out_dir, "per_class_deltas.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot → {path}")


def _plot_controlled_study(results_dir: str) -> None:
    """
    If controlled-study result files exist (from train.py --study),
    generate a comparison plot.
    """
    # DAN lambda_mmd study
    study_data = []
    for lmmd in [0.1, 1.0, 10.0]:
        sv_path = os.path.join(results_dir, f"dan_lambda_mmd{lmmd}_source_val.json")
        if os.path.isfile(sv_path):
            with open(sv_path) as f:
                sv = json.load(f)
            study_data.append({
                "lambda": lmmd,
                "source_mean_f1": sv.get("mean", {}).get("macro_f1", float("nan")),
            })

    if len(study_data) > 1:
        lambdas  = [d["lambda"] for d in study_data]
        src_f1s  = [d["source_mean_f1"] for d in study_data]

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(lambdas, src_f1s, "o-", color="#e74c3c", label="Mean src val macro-F1")
        ax.set_xscale("log")
        ax.set_xlabel("λ_MMD")
        ax.set_ylabel("Mean Source Val Macro-F1")
        ax.set_title("DAN Controlled Study: Effect of λ_MMD")
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        path = os.path.join(results_dir, "controlled_study_dan_lambda.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Controlled study plot → {path}")

    # DANN grl_alpha study
    dann_study = []
    for alpha in [0.25, 0.5, 1.0]:
        sv_path = os.path.join(results_dir, f"dann_grl_max_alpha{alpha}_source_val.json")
        if os.path.isfile(sv_path):
            with open(sv_path) as f:
                sv = json.load(f)
            dann_study.append({
                "alpha": alpha,
                "source_mean_f1": sv.get("mean", {}).get("macro_f1", float("nan")),
            })

    if len(dann_study) > 1:
        alphas  = [d["alpha"]         for d in dann_study]
        src_f1s = [d["source_mean_f1"] for d in dann_study]
        
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(alphas, src_f1s, "o-", color="#8e44ad", label="Mean src val macro-F1")
        ax.set_xlabel("GRL max alpha")
        ax.set_ylabel("Mean Source Val Macro-F1")
        ax.set_title("DANN Controlled Study: Effect of GRL Max Alpha")
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        path = os.path.join(results_dir, "controlled_study_dann_alpha.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Controlled study plot → {path}")


# ─── CLI ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Task 2 – Final evaluation (target labels used here only)"
    )
    p.add_argument("--data_root",       default=None,
                   help="Override data_root from config")
    p.add_argument("--checkpoints_dir", default=None,
                   help="Directory containing best checkpoints")
    p.add_argument("--results_dir",     default=None,
                   help="Directory to save evaluation outputs")
    p.add_argument("--splits_path",     default=None,
                   help="Full path to the splits JSON file")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_evaluation(args)
