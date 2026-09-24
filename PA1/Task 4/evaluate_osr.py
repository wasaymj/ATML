"""
evaluate_osr.py — Task 4: Open-Set Recognition evaluation.
Computes AUROC, FPR@95TPR, score distributions, and failure analysis.
"""
from __future__ import annotations

import warnings
warnings.filterwarnings("ignore", message=".*DataLoader.*worker.*")

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import roc_curve

from scores.msp import get_msp_score
from scores.mls import get_mls_score
from scores.energy import get_energy_score
from scores.mahalanobis import compute_mahalanobis_params, get_mahalanobis_score
from scores.proser_score import get_proser_score
from evaluation.metrics import compute_metrics
from evaluation.thresholds import get_threshold
from evaluation.failure_analysis import save_failures
from data.cifar100_unknowns import get_cifar100_unknowns

def get_scores(method: str, score_type: str, outputs: dict, means=None, var=None):
    if score_type == "msp":
        return get_msp_score(outputs["logits"][:, :10])
    elif score_type == "mls":
        return get_mls_score(outputs["logits"][:, :10])
    elif score_type == "energy":
        return get_energy_score(outputs["logits"][:, :10])
    elif score_type == "mahalanobis":
        return get_mahalanobis_score(outputs["feats"], means, var)
    elif score_type == "proser":
        return get_proser_score(outputs["logits"])
    else:
        raise ValueError(f"Unknown score type: {score_type}")

def main(args):
    os.makedirs(args.results_dir, exist_ok=True)
    cifar100 = get_cifar100_unknowns(args.data_root)

    methods = ["vanilla", "gcsc", "proser"]
    results = {}
    
    vanilla_means, vanilla_var = None, None

    for method in methods:
        cache_path = os.path.join(args.cache_dir, f"{method}_outputs.pt")
        if not os.path.exists(cache_path):
            print(f"[eval] Skipping {method} (cache not found)")
            sys.stdout.flush()
            continue

        print(f"\n[eval] Processing {method}...")
        sys.stdout.flush()

        outputs = torch.load(cache_path)
        test_logits = outputs["test"]["logits"][:, :10]
        test_preds = test_logits.argmax(dim=1)
        csa = (test_preds == outputs["test"]["labels"]).float().mean().item()

        means, var = None, None
        if method == "vanilla":
            means, var = compute_mahalanobis_params(
                outputs["train_unaug"]["feats"], outputs["train_unaug"]["labels"]
            )
            vanilla_means, vanilla_var = means, var
            score_types = ["msp", "mls", "energy", "mahalanobis"]
        elif method == "gcsc":
            score_types = ["mls"]
        elif method == "proser":
            score_types = ["mls", "proser"]

        results[method] = {"csa": csa, "scores": {}}

        for st in score_types:
            val_s  = get_scores(method, st, outputs["val"], means, var)
            test_s = get_scores(method, st, outputs["test"], means, var)
            near_s = get_scores(method, st, outputs["near"], means, var)
            far_s  = get_scores(method, st, outputs["far"], means, var)
            all_s  = torch.cat([near_s, far_s])

            threshold = get_threshold(val_s, 0.95)

            near_metrics = compute_metrics(test_s, near_s, threshold)
            far_metrics  = compute_metrics(test_s, far_s, threshold)
            all_metrics  = compute_metrics(test_s, all_s, threshold)
            test_accept  = (test_s <= threshold).float().mean().item()

            results[method]["scores"][st] = {
                "threshold": threshold,
                "near": near_metrics,
                "far": far_metrics,
                "all": all_metrics,
                "test_acceptance_rate": test_accept,
            }

            if method == "vanilla" and st == "mls":
                save_failures(outputs["near"], near_s, threshold, "near", args.results_dir, cifar100["near"])
                save_failures(outputs["far"], far_s, threshold, "far", args.results_dir, cifar100["far"])

    if "vanilla" in results:
        print("\n" + "=" * 100)
        print("TABLE 1: Post-hoc Novelty Scores on Frozen Vanilla Model")
        print(f"Closed-Set Accuracy (CSA): {results['vanilla']['csa']*100:.2f}%")
        print("=" * 100)
        header = (f"{'Score':<15} | {'Near AUROC':<12} | {'Far AUROC':<12} | "
                  f"{'All AUROC':<12} | {'Near FPR95':<12} | {'Far FPR95':<12} | "
                  f"{'Test Accept':<12}")
        print(header)
        print("-" * 100)
        for st, mets in results["vanilla"]["scores"].items():
            print(
                f"{st:<15} | {mets['near']['auroc']:.4f}       | "
                f"{mets['far']['auroc']:.4f}       | "
                f"{mets['all']['auroc']:.4f}       | "
                f"{mets['near']['fpr95']:.4f}       | "
                f"{mets['far']['fpr95']:.4f}       | "
                f"{mets['test_acceptance_rate']:.4f}"
            )
        sys.stdout.flush()

    print("\n" + "=" * 100)
    print("TABLE 2: Model Comparison — Vanilla vs GCSC vs PROSER")
    print("=" * 100)
    header2 = (f"{'Method (Score)':<25} | {'CSA':<8} | {'Near AUROC':<12} | "
               f"{'Far AUROC':<12} | {'Near FPR95':<12} | {'Far FPR95':<12} | "
               f"{'Test Accept':<12}")
    print(header2)
    print("-" * 100)
    for m in ["vanilla", "gcsc", "proser"]:
        if m not in results:
            print(f"  [SKIP] {m} not in results — cache may be missing.")
            continue
        csa = results[m]["csa"]
        for st, mets in results[m]["scores"].items():
            name = f"{m} ({st})"
            print(
                f"{name:<25} | {csa:.4f}   | "
                f"{mets['near']['auroc']:.4f}       | "
                f"{mets['far']['auroc']:.4f}       | "
                f"{mets['near']['fpr95']:.4f}       | "
                f"{mets['far']['fpr95']:.4f}       | "
                f"{mets['test_acceptance_rate']:.4f}"
            )
    sys.stdout.flush()

    if "vanilla" in results:
        cache_path = os.path.join(args.cache_dir, "vanilla_outputs.pt")
        outputs = torch.load(cache_path)
        means, var = vanilla_means, vanilla_var

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        scores_to_plot = ["msp", "mls", "mahalanobis"]
        for i, st in enumerate(scores_to_plot):
            ax = axes[i]
            test_s = get_scores("vanilla", st, outputs["test"], means, var).numpy()
            near_s = get_scores("vanilla", st, outputs["near"], means, var).numpy()
            far_s  = get_scores("vanilla", st, outputs["far"], means, var).numpy()

            ax.hist(test_s, bins=50, alpha=0.5, density=True, label="Known (Test)")
            ax.hist(near_s, bins=50, alpha=0.5, density=True, label="Near Unknown")
            ax.hist(far_s, bins=50, alpha=0.5, density=True, label="Far Unknown")
            ax.set_title(f"{st.upper()} Score Distribution")
            ax.set_xlabel("Unknownness Score")
            ax.set_ylabel("Density")
            ax.legend(fontsize=8)

        plt.tight_layout()
        plot_path = os.path.join(args.results_dir, "vanilla_score_distributions.png")
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"\n[eval] Score distributions saved → {plot_path}")
        sys.stdout.flush()

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for st in ["msp", "mls", "energy", "mahalanobis"]:
            test_s = get_scores("vanilla", st, outputs["test"], means, var).numpy()
            near_s = get_scores("vanilla", st, outputs["near"], means, var).numpy()
            far_s  = get_scores("vanilla", st, outputs["far"],  means, var).numpy()

            y_near = np.concatenate([np.zeros(len(test_s)), np.ones(len(near_s))])
            s_near = np.concatenate([test_s, near_s])
            fpr_n, tpr_n, _ = roc_curve(y_near, s_near)
            axes[0].plot(fpr_n, tpr_n, label=st.upper())

            y_far  = np.concatenate([np.zeros(len(test_s)), np.ones(len(far_s))])
            s_far  = np.concatenate([test_s, far_s])
            fpr_f, tpr_f, _ = roc_curve(y_far, s_far)
            axes[1].plot(fpr_f, tpr_f, label=st.upper())

        for ax, title in zip(axes, ["Near Unknowns", "Far Unknowns"]):
            ax.plot([0,1],[0,1],'k--',alpha=0.3)
            ax.set_xlabel("FPR")
            ax.set_ylabel("TPR")
            ax.set_title(f"Vanilla ROC — {title}")
            ax.legend()
            ax.grid(alpha=0.3)

        plt.tight_layout()
        roc_path = os.path.join(args.results_dir, "vanilla_roc_curves.png")
        plt.savefig(roc_path, dpi=150)
        plt.close()
        print(f"[eval] ROC curves saved → {roc_path}")
        sys.stdout.flush()

    json_path = os.path.join(args.results_dir, "osr_results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[eval] Results saved → {json_path}")
    sys.stdout.flush()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Task 4 — OSR Evaluation")
    parser.add_argument("--cache_dir", type=str, required=True)
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--data_root", type=str, required=True)
    args = parser.parse_args()
    main(args)
