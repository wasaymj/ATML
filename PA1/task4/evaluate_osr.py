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
from sklearn.metrics import roc_auc_score, roc_curve

from scores.definitions import (
    get_msp_score, get_mls_score, get_energy_score,
    compute_mahalanobis_params, get_mahalanobis_score, get_proser_score
)
from data.cifar100_unknowns import get_cifar100_unknowns

# ── Class name mappings for human-readable failure analysis ──────────────────

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

CIFAR100_CLASSES = [
    "apple", "aquarium_fish", "baby", "bear", "beaver",
    "bed", "bee", "beetle", "bicycle", "bottle",
    "bowl", "boy", "bridge", "bus", "butterfly",
    "camel", "can", "castle", "caterpillar", "cattle",
    "chair", "chimpanzee", "clock", "cloud", "cockroach",
    "couch", "crab", "crocodile", "cup", "dinosaur",
    "dolphin", "elephant", "flatfish", "forest", "fox",
    "girl", "hamster", "house", "kangaroo", "keyboard",
    "lamp", "lawn_mower", "leopard", "lion", "lizard",
    "lobster", "man", "maple_tree", "motorcycle", "mountain",
    "mouse", "mushroom", "oak_tree", "orange", "orchid",
    "otter", "palm_tree", "pear", "pickup_truck", "pine_tree",
    "plain", "plate", "poppy", "porcupine", "possum",
    "rabbit", "raccoon", "ray", "road", "rocket",
    "rose", "sea", "seal", "shark", "shrew",
    "skunk", "skyscraper", "snail", "snake", "spider",
    "squirrel", "streetcar", "sunflower", "sweet_pepper", "table",
    "tank", "telephone", "television", "tiger", "tractor",
    "train", "trout", "tulip", "turtle", "wardrobe",
    "whale", "willow_tree", "wolf", "woman", "worm",
]


# ── Score extraction helper ──────────────────────────────────────────────────

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


# ── Metric computation ───────────────────────────────────────────────────────

def compute_metrics(known_scores, unknown_scores, threshold):
    y_true = np.concatenate([np.zeros(len(known_scores)), np.ones(len(unknown_scores))])
    y_scores = np.concatenate([known_scores.numpy(), unknown_scores.numpy()])
    auroc = roc_auc_score(y_true, y_scores)
    fpr95 = (unknown_scores <= threshold).float().mean().item()
    return {"auroc": auroc, "fpr95": fpr95}


# ── Failure analysis with human-readable class names and image grid ──────────

def save_failures(data, scores, threshold, group_name, out_dir, dataset):
    """Save the first 3 incorrectly accepted unknowns as a JSON and an Image Grid."""
    accepted_mask = (scores <= threshold)
    accepted_indices = accepted_mask.nonzero(as_tuple=True)[0]

    failures = []
    failed_imgs = []
    
    # Normalization parameters used in cifar10.py
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
    std = torch.tensor([0.2023, 0.1994, 0.2010]).view(3, 1, 1)

    for idx in accepted_indices[:3]:
        pred_class_id = data["logits"][idx, :10].argmax().item()
        true_class_id = data["labels"][idx].item()
        score = scores[idx].item()

        # Un-normalize image for plotting
        img_tensor = dataset[idx.item()][0]
        img = img_tensor * std + mean
        img = torch.clamp(img, 0, 1)
        failed_imgs.append(img)

        predicted_known_class = CIFAR10_CLASSES[pred_class_id]
        unknown_class = (CIFAR100_CLASSES[true_class_id]
                         if true_class_id < len(CIFAR100_CLASSES)
                         else f"cifar100_class_{true_class_id}")

        failures.append({
            "unknown_class": unknown_class,
            "predicted_known_class": predicted_known_class,
            "score": score,
            "threshold": threshold,
        })

    # Save JSON
    path_json = os.path.join(out_dir, f"failures_vanilla_mls_{group_name}.json")
    with open(path_json, "w") as f:
        json.dump(failures, f, indent=2)

    # Save Image Grid
    if len(failed_imgs) > 0:
        fig, axes = plt.subplots(1, len(failed_imgs), figsize=(3.5 * len(failed_imgs), 3.5))
        if len(failed_imgs) == 1: axes = [axes]
        
        for i, ax in enumerate(axes):
            ax.imshow(failed_imgs[i].permute(1, 2, 0).numpy())
            f = failures[i]
            title = f"True: {f['unknown_class']}\nPred: {f['predicted_known_class']}\nScore: {f['score']:.2f}"
            ax.set_title(title, fontsize=10)
            ax.axis('off')
            
        plt.tight_layout()
        path_img = os.path.join(out_dir, f"failures_vanilla_mls_{group_name}.png")
        plt.savefig(path_img, dpi=150)
        plt.close()

    print(f"  [failures] Saved {len(failures)} {group_name} failures (JSON & PNG)")
    sys.stdout.flush()


# ── Main evaluation ──────────────────────────────────────────────────────────

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

        # Compute Mahalanobis params from unaugmented training features
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

            # Threshold = 95th percentile of unknownness on CIFAR-10 validation
            threshold = torch.quantile(val_s, 0.95).item()

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

            # Failure analysis: vanilla MLS only (per manual)
            if method == "vanilla" and st == "mls":
                save_failures(outputs["near"], near_s, threshold, "near", args.results_dir, cifar100["near"])
                save_failures(outputs["far"], far_s, threshold, "far", args.results_dir, cifar100["far"])

    # ── TABLE 1: Post-hoc scores on frozen Vanilla model ─────────────────
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

    # ── TABLE 2: Model comparison ────────────────────────────────────────
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

    # ── Plots (Distributions & ROC) ───────────────────────────────────────
    if "vanilla" in results:
        cache_path = os.path.join(args.cache_dir, "vanilla_outputs.pt")
        outputs = torch.load(cache_path)
        means, var = vanilla_means, vanilla_var

        # 1. Score Distributions
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

        # 2. ROC Curves
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

    # ── Save all results to JSON ─────────────────────────────────────────
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
