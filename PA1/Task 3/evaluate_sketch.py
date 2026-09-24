import argparse
import json
import os
import torch
import yaml

from shared.pacs_protocol import build_source_loaders, build_target_loader
from shared.pacs_protocol import get_or_create_splits
from models.backbone import BNFrozenResNet18
from models.classifier_head import ClassifierHead
from evaluation.domain_metrics import evaluate_target, print_comparison_table
from evaluation.source_domain_separability import compute_domain_separability
from evaluation.sharpness import compute_sharpness
from selection.source_validation import evaluate_source_domains
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix as sk_cm
import numpy as np

PACS_CLASSES = ['dog','elephant','giraffe','guitar','horse','house','person']

@torch.no_grad()
def per_class_target(backbone, head, loader, device):
    backbone.eval(); head.eval()
    all_p, all_l = [], []
    for imgs, lbls in loader:
        logits = head(backbone(imgs.to(device)))
        all_p.append(logits.argmax(1).cpu().numpy())
        all_l.append(lbls.numpy())
    preds = np.concatenate(all_p); labels = np.concatenate(all_l)
    per_cls = {}
    for c, name in enumerate(PACS_CLASSES):
        mask = labels == c
        per_cls[name] = float((preds[mask] == c).mean()) if mask.sum() > 0 else float('nan')
    return per_cls, preds, labels

METHODS = ["erm", "dan_dg", "sam"]

def get_checkpoint_path(method, cfg):
    if method == "erm":
        # Load from Task 2 checkpoints directly
        return os.path.join(cfg["task2_checkpoints_dir"], "source_only_best.pth")
    else:
        return os.path.join(cfg["checkpoints_dir"], f"{method}_best.pth")

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[eval] Using device: {device}")

    # Load splits and datasets
    from shared.pacs_protocol import get_train_transform, get_eval_transform, build_source_datasets, build_target_dataset, download_pacs
    pacs_root = download_pacs(args.data_root)
    splits = get_or_create_splits(args.splits_path, pacs_root)
    
    train_tf = get_train_transform()
    eval_tf = get_eval_transform()
    source_datasets = build_source_datasets(pacs_root, splits, train_tf, eval_tf)
    source_loaders = build_source_loaders(source_datasets, batch_size_per_source=8, num_workers=4)
    val_loaders = {d: source_loaders[d]["val"] for d in source_loaders}
    
    target_ds = build_target_dataset(pacs_root, eval_tf, return_labels=True)
    target_loader = build_target_loader(target_ds, batch_size=64, shuffle=False, num_workers=4, drop_last=False)
    
    cfg = {
        "checkpoints_dir": args.checkpoints_dir,
        "task2_checkpoints_dir": args.task2_checkpoints_dir,
        "results_dir": args.results_dir,
        "num_classes": 7
    }
    
    os.makedirs(cfg["results_dir"], exist_ok=True)
    
    all_results = {}
    
    backbone = BNFrozenResNet18().to(device)
    head = ClassifierHead(feat_dim=512, num_classes=cfg["num_classes"]).to(device)
    
    # 1. Main Methods
    for method in METHODS:
        ckpt_path = get_checkpoint_path(method, cfg)
        if not os.path.isfile(ckpt_path):
            print(f"  [!] Missing checkpoint for {method} at {ckpt_path}. Skipping.")
            continue
            
        print(f"\n[eval] Evaluating {method}...")
        ckpt = torch.load(ckpt_path, map_location=device)
        
        # Determine state dict structure
        if "state_dict" in ckpt:
            sd = ckpt["state_dict"]
            backbone.load_state_dict(sd["backbone"])
            head.load_state_dict(sd["head"])

            
        source_val_results = evaluate_source_domains(backbone, head, val_loaders, device)
        
        # Target Evaluation
        tgt_metrics = evaluate_target(backbone, head, target_loader, device)
        per_class_acc, preds, labels = per_class_target(backbone, head, target_loader, device)
        tgt_metrics['per_class'] = per_class_acc
        
        if method == "erm":
            cm = sk_cm(labels, preds)
            plt.figure(figsize=(8,6))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=PACS_CLASSES, yticklabels=PACS_CLASSES)
            plt.title("ERM (Source-Only) - Sketch Confusion Matrix")
            plt.ylabel("True Class")
            plt.xlabel("Predicted Class")
            plt.tight_layout()
            cm_path = os.path.join(cfg["results_dir"], "erm_confusion_matrix.png")
            plt.savefig(cm_path, dpi=150)
            plt.close()
            print(f"  -> ERM Confusion matrix saved to {cm_path}")

        
        # Domain Separability
        dom_sep = compute_domain_separability(backbone, val_loaders, device, seed=6304)
        
        # Sharpness
        sharpness = compute_sharpness(backbone, head, val_loaders, device, rho=0.05, seed=6304)
        
        all_results[method] = {
            "source_val": source_val_results,
            "target": tgt_metrics,
            "domain_separability": dom_sep,
            "sharpness": sharpness
        }
        
    # 2. Controlled Study (lambda_dg)
    study_results = {}
    lambdas = ["0.1", "1.0", "10.0"]
    print("\n[eval] Evaluating Controlled Study (lambda_dg)...")
    for lam in lambdas:
        method = f"dan_dg_lambda_dg{lam}"
        ckpt_path = os.path.join(cfg["checkpoints_dir"], f"{method}_best.pth")
        if not os.path.isfile(ckpt_path):
            continue
        ckpt = torch.load(ckpt_path, map_location=device)
        if "state_dict" in ckpt:
            sd = ckpt["state_dict"]
            backbone.load_state_dict(sd["backbone"])
            head.load_state_dict(sd["head"])
        
        source_val_results = evaluate_source_domains(backbone, head, val_loaders, device)
        tgt_metrics = evaluate_target(backbone, head, target_loader, device)
        dom_sep = compute_domain_separability(backbone, val_loaders, device, seed=6304)
        
        study_results[method] = {
            "source_val": source_val_results,
            "target": tgt_metrics,
            "domain_separability": dom_sep
        }
    
    all_results["controlled_study"] = study_results
        
    
    # Compute per-class deltas vs ERM
    if "erm" in all_results and "per_class" in all_results["erm"]["target"]:
        erm_pc = all_results["erm"]["target"]["per_class"]
        for method in METHODS:
            if method == "erm" or method not in all_results: continue
            
            pc = all_results[method]["target"].get("per_class", {})
            if not pc: continue
            
            deltas = {c: pc[c] - erm_pc.get(c, 0.0) for c in PACS_CLASSES if c in pc and c in erm_pc}
            all_results[method]["target"]["per_class_deltas_vs_erm"] = deltas
            
            sorted_deltas = sorted(deltas.items(), key=lambda x: x[1], reverse=True)
            top_3_gains = sorted_deltas[:3]
            top_3_losses = sorted_deltas[-3:]
            top_3_losses.reverse()
            
            print(f"\n--- {method} vs ERM Per-Class Deltas ---")
            print("Top 3 Gains:")
            for c, d in top_3_gains: print(f"  {c}: +{d:.4f}")
            print("Top 3 Losses:")
            for c, d in top_3_losses: print(f"  {c}: {d:.4f}")
            
            all_results[method]["target"]["top_3_gains"] = top_3_gains
            all_results[method]["target"]["top_3_losses"] = top_3_losses

    out_path = os.path.join(cfg["results_dir"], "final_evaluation.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
        
    print_comparison_table(all_results)
    
    if study_results:
        print("\n--- Controlled Study Results ---")
        lams, accs = [], []
        for k, v in study_results.items():
            print(f"{k}: Target Acc = {v['target']['accuracy']:.4f}, DomSep = {v['domain_separability']:.4f}")
            lams.append(k.replace("dan_dg_lambda_dg", ""))
            accs.append(v['target']['accuracy'])
            
        plt.figure(figsize=(6,4))
        plt.plot(lams, accs, marker='o', linestyle='-', color='b')
        plt.title("DAN-DG Controlled Study")
        plt.xlabel("Lambda DG")
        plt.ylabel("Sketch Accuracy")
        plt.grid(True)
        plot_path = os.path.join(cfg["results_dir"], "controlled_study_plot.png")
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"  -> Controlled study plot saved to {plot_path}")
            
    print(f"\nResults saved to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--splits_path", type=str, required=True)
    parser.add_argument("--checkpoints_dir", type=str, required=True)
    parser.add_argument("--task2_checkpoints_dir", type=str, required=True)
    parser.add_argument("--results_dir", type=str, required=True)
    args = parser.parse_args()
    main(args)
