import json
import os
import torch
from typing import Dict
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

def evaluate_target(backbone, head, target_loader, device):
    backbone.eval()
    head.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in target_loader:
            imgs = imgs.to(device)
            feats = backbone(imgs)
            logits = head(feats)
            preds = logits.argmax(dim=1).cpu()
            all_preds.append(preds)
            all_labels.append(labels)
            
    preds_t = torch.cat(all_preds).numpy()
    labels_t = torch.cat(all_labels).numpy()
    
    acc = (preds_t == labels_t).mean()
    macro_f1 = f1_score(labels_t, preds_t, average="macro", zero_division=0)
    
    return {"accuracy": float(acc), "macro_f1": float(macro_f1)}

def print_comparison_table(results: Dict):
    print("
" + "="*110)
    print("TASK 3 FINAL COMPARISON TABLE")
    print("="*110)
    headers = ["Method", "Photo", "Art", "Cartoon", "Mean Src", "Target Acc", "Target F1", "Delta Acc", "Domain Sep", "Sharpness"]
    row_fmt = "{:<12} | {:<7} | {:<7} | {:<7} | {:<8} | {:<10} | {:<10} | {:<9} | {:<10} | {:<10}"
    print(row_fmt.format(*headers))
    print("-" * 110)
    
    baseline_acc = results.get("erm", {}).get("target", {}).get("accuracy", 0.0)
    
    for method, metrics in results.items():
        if "target" not in metrics: continue
        if method == "controlled_study": continue
        
        src_val = metrics.get("source_val", {})
        photo_acc = src_val.get("photo", {}).get("accuracy", 0.0)
        art_acc = src_val.get("art_painting", {}).get("accuracy", 0.0)
        cart_acc = src_val.get("cartoon", {}).get("accuracy", 0.0)
        mean_acc = src_val.get("mean", {}).get("accuracy", 0.0)
        
        tgt = metrics["target"]
        delta = tgt["accuracy"] - baseline_acc
        
        sep = metrics.get("domain_separability", 0.0)
        sharp = metrics.get("sharpness", 0.0)
        
        print(row_fmt.format(
            method,
            f"{photo_acc:.4f}",
            f"{art_acc:.4f}",
            f"{cart_acc:.4f}",
            f"{mean_acc:.4f}",
            f"{tgt['accuracy']:.4f}",
            f"{tgt['macro_f1']:.4f}",
            f"{delta:+.4f}",
            f"{sep:.4f}",
            f"{sharp:.4f}"
        ))
    print("="*110)
    headers = ["Method", "Mean Src Acc", "Worst Src Acc", "Target Acc", "Target F1", "Δ Target Acc", "Domain Sep", "Sharpness"]
    row_fmt = "{:<12} | {:<12} | {:<13} | {:<10} | {:<10} | {:<12} | {:<10} | {:<10}"
    print(row_fmt.format(*headers))
    print("-" * 80)
    
    baseline_acc = results.get("erm", {}).get("target", {}).get("accuracy", 0.0)
    
    for method, metrics in results.items():
        if "target" not in metrics: continue
        
        src_val = metrics.get("source_val", {})
        mean_acc = src_val.get("mean", {}).get("accuracy", 0.0)
        worst_dom = src_val.get("worst", {}).get("domain", "")
        worst_acc = src_val.get(worst_dom, {}).get("accuracy", 0.0) if worst_dom else 0.0
        
        tgt = metrics["target"]
        delta = tgt["accuracy"] - baseline_acc
        
        sep = metrics.get("domain_separability", 0.0)
        sharp = metrics.get("sharpness", 0.0)
        
        print(row_fmt.format(
            method,
            f"{mean_acc:.4f}",
            f"{worst_acc:.4f}",
            f"{tgt['accuracy']:.4f}",
            f"{tgt['macro_f1']:.4f}",
            f"{delta:+.4f}",
            f"{sep:.4f}",
            f"{sharp:.4f}"
        ))
    print("="*80)
