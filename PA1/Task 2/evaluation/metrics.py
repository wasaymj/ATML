"""
evaluation/metrics.py
=====================
Shared evaluation helpers for Task 2 (and Task 3).
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader


@torch.no_grad()
def evaluate_loader(
    backbone: nn.Module,
    head:     nn.Module,
    loader:   DataLoader,
    device:   torch.device,
) -> Dict[str, float]:
    """
    Evaluate backbone + head on a DataLoader (labeled).

    Returns
    -------
    dict with keys: 'accuracy', 'macro_f1', 'n_samples'
    """
    backbone.eval()
    head.eval()

    all_preds:  List[np.ndarray] = []
    all_labels: List[np.ndarray] = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        feats  = backbone(images)
        logits = head(feats)
        preds  = logits.argmax(dim=1).cpu().numpy()
        all_preds.append(preds)
        all_labels.append(labels.numpy())

    preds_np  = np.concatenate(all_preds)
    labels_np = np.concatenate(all_labels)

    acc      = float((preds_np == labels_np).mean())
    macro_f1 = float(f1_score(labels_np, preds_np, average="macro", zero_division=0))

    return {"accuracy": acc, "macro_f1": macro_f1, "n_samples": len(labels_np)}


@torch.no_grad()
def evaluate_source_domains(
    backbone:      nn.Module,
    head:          nn.Module,
    val_loaders:   Dict[str, DataLoader],   # {domain: loader}
    device:        torch.device,
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate on each source validation domain.

    Returns
    -------
    dict: {domain: {'accuracy': float, 'macro_f1': float}} plus
          a 'mean' key with mean accuracy and mean macro_f1 across domains.
    """
    results: Dict[str, Dict[str, float]] = {}

    for domain, loader in val_loaders.items():
        results[domain] = evaluate_loader(backbone, head, loader, device)

    # Mean across source domains (for early stopping and reporting)
    mean_acc = float(np.mean([r["accuracy"]  for r in results.values()]))
    mean_f1  = float(np.mean([r["macro_f1"]  for r in results.values()]))
    results["mean"] = {"accuracy": mean_acc, "macro_f1": mean_f1}

    return results


def print_source_val_table(results: Dict[str, Dict[str, float]]) -> None:
    """Pretty-print source validation results."""
    header = f"{'Domain':<18} {'Accuracy':>10} {'Macro-F1':>10}"
    print(header)
    print("-" * len(header))
    for domain, r in results.items():
        if domain == "mean":
            continue
        print(f"  {domain:<16} {r['accuracy']:>10.4f} {r['macro_f1']:>10.4f}")
    m = results["mean"]
    print("-" * len(header))
    print(f"  {'Mean':<16} {m['accuracy']:>10.4f} {m['macro_f1']:>10.4f}")
