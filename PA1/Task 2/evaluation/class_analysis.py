"""
evaluation/class_analysis.py
=============================
Per-class accuracy analysis on the target (Sketch) domain.

Produces, for each method:
* Per-class accuracy compared to source-only baseline.
* Classes with the largest improvement and degradation.
* Dominant confusion for each class (which source class is predicted most).
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from shared.pacs import PACS_CLASSES

NUM_CLASSES  = len(PACS_CLASSES)


# ─── prediction collection ────────────────────────────────────────────────────

@torch.no_grad()
def collect_predictions(
    backbone: nn.Module,
    head:     nn.Module,
    loader:   DataLoader,
    device:   torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (predictions, true_labels) as integer numpy arrays.
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

    return np.concatenate(all_preds), np.concatenate(all_labels)


# ─── per-class accuracy ───────────────────────────────────────────────────────

def per_class_accuracy(
    preds:  np.ndarray,
    labels: np.ndarray,
    num_classes: int = NUM_CLASSES,
) -> Dict[str, float]:
    """
    Compute per-class accuracy.

    Returns
    -------
    dict: {class_name: accuracy}
    """
    result: Dict[str, float] = {}
    for c, cname in enumerate(PACS_CLASSES):
        mask = labels == c
        if mask.sum() == 0:
            result[cname] = float("nan")
        else:
            result[cname] = float((preds[mask] == c).mean())
    return result


# ─── confusion matrix and dominant confusions ─────────────────────────────────

def confusion_matrix(
    preds:  np.ndarray,
    labels: np.ndarray,
    num_classes: int = NUM_CLASSES,
) -> np.ndarray:
    """Return confusion matrix of shape (num_classes, num_classes)."""
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(labels, preds):
        cm[int(t), int(p)] += 1
    return cm


def dominant_confusions(
    cm: np.ndarray,
    top_k: int = 3,
) -> Dict[str, List[Tuple[str, int]]]:
    """
    For each true class, return the top-k most predicted other classes.

    Returns
    -------
    dict: {true_class_name: [(predicted_class_name, count), ...]}
    """
    result: Dict[str, List[Tuple[str, int]]] = {}
    for c, cname in enumerate(PACS_CLASSES):
        row    = cm[c].copy()
        row[c] = 0                                   # exclude correct predictions
        order  = np.argsort(row)[::-1][:top_k]
        result[cname] = [(PACS_CLASSES[j], int(row[j])) for j in order if row[j] > 0]
    return result


# ─── comparison vs baseline ───────────────────────────────────────────────────

def compare_to_baseline(
    method_acc:   Dict[str, float],
    baseline_acc: Dict[str, float],
    top_k: int = 3,
) -> Dict:
    """
    Identify the classes with the largest improvement and degradation
    relative to the source-only baseline.

    Returns
    -------
    dict with:
        'deltas':      {class_name: delta_accuracy}
        'top_gains':   [(class_name, delta), ...]   # top_k largest gains
        'top_losses':  [(class_name, delta), ...]   # top_k largest drops
    """
    deltas: Dict[str, float] = {}
    for cname in PACS_CLASSES:
        ma = method_acc.get(cname, float("nan"))
        ba = baseline_acc.get(cname, float("nan"))
        if np.isnan(ma) or np.isnan(ba):
            deltas[cname] = float("nan")
        else:
            deltas[cname] = ma - ba

    valid_deltas = {k: v for k, v in deltas.items() if not np.isnan(v)}
    sorted_by_delta = sorted(valid_deltas.items(), key=lambda kv: kv[1], reverse=True)

    return {
        "deltas":     deltas,
        "top_gains":  sorted_by_delta[:top_k],
        "top_losses": sorted_by_delta[-top_k:],
    }


# ─── full analysis ────────────────────────────────────────────────────────────

def full_class_analysis(
    backbone:    nn.Module,
    head:        nn.Module,
    target_loader: DataLoader,
    device:      torch.device,
    baseline_per_class: Optional[Dict[str, float]] = None,
) -> Dict:
    """
    Run the complete per-class analysis for one method.

    Parameters
    ----------
    baseline_per_class:
        Per-class accuracy of the source-only model (for delta comparison).
        If None, deltas are not computed.

    Returns
    -------
    dict with keys: 'per_class_accuracy', 'confusion_matrix',
                    'dominant_confusions', 'comparison' (if baseline given)
    """
    preds, labels = collect_predictions(backbone, head, target_loader, device)

    pc_acc    = per_class_accuracy(preds, labels)
    cm        = confusion_matrix(preds, labels)
    dom_conf  = dominant_confusions(cm)

    out: Dict = {
        "per_class_accuracy":   pc_acc,
        "confusion_matrix":     cm.tolist(),
        "dominant_confusions":  dom_conf,
    }

    if baseline_per_class is not None:
        out["comparison"] = compare_to_baseline(pc_acc, baseline_per_class)

    return out


def print_class_analysis(analysis: Dict) -> None:
    """Pretty-print per-class accuracy and top changes."""
    print(f"\n{'Class':<12} {'Accuracy':>10}")
    print("-" * 24)
    for cname, acc in analysis["per_class_accuracy"].items():
        mark = ""
        if "comparison" in analysis:
            delta = analysis["comparison"]["deltas"].get(cname, 0.0)
            if not np.isnan(delta):
                mark = f"  ({delta:+.3f})"
        print(f"  {cname:<10} {acc:>10.4f}{mark}")

    if "comparison" in analysis:
        comp = analysis["comparison"]
        print("\n  Top gains :")
        for cname, d in comp["top_gains"]:
            print(f"    {cname}: {d:+.3f}")
        print("  Top losses:")
        for cname, d in comp["top_losses"]:
            print(f"    {cname}: {d:+.3f}")
