"""
evaluation/domain_separability.py
==================================
Measures residual domain information in learned representations via a
logistic-regression probe.

Protocol (Task 2, Step 5 — exactly as specified in the assignment)
------------------------------------------------------------------
1. Freeze backbone; collect features from source-validation split and
   from the target (Sketch) domain.
2. Take equal numbers from source and target
   (min(n_source_val, n_target) samples each).
3. Create a 70/30 stratified split using seed 6304.
4. Train a balanced LogisticRegression with C=1 on the 70% split.
5. Report held-out accuracy as the domain separability score.
   50% = chance performance (domains indistinguishable).

A lower score indicates less domain information in the representation,
but does NOT by itself imply that class information has been preserved.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader

SEED = 6304


@torch.no_grad()
def _extract_features(
    backbone: nn.Module,
    loader:   DataLoader,
    device:   torch.device,
    max_samples: Optional[int] = None,
) -> np.ndarray:
    """Extract backbone features for all examples in *loader*."""
    backbone.eval()
    feats: List[np.ndarray] = []

    for images, _ in loader:
        images = images.to(device, non_blocking=True)
        f = backbone(images).cpu().numpy()
        feats.append(f)
        if max_samples is not None and sum(len(x) for x in feats) >= max_samples:
            break

    arr = np.concatenate(feats, axis=0)
    if max_samples is not None:
        arr = arr[:max_samples]
    return arr


def compute_domain_separability(
    backbone:        nn.Module,
    source_val_loaders: Dict[str, DataLoader],   # {domain: val_loader}
    target_eval_loader: DataLoader,
    device:          torch.device,
) -> Dict[str, float]:
    """
    Compute the domain separability score (source vs target).

    Parameters
    ----------
    backbone:
        Feature extractor (frozen during this call).
    source_val_loaders:
        Validation loaders for all source domains.
        Features are concatenated across all source domains.
    target_eval_loader:
        Full target (Sketch) loader.

    Returns
    -------
    dict with keys:
        'domain_separability': float  (held-out accuracy, 0.5 = chance)
        'n_source':            int
        'n_target':            int
        'n_train':             int
        'n_test':              int
    """
    # ── Extract source-val features ───────────────────────────────────────
    src_feats_list: List[np.ndarray] = []
    for loader in source_val_loaders.values():
        src_feats_list.append(_extract_features(backbone, loader, device))
    src_feats = np.concatenate(src_feats_list, axis=0)

    # ── Extract target features ───────────────────────────────────────────
    tgt_feats = _extract_features(backbone, target_eval_loader, device)

    # ── Balance: equal numbers from source and target ─────────────────────
    n_equal  = min(len(src_feats), len(tgt_feats))
    rng      = np.random.default_rng(SEED)
    src_idx  = rng.choice(len(src_feats), n_equal, replace=False)
    tgt_idx  = rng.choice(len(tgt_feats), n_equal, replace=False)
    src_feats = src_feats[src_idx]
    tgt_feats = tgt_feats[tgt_idx]

    # ── Build binary classification dataset ───────────────────────────────
    X = np.concatenate([src_feats, tgt_feats], axis=0).astype(np.float32)
    y = np.array([0] * n_equal + [1] * n_equal, dtype=np.int64)  # 0=src, 1=tgt

    # ── 70/30 stratified split, seed 6304 ────────────────────────────────
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.30, random_state=SEED)
    train_idx, test_idx = next(sss.split(X, y))

    # ── Train LogisticRegression(C=1) ─────────────────────────────────────
    clf = LogisticRegression(C=1.0, max_iter=1000, random_state=SEED, solver="lbfgs")
    clf.fit(X[train_idx], y[train_idx])
    score = float(clf.score(X[test_idx], y[test_idx]))

    return {
        "domain_separability": score,
        "n_source":            n_equal,
        "n_target":            n_equal,
        "n_train":             len(train_idx),
        "n_test":              len(test_idx),
    }
