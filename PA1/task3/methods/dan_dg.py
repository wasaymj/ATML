"""
methods/dan_dg.py
=================
DAN-DG: Pairwise MMD alignment across source domains for Domain Generalization.

Loss
----
    L = L_cls + lambda_dg * (1/3) * Σ_{i<j} MMD²(F(x_i), F(x_j))

MMD kernel and bandwidth computation are copied directly from Task 2's
proven dan.py implementation to ensure mathematical correctness.
"""
from __future__ import annotations

import math
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW


# ─── BN freeze (same as Task 2) ─────────────────────────────────────────────

def _freeze_bn(model: nn.Module) -> None:
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.eval()


# ─── MMD computation (copied from Task 2 dan.py, proven correct) ─────────────

def _pairwise_sq_dist(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Compute pairwise squared Euclidean distances between rows of a and b.
    Returns Tensor of shape (n, m).
    """
    a_sq = (a ** 2).sum(dim=1, keepdim=True)   # (n, 1)
    b_sq = (b ** 2).sum(dim=1, keepdim=True)   # (m, 1)
    return (a_sq + b_sq.t() - 2.0 * a @ b.t()).clamp(min=0.0)


def _adaptive_sigmas(
    src_feats: torch.Tensor,
    tgt_feats: torch.Tensor,
    kernel_muls: List[float] = (0.5, 1.0, 2.0),
) -> List[float]:
    """
    Compute RBF bandwidths as {0.5, 1, 2} × median pairwise squared
    distance in the combined batch.  Uses upper-triangular distances only
    (excludes diagonal zeros that would drag the median down).
    """
    combined = torch.cat([src_feats.detach(), tgt_feats.detach()], dim=0)
    n = combined.size(0)

    dists = _pairwise_sq_dist(combined, combined)
    rows, cols = torch.triu_indices(n, n, offset=1, device=combined.device)
    upper = dists[rows, cols]

    median_sq = float(upper.median().item()) if upper.numel() > 0 else 1.0
    if median_sq <= 0.0 or math.isnan(median_sq) or math.isinf(median_sq):
        median_sq = 1.0

    return [mul * median_sq for mul in kernel_muls]


def _rbf_kernel(a: torch.Tensor, b: torch.Tensor, sigmas: List[float]) -> torch.Tensor:
    """Multi-scale RBF kernel: K[i,j] = Σ_σ exp(-||a_i - b_j||² / σ)."""
    dists = _pairwise_sq_dist(a, b)
    K = torch.zeros_like(dists)
    for sigma in sigmas:
        K += torch.exp(-dists / sigma)
    return K


def compute_mmd2(
    src_feats:   torch.Tensor,
    tgt_feats:   torch.Tensor,
    kernel_muls: List[float] = (0.5, 1.0, 2.0),
) -> torch.Tensor:
    """Biased MMD² estimate using three adaptive RBF kernels."""
    sigmas = _adaptive_sigmas(src_feats, tgt_feats, kernel_muls)
    K_ss   = _rbf_kernel(src_feats, src_feats, sigmas)
    K_tt   = _rbf_kernel(tgt_feats, tgt_feats, sigmas)
    K_st   = _rbf_kernel(src_feats, tgt_feats, sigmas)

    mmd2 = K_ss.mean() - 2.0 * K_st.mean() + K_tt.mean()
    return mmd2.clamp(min=0.0)


# ─── DAN-DG method ──────────────────────────────────────────────────────────

class DANDGMethod:
    """
    DAN-DG: Domain Alignment Network for Domain Generalization.
    Aligns the 3 observed source domains pairwise using MMD.
    """

    def __init__(
        self,
        backbone: nn.Module,
        head: nn.Module,
        device: torch.device,
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        lambda_dg: float = 1.0,
        kernel_muls: List[float] = (0.5, 1.0, 2.0),
    ) -> None:
        self.backbone = backbone.to(device)
        self.head = head.to(device)
        self.device = device
        self.lambda_dg = lambda_dg
        self.kernel_muls = list(kernel_muls)

        self.optimizer = AdamW(
            list(self.backbone.parameters()) + list(self.head.parameters()),
            lr=lr,
            weight_decay=weight_decay,
        )
        self.history: Dict[str, List[float]] = {
            "cls_loss": [], "mmd_loss": [], "total_loss": []
        }

    def train_step(
        self, src_batches: List[Tuple[torch.Tensor, torch.Tensor]]
    ) -> Dict[str, float]:
        self.backbone.train()
        self.head.train()
        _freeze_bn(self.backbone)

        # Forward each source domain separately to get per-domain features
        feats_list = []
        labels_list = []
        for img, lbl in src_batches:
            feats_list.append(self.backbone(img.to(self.device)))
            labels_list.append(lbl.to(self.device))

        # Classification on all source samples combined
        src_feats = torch.cat(feats_list, dim=0)
        src_labels = torch.cat(labels_list, dim=0)
        logits = self.head(src_feats)
        cls_loss = F.cross_entropy(logits, src_labels)

        # Pairwise MMD across source domains
        # L2-normalize features before MMD to stabilize gradient magnitudes
        # (same fix applied to Task 2's DAN for gradient explosion prevention)
        mmd_loss = torch.tensor(0.0, device=self.device)
        pairs = [(0, 1), (0, 2), (1, 2)]
        for i, j in pairs:
            fi = F.normalize(feats_list[i], dim=1)
            fj = F.normalize(feats_list[j], dim=1)
            mmd_loss = mmd_loss + compute_mmd2(fi, fj, self.kernel_muls)
        mmd_loss = mmd_loss / len(pairs)

        total = cls_loss + self.lambda_dg * mmd_loss

        # Backward
        self.optimizer.zero_grad()
        total.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.backbone.parameters()) + list(self.head.parameters()),
            max_norm=1.0,
        )
        self.optimizer.step()

        return {
            "cls_loss":   cls_loss.item(),
            "mmd_loss":   mmd_loss.item(),
            "total_loss": total.item(),
        }

    def state_dict(self) -> Dict:
        return {
            "backbone": self.backbone.state_dict(),
            "head":     self.head.state_dict(),
        }

    def load_state_dict(self, sd: Dict) -> None:
        self.backbone.load_state_dict(sd["backbone"])
        self.head.load_state_dict(sd["head"])
