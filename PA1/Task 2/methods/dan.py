"""
methods/dan.py
==============
Step 2: DAN – Maximum Mean Discrepancy (MMD) domain alignment.

Loss
----
    L_DAN = L_cls + lambda_mmd * MMD²(F(x^s), F(x^t))

MMD is applied to the 512-dim backbone feature before the classifier head.

Kernel
------
Sum of three RBF kernels whose bandwidths are 0.5×, 1×, and 2× the median
pairwise squared feature distance of the current combined batch (kernel trick
— no explicit feature map is computed).

    k(x, y) = Σ_σ  exp(-||x-y||² / (2σ²))

    MMD² = E_ss[k] - 2·E_st[k] + E_tt[k]
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

from .source_only import _freeze_bn


# ─── MMD computation ─────────────────────────────────────────────────────────

def _pairwise_sq_dist(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Compute pairwise squared Euclidean distances between rows of *a* and *b*.

    Returns
    -------
    Tensor of shape (n, m) where n=len(a), m=len(b).
    """
    # ||a - b||² = ||a||² + ||b||² - 2 a·bᵀ
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
    distance in the combined batch.

    If the median is zero (constant features), fall back to 1.0.
    """
    combined = torch.cat([src_feats.detach(), tgt_feats.detach()], dim=0)
    n = combined.size(0)

    # Upper-triangular pairwise distances (excluding diagonal)
    dists = _pairwise_sq_dist(combined, combined)
    # Indices of upper triangle
    rows, cols = torch.triu_indices(n, n, offset=1, device=combined.device)
    upper = dists[rows, cols]

    median_sq = float(upper.median().item()) if upper.numel() > 0 else 1.0
    import math
    if median_sq <= 0.0 or math.isnan(median_sq) or math.isinf(median_sq):
        median_sq = 1.0

    return [mul * median_sq for mul in kernel_muls]


def _rbf_kernel(a: torch.Tensor, b: torch.Tensor, sigmas: List[float]) -> torch.Tensor:
    """
    Multi-scale RBF kernel matrix K[i,j] = Σ_σ exp(-||a_i - b_j||² / (2σ²)).
    """
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
    """
    Biased MMD² estimate using three adaptive RBF kernels.

    Parameters
    ----------
    src_feats:   (n, D) source features (in computation graph)
    tgt_feats:   (m, D) target features (in computation graph)
    kernel_muls: bandwidth multipliers relative to median sq dist

    Returns
    -------
    Scalar tensor (differentiable w.r.t. src_feats and tgt_feats).
    """
    sigmas   = _adaptive_sigmas(src_feats, tgt_feats, kernel_muls)
    K_ss     = _rbf_kernel(src_feats, src_feats, sigmas)
    K_tt     = _rbf_kernel(tgt_feats, tgt_feats, sigmas)
    K_st     = _rbf_kernel(src_feats, tgt_feats, sigmas)

    mmd2 = K_ss.mean() - 2.0 * K_st.mean() + K_tt.mean()
    return mmd2.clamp(min=0.0)


# ─── DAN method ──────────────────────────────────────────────────────────────

class DAN:
    """
    DAN: Classification loss + MMD alignment on the 512-dim feature.

    Parameters
    ----------
    backbone, head, device, lr, weight_decay:
        Same as SourceOnly.
    lambda_mmd:
        MMD penalty weight (1.0 for the main comparison).
    kernel_muls:
        Bandwidth multipliers for the three RBF kernels.
    """

    def __init__(
        self,
        backbone:     nn.Module,
        head:         nn.Module,
        device:       torch.device,
        lr:           float = 1e-4,
        weight_decay: float = 1e-4,
        lambda_mmd:   float = 1.0,
        kernel_muls:  List[float] = (0.5, 1.0, 2.0),
    ) -> None:
        self.backbone    = backbone.to(device)
        self.head        = head.to(device)
        self.device      = device
        self.lambda_mmd  = lambda_mmd
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
        self,
        src_batches: List[Tuple[torch.Tensor, torch.Tensor]],
        tgt_batch:   Tuple[torch.Tensor, torch.Tensor],
        p:           float,
    ) -> Dict[str, float]:
        self.backbone.train()
        self.head.train()
        _freeze_bn(self.backbone)

        src_imgs   = torch.cat([b[0] for b in src_batches], dim=0).to(self.device)
        src_labels = torch.cat([b[1] for b in src_batches], dim=0).to(self.device)
        tgt_imgs   = tgt_batch[0].to(self.device)

        # ── Forward ──────────────────────────────────────────────────────
        src_feats = self.backbone(src_imgs)    # (n_src, 512)
        tgt_feats = self.backbone(tgt_imgs)    # (n_tgt, 512)
        logits    = self.head(src_feats)       # (n_src, 7)

        cls_loss  = F.cross_entropy(logits, src_labels)
        src_feats_norm = F.normalize(src_feats, dim=1)   # ← add
        tgt_feats_norm = F.normalize(tgt_feats, dim=1)   # ← add
        mmd_loss = compute_mmd2(src_feats_norm, tgt_feats_norm, self.kernel_muls)
        total     = cls_loss + self.lambda_mmd * mmd_loss

        # ── Backward ─────────────────────────────────────────────────────
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
