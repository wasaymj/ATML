"""
methods/source_only.py
=======================
Step 1: Source-Only ERM baseline.

Trains ResNet-18 + 7-class head with cross-entropy over domain-balanced
source batches only (no target signal used at all).

This checkpoint is saved and reused unchanged as the Task 3 ERM baseline
— do NOT retrain it with different settings.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW


class SourceOnly:
    """
    Source-only ERM training step.

    Parameters
    ----------
    backbone:   BNFrozenResNet18 (or any nn.Module outputting (B, feat_dim))
    head:       ClassifierHead
    device:     torch.device
    lr:         AdamW learning rate
    weight_decay: AdamW weight decay
    """

    def __init__(
        self,
        backbone:     nn.Module,
        head:         nn.Module,
        device:       torch.device,
        lr:           float = 1e-4,
        weight_decay: float = 1e-4,
    ) -> None:
        self.backbone = backbone.to(device)
        self.head     = head.to(device)
        self.device   = device

        self.optimizer = AdamW(
            list(self.backbone.parameters()) + list(self.head.parameters()),
            lr=lr,
            weight_decay=weight_decay,
        )

        # Training history (lists of per-epoch mean values)
        self.history: Dict[str, List[float]] = {"cls_loss": [], "total_loss": []}

    # ── single training step ──────────────────────────────────────────────────

    def train_step(
        self,
        src_batches: List[Tuple[torch.Tensor, torch.Tensor]],
        tgt_batch:   Tuple[torch.Tensor, torch.Tensor],   # labels ignored
        p:           float,                                # training progress [0,1]
    ) -> Dict[str, float]:
        """
        One gradient update.

        Parameters
        ----------
        src_batches: list of (images, labels) tensors, one per source domain.
        tgt_batch:   (images, labels=-1)  — target images ignored by source-only.
        p:           training progress in [0,1] (unused here; kept for API consistency).

        Returns
        -------
        dict of scalar loss values (detached from graph).
        """
        self.backbone.train()
        self.head.train()
        # Enforce BN freeze (backbone.train() might be overridden, but call explicitly)
        _freeze_bn(self.backbone)

        # ── concatenate all source batches ───────────────────────────────
        src_imgs   = torch.cat([b[0] for b in src_batches], dim=0).to(self.device)
        src_labels = torch.cat([b[1] for b in src_batches], dim=0).to(self.device)

        # ── forward ──────────────────────────────────────────────────────
        feats  = self.backbone(src_imgs)
        logits = self.head(feats)
        loss   = F.cross_entropy(logits, src_labels)

        # ── backward ─────────────────────────────────────────────────────
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.backbone.parameters()) + list(self.head.parameters()),
            max_norm=1.0,
        )
        self.optimizer.step()

        return {"cls_loss": loss.item(), "total_loss": loss.item()}

    # ── state dict helpers ────────────────────────────────────────────────────

    def state_dict(self) -> Dict:
        return {
            "backbone": self.backbone.state_dict(),
            "head":     self.head.state_dict(),
        }

    def load_state_dict(self, sd: Dict) -> None:
        self.backbone.load_state_dict(sd["backbone"])
        self.head.load_state_dict(sd["head"])


# ─── shared BN freeze helper ─────────────────────────────────────────────────

def _freeze_bn(model: nn.Module) -> None:
    """Set all BN modules to eval() to prevent running-stat updates."""
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.eval()
