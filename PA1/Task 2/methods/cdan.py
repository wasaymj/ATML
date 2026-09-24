"""
methods/cdan.py
===============
Step 4: CDAN – Class-Conditional Adversarial Domain Adaptation.

Discriminator input
-------------------
    g(x) = vec(f ⊗ p)

where
    f = backbone feature,  shape (B, 512)
    p = softmax(classifier(f)), shape (B, 7)
    ⊗ = outer product per sample → (B, 512, 7) → flattened to (B, 3584)

As required by the assignment:
  * f and p are NOT detached from the computational graph.
  * No entropy conditioning.

Everything else (GRL schedule, discriminator architecture, loss weights,
domain labels, BN freeze) is identical to DANN.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

from models.domain_discriminator import (
    DomainDiscriminator,
    GradientReversal,
    cdan_input,
    grl_alpha,
)
from .source_only import _freeze_bn


class CDAN:
    """
    CDAN training step.

    Parameters
    ----------
    feat_dim:    Backbone feature dimension (512).
    num_classes: Number of source/target classes (7).
    All other parameters: same as DANN.
    """

    def __init__(
        self,
        backbone:      nn.Module,
        head:          nn.Module,
        device:        torch.device,
        lr:            float = 1e-4,
        weight_decay:  float = 1e-4,
        lambda_domain: float = 1.0,
        grl_max_alpha: float = 1.0,
        disc_hidden:   int   = 256,
        disc_dropout:  float = 0.5,
        feat_dim:      int   = 512,
        num_classes:   int   = 7,
    ) -> None:
        self.backbone      = backbone.to(device)
        self.head          = head.to(device)
        self.device        = device
        self.lambda_domain = lambda_domain
        self.grl_max_alpha = grl_max_alpha

        # CDAN input: feat_dim × num_classes = 3584
        cdan_in_dim = feat_dim * num_classes
        self.discriminator = DomainDiscriminator(
            in_dim=cdan_in_dim, hidden=disc_hidden, dropout=disc_dropout,
        ).to(device)
        self.grl = GradientReversal()

        self.optimizer = AdamW(
            list(self.backbone.parameters())
            + list(self.head.parameters())
            + list(self.discriminator.parameters()),
            lr=lr,
            weight_decay=weight_decay,
        )

        self.history: Dict[str, List[float]] = {
            "cls_loss": [], "domain_loss": [], "total_loss": [], "alpha": [],
        }

    def train_step(
        self,
        src_batches: List[Tuple[torch.Tensor, torch.Tensor]],
        tgt_batch:   Tuple[torch.Tensor, torch.Tensor],
        p:           float,
    ) -> Dict[str, float]:
        self.backbone.train()
        self.head.train()
        self.discriminator.train()
        _freeze_bn(self.backbone)

        alpha = grl_alpha(p, self.grl_max_alpha)

        src_imgs   = torch.cat([b[0] for b in src_batches], dim=0).to(self.device)
        src_labels = torch.cat([b[1] for b in src_batches], dim=0).to(self.device)
        tgt_imgs   = tgt_batch[0].to(self.device)
        n_src      = src_imgs.size(0)
        n_tgt      = tgt_imgs.size(0)

        # ── Forward: all images through backbone + head ───────────────────
        all_imgs   = torch.cat([src_imgs, tgt_imgs], dim=0)
        all_feats  = self.backbone(all_imgs)        # (N, 512) — in graph
        all_logits = self.head(all_feats)           # (N, 7)   — in graph

        # ── Classification loss (source examples only) ────────────────────
        cls_loss = F.cross_entropy(all_logits[:n_src], src_labels)

        # ── CDAN input: outer product f ⊗ p (no detach) ──────────────────
        g = cdan_input(all_feats, all_logits)       # (N, 3584) — in graph

        # ── Domain loss through GRL ───────────────────────────────────────
        all_feats_norm = F.normalize(all_feats, dim=1)           # ← add this
        g              = cdan_input(all_feats_norm, all_logits)  # ← change all_feats to all_feats_norm
        grl_g         = self.grl(g, alpha)
        domain_logits = self.discriminator(grl_g)   # (N, 2)

        domain_labels = torch.cat([
            torch.zeros(n_src, dtype=torch.long),
            torch.ones(n_tgt,  dtype=torch.long),
        ]).to(self.device)

        domain_loss = F.cross_entropy(domain_logits, domain_labels)
        total       = cls_loss + self.lambda_domain * domain_loss

        # ── Backward ─────────────────────────────────────────────────────
        self.optimizer.zero_grad()
        total.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.backbone.parameters())
            + list(self.head.parameters())
            + list(self.discriminator.parameters()),
            max_norm=1.0,
        )
        self.optimizer.step()

        return {
            "cls_loss":    cls_loss.item(),
            "domain_loss": domain_loss.item(),
            "total_loss":  total.item(),
            "alpha":       alpha,
        }

    def state_dict(self) -> Dict:
        return {
            "backbone":      self.backbone.state_dict(),
            "head":          self.head.state_dict(),
            "discriminator": self.discriminator.state_dict(),
        }

    def load_state_dict(self, sd: Dict) -> None:
        self.backbone.load_state_dict(sd["backbone"])
        self.head.load_state_dict(sd["head"])
        if "discriminator" in sd:
            self.discriminator.load_state_dict(sd["discriminator"])
