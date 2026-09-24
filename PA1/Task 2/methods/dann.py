"""
methods/dann.py
===============
Step 3: DANN – Adversarial domain adaptation with Gradient Reversal.

Architecture
------------
* Domain discriminator:  Linear(512, 256) → ReLU → Dropout(0.5) → Linear(256, 2)
* GRL between backbone features and discriminator.
* alpha(p) = 2 / (1 + exp(-10p)) - 1,  p ∈ [0,1] = training progress.

Loss (unit weight for domain loss)
-----------------------------------
    L_total = L_cls + L_domain

    L_cls    : source examples only → backbone + head.
    L_domain : source + target → GRL → discriminator.
               Normal gradient reaches the discriminator (it learns to classify).
               Reversed gradient (scaled by alpha) reaches the backbone (it learns
               to confuse the discriminator).

Domain labels: source = 0, target = 1  (consistent throughout).
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

from models.domain_discriminator import DomainDiscriminator, GradientReversal, grl_alpha
from .source_only import _freeze_bn


class DANN:
    """
    DANN training step.

    Parameters
    ----------
    backbone, head, device, lr, weight_decay:
        Standard arguments.
    lambda_domain:
        Domain loss coefficient (1.0 per assignment spec).
    grl_max_alpha:
        Maximum GRL alpha (1.0 for main comparison; {0.25, 0.5, 1.0} for study).
    disc_hidden:
        Discriminator hidden width (256).
    disc_dropout:
        Discriminator dropout rate (0.5).
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
    ) -> None:
        self.backbone      = backbone.to(device)
        self.head          = head.to(device)
        self.device        = device
        self.lambda_domain = lambda_domain
        self.grl_max_alpha = grl_max_alpha

        # ── Domain discriminator + GRL ────────────────────────────────────
        self.discriminator = DomainDiscriminator(
            in_dim=feat_dim, hidden=disc_hidden, dropout=disc_dropout,
        ).to(device)
        self.grl = GradientReversal()

        # ── Optimizer: all parameters in one group ────────────────────────
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
        """
        Parameters
        ----------
        p: training progress in [0, 1].
        """
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

        # ── Forward: all images through backbone in one pass ──────────────
        all_imgs  = torch.cat([src_imgs, tgt_imgs], dim=0)    # (n_src+n_tgt, 3, H, W)
        all_feats = self.backbone(all_imgs)                    # (n_src+n_tgt, 512)

        # ── Classification loss (source only) ─────────────────────────────
        src_logits = self.head(all_feats[:n_src])
        cls_loss   = F.cross_entropy(src_logits, src_labels)

        # ── Domain loss (source + target through GRL) ─────────────────────
        all_feats_norm = F.normalize(all_feats, dim=1) 
        grl_feats      = self.grl(all_feats_norm, alpha)
        domain_logits  = self.discriminator(grl_feats)          # (n_src+n_tgt, 2)

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
