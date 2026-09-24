"""
models/domain_discriminator.py
===============================
Gradient Reversal Layer (GRL) and Domain Discriminator for DANN and CDAN.

Architecture (as specified in the assignment)
---------------------------------------------
  Linear(in_dim, 256) → ReLU → Dropout(0.5) → Linear(256, 2)

GRL schedule (DANN / CDAN)
---------------------------
  alpha(p) = 2 / (1 + exp(-10 * p)) - 1,   p in [0, 1] is training progress.
  The GRL multiplies the incoming gradient by -alpha in the backward pass,
  so the feature extractor learns to confuse the discriminator while the
  discriminator still tries to distinguish source from target.

CDAN input dimension
---------------------
  g(x) = vec(f ⊗ p)   where f: (B, 512), p: (B, 7)
  Outer product: (B, 512, 7) → flattened to (B, 3584).
  The discriminator is instantiated with in_dim = feat_dim * num_classes = 3584.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─── Gradient Reversal Layer ──────────────────────────────────────────────────

class _GRLFunction(torch.autograd.Function):
    """Autograd function that acts as identity forward / gradient negation backward."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, alpha: float) -> torch.Tensor:   # type: ignore[override]
        ctx.alpha = alpha
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):                        # type: ignore[override]
        return -ctx.alpha * grad_output, None


class GradientReversal(nn.Module):
    """
    Stateless GRL.  Pass the current `alpha` at each forward call.

    Usage::
        grl = GradientReversal()
        reversed_feats = grl(feats, alpha)
    """

    def forward(self, x: torch.Tensor, alpha: float) -> torch.Tensor:
        return _GRLFunction.apply(x, alpha)


def grl_alpha(p: float, max_alpha: float = 1.0) -> float:
    """
    Annealed GRL alpha from the DANN paper.

    Parameters
    ----------
    p:         Training progress in [0, 1].
    max_alpha: Maximum alpha value (1.0 for the main comparison,
               {0.25, 0.5, 1.0} for the controlled study).

    Returns
    -------
    float: alpha = max_alpha * (2 / (1 + exp(-10 * p)) - 1)
    """
    return max_alpha * (2.0 / (1.0 + math.exp(-10.0 * p)) - 1.0)


# ─── Domain Discriminator ─────────────────────────────────────────────────────

class DomainDiscriminator(nn.Module):
    """
    Binary domain discriminator (source vs target).

    Architecture::
        Linear(in_dim, hidden) → ReLU → Dropout(dropout) → Linear(hidden, 2)

    For DANN  : in_dim = feat_dim  (512)
    For CDAN  : in_dim = feat_dim * num_classes  (3584)

    Domain labels convention: source = 0, target = 1.
    """

    def __init__(
        self,
        in_dim:  int   = 512,
        hidden:  int   = 256,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden, 2),
        )
        # Xavier init for linear layers
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x: (B, in_dim) – gradient-reversed features from GRL.

        Returns
        -------
        logits: (B, 2)  [source-logit, target-logit]
        """
        return self.net(x)


# ─── CDAN input construction ─────────────────────────────────────────────────

def cdan_input(
    features: torch.Tensor,
    logits:   torch.Tensor,
) -> torch.Tensor:
    """
    Build the CDAN discriminator input: vec(f ⊗ p).

    As required by the assignment:
      * f and p are NOT detached (remain in the computational graph).
      * No entropy conditioning is applied.

    Parameters
    ----------
    features: (B, D)   backbone feature, D = 512
    logits:   (B, C)   classifier logits, C = 7

    Returns
    -------
    g: (B, D*C)  flattened outer product, D*C = 3584
    """
    probs = F.softmax(logits, dim=1)             # (B, C)
    # Outer product per sample: (B, D, 1) × (B, 1, C) → (B, D, C)
    outer = torch.bmm(features.unsqueeze(2), probs.unsqueeze(1))
    return outer.reshape(features.size(0), -1)   # (B, D*C)
